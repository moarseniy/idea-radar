"""Ретро-строки обучающей выборки из исторического прогона радара (машина времени).

Протокол тот же, что у ml/datasets/labeled.csv, но на дату среза T прогона:
  1. Кандидаты — все проверенные кандидаты прогона радара на T. Его документы уже
     отсечены по дате (SourceFetcher), поэтому судья видит только то, что было известно в T.
  2. Позитив — судья стадии (stage_judge.ask, та же модель, что у основной разметки)
     не признал кандидата зрелым: читает документы кандидата из прогона (их 1–2, по ним
     одним судья не уверен) и заголовки Google News за год до T — те же, что медиа-признаки.
     Порога уверенности нет, как и в исходной разметке: судья — вето на стадию 5.
     Вето зрелости — ещё и статья Википедии В РЕДАКЦИИ НА T: стадия 5 по ней → негатив;
     статья, созданная больше чем за OLD_YEARS до T, — тоже (правило исходной разметки,
     ml/datasets/manifest.md). Статьи часто называются не как термин ("C4.5 algorithm"),
     а поиск Википедии подсовывает общего родителя ("Anomaly detection" для квантового
     детектора аномалий), поэтому давний метод ловит и вопрос LLM «с какого года подход
     общеизвестен» — это знание о прошлом T, не о будущем; порог тот же, OLD_YEARS.
     Текущая статья для этого не годится: она описывает то, что стало потом.
  3. Трудные негативы — зрелые «родители» позитивов (contrastive_negatives.parents),
     стадия 5 по статье Википедии в редакции на T.
  4. Признаки — featurize(name, domain, T), тот же модуль и схема, что у training.csv.
  5. Исход — признаки позитивов на сегодня: только для отчёта «что стало потом», в
     метку и в признаки не входит.
  6. Технологии, уже входящие в обучение или золото, отбрасываются (по термину).
  7. Ярус позитива — по «движению» на T, как у исходных позитивов из отраслевых новостей:
     silver — не меньше NEWS_SILVER заголовков за год до T (на пилоте выросли к 2026 году 75 %);
     bronze — 1–2 заголовка или денежное событие на T (выросли 50 %), в обучение с весом 0,3;
     без новостей и денег — отбрасывается как неопределённый (выросли 33 %). Негативы — silver.

Запуск: python ml/scripts/retro_dataset.py --run-dir DIR --domain Финтех [--out CSV]
DIR — папка прогона с run.json и radar.db (STORAGE_DIR сервиса радара).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

# Признаки по всем источникам, включая новые денежные (по умолчанию выключены в реестре):
# так же будут пересчитаны и существующие строки обучения.
os.environ.setdefault("RADAR_EXTRA_CONNECTORS", "all")
os.environ.setdefault("FEATURE_LLM_TRIES", "2")  # сеть до OpenRouter бывает нестабильной
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_training_set as bts
import contrastive_negatives
import stage_judge

from app.ml.features.evidence import media
from app.ml.features.extract import derive_query, featurize, llm_json
from app.ml.features.schema import FEATURE_NAMES

ROOT = bts.ROOT
CACHE = ROOT / "ml" / ".cache" / "retro"
OUT = ROOT / "ml" / "datasets" / "retro" / "retro_rows.csv"
NEWS_LIMIT = 12
NEWS_SILVER = 3
MONEY_FIELDS = ("funding_round_count_24m", "public_grants_log_36m", "funding_log_24m")
OLD_YEARS = 7
ESTABLISHED_PROMPT = """Технология или метод: «{term}» ({name}).
С какого года этот конкретный подход общеизвестен в профессиональном сообществе (учебники,
стандартные библиотеки, массовые применения)? Если это новое направление, которое ещё не
устоялось, ответь null. Не путай подход с его более общей областью: для «квантового
детектора аномалий» важен год квантового детектора, а не детекции аномалий вообще.
Ответ строго JSON: {{"established_year": 1995, "why": "коротко"}}"""
OUTCOME_FIELDS = ["papers_log_3y", "paper_growth_2y", "funding_log_24m", "media_volume_normalized_12m", "llm_stage"]
WIKI_UA = {"User-Agent": f"weak-signal-radar/0.1 (hackathon research; {os.getenv('OPENALEX_MAILTO', '')})"}


def _cached(kind: str, key: str, fn):
    path = CACHE / kind / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def _wiki(params: dict) -> dict:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=WIKI_UA), timeout=30) as response:
        return json.loads(response.read())


def wiki_text_at(title: str, as_of: date) -> dict | None:
    """Текст статьи английской Википедии в редакции на as_of; None — статьи тогда не было."""
    def load():
        pages = _wiki({"action": "query", "prop": "revisions", "titles": title, "redirects": 1, "rvlimit": 1,
                       "rvstart": f"{as_of.isoformat()}T23:59:59Z", "rvdir": "older", "rvprop": "ids|timestamp"})
        page = next(iter(pages["query"]["pages"].values()))
        revisions = page.get("revisions") or []
        if "missing" in page or not revisions:
            return None
        first = _wiki({"action": "query", "prop": "revisions", "pageids": page["pageid"], "rvlimit": 1,
                       "rvdir": "newer", "rvprop": "timestamp"})
        created = next(iter(first["query"]["pages"].values()))["revisions"][0]["timestamp"]
        parsed = _wiki({"action": "parse", "oldid": revisions[0]["revid"], "prop": "text", "formatversion": 2})
        raw = parsed.get("parse", {}).get("text", "")
        raw = re.sub(r"(?is)<(style|script|table|sup)[^>]*>.*?</\1>", " ", raw)
        paragraphs = re.findall(r"(?is)<p[^>]*>(.*?)</p>", raw)
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", " ".join(paragraphs)))).strip()
        if len(text) < 500 or "may refer to" in text[:300]:
            return None
        return {"title": page["title"], "revision_at": revisions[0]["timestamp"], "created_at": created,
                "text": text[:6000]}
    return _cached("wiki_at2", f"{title}|{as_of}", load)


def judge(name: str, articles: list[dict], as_of: date) -> dict:
    key = name + "|" + as_of.isoformat() + "|" + "|".join(a["title"] for a in articles)
    return _cached("judge", key, lambda: stage_judge.ask(name, articles))


def wiki_judge(term: str, as_of: date) -> dict | None:
    article = wiki_text_at(term, as_of)
    if not article:
        return None
    verdict = judge(term, [{"title": f"Wikipedia: {article['title']} ({article['revision_at'][:10]})",
                            "raw_content": article["text"]}], as_of)
    age = (as_of - date.fromisoformat(article["created_at"][:10])).days / 365.25
    return {**verdict, "wiki_title": article["title"], "revision_at": article["revision_at"], "age_years": age}


def run_sources(run_dir: Path, run_id: str) -> dict[str, dict]:
    with sqlite3.connect(run_dir / "radar.db") as conn:
        rows = conn.execute("SELECT id, payload FROM radar_sources WHERE run_id=?", (run_id,)).fetchall()
    return {sid: json.loads(payload) for sid, payload in rows}


def known_terms() -> set[str]:
    """Термины обучения, золота и негативов валидации: ретро-строка с таким термином — дубль
    или утечка в валидацию."""
    terms = set()
    compare = ROOT / "ml/datasets/model_compare/gpt-5.6-luna"
    for path in (compare / "training.csv", ROOT / "ml/datasets/gold_features.csv", compare / "validation.csv", OUT):
        if path.exists():
            for row in csv.DictReader(path.open(encoding="utf-8")):
                for term in [row.get("search_term", ""), *(row.get("search_aliases") or "").split(";")]:
                    if term.strip():
                        terms.add(bts.norm_term(term))
    return terms


def features_row(row: dict, as_of: date) -> dict:
    query = derive_query(row["name_ru"], row["domain"])
    if not query["term"]:
        raise ValueError("нет поискового термина")
    result = featurize(row["name_ru"], row["domain"], as_of, query=query)
    return {**row, **result, "search_term": query["term"], "search_aliases": "; ".join(query["aliases"]),
            "technology_group_id": "tech_" + hashlib.sha256(bts.norm_term(query["term"]).encode()).hexdigest()[:12]}


def news_at(name: str, domain: str, as_of: date) -> list[dict]:
    """Свежие заголовки Google News за год до as_of по терминам featurize (кэш медиа-признаков)."""
    query = derive_query(name, domain)
    if not query["term"]:
        return []
    feed = media([query["term"], *query["aliases"]], as_of)
    items = [*feed.windows.get("0-6m", []), *feed.windows.get("6-12m", [])]
    items = sorted({i["title"]: i for i in items}.values(), key=lambda i: i["date"], reverse=True)
    return [{"title": i["title"], "published_date": i["date"], "raw_content": f"{i['title']}. {i['publisher']}"}
            for i in items[:NEWS_LIMIT]]


def established_year(term: str, name: str) -> int | None:
    data = llm_json(ESTABLISHED_PROMPT.format(term=term, name=name))
    try:
        year = int(data.get("established_year"))
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= date.today().year else None  # noqa: DTZ011


def label_candidate(assessment: dict, sources: dict[str, dict], domain: str, as_of: date) -> dict:
    name = assessment["title"]
    docs = [sources[sid] for sid in assessment.get("source_ids", []) if sid in sources
            and sources[sid].get("status") == "read"]
    articles = [{"title": d["title"], "published_date": d.get("published_at") or "",
                 "raw_content": d.get("text", "")} for d in docs[:4]]
    news = news_at(name, domain, as_of)
    articles += news
    base = {"id": "retro_" + hashlib.sha256(f"{name}|{as_of}".encode()).hexdigest()[:12], "source": "retro",
            "source_file": "", "name_ru": name, "name_original": name, "domain": domain,
            "evidence_urls": " ".join(d["url"] for d in docs[:6]), "stage_ref": "", "trend_ref": ""}
    if not articles:
        return {**base, "decision": "drop", "reason": "нет документов на дату среза"}
    verdict = judge(name, articles, as_of)
    base.update(stage_ref=verdict["stage"], trend_ref=verdict["trend"], judge_confidence=verdict["confidence"],
                judge_articles=len(articles), news_at_T=len(news))
    term = (assessment.get("model") or {}).get("search_term") or name
    wiki = wiki_judge(term, as_of)
    established = established_year(term, name)
    old = bool(wiki and wiki["age_years"] > OLD_YEARS) or bool(established and as_of.year - established > OLD_YEARS)
    if verdict["stage"] == 5 or (wiki and wiki["stage"] == 5) or old:
        why = ("статьи" if verdict["stage"] == 5 else f"Википедия на {wiki['revision_at'][:10]}"
               if wiki and wiki["stage"] == 5 else f"статье Википедии {wiki['age_years']:.0f} лет"
               if wiki and wiki["age_years"] > OLD_YEARS else f"общеизвестен с {established}")
        return {**base, "label": "0", "negative_type": "N1_mature_at_T", "decision": "keep",
                "reason": f"зрелая на T ({why})"}
    return {**base, "label": "1", "negative_type": "", "decision": "keep",
            "reason": f"стадия {verdict['stage']} на T: {verdict['stage_quote'][:120]}"}


def parent_negatives(positive: dict, as_of: date) -> list[dict]:
    out = []
    for parent in _cached("parents", positive["name_ru"] + "|" + positive["domain"],
                          lambda: contrastive_negatives.parents({"name": positive["name_ru"],
                                                                 "domain": positive["domain"]}, k=1)):
        wiki = wiki_judge(parent["wiki_title"], as_of)
        if wiki and wiki["stage"] == 5:
            out.append({"id": "retro_" + hashlib.sha256(f"{parent['name_ru']}|{as_of}".encode()).hexdigest()[:12],
                        "source": "retro", "source_file": "", "label": "0", "name_ru": parent["name_ru"],
                        "name_original": parent["wiki_title"], "domain": positive["domain"],
                        "negative_type": "N1_parent_at_T", "stage_ref": 5, "trend_ref": wiki["trend"],
                        "evidence_urls": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(wiki["wiki_title"]),
                        "decision": "keep", "reason": f"зрелая пара к «{positive['name_ru'][:60]}»"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--domain", required=True, help="одна из областей обучающей выборки")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-outcome", action="store_true")
    args = parser.parse_args()
    bts.load_dotenv()

    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    as_of = date.fromisoformat(run["as_of"])
    if as_of >= date.today():  # noqa: DTZ011
        raise SystemExit("нужен исторический прогон: дата оценки раньше сегодняшней")
    sources = run_sources(args.run_dir, run["id"])
    print(f"прогон {run['id'][:8]} на {as_of}: «{run['query']}», кандидатов {len(run['assessments'])}")

    with ThreadPoolExecutor(args.workers) as pool:
        decided = list(pool.map(lambda a: _retried(label_candidate, a, sources, args.domain, as_of,
                                                   fallback={"name_ru": a["title"], "decision": "drop"}),
                                run["assessments"]))
        positives = [r for r in decided if r.get("label") == "1"]
        parents = [n for batch in pool.map(lambda p: _retried(parent_negatives, p, as_of, fallback=[]), positives)
                   for n in batch]
    rows = [r for r in decided if r["decision"] == "keep"] + parents

    taken = known_terms()
    with ThreadPoolExecutor(args.workers) as pool:
        featured = list(pool.map(lambda r: _safe(features_row, r, as_of), rows))
    kept, removed, seen = [], [r for r in decided if r["decision"] == "drop"], set()
    for row in featured:
        key = bts.norm_term(row.get("search_term", ""))
        if row.get("error"):
            removed.append({**row, "reason": "ошибка признаков: " + row["error"]})
        elif key in taken or key in seen:
            removed.append({**row, "reason": "технология уже в обучении, золоте или ретро-выборке"})
        else:
            seen.add(key)
            tier = assign_tier(row)
            if tier == "drop":
                removed.append({**row, "reason": "позитив без движения на T: нет новостей и денег"})
                continue
            kept.append({**row, "tier": tier, "snapshot_date": as_of.isoformat(), "run_id": run["id"],
                         "run_query": run["query"]})

    if not args.no_outcome:  # что стало потом: позитивы на сегодня, только в отчёт
        today = date.today()  # noqa: DTZ011
        with ThreadPoolExecutor(args.workers) as pool:
            outcomes = list(pool.map(lambda r: _safe(lambda x, d: featurize(x["name_ru"], x["domain"], d), r, today)
                                     if r["label"] == "1" else {}, kept))
        for row, outcome in zip(kept, outcomes):
            for field in OUTCOME_FIELDS:
                row[f"outcome_{field}"] = outcome.get(field, "") if outcome and not outcome.get("error") else ""

    fields = bts.META + bts.SERVICE + FEATURE_NAMES + ["tier", "news_at_T", "money_at_T", "run_id", "run_query",
                                                       "reason", "judge_confidence", "judge_articles"] + \
        [f"outcome_{f}" for f in OUTCOME_FIELDS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with args.out.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(kept)
    removed_path = args.out.with_name(args.out.stem + "_removed.csv")
    with removed_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "name_ru", "search_term", "label", "reason", "run_id"],
                                extrasaction="ignore")
        if removed_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows({**r, "run_id": run["id"]} for r in removed)
    print(f"строк: {len(kept)} {dict(Counter((r['label'], r['negative_type']) for r in kept))}; "
          f"отброшено {len(removed)} {dict(Counter(r['reason'].split(' (')[0].split(':')[0] for r in removed))}")
    return 0


def assign_tier(row: dict) -> str:
    def number(value) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0
    row["money_at_T"] = int(any(number(row.get(field)) > 0 for field in MONEY_FIELDS))
    if row["label"] != "1":
        return "silver"
    news = int(row.get("news_at_T") or 0)
    if news >= NEWS_SILVER:
        return "silver"
    return "bronze" if news or row["money_at_T"] else "drop"


def _retried(fn, *args, fallback, tries: int = 3):
    """Повтор при сетевом сбое (ответы LLM кэшируются, повтор дешёвый). Не помогло —
    кандидат уходит в отброшенные с причиной, а не роняет разметку всей темы."""
    for attempt in range(tries):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"сбой {fn.__name__}: {str(exc)[:120]}", flush=True)
                if isinstance(fallback, dict):
                    return {**fallback, "reason": "сетевой сбой: " + str(exc)[:120]}
                return fallback
            time.sleep(20 * (attempt + 1))


def _safe(fn, row: dict, *args) -> dict:
    try:
        return fn(row, *args)
    except Exception as exc:  # noqa: BLE001 — строка уходит в отброшенные с причиной
        return {**row, "error": str(exc)[:160]}


if __name__ == "__main__":
    raise SystemExit(main())
