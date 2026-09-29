"""Засев пула кандидатов на разметку из OpenAlex.

Что делает: берёт заголовки статей по шести областям датасета в двух временных окнах,
майнит из них n-граммы и раскладывает по четырём стратам на основе контраста частот.
Результат — ml/datasets/candidates_pool.csv.

Зачем так, а не через ключевые слова OpenAlex: поле keywords содержит устаревшие
концепты Wikidata («Key (lock)», «Enhanced Data Rates for GSM Evolution»), непригодные
как названия технологий. Заголовки свежих статей дают реальные формулировки.

ВАЖНО, два ограничения, о которых должен знать аннотатор:

1. Выход скрипта — это ПОДСКАЗКИ, а не названия технологий. Человек переписывает фразу
   в нормальное название в стиле gold.csv. «federated learning edge devices» — подсказка,
   «Федеративное обучение на edge-устройствах в промышленных сетях» — название.

2. Страта — это НЕ метка. Она нужна координатору для балансировки квот по типам
   негативов. Чтобы страта не якорила аннотатора, раздавайте пул с ключом --blind.

Настройка вежливого пула OpenAlex:
    export OPENALEX_MAILTO="почта-команды@example.org"
Без него запросы идут в общий пул и часто получают HTTP 429.

Запуск:
    python ml/scripts/seed_candidates.py                    # все области
    python ml/scripts/seed_candidates.py --domain Финтех    # одна область
    python ml/scripts/seed_candidates.py --blind            # без страты в выдаче
    python ml/scripts/seed_candidates.py --offline          # только из кэша, без сети
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "datasets" / "candidates_pool.csv"
CACHE = ROOT / "ml" / ".cache" / "openalex"

API = "https://api.openalex.org"
USER_AGENT = "weak-signal-radar/0.1 (hackathon research prototype)"

# Окна сравнения. Свежее — «что происходит сейчас», базовое — «что было три года назад».
RECENT_FROM, RECENT_TO = "2025-06-01", "2026-09-01"
BASE_FROM, BASE_TO = "2022-01-01", "2023-06-30"

# Области датасета заказчика и поисковые выражения к ним.
DOMAIN_QUERIES: dict[str, tuple[str, ...]] = {
    "Индустриальный ИИ": (
        "industrial AI manufacturing process control",
        "predictive maintenance machine learning industry",
        "digital twin industrial optimization",
    ),
    "Роботы": (
        "robot actuator manipulation hardware",
        "humanoid robot locomotion control",
        "soft robotics materials actuation",
    ),
    "Инфраструктура ИИ": (
        "AI accelerator datacenter interconnect",
        "large language model serving infrastructure",
        "memory bandwidth training cluster",
    ),
    "Финтех": (
        "payment infrastructure settlement technology",
        "credit risk model alternative data",
        "digital currency financial market infrastructure",
    ),
    "Защита ИИ": (
        "AI agent security identity",
        "machine learning model attack defense",
        "LLM guardrails adversarial robustness",
    ),
    "Edge": (
        "edge AI inference accelerator",
        "on-device machine learning efficiency",
        "tinyML embedded neural network",
    ),
}

STOPWORDS = set("""
the a an of for with and or in on to by using via based from at as is are was were be been
new novel toward towards study analysis approach method methods system systems model models
framework review survey paper we its their this that these those our can it into over under
between among through during more most less than not but also case study results evaluation
performance application applications design development research use used using high low
against role age era impact potential future current modern various different several
""".split())

# Слишком общие фразы: проходят частотный фильтр, но технологиями не являются.
GENERIC = {
    "machine learning", "deep learning", "neural network", "neural networks",
    "artificial intelligence", "large language", "language model", "language models",
    "large language model", "large language models", "computer vision", "data driven",
    "state art", "real time", "case study", "generative ai", "agentic ai", "ai agents",
    "ai agent", "llm agents", "autonomous ai", "ai security", "ai adoption", "ai systems",
    "ai models", "ai model", "foundation models", "decision support", "threat detection",
    "attack vectors", "security operations", "regulatory compliance", "risk management",
    "digital transformation", "future directions", "open challenges", "research agenda",
}

# Слова-обёртки из названий обзоров: «... : a comprehensive survey» и подобное.
BOILERPLATE = {
    "comprehensive", "survey", "review", "systematic", "overview", "taxonomy",
    "empirical", "preliminary", "recent", "advances", "advancements", "perspectives",
    "challenges", "opportunities", "directions", "insights", "lessons", "roadmap",
    "tutorial", "primer", "position", "vision", "towards", "exploring", "understanding",
    "investigating", "rethinking", "revisiting", "benchmarking", "comparative",
    "integrating", "leveraging", "enhancing", "improving", "enabling", "advancing",
    "defending", "addressing", "harnessing", "unlocking",
}

# Слова, которые сами по себе не делают фразу технологией. Фраза, состоящая
# только из них, отбрасывается: нужен хотя бы один содержательный термин.
WEAK_TOKENS = {
    "ai", "ml", "llm", "llms", "model", "models", "data", "system", "systems",
    "security", "secure", "learning", "network", "networks", "based", "aware",
    "driven", "enabled", "assisted", "powered", "large", "small", "smart",
    "intelligent", "autonomous", "digital", "advanced", "efficient", "robust",
    "generative", "agentic", "agent", "agents", "framework", "platform", "solution",
    "application", "applications", "technology", "technologies", "management",
    "detection", "analysis", "control", "design", "development", "evaluation",
}

MIN_RECENT_COUNT = 6          # реже — отношение частот неустойчиво
MAX_PHRASE_WORDS = 4
MIN_PHRASE_WORDS = 2
PAGES_PER_QUERY = 3           # 200 записей на страницу
RATE_LIMIT_SLEEP = 2.5        # пауза между запросами, секунды


@dataclass(frozen=True)
class Candidate:
    phrase: str
    domain: str
    recent: int
    base: int
    growth: float
    stratum: str
    query: str


class OpenAlexError(RuntimeError):
    pass


def _cache_path(url: str) -> Path:
    return CACHE / f"{hashlib.sha1(url.encode()).hexdigest()}.json"


def fetch(path: str, *, offline: bool, tries: int = 8, **params) -> dict:
    """GET к OpenAlex с кэшем на диске, ретраями и вежливым пулом."""
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"

    cached = _cache_path(url)
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    if offline:
        raise OpenAlexError("режим --offline, а ответа нет в кэше")

    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read())
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise OpenAlexError(f"HTTP {exc.code} на {path}") from exc
            time.sleep(5 * (attempt + 1) + random.random())
        except Exception as exc:  # noqa: BLE001 — сетевые сбои любого рода
            last = exc
            time.sleep(5 * (attempt + 1) + random.random())
    raise OpenAlexError(f"не удалось получить {path} за {tries} попыток: {last}")


def fetch_titles(query: str, date_from: str, date_to: str, *, offline: bool) -> list[str]:
    titles: list[str] = []
    cursor = "*"
    for _ in range(PAGES_PER_QUERY):
        payload = fetch(
            "/works",
            offline=offline,
            search=query,
            per_page=200,
            cursor=cursor,
            filter=f"from_publication_date:{date_from},to_publication_date:{date_to},type:article",
        )
        titles.extend(w["title"] for w in payload.get("results", []) if w.get("title"))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(RATE_LIMIT_SLEEP)
    return titles


# Единственный служебный предлог, допустимый внутри фразы: «internet of things»,
# «quality of service». Остальные означают, что n-грамма порвала фразу заголовка
# по границе — «attacks in large language», «learning and deep learning».
INTERNAL_ALLOWED = {"of"}

# Хвосты-модификаторы: «machine learning-driven», «deep learning-based» — это не
# название технологии, а определение к следующему слову, которое в n-грамму не попало.
TRAILING_MODIFIERS = ("-based", "-driven", "-aware", "-enabled", "-assisted", "-powered")


def _is_usable(gram: list[str]) -> bool:
    """Фраза годится, если она не обёртка обзора и несёт хотя бы один содержательный термин."""
    if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
        return False
    if any(w in STOPWORDS and w not in INTERNAL_ALLOWED for w in gram[1:-1]):
        return False
    if gram[-1].endswith(TRAILING_MODIFIERS):
        return False
    if any(w in BOILERPLATE for w in gram):
        return False
    if all(w in WEAK_TOKENS for w in gram):
        return False
    return " ".join(gram) not in GENERIC


def extract_phrases(titles: list[str]) -> collections.Counter[str]:
    """N-граммы из заголовков после лексической фильтрации."""
    counter: collections.Counter[str] = collections.Counter()
    for title in titles:
        words = re.findall(r"[a-z][a-z0-9\-]+", title.lower())
        for size in range(MIN_PHRASE_WORDS, MAX_PHRASE_WORDS + 1):
            for i in range(len(words) - size + 1):
                gram = words[i:i + size]
                if _is_usable(gram):
                    counter[" ".join(gram)] += 1
    return counter


def collapse_subphrases(counter: collections.Counter[str]) -> collections.Counter[str]:
    """Схлопывание вложенных фраз.

    «prompt injection» и «prompt injection attacks» с близкими частотами — одно и то же
    наблюдение. Оставляем более длинную формулировку: она информативнее для аннотатора.
    """
    phrases = sorted(counter, key=lambda p: (-len(p.split()), -counter[p]))
    kept: dict[str, int] = {}
    for phrase in phrases:
        absorbed = False
        for longer in kept:
            if phrase in longer and counter[longer] >= 0.6 * counter[phrase]:
                absorbed = True
                break
        if not absorbed:
            kept[phrase] = counter[phrase]
    return collections.Counter(kept)


def classify(recent: int, base: int, growth: float, recent_median: int) -> str:
    """Страта задаёт, какой тип примера здесь вероятнее. Это подсказка, не метка.

    rising_rare        — редко и быстро растёт: вероятные кандидаты в слабые сигналы
    high_volume_rising — много и растёт: вероятные N3d (растёт, но уже массовое)
    established_flat   — много и стабильно: вероятные N1 (зрелые)
    stagnant           — мало и без движения: вероятные N3e (исследование без движения)
    """
    voluminous = recent > recent_median
    if growth >= 2.0:
        return "high_volume_rising" if voluminous else "rising_rare"
    if growth <= 1.2:
        return "established_flat" if voluminous else "stagnant"
    return "established_flat" if voluminous else "rising_rare"


def collect_domain(domain: str, queries: tuple[str, ...], *, offline: bool) -> list[Candidate]:
    recent_counter: collections.Counter[str] = collections.Counter()
    base_counter: collections.Counter[str] = collections.Counter()
    phrase_query: dict[str, str] = {}

    for query in queries:
        try:
            recent_titles = fetch_titles(query, RECENT_FROM, RECENT_TO, offline=offline)
            base_titles = fetch_titles(query, BASE_FROM, BASE_TO, offline=offline)
        except OpenAlexError as exc:
            print(f"    [!] запрос «{query}» пропущен: {exc}", file=sys.stderr)
            continue

        recent_phrases = extract_phrases(recent_titles)
        recent_counter.update(recent_phrases)
        base_counter.update(extract_phrases(base_titles))
        for phrase in recent_phrases:
            phrase_query.setdefault(phrase, query)
        print(f"    «{query}»: {len(recent_titles)} свежих, {len(base_titles)} базовых заголовков")

    recent_counter = collapse_subphrases(recent_counter)
    frequent = {p: n for p, n in recent_counter.items() if n >= MIN_RECENT_COUNT}
    if not frequent:
        return []
    counts = sorted(frequent.values())
    median = counts[len(counts) // 2]

    candidates = []
    for phrase, recent in frequent.items():
        base = base_counter.get(phrase, 0)
        growth = (recent + 1) / (base + 1)
        candidates.append(Candidate(
            phrase=phrase,
            domain=domain,
            recent=recent,
            base=base,
            growth=growth,
            stratum=classify(recent, base, growth, median),
            query=phrase_query.get(phrase, ""),
        ))
    return candidates


def select(candidates: list[Candidate], per_stratum: int) -> list[Candidate]:
    """Квоты по стратам: пул должен содержать и позитивы, и все типы негативов."""
    by_stratum: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate.stratum].append(candidate)

    selected: list[Candidate] = []
    for stratum, items in by_stratum.items():
        key = (lambda c: -c.growth) if stratum in ("rising_rare", "high_volume_rising") else (lambda c: -c.recent)
        selected.extend(sorted(items, key=key)[:per_stratum])
    return selected


def merge_with_existing(rows: list[dict[str, str]], touched: set[str]) -> list[dict[str, str]]:
    """Дополнить пул, а не перезаписать его.

    Запуск с --domain обновляет только указанные области. Строки остальных областей
    сохраняются, вместе с уже проставленными assigned_to и status: терять работу
    аннотаторов из-за повторного засева нельзя.
    """
    if not OUT.exists():
        return rows

    with OUT.open(encoding="utf-8") as fh:
        existing = [r for r in csv.DictReader(fh) if r.get("id")]

    kept = [r for r in existing if r.get("domain") not in touched]
    # В обновляемых областях сохраняем назначение и статус уже розданных строк.
    previous = {r["id"]: r for r in existing if r.get("domain") in touched}
    for row in rows:
        old = previous.get(row["id"])
        if old:
            row["assigned_to"] = old.get("assigned_to", "")
            row["status"] = old.get("status", "new")
    return kept + rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", action="append", help="ограничить одной областью (можно повторять)")
    parser.add_argument("--per-stratum", type=int, default=15, help="сколько кандидатов на страту и область")
    parser.add_argument("--blind", action="store_true", help="не выводить страту — защита от якорения аннотатора")
    parser.add_argument("--offline", action="store_true", help="работать только из кэша")
    args = parser.parse_args()

    domains = {d: q for d, q in DOMAIN_QUERIES.items() if not args.domain or d in args.domain}
    if not domains:
        print(f"неизвестная область. Доступны: {', '.join(DOMAIN_QUERIES)}", file=sys.stderr)
        return 1

    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    print(f"вежливый пул OpenAlex: {'да, ' + mailto if mailto else 'НЕТ — ожидайте HTTP 429 и долгие ретраи'}\n")

    rows: list[dict[str, str]] = []
    for domain, queries in domains.items():
        print(f"[{domain}]")
        candidates = collect_domain(domain, queries, offline=args.offline)
        if not candidates:
            print("    ничего не собрано\n")
            continue
        chosen = select(candidates, args.per_stratum)
        for i, candidate in enumerate(chosen, start=1):
            slug = re.sub(r"[^a-z0-9]+", "-", candidate.phrase)[:40].strip("-")
            note = (f"свежих {candidate.recent}, базовых {candidate.base}, "
                    f"рост x{candidate.growth:.1f}")
            if not args.blind:
                note = f"{candidate.stratum}; {note}"
            rows.append({
                "id": f"cand-{abs(hash((domain, slug))) % 10**6:06d}",
                "name": candidate.phrase,
                "domain": domain,
                "seed_source": "openalex:titles",
                "seed_note": note,
                "assigned_to": "",
                "status": "new",
            })
        counts = collections.Counter(c.stratum for c in chosen)
        print(f"    отобрано {len(chosen)}: {dict(counts)}\n")

    if not rows:
        print("пул пуст — ничего не записано", file=sys.stderr)
        return 1

    merged = merge_with_existing(rows, touched=set(domains))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)

    print(f"собрано в этот заход: {len(rows)}; всего в пуле: {len(merged)} -> {OUT.relative_to(ROOT)}")
    print("Напоминание: это подсказки, а не названия. Человек переписывает фразу в")
    print("название технологии в стиле gold.csv и ставит метку по протоколу.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
