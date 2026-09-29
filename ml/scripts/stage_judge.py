"""Стадия внедрения и тренд по тексту статей — рубрика заказчика, прочитанная моделью.

Почему модель, а не научные базы. Проверка на золоте показала: у методологов слабый
сигнал — это новое движение (раунд, пилот, новое поколение продукта), в том числе в
старой научной области. Нейроморфные процессоры и роевая робототехника существуют с
1990-х, но в их списке есть. Научные базы измеряют возраст идеи; стадию внедрения
видно только в отраслевых медиа. Поэтому стадию читает модель по статьям, а научные
показатели остаются признаками классификатора, а не судьёй зрелости.

Модель не решает, слабый ли это сигнал. Она извлекает стадию (1-5) и тренд (1-3) с
цитатами, метку назначает правило заказчика: стадия <= 4 и тренд >= 2.

Прежде чем доверять, проверяем на золоте:
    python ml/scripts/stage_judge.py --validate-gold 50
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402 — .env, запросы к Tavily и OpenRouter
from autolabel import weighted_kappa  # noqa: E402

ROOT = collect.ROOT
GOLD = ROOT / "ml" / "datasets" / "gold.csv"
GOLD_TERMS = ROOT / "ml" / ".cache" / "gold_terms.json"
CACHE = ROOT / "ml" / ".cache" / "stage"
REPORT = ROOT / "ml" / "reports" / "stage_judge_validation.md"
MODEL = os.getenv("STAGE_MODEL", collect.MODEL)

PROMPT = """Ты аналитик технологической разведки. Определи по статьям ниже СТАДИЮ ВНЕДРЕНИЯ
и ТРЕНД для технологии:

«{name}»

Стадия — это состояние внедрения ИМЕННО ЭТОЙ технологии на дату самых свежих статей,
а не возраст научной области. Старое научное направление, в котором сейчас идут первые
пилоты нового поколения, — это стадия 3, а не 5.

Стадия:
1 — Концепция/Исследование: только научные работы и лабораторные результаты
2 — Прототип/PoC: работающий прототип или демонстрация, коммерческих клиентов нет
3 — Пилот: пилотные внедрения у заказчиков, контракты на испытания
4 — Раннее внедрение: первые коммерческие поставки, первые платящие клиенты
5 — Массовое внедрение: сформированный рынок, выраженные лидеры, отраслевые стандарты

Тренд упоминаний и активности (раунды, анонсы, новые игроки) за последние 12-18 месяцев:
1 — Стабильный: без заметного роста
2 — Растёт: заметный рост
3 — Растёт быстро: кратный рост, серия раундов, выход многих новых игроков

Правила:
- Опирайся только на статьи. Для каждого вывода дай короткую цитату и номер статьи.
- Если статьи не про эту технологию или сведений мало — confidence ниже 0.4.

Статьи:
{articles}

Ответ строго JSON без markdown:
{{"stage": 1, "stage_quote": "...", "stage_source": 1, "trend": 1, "trend_quote": "...",
  "trend_source": 1, "confidence": 0.0}}"""


def ask(name: str, articles: list[dict]) -> dict:
    listing = "\n\n".join(
        f"[{i}] {a.get('title', '')} ({(a.get('published_date') or 'дата неизвестна')[:16]})\n"
        f"{(a.get('raw_content') or a.get('content') or '')[:2000]}"
        for i, a in enumerate(articles, start=1))
    for _ in range(2):
        response = collect._post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": PROMPT.format(name=name, articles=listing)}]},
            {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content")
        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
            return {"stage": int(data["stage"]), "trend": int(data["trend"]),
                    "confidence": float(data.get("confidence", 0)),
                    "stage_quote": str(data.get("stage_quote", ""))[:300],
                    "trend_quote": str(data.get("trend_quote", ""))[:300]}
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
    raise RuntimeError("модель не вернула стадию")


def gold_articles(row: dict, term: str) -> list[dict]:
    """Один запрос Tavily на золотую запись, результат кэшируется: бюджет ограничен."""
    cached = CACHE / f"gold_{row['id']}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    company = (row.get("companies") or "").split(",")[0].split("(")[0].strip()
    query = f"{term} {company}".strip()
    collect.TAVILY_LIMIT.wait()
    payload = collect._post_json(
        "https://api.tavily.com/search",
        {"query": query, "topic": "general", "max_results": 6, "search_depth": "basic",
         "include_raw_content": True},
        {"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"}, timeout=40)
    articles = [a for a in payload.get("results", []) if collect.trusted(a)]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
    return articles


def validate_gold(n: int) -> int:
    gold = [r for r in csv.DictReader(GOLD.open(encoding="utf-8")) if r["stage"] and r["trend"]]
    terms = json.loads(GOLD_TERMS.read_text(encoding="utf-8")) if GOLD_TERMS.exists() else {}
    random.Random(42).shuffle(gold)
    sample = gold[:n]
    print(f"проверка на {len(sample)} золотых записях, модель {MODEL}, Tavily: до {len(sample)} запросов")

    def one(row):
        term = terms.get(row["id"]) or row["name"]
        try:
            articles = gold_articles(row, term)
            if not articles:
                return row, None, "статей не найдено"
            return row, ask(row["name"], articles), ""
        except Exception as exc:  # noqa: BLE001
            return row, None, str(exc)[:80]

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(one, sample))

    ok = [(r, p) for r, p, _ in results if p]
    failed = [(r["id"], why) for r, p, why in results if not p]
    stage_pairs = [(int(r["stage"]), p["stage"]) for r, p in ok]
    trend_pairs = [(int(r["trend"]), p["trend"]) for r, p in ok]
    label_ok = sum(1 for r, p in ok if p["stage"] <= 4 and p["trend"] >= 2)

    def acc(pairs):
        return (sum(a == b for a, b in pairs) / len(pairs), sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs))

    s_exact, s_near = acc(stage_pairs)
    t_exact, t_near = acc(trend_pairs)
    s_kappa = weighted_kappa(stage_pairs, [1, 2, 3, 4, 5])
    t_kappa = weighted_kappa(trend_pairs, [1, 2, 3])
    mature_calls = sum(1 for _, p in ok if p["stage"] == 5)
    confusion = collections.Counter(stage_pairs)

    lines = [
        "# Проверка определения стадии по медиа на золоте", "",
        f"Дата: {date.today().isoformat()}  ", f"Модель: `{MODEL}`  ",
        f"Записей: {len(sample)}, оценено: {len(ok)}, без оценки: {len(failed)}", "",
        "Стадия и тренд методологов модели не передавались.", "",
        "| Поле | Точно | ±1 | Каппа (взвеш.) |", "|---|---|---|---|",
        f"| Стадия | {s_exact:.0%} | {s_near:.0%} | {s_kappa:.3f} |",
        f"| Тренд | {t_exact:.0%} | {t_near:.0%} | {t_kappa:.3f} |", "",
        f"**Метка по правилу «стадия ≤ 4 и тренд ≥ 2» совпала с золотом: {label_ok}/{len(ok)} "
        f"({label_ok / len(ok):.0%}).**  ",
        f"Ложно названо зрелым (стадия 5): {mature_calls}/{len(ok)}.", "",
        "## Матрица стадий (строки — методологи, столбцы — модель)", "",
        "| | " + " | ".join(str(c) for c in range(1, 6)) + " |",
        "|---|" + "---|" * 5,
    ] + [f"| **{g}** | " + " | ".join(str(confusion.get((g, m), 0)) for m in range(1, 6)) + " |"
         for g in range(1, 5)]
    if failed:
        lines += ["", "## Без оценки", ""] + [f"- {i}: {why}" for i, why in failed]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with REPORT.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for r, p in ok:
            fh.write(json.dumps({"id": r["id"], "gold_stage": int(r["stage"]), "gold_trend": int(r["trend"]),
                                 **p}, ensure_ascii=False) + "\n")
    print("\n".join(lines))
    return 0


UA = {"User-Agent": "Mozilla/5.0 (compatible; weak-signal-radar/0.1; hackathon research)"}
LABELED = ROOT / "ml" / "datasets" / "labeled.csv"
STAGE_REJECTED = ROOT / "ml" / "datasets" / "stage_rejected.csv"


def fetch_text(url: str, limit: int = 6000) -> str:
    """Текст статьи по ссылке обычным запросом, без Tavily. Кэшируется."""
    cached = CACHE / "pages" / (re.sub(r"[^a-z0-9]+", "_", url.lower())[:120] + ".txt")
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as response:
            raw = response.read(2_000_000).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — недоступная страница не ошибка прогона
        raw = ""
    raw = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", raw)
    paragraphs = re.findall(r"(?is)<p[^>]*>(.*?)</p>", raw)
    text = " ".join(re.sub(r"<[^>]+>", " ", p) for p in paragraphs) or re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()[:limit]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def tavily_extract(urls: list[str]) -> dict[str, str]:
    """Текст страниц через Tavily extract — для сайтов, закрытых от обычного запроса.
    Стоит заметно дешевле поиска; результат кэшируется в тот же кэш страниц."""
    todo = [u for u in urls if not (CACHE / "pages" / (re.sub(r"[^a-z0-9]+", "_", u.lower())[:120] + ".txt")).exists()
            or len(fetch_text(u)) <= 300]
    if not todo:
        return {}
    payload = collect._post_json("https://api.tavily.com/extract", {"urls": todo[:20], "extract_depth": "basic"},
                                 {"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"}, timeout=90)
    out = {}
    for item in payload.get("results", []):
        text = re.sub(r"\s+", " ", item.get("raw_content") or "").strip()[:6000]
        if len(text) > 300:
            out[item["url"]] = text
            cached = CACHE / "pages" / (re.sub(r"[^a-z0-9]+", "_", item["url"].lower())[:120] + ".txt")
            cached.write_text(text, encoding="utf-8")
    return out


def wiki_stage(term: str) -> tuple[str, dict] | None:
    """Стадия технологии по её статье в Википедии — описание технологии в целом,
    а не одной компании. Нет статьи с таким названием — None. Кэшируется."""
    if not term:
        return None
    cached = CACHE / "wiki" / (re.sub(r"[^a-z0-9]+", "_", term.lower())[:80] + ".json")
    if cached.exists():
        blob = json.loads(cached.read_text(encoding="utf-8"))
        return (blob["title"], blob["judgement"]) if blob else None
    import urllib.parse
    import urllib.request
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "titles": term,
        "format": "json", "redirects": 1})
    wiki_ua = {"User-Agent": f"weak-signal-radar/0.1 (hackathon research; {os.getenv('OPENALEX_MAILTO', '')})"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=wiki_ua), timeout=30) as response:
        page = next(iter(json.loads(response.read())["query"]["pages"].values()))
    result = None
    extract = page.get("extract") or ""
    if "missing" not in page and len(extract) > 500 and "may refer to" not in extract[:300]:
        result = (page["title"], ask(term, [{"title": f"Wikipedia: {page['title']}", "raw_content": extract[:6000]}]))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"title": result[0], "judgement": result[1]} if result else None,
                                 ensure_ascii=False), encoding="utf-8")
    return result


def recheck_labeled(use_extract: bool = False) -> int:
    """Перепроверка стадии у собранных позитивов по их же источникам.

    Стадия 5 (массовое внедрение) — отбраковка: проверено на золоте, что модель не
    называет зрелыми настоящие слабые сигналы (0 из 49), а зрелые узнаёт (13 из 14).
    Тренд записывается, но метку не определяет: его модель читает плохо (каппа 0.2).
    """
    rows = list(csv.DictReader(LABELED.open(encoding="utf-8")))
    fields = list(rows[0])
    todo = [r for r in rows if r["label"] == "1" and not r["stage"]]
    print(f"позитивов без стадии: {len(todo)}")

    def one(row):
        urls = [u for u in row["evidence_urls"].split(";") if u.startswith("http")][:3]
        articles = [{"title": u, "raw_content": fetch_text(u)} for u in urls]
        articles = [a for a in articles if len(a["raw_content"]) > 300]
        if not articles and use_extract:
            extracted = tavily_extract(urls)
            articles = [{"title": u, "raw_content": text} for u, text in extracted.items()]
        if not articles:
            return row, None, "источники недоступны"
        try:
            return row, ask(row["name"], articles), ""
        except Exception as exc:  # noqa: BLE001
            return row, None, str(exc)[:60]

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(one, todo))

    rejected, judged, unavailable = [], 0, 0
    verdict_by_id = {}
    for row, judgement, why in results:
        if judgement is None:
            unavailable += 1
            continue
        judged += 1
        verdict_by_id[row["id"]] = judgement
        if judgement["stage"] == 5:
            rejected.append({"id": row["id"], "name": row["name"],
                             "reason": f"стадия 5 по источникам: {judgement['stage_quote'][:200]}"})

    drop = {r["id"] for r in rejected}
    out = []
    for row in rows:
        if row["id"] in drop:
            continue
        judgement = verdict_by_id.get(row["id"])
        if judgement:
            row["stage"], row["trend"] = judgement["stage"], judgement["trend"]
            row["rationale"] += (f" | стадия по источникам: {judgement['stage']} "
                                 f"(уверенность {judgement['confidence']:.1f}): {judgement['stage_quote'][:150]}")
        out.append(row)
    with LABELED.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    if rejected:
        with STAGE_REJECTED.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "name", "reason"])
            if fh.tell() == 0:
                writer.writeheader()
            writer.writerows(rejected)

    stages = collections.Counter(v["stage"] for v in verdict_by_id.values())
    print(f"оценено: {judged}, источники недоступны: {unavailable}, отбраковано как зрелые: {len(rejected)}")
    print(f"распределение стадий: {dict(sorted(stages.items()))}")
    for item in rejected[:10]:
        print(f"  - {item['name'][:70]}\n      {item['reason'][:150]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate-gold", type=int, metavar="N")
    parser.add_argument("--recheck-labeled", action="store_true",
                        help="перепроверить стадию у собранных позитивов по их источникам")
    parser.add_argument("--extract", action="store_true",
                        help="недоступные страницы доставать через Tavily extract (тратит кредиты)")
    args = parser.parse_args()
    if args.recheck_labeled:
        return recheck_labeled(use_extract=args.extract)
    if args.validate_gold:
        return validate_gold(args.validate_gold)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
