"""Авторазметка обучающей выборки: извлечение стадии и тренда двумя моделями.

Модель здесь НЕ решает, слабый ли это сигнал. Она читает собранные документы и
извлекает два факта — какая стадия развития в них описана и какая динамика упоминаний,
каждый с цитатой-подтверждением. Метку выводит правило заказчика, восстановленное из
их датасета (docs/02-methodology.md §1):

    label = 1, если стадия <= 4 И тренд >= 2

Разделение существенно: извлечение факта из текста проверяемо цитатой, суждение
«перспективно ли это» — нет. Правило перевода в метку принадлежит методологам ГПБ,
а не нам и не языковой модели.

Два независимых прохода разными моделями. Где модели разошлись — запись не попадает
в обучение, а уходит в disputed.csv. Это тот же принцип, что и с двумя аннотаторами.

РЕЖИМ ПРОВЕРКИ САМОГО РАЗМЕТЧИКА (--validate-on-gold) прогоняет пайплайн по 100 золотым
записям вслепую: стадия и тренд методологов в промпт не попадают. Сравнение с эталоном
даёт измеренную точность авторазметки против настоящей человеческой разметки. Это
единственная честная оценка качества выборки, когда аннотаторов нет.

Запуск:
    export OPENROUTER_API_KEY="..."
    python ml/scripts/autolabel.py --validate-on-gold      # сначала это
    python ml/scripts/autolabel.py --limit 50              # потом разметка пула

Модели задаются через AUTOLABEL_MODEL_A / AUTOLABEL_MODEL_B.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv() -> None:
    """Подхватить ROOT/.env, не перетирая уже заданные переменные окружения."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


load_dotenv()

GOLD = ROOT / "ml" / "datasets" / "gold.csv"
POOL = ROOT / "ml" / "datasets" / "candidates_pool.csv"
LABELED = ROOT / "ml" / "datasets" / "labeled.csv"
DISPUTED = ROOT / "ml" / "datasets" / "disputed.csv"
REPORT = ROOT / "ml" / "reports" / "autolabel_validation.md"
CACHE = ROOT / "ml" / ".cache"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENALEX = "https://api.openalex.org"
USER_AGENT = "weak-signal-radar/0.1"

MODEL_A = os.getenv("AUTOLABEL_MODEL_A", "z-ai/glm-5.3")
MODEL_B = os.getenv("AUTOLABEL_MODEL_B", "google/gemini-3.8-flash")

EVIDENCE_WORKS = 12          # сколько публикаций показывать модели
HISTOGRAM_YEARS = 8

STAGE_NAMES = {
    1: "Концепция/Исследование",
    2: "Прототип/PoC",
    3: "Пилот",
    4: "Раннее внедрение",
    5: "Массовое внедрение",
}
TREND_NAMES = {1: "Стабильный", 2: "Растёт", 3: "Растёт быстро"}

EXTRACTION_SCHEMA = """{
  "is_technology": true|false,
  "stage": 1..5,
  "stage_quote": "цитата из документа, обосновывающая стадию",
  "stage_source": "URL документа с цитатой",
  "trend": 1..3,
  "trend_quote": "цитата или числовое наблюдение по динамике",
  "trend_source": "URL",
  "confidence": 0.0..1.0
}"""

INSTRUCTIONS = f"""Ты аналитик технологической разведки. Тебе дают название технологии и
набор реально найденных публикаций с годами. Твоя задача — ИЗВЛЕЧЬ два факта из этих
документов, а не оценить перспективность технологии.

Стадия развития:
1 — {STAGE_NAMES[1]}: только научные работы, лабораторные результаты
2 — {STAGE_NAMES[2]}: есть работающий прототип или proof of concept
3 — {STAGE_NAMES[3]}: есть пилотные внедрения у заказчиков
4 — {STAGE_NAMES[4]}: первые коммерческие поставки, ранние клиенты
5 — {STAGE_NAMES[5]}: массовое применение, отраслевой стандарт, сформированный рынок

Динамика упоминаний:
1 — {TREND_NAMES[1]}: объём публикаций примерно постоянен
2 — {TREND_NAMES[2]}: заметный рост
3 — {TREND_NAMES[3]}: кратный рост за последние 1-2 года

Правила:
- Каждый вывод подкрепляй цитатой из предоставленных документов и ссылкой на документ.
- Не используй знания вне предоставленных документов.
- Если документов недостаточно для вывода, ставь confidence ниже 0.5.
- is_technology = false, если объект является компанией, продуктом, отраслью или
  общей концепцией, а не технологией.

Ответ — строго JSON по схеме, без markdown:
{EXTRACTION_SCHEMA}"""


@dataclass
class Extraction:
    is_technology: bool
    stage: int
    trend: int
    stage_quote: str
    stage_source: str
    trend_quote: str
    trend_source: str
    confidence: float
    model: str


class LLMError(RuntimeError):
    pass


def _http_json(url: str, *, data: dict | None = None, headers: dict | None = None,
               tries: int = 5, timeout: int = 90) -> dict:
    payload = json.dumps(data).encode() if data is not None else None
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"User-Agent": USER_AGENT, **(headers or {})},
                method="POST" if payload else "GET",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(5 * (attempt + 1) + random.random())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1) + random.random())
    raise LLMError(f"{url}: не удалось за {tries} попыток: {last}")


QUERY_INSTRUCTIONS = """Ты помогаешь искать научные публикации по технологии.
Тебе дают название технологии по-русски и список связанных компаний.
Верни 2-3 коротких ПОИСКОВЫХ ЗАПРОСА НА АНГЛИЙСКОМ, по которым эту технологию можно найти
в базе научных публикаций. Используй принятую в отрасли англоязычную терминологию, а не
дословный перевод. Без кавычек и операторов, 3-6 слов в запросе.

Ответ — строго JSON: {"queries": ["...", "...", "..."]}"""


def english_queries(name: str, companies: str = "") -> list[str]:
    """Русское название -> англоязычные поисковые запросы.

    Датасет заказчика на русском, а OpenAlex и Crossref — англоязычные корпуса.
    Без этого шага поиск по золотым записям не находит ничего. Это же стадия S1
    основного пайплайна (расширение запроса), поэтому работа не одноразовая.
    """
    cache_file = CACHE / "queries" / f"{re.sub(r'[^a-zа-я0-9]+', '_', name.lower())[:80]}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if not re.search(r"[а-яА-Я]", name):
        return [name]

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise LLMError("OPENROUTER_API_KEY не задан")
    prompt = f"Технология: {name}" + (f"\nКомпании: {companies}" if companies else "")
    response = _http_json(
        OPENROUTER_URL,
        data={"model": MODEL_A, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": QUERY_INSTRUCTIONS},
                           {"role": "user", "content": prompt}]},
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    text = response["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    queries = json.loads(match.group(0))["queries"] if match else []
    queries = [q for q in queries if isinstance(q, str) and q.strip()][:3] or [name]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
    return queries


def _crossref_evidence(query: str) -> tuple[list[dict], dict[int, int]]:
    """Резервный источник: Crossref заметно терпимее к нагрузке, чем OpenAlex."""
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    params = {"query.bibliographic": query, "rows": 60,
              "filter": "type:journal-article,from-pub-date:2018-01-01"}
    if mailto:
        params["mailto"] = mailto
    payload = _http_json("https://api.crossref.org/works?" + urllib.parse.urlencode(params))
    items = payload.get("message", {}).get("items", [])

    works, histogram = [], collections.Counter()
    current = date.today().year
    for item in items:
        parts = (item.get("issued") or {}).get("date-parts") or [[None]]
        year = parts[0][0] if parts and parts[0] else None
        if isinstance(year, int) and current - HISTOGRAM_YEARS < year <= current:
            histogram[year] += 1
        if len(works) < EVIDENCE_WORKS:
            works.append({
                "title": (item.get("title") or [""])[0],
                "year": year,
                "url": f"https://doi.org/{item['DOI']}" if item.get("DOI") else "",
                "venue": (item.get("container-title") or [""])[0],
            })
    return works, dict(histogram)


def gather_evidence(name: str, companies: str = "") -> tuple[list[dict], dict[int, int]]:
    """Публикации и гистограмма по годам. OpenAlex, при отказе — Crossref."""
    slug = re.sub(r"[^a-zа-я0-9]+", "_", name.lower())[:80]
    cache_file = CACHE / "evidence" / f"{slug}.json"
    if cache_file.exists():
        blob = json.loads(cache_file.read_text(encoding="utf-8"))
        return blob["works"], {int(k): v for k, v in blob["histogram"].items()}

    queries = english_queries(name, companies)
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    works: list[dict] = []
    histogram: dict[int, int] = {}
    source = ""

    # EVIDENCE_SOURCE=crossref пропускает OpenAlex: при устойчивом 429 он только
    # съедает время на ретраях, ничего не возвращая.
    skip_openalex = os.getenv("EVIDENCE_SOURCE", "").strip().lower() == "crossref"

    for query in ([] if skip_openalex else queries):
        common = {"search": query, "mailto": mailto} if mailto else {"search": query}
        try:
            works_payload = _http_json(
                f"{OPENALEX}/works?" + urllib.parse.urlencode({
                    **common, "per_page": EVIDENCE_WORKS,
                    "sort": "relevance_score:desc", "filter": "type:article",
                }), tries=3)
            works = [{
                "title": w.get("title") or "",
                "year": w.get("publication_year"),
                "url": w.get("doi") or (w.get("ids", {}) or {}).get("openalex", ""),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
            } for w in works_payload.get("results", [])]
            hist_payload = _http_json(
                f"{OPENALEX}/works?" + urllib.parse.urlencode({**common, "group_by": "publication_year"}),
                tries=3)
            current = date.today().year
            histogram = {
                int(g["key"]): g["count"] for g in hist_payload.get("group_by", [])
                if g["key"].isdigit() and current - HISTOGRAM_YEARS < int(g["key"]) <= current
            }
            source = "openalex"
            break
        except Exception:  # noqa: BLE001 — переходим к следующему запросу или к Crossref
            continue

    if not works:
        for query in queries:
            try:
                works, histogram = _crossref_evidence(query)
                source = "crossref"
                break
            except Exception:  # noqa: BLE001
                continue

    if not works:
        raise LLMError(f"источники недоступны по запросам: {queries}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(
        {"works": works, "histogram": histogram, "queries": queries, "source": source},
        ensure_ascii=False), encoding="utf-8")
    return works, histogram


def structural_trend(histogram: dict[int, int]) -> int | None:
    """Тренд из арифметики по годам — базовая линия без участия модели.

    Сравнивается с извлечённым моделью трендом в режиме проверки: если арифметика
    ближе к разметке методологов, то LLM для этого поля вообще не нужна.
    """
    years = sorted(histogram)
    if len(years) < 4:
        return None
    half = len(years) // 2
    early = statistics.mean(histogram[y] for y in years[:half]) or 0.5
    late = statistics.mean(histogram[y] for y in years[half:])
    ratio = (late + 1) / (early + 1)
    if ratio >= 2.5:
        return 3
    if ratio >= 1.25:
        return 2
    return 1


def build_prompt(name: str, works: list[dict], histogram: dict[int, int]) -> str:
    lines = [f"Технология: {name}", "", "Найденные публикации:"]
    for i, work in enumerate(works, start=1):
        lines.append(f"{i}. [{work['year']}] {work['title']}")
        if work["venue"]:
            lines.append(f"   издание: {work['venue']}")
        if work["url"]:
            lines.append(f"   ссылка: {work['url']}")
    if histogram:
        series = ", ".join(f"{y}: {histogram[y]}" for y in sorted(histogram))
        lines += ["", f"Публикаций по годам: {series}"]
    return "\n".join(lines)


def extract(model: str, name: str, works: list[dict], histogram: dict[int, int]) -> Extraction:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise LLMError("OPENROUTER_API_KEY не задан")

    response = _http_json(
        OPENROUTER_URL,
        data={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": build_prompt(name, works, histogram)},
            ],
        },
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    text = response["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise LLMError(f"{model}: ответ без JSON")
    data = json.loads(match.group(0))
    return Extraction(
        is_technology=bool(data.get("is_technology", True)),
        stage=int(data.get("stage", 0)),
        trend=int(data.get("trend", 0)),
        stage_quote=str(data.get("stage_quote", ""))[:400],
        stage_source=str(data.get("stage_source", ""))[:300],
        trend_quote=str(data.get("trend_quote", ""))[:400],
        trend_source=str(data.get("trend_source", ""))[:300],
        confidence=float(data.get("confidence", 0.0)),
        model=model,
    )


def derive_label(stage: int, trend: int, is_technology: bool) -> int:
    """Правило заказчика. Восстановлено из их датасета, сходится на 99 строках из 100."""
    if not is_technology:
        return 0
    return int(1 <= stage <= 4 and trend >= 2)


def weighted_kappa(pairs: list[tuple[int, int]], categories: list[int]) -> float | None:
    """Каппа Коэна с квадратичными весами: для порядковых шкал ошибка на 1 градацию
    должна штрафоваться слабее, чем на 3."""
    if not pairs:
        return None
    n = len(pairs)
    size = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    max_distance = (size - 1) ** 2 or 1

    observed = sum((index[a] - index[b]) ** 2 for a, b in pairs) / (n * max_distance)
    row = collections.Counter(a for a, _ in pairs)
    col = collections.Counter(b for _, b in pairs)
    expected = sum(
        (row[x] / n) * (col[y] / n) * ((index[x] - index[y]) ** 2) / max_distance
        for x in categories for y in categories
    )
    return 1 - observed / expected if expected else None


def validate_on_gold(limit: int | None) -> int:
    """Прогон по золотым записям вслепую и сравнение с разметкой методологов."""
    with GOLD.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["stage"] and r["trend"]]
    if limit:
        rows = rows[:limit]

    print(f"проверка авторазметчика на {len(rows)} золотых записях")
    print(f"модели: {MODEL_A} / {MODEL_B}\n")

    results = []
    for i, row in enumerate(rows, start=1):
        try:
            works, histogram = gather_evidence(row["name"], row.get("companies", ""))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(rows)}] {row['id']}: источники недоступны — {exc}", file=sys.stderr)
            continue
        entry = {
            "id": row["id"], "name": row["name"],
            "gold_stage": int(row["stage"]), "gold_trend": int(row["trend"]),
            "structural_trend": structural_trend(histogram),
        }
        for tag, model in (("a", MODEL_A), ("b", MODEL_B)):
            try:
                extraction = extract(model, row["name"], works, histogram)
                entry[f"{tag}_stage"] = extraction.stage
                entry[f"{tag}_trend"] = extraction.trend
                entry[f"{tag}_conf"] = extraction.confidence
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(rows)}] {row['id']}: {model} — {exc}", file=sys.stderr)
        results.append(entry)
        if i % 10 == 0:
            print(f"  обработано {i}/{len(rows)}")

    if not results:
        print("ничего не получилось обработать", file=sys.stderr)
        return 1

    # Сырые результаты нужны для разбора ошибок: без них по агрегатам не понять,
    # вырождены ли предсказания (модель ставит всем одну стадию) или просто шумны.
    raw = REPORT.with_suffix(".jsonl")
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w", encoding="utf-8") as fh:
        for entry in results:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"сырые результаты -> {raw.relative_to(ROOT)}")

    write_validation_report(results)
    return 0


def _accuracy(pairs: list[tuple[int, int]]) -> tuple[float, float]:
    """Доля точных совпадений и доля совпадений с допуском в одну градацию."""
    exact = sum(1 for a, b in pairs if a == b) / len(pairs)
    near = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
    return exact, near


def write_validation_report(results: list[dict]) -> None:
    lines = ["# Проверка авторазметчика на золотом ярусе", "",
             f"Дата: {date.today().isoformat()}  ",
             f"Модели: `{MODEL_A}` и `{MODEL_B}`  ",
             f"Записей обработано: {len(results)}", "",
             "Прогон вслепую: стадия и тренд методологов в промпт не передавались.",
             "", "## Точность против разметки методологов", "",
             "| Источник оценки | Поле | Точно | ±1 градация | Каппа (взвеш.) |",
             "|---|---|---|---|---|"]

    for tag, title in (("a", MODEL_A), ("b", MODEL_B)):
        for field, categories in (("stage", [1, 2, 3, 4, 5]), ("trend", [1, 2, 3])):
            pairs = [(r[f"gold_{field}"], r[f"{tag}_{field}"])
                     for r in results if r.get(f"{tag}_{field}")]
            if not pairs:
                continue
            exact, near = _accuracy(pairs)
            kappa = weighted_kappa(pairs, categories)
            kappa_text = f"{kappa:.3f}" if kappa is not None else "—"
            lines.append(f"| {title} | {field} | {exact:.0%} | {near:.0%} | {kappa_text} |")

    structural = [(r["gold_trend"], r["structural_trend"])
                  for r in results if r.get("structural_trend")]
    if structural:
        exact, near = _accuracy(structural)
        kappa = weighted_kappa(structural, [1, 2, 3])
        kappa_text = f"{kappa:.3f}" if kappa is not None else "—"
        lines.append(f"| арифметика по годам (без LLM) | trend | {exact:.0%} | {near:.0%} | {kappa_text} |")

    agreement = [(r["a_stage"], r["b_stage"]) for r in results
                 if r.get("a_stage") and r.get("b_stage")]
    if agreement:
        exact, near = _accuracy(agreement)
        kappa = weighted_kappa(agreement, [1, 2, 3, 4, 5])
        lines += ["", "## Согласие моделей между собой", "",
                  f"Стадия: точных совпадений {exact:.0%}, с допуском ±1 — {near:.0%}, "
                  f"каппа {kappa:.3f}" if kappa is not None else ""]

    label_pairs = [(1, derive_label(r["a_stage"], r["a_trend"], True))
                   for r in results if r.get("a_stage") and r.get("a_trend")]
    if label_pairs:
        recall = sum(1 for _, p in label_pairs if p == 1) / len(label_pairs)
        lines += ["", "## Полнота на золоте", "",
                  f"Все 100 золотых записей — слабые сигналы. Правило «стадия ≤ 4 и тренд ≥ 2» "
                  f"поверх извлечений модели {MODEL_A} воспроизводит метку в **{recall:.0%}** случаев.",
                  "", "Это верхняя оценка полноты авторазметки. Ошибку на зрелых технологиях "
                  "так измерить нельзя: в золотом ярусе нет ни одной записи со стадией 5, "
                  "их закрывают структурные негативы."]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nотчёт -> {REPORT.relative_to(ROOT)}")
    for line in lines[8:24]:
        print(line)


def label_pool(limit: int | None) -> int:
    with POOL.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") != "rejected"]
    if limit:
        rows = rows[:limit]

    accepted: list[dict] = []
    disputed: list[dict] = []
    print(f"разметка {len(rows)} кандидатов моделями {MODEL_A} / {MODEL_B}\n")

    for i, row in enumerate(rows, start=1):
        try:
            works, histogram = gather_evidence(row["name"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {row['name']}: источники недоступны — {exc}", file=sys.stderr)
            continue

        extractions: list[Extraction] = []
        for model in (MODEL_A, MODEL_B):
            try:
                extractions.append(extract(model, row["name"], works, histogram))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] {row['name']}: {model} — {exc}", file=sys.stderr)
        if len(extractions) < 2:
            continue

        first, second = extractions
        label_1 = derive_label(first.stage, first.trend, first.is_technology)
        label_2 = derive_label(second.stage, second.trend, second.is_technology)

        if label_1 != label_2 or abs(first.stage - second.stage) > 1:
            disputed.append({
                "id": row["id"], "name": row["name"], "domain": row["domain"],
                "model_1": first.model, "label_1": label_1,
                "stage_1": first.stage, "trend_1": first.trend,
                "model_2": second.model, "label_2": label_2,
                "stage_2": second.stage, "trend_2": second.trend,
            })
            continue

        for extraction, label in ((first, label_1), (second, label_2)):
            accepted.append({
                "id": row["id"], "name": row["name"], "domain": row["domain"],
                "label": label, "stage": extraction.stage, "trend": extraction.trend,
                "negative_type": "", "rationale": extraction.stage_quote,
                "annotator": extraction.model,
                "evidence_urls": ";".join(filter(None, [extraction.stage_source, extraction.trend_source])),
                "added_at": date.today().isoformat(),
            })
        if i % 10 == 0:
            print(f"  обработано {i}/{len(rows)}, согласовано {len(accepted)//2}, спорных {len(disputed)}")

    if accepted:
        exists = LABELED.exists() and LABELED.stat().st_size > 0
        with LABELED.open("a" if exists else "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(accepted[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(accepted)
    if disputed:
        with DISPUTED.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(disputed[0]))
            writer.writeheader()
            writer.writerows(disputed)

    total = len(accepted) // 2 + len(disputed)
    rate = len(disputed) / total if total else 0
    print(f"\nсогласовано: {len(accepted)//2}, спорных: {len(disputed)} ({rate:.0%})")
    print(f"-> {LABELED.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate-on-gold", action="store_true",
                        help="проверить разметчик на золотом ярусе (делать это первым)")
    parser.add_argument("--limit", type=int, help="ограничить число записей")
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        print("OPENROUTER_API_KEY не задан", file=sys.stderr)
        return 1
    return validate_on_gold(args.limit) if args.validate_on_gold else label_pool(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
