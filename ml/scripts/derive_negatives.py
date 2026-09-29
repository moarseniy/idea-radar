"""Структурные негативы: метка из измеримых фактов, без суждения модели.

Отрицательный пример не требует чьего-либо мнения, если он опознаётся по проверяемым
свойствам данных:

  N1  зрелая          — статья в Википедии старше 7 лет, либо высокий объём публикаций
                        при первой публикации больше 10 лет назад
  N3d растёт, но
      массовая        — объём в верхнем квартиле пула И заметный рост
  N3e исследование
      без движения    — объём достаточный для оценки, но ряд плоский на горизонте 8+ лет

Источники фактов:
  OpenAlex  — счётчики публикаций по годам через ФРАЗОВЫЙ фильтр
              title_and_abstract.search:"...". Параметр search для этого непригоден:
              он матчит слова по отдельности, и «retrieval-augmented generation»
              возвращает сотни тысяч работ начиная с 2016 года.
  Wikipedia — дата создания статьи как индикатор укоренённости термина.

Порог по объёму берётся от распределения самого пула, а не назначается произвольно.

Кандидат, не попавший ни в одну категорию, НЕ размечается: угадывать здесь нечего,
такие записи остаются для авторазметчика.

ВНИМАНИЕ, риск тавтологии. Негативы определяются через объём и наклон, а объём и наклон
есть среди признаков модели. Классификатор может выучить определение вместо сути и показать
завышенный F1. Контроль — проверка на 100 золотых записях и отдельный разбор этих признаков
в абляции (docs/02-methodology.md §5.2).

Запуск:
    python ml/scripts/derive_negatives.py                # весь пул
    python ml/scripts/derive_negatives.py --limit 20     # проба
    python ml/scripts/derive_negatives.py --dry-run      # без записи, только статистика
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "ml" / "datasets" / "candidates_pool.csv"
LABELED = ROOT / "ml" / "datasets" / "labeled.csv"
CACHE = ROOT / "ml" / ".cache" / "facts"

from evidence_sources import assess, preflight

CONTACT = os.getenv("OPENALEX_MAILTO", "").strip() or "hackathon@example.org"
UA = {"User-Agent": f"weak-signal-radar/0.1 (hackathon research; {CONTACT})"}

YEARS_BACK = 10
WIKI_MATURE_AGE_YEARS = 7
FIRST_PUB_MATURE_AGE = 10
MIN_VOLUME_FOR_JUDGEMENT = 15   # ниже — данных мало, вывод не делаем
FLAT_SLOPE_MAX = 0.08           # наклон log(1+n) на год: почти горизонталь
RISING_SLOPE_MIN = 0.25
REQUEST_SPACING = 1.5


@dataclass
class Facts:
    name: str
    histogram: dict[int, int]
    wiki_created: str | None
    total_recent: int
    slope: float | None
    first_year: int | None
    verdict: str = "unverified"
    note: str = ""


def _get(url: str, tries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(4 * (attempt + 1) + random.random())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(4 * (attempt + 1) + random.random())
    raise RuntimeError(f"{url}: {last}")


def log_slope(histogram: dict[int, int]) -> float | None:
    """Наклон линейной регрессии по log(1+n) — рост в разах, а не в штуках."""
    years = sorted(y for y in histogram if histogram[y] >= 0)
    if len(years) < 4:
        return None
    xs = [float(y) for y in years]
    ys = [math.log1p(histogram[y]) for y in years]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def collect_facts(name: str) -> Facts:
    """Факты о технологии через общий модуль + ряд по годам, если OpenAlex доступен."""
    maturity = assess(name, verbose=True)
    histogram: dict[int, int] = {}
    if maturity.openalex_ok:
        # Полный ряд по годам есть только у OpenAlex; при его недоступности
        # наклон не считается, и вывод делается по объёму и возрасту термина.
        try:
            histogram = _openalex_histogram(name)
        except Exception:  # noqa: BLE001
            histogram = {}
    volume = maturity.openalex_recent if maturity.openalex_ok else maturity.s2_recent
    return Facts(
        name=name,
        histogram=histogram,
        wiki_created=maturity.wiki_created if maturity.wiki_ok else None,
        total_recent=volume or 0,
        slope=log_slope(histogram) if histogram else None,
        first_year=min(histogram) if histogram else None,
        verdict=maturity.verdict,
        note=maturity.note,
    )


def _openalex_histogram(name: str) -> dict[int, int]:
    params = {"filter": f'title_and_abstract.search:"{name}"',
              "group_by": "publication_year", "mailto": CONTACT}
    payload = _get("https://api.openalex.org/works?" + urllib.parse.urlencode(params))
    current = date.today().year
    return {int(g["key"]): g["count"] for g in payload.get("group_by", [])
            if g["key"].isdigit() and current - YEARS_BACK <= int(g["key"]) <= current}


def classify(facts: Facts, high_volume: float, top_quartile: float) -> tuple[str, str] | None:
    """Тип негатива и обоснование в виде фактов. None — вывод не делается.

    По непроверенным данным вывод не делается никогда: verdict == "unverified"
    означает, что источники молчали, а не что признаков зрелости нет.
    """
    if facts.verdict == "unverified":
        return None
    if facts.verdict == "mature":
        return "N1", facts.note
    current = date.today().year
    wiki_age = None
    if facts.wiki_created:
        wiki_age = (datetime.now() - datetime.strptime(facts.wiki_created, "%Y-%m-%d")).days / 365.25

    if wiki_age is not None and wiki_age >= WIKI_MATURE_AGE_YEARS:
        return "N1", (f"статья в Википедии создана {facts.wiki_created} "
                      f"({wiki_age:.0f} лет назад); термин укоренён")

    if facts.total_recent < MIN_VOLUME_FOR_JUDGEMENT:
        return None

    if (facts.first_year is not None and current - facts.first_year >= FIRST_PUB_MATURE_AGE
            and facts.total_recent >= high_volume):
        return "N1", (f"публикации с {facts.first_year} года, {facts.total_recent} работ "
                      f"за последние 3 года — объём выше {high_volume:.0f} (верхний дециль пула)")

    if facts.slope is None:
        return None

    if facts.slope <= FLAT_SLOPE_MAX:
        return "N3e", (f"{facts.total_recent} работ за 3 года при наклоне {facts.slope:+.3f} "
                       f"на год — ряд практически горизонтален")

    if facts.total_recent >= top_quartile and facts.slope >= RISING_SLOPE_MIN:
        return "N3d", (f"{facts.total_recent} работ за 3 года (верхний квартиль пула) "
                       f"при наклоне {facts.slope:+.3f} — растёт, но уже массовая тема")

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with POOL.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") != "rejected"]
    if args.limit:
        rows = rows[:args.limit]

    if not preflight():
        return 2
    print(f"сбор фактов по {len(rows)} кандидатам\n")
    collected: list[tuple[dict, Facts]] = []
    for i, row in enumerate(rows, start=1):
        try:
            collected.append((row, collect_facts(row["name"])))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(rows)}] {row['name']}: пропуск — {exc}", file=sys.stderr)
        if i % 20 == 0:
            print(f"  обработано {i}/{len(rows)}")

    if not collected:
        print("фактов не собрано", file=sys.stderr)
        return 1

    volumes = sorted(f.total_recent for _, f in collected)
    high_volume = volumes[int(len(volumes) * 0.9)] if volumes else 0
    top_quartile = volumes[int(len(volumes) * 0.75)] if volumes else 0
    print(f"\nпороги по распределению пула: верхний дециль {high_volume}, квартиль {top_quartile}")

    negatives, counts = [], {}
    for row, facts in collected:
        verdict = classify(facts, high_volume, top_quartile)
        if verdict is None:
            counts["не классифицирован"] = counts.get("не классифицирован", 0) + 1
            continue
        negative_type, rationale = verdict
        counts[negative_type] = counts.get(negative_type, 0) + 1
        negatives.append({
            "id": row["id"], "name": row["name"], "domain": row["domain"],
            "label": 0, "stage": 5 if negative_type == "N1" else "",
            "trend": "", "negative_type": negative_type, "rationale": rationale,
            "annotator": "structural", "evidence_urls": "",
            "added_at": date.today().isoformat(),
        })

    print(f"\nитог: {dict(sorted(counts.items()))}")
    if args.dry_run:
        print("\n--dry-run: ничего не записано. Примеры:")
        for item in negatives[:8]:
            print(f"  [{item['negative_type']}] {item['name']}\n        {item['rationale']}")
        return 0

    if negatives:
        exists = LABELED.exists() and LABELED.stat().st_size > 0
        with LABELED.open("a" if exists else "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(negatives[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(negatives)
        print(f"\nзаписано {len(negatives)} негативов -> {LABELED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
