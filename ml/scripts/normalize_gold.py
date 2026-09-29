"""Нормализация золотого яруса: xlsx заказчика -> gold.csv + gold_sources_reference.csv.

Стадия и тренд разбираются из текста свободной формы в числовые шкалы 1..5 и 1..3.
Разбор проверяется на согласованность с колонкой «Балл (стадия+тренд)»: несовпадение
означает составную или неоднозначную формулировку и помечается parse_ok=0 — такие
строки идут на ручную проверку аннотаторами, а не исправляются автоматически.

Колонка «Источники» выносится в отдельный файл и НЕ участвует в признаках:
86 из 100 записей содержат ровно 3 ссылки, это артефакт заполнения таблицы
(см. docs/06-labeling-protocol.md §6).

Запуск:  python ml/scripts/normalize_gold.py
Требует: openpyxl
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "100_слабых_технологических_сигналов_сентябрь_2026.xlsx"
OUT_GOLD = ROOT / "ml" / "datasets" / "gold.csv"
OUT_SOURCES = ROOT / "ml" / "datasets" / "gold_sources_reference.csv"

# Шкала стадии. Засчитываются только канонические ярлыки: описательные хвосты после
# стрелки («первые полисы», «ранние поставки», «ранняя серия») методологи трактуют как
# комментарий, а не как повышение стадии. Проверено на 100 строках: правило «максимум по
# каноническим ярлыкам» даёт 99 совпадений с колонкой «Балл», правило «левая часть» — 58,
# правило «максимум с учётом хвостов» — 90.
STAGE_PATTERNS: list[tuple[int, tuple[str, ...]]] = [
    (5, ("массовое внедрение",)),
    (4, ("раннее внедрение",)),
    (3, ("пилот",)),
    (2, ("прототип", "poc")),
    (1, ("концепц", "исследован")),
]

# Ярлык тренда отделён от пояснения методолога длинным тире. Разбирать надо только
# ярлык: в пояснении встречаются слова вроде «двукратным», сбивающие разбор.
TREND_LABEL_SPLIT = re.compile(r"\s[—–-]\s")


def parse_stage(text: str) -> int | None:
    """Максимальная каноническая стадия, упомянутая в формулировке.

    «Прототип/PoC → Раннее внедрение»            -> 4
    «Прототип/PoC → первые полисы (4+ андеррайтера)» -> 2 (хвост — комментарий)
    """
    if not text:
        return None
    low = text.lower()
    found = [value for value, keys in STAGE_PATTERNS if any(k in low for k in keys)]
    return max(found) if found else None


def parse_trend(text: str) -> int | None:
    if not text:
        return None
    label = TREND_LABEL_SPLIT.split(text, maxsplit=1)[0].lower()
    if "раст" in label:
        return 3 if ("быстро" in label or "кратн" in label) else 2
    if "стабильн" in label:
        return 1
    return None


def main() -> int:
    if not SRC.exists():
        print(f"не найден исходный файл: {SRC}", file=sys.stderr)
        return 1

    wb = openpyxl.load_workbook(SRC)
    ws = wb.active
    rows = [r for r in ws.iter_rows(min_row=3, values_only=True) if r[1] is not None]

    gold: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    mismatched: list[int] = []

    for row in rows:
        idx, name, domain, companies, why, stage_text, trend_text, score, src = row[1:10]
        stage = parse_stage(stage_text or "")
        trend = parse_trend(trend_text or "")
        parse_ok = int(
            stage is not None and trend is not None and score is not None
            and stage + trend == int(score)
        )
        if not parse_ok:
            mismatched.append(int(idx))

        gold.append({
            "id": f"gold-{int(idx):03d}",
            "name": (name or "").strip(),
            "domain": (domain or "").strip(),
            "companies": (companies or "").strip(),
            "label": 1,
            "stage": stage if stage is not None else "",
            "trend": trend if trend is not None else "",
            "score_reference": int(score) if score is not None else "",
            "stage_text": (stage_text or "").strip(),
            "trend_text": (trend_text or "").strip(),
            "parse_ok": parse_ok,
        })
        # Отдельный файл: справочно для проверки покрытия ETL, в признаки не идёт.
        sources.append({
            "id": f"gold-{int(idx):03d}",
            "name": (name or "").strip(),
            "source_count": len(re.findall(r"\]\(http", src or "")),
            "sources_raw": (src or "").strip(),
            "why_weak_signal_raw": (why or "").strip(),
        })

    OUT_GOLD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_GOLD.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(gold[0]))
        writer.writeheader()
        writer.writerows(gold)

    with OUT_SOURCES.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sources[0]))
        writer.writeheader()
        writer.writerows(sources)

    ok = sum(r["parse_ok"] for r in gold)
    print(f"записей: {len(gold)}")
    print(f"разбор согласован с баллом: {ok}/{len(gold)}")
    if mismatched:
        print(f"на ручную проверку (parse_ok=0): {mismatched}")
    print(f"-> {OUT_GOLD.relative_to(ROOT)}")
    print(f"-> {OUT_SOURCES.relative_to(ROOT)}  (в признаки не включать)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
