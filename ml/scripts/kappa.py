"""Согласие аннотаторов: Cohen's kappa по бинарной метке, стадии и тренду.

Вход: ml/datasets/labeled.csv, где одна и та же запись размечена дважды —
две строки с одинаковым id и разными значениями в колонке annotator.

Порог приёмки из docs/06-labeling-protocol.md: kappa >= 0.7 по метке.
Расхождения выгружаются в ml/datasets/disputed.csv.

Запуск: python ml/scripts/kappa.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "ml" / "datasets" / "labeled.csv"
DISPUTED = ROOT / "ml" / "datasets" / "disputed.csv"

KAPPA_THRESHOLD = 0.7


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Каппа Коэна для двух наборов категориальных оценок."""
    if not pairs:
        return None
    categories = sorted({value for pair in pairs for value in pair})
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum(
        (sum(1 for a, _ in pairs if a == c) / n) * (sum(1 for _, b in pairs if b == c) / n)
        for c in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def interpret(kappa: float) -> str:
    if kappa >= 0.8:
        return "почти полное согласие"
    if kappa >= 0.7:
        return "приемлемо"
    if kappa >= 0.6:
        return "умеренное — протокол требует уточнения"
    return "низкое — перерамечать после уточнения протокола"


def main() -> int:
    if not LABELED.exists():
        print(f"нет файла {LABELED}", file=sys.stderr)
        return 1

    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    with LABELED.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("id"):
                by_id[row["id"]].append(row)

    doubled = {rid: rows for rid, rows in by_id.items() if len(rows) >= 2}
    if not doubled:
        print("нет записей, размеченных дважды — считать согласие не на чем")
        print(f"всего записей: {sum(len(v) for v in by_id.values())}")
        return 0

    disputed: list[dict[str, str]] = []
    print(f"записей с двойной разметкой: {len(doubled)}\n")

    for field in ("label", "stage", "trend"):
        pairs = [(r[0].get(field, ""), r[1].get(field, "")) for r in doubled.values()]
        pairs = [p for p in pairs if p[0] != "" and p[1] != ""]
        kappa = cohens_kappa(pairs)
        if kappa is None:
            print(f"{field:>6}: нет данных")
            continue
        agreement = sum(1 for a, b in pairs if a == b) / len(pairs)
        flag = "" if field != "label" or kappa >= KAPPA_THRESHOLD else "   ← НИЖЕ ПОРОГА 0.7"
        print(f"{field:>6}: kappa = {kappa:.3f}  ({interpret(kappa)}), "
              f"совпадений {agreement:.0%} из {len(pairs)}{flag}")

    for rid, rows in doubled.items():
        a, b = rows[0], rows[1]
        if a.get("label") != b.get("label"):
            disputed.append({
                "id": rid,
                "name": a.get("name", ""),
                "domain": a.get("domain", ""),
                "annotator_1": a.get("annotator", ""),
                "label_1": a.get("label", ""),
                "rationale_1": a.get("rationale", ""),
                "annotator_2": b.get("annotator", ""),
                "label_2": b.get("label", ""),
                "rationale_2": b.get("rationale", ""),
            })

    if disputed:
        with DISPUTED.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(disputed[0]))
            writer.writeheader()
            writer.writerows(disputed)
        print(f"\nрасхождений по метке: {len(disputed)} -> {DISPUTED.relative_to(ROOT)}")
        print("их разрешает третий участник (протокол §4)")
    else:
        print("\nрасхождений по метке нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
