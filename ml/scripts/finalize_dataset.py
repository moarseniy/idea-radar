"""Финальная сборка датасета: сверка пересечений и сбалансированный отбор.

  overlaps  — удалить строки, чей английский термин совпадает с термином золота
              (валидация) или датасета Арсения (слияние). Сверка по русским названиям
              пропускала одну и ту же технологию в других формулировках.
  select    — отобрать N строк класса с выравниванием по областям. Перекос по областям —
              утечка: при 408 позитивах робототехники против 39 в индустриальном ИИ модель
              выучит «робототехника → слабый сигнал». Внутри области приоритет — научному
              подтверждению, затем разнообразию терминов. Остальное — в резерв, не удаляется.

Запуск:
    python ml/scripts/finalize_dataset.py overlaps
    python ml/scripts/finalize_dataset.py select --label 1 --target 400
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402

RESERVE = collect.ROOT / "ml" / "datasets" / "reserve.csv"
REMOVED = collect.ROOT / "ml" / "datasets" / "dedupe_removed.csv"


def term_of(row: dict) -> str:
    match = re.search(r"термин: ([^|]+)", row["rationale"])
    return match.group(1).strip() if match else ""


def load() -> tuple[list[dict], list[str]]:
    rows = list(csv.DictReader(collect.LABELED.open(encoding="utf-8")))
    return rows, list(rows[0])


def save(rows: list[dict], fields: list[str]) -> None:
    with collect.LABELED.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlaps() -> int:
    rows, fields = load()
    terms = [r["term"] for f in sorted(collect.EXCLUSIONS.glob("*_terms.csv"))
             for r in csv.DictReader(f.open(encoding="utf-8")) if r.get("term")]
    seen = collect.Seen([], terms)
    drop = [r for r in rows if r["label"] == "1" and seen.excluded_term(term_of(r))]
    save([r for r in rows if r not in drop], fields)
    collect.Writer().append(REMOVED, [{"id": r["id"], "name": r["name"],
                                       "reason": f"термин «{term_of(r)}» пересекается с золотом или Арсением"}
                                      for r in drop], ["id", "name", "reason"])
    print(f"удалено пересечений: {len(drop)}; осталось строк: {len(rows) - len(drop)}")
    return 0


def _words(term: str) -> set[str]:
    return collect.Seen._term_words(term)


def select(label: str, target: int, seed: int = 42, mirror: bool = False) -> int:
    rows, fields = load()
    pool = [r for r in rows if r["label"] == label]
    others = [r for r in rows if r["label"] != label]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_domain[row["domain"]].append(row)

    rng = random.Random(seed)
    ranked: dict[str, list[dict]] = {}
    for domain, items in by_domain.items():
        rng.shuffle(items)
        # Научное подтверждение раньше «только медиа»; внутри — случайный порядок.
        items.sort(key=lambda r: "media_only" in r["annotator"])
        chosen, backlog, taken_terms = [], [], []
        for row in items:  # сначала разнообразие: близкие термины откладываются
            words = _words(term_of(row))
            if words and any(len(words & t) / len(words | t) >= 0.5 for t in taken_terms):
                backlog.append(row)
            else:
                chosen.append(row)
                if words:
                    taken_terms.append(words)
        ranked[domain] = chosen + backlog

    if mirror:
        # Столько же строк в области, сколько у другого класса: доля класса в каждой
        # области 50%, и область не несёт информации о метке.
        other = defaultdict(int)
        for row in others:
            other[row["domain"]] += 1
        quota = {d: min(other[d], len(ranked[d])) for d in ranked}
        short = {d: other[d] - quota[d] for d in ranked if other[d] > quota[d]}
        if short:
            print(f"  нехватка до зеркала: {short}")
        selected = [r for d in ranked for r in ranked[d][:quota[d]]]
        reserve = [r for d in ranked for r in ranked[d][quota[d]:]]
        save(others + selected, fields)
        collect.Writer().append(RESERVE, reserve, fields)
        print(f"класс {label}: отобрано {len(selected)} зеркально другому классу, в резерв {len(reserve)}")
        for d in sorted(ranked):
            print(f"  {d:20} {quota[d]:4} (другой класс {other[d]})")
        return 0

    # Квоты: поровну, недобор малых областей перераспределяется на остальные.
    quota = {d: 0 for d in ranked}
    remaining = target
    while remaining > 0:
        open_domains = [d for d in ranked if quota[d] < len(ranked[d])]
        if not open_domains:
            break
        share = max(1, remaining // len(open_domains))
        for d in open_domains:
            add = min(share, len(ranked[d]) - quota[d], remaining)
            quota[d] += add
            remaining -= add
            if remaining == 0:
                break

    selected = [r for d in ranked for r in ranked[d][:quota[d]]]
    reserve = [r for d in ranked for r in ranked[d][quota[d]:]]
    save(others + selected, fields)
    collect.Writer().append(RESERVE, reserve, fields)
    print(f"класс {label}: отобрано {len(selected)}, в резерв {len(reserve)}")
    for d in sorted(ranked):
        print(f"  {d:20} {quota[d]:4} из {len(ranked[d])}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("overlaps")
    s = sub.add_parser("select")
    s.add_argument("--label", choices=["0", "1"], required=True)
    s.add_argument("--target", type=int, default=0)
    s.add_argument("--mirror", action="store_true",
                   help="в каждой области столько же строк, сколько у другого класса")
    args = parser.parse_args()
    return overlaps() if args.cmd == "overlaps" else select(args.label, args.target, mirror=args.mirror)


if __name__ == "__main__":
    raise SystemExit(main())
