"""Финальная ретро-выборка для обучения: из retro_rows.csv — retro_training.csv.

  1. Убирает строки, чей термин или алиас совпадает с термином пересчитанной выборки
     (ml/datasets/enriched: обучение, золото, негативы валидации) — дубль или утечка
     в валидацию. Первые прогоны сверялись без негативов валидации.
  2. Строки с одной статьёй Википедии (name_original — у «родителей» это её заголовок)
     получают общий technology_group_id: варианты одного понятия ("Automated Guided
     Vehicle", "AGV routing") не должны оказаться по разные стороны group k-fold.
  3. Вес строки для обучения: silver — 1, bronze — BRONZE_WEIGHT (метка по новостям слабее).

Запуск: python ml/scripts/retro_finalize.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_training_set as bts  # noqa: E402

RETRO = bts.DATASETS / "retro"
ENRICHED = bts.DATASETS / "enriched"
BRONZE_WEIGHT = 0.3


def terms(row: dict) -> set[str]:
    return {bts.norm_term(t) for t in [row.get("search_term", ""), *(row.get("search_aliases") or "").split(";")]
            if t.strip()}


def main() -> int:
    rows = bts.read(RETRO / "retro_rows.csv")
    fields = list(rows[0])
    taken = {t for part in ("training", "gold", "validation") for r in bts.read(ENRICHED / f"{part}.csv")
             for t in terms(r)}
    kept = [r for r in rows if not terms(r) & taken]
    removed = [r for r in rows if terms(r) & taken]

    by_article = defaultdict(list)
    for r in kept:
        by_article[r["name_original"].strip().lower()].append(r)
    for name, group in by_article.items():
        if len(group) > 1:
            gid = "tech_" + hashlib.sha256(("wiki:" + name).encode()).hexdigest()[:12]
            for r in group:
                r["technology_group_id"] = gid
    for r in kept:
        r["sample_weight"] = "1" if r["tier"] == "silver" or r["label"] == "0" else str(BRONZE_WEIGHT)

    bts.write(RETRO / "retro_training.csv", kept, fields + ["sample_weight"])
    report = {
        "rows": len(kept), "removed_overlap": [f"{r['search_term']} ({r['label']})" for r in removed],
        "labels": Counter(r["label"] for r in kept), "tiers": Counter(f"{r['label']}:{r['tier']}" for r in kept),
        "negative_types": Counter(r["negative_type"] for r in kept if r["label"] == "0"),
        "by_domain": {d: dict(Counter(r["label"] for r in kept if r["domain"] == d))
                      for d in sorted({r["domain"] for r in kept})},
        "topics": len({r["run_id"] for r in kept}), "groups": len({r["technology_group_id"] for r in kept}),
    }
    (RETRO / "retro_training.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
