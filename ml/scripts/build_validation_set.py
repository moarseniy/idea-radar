"""Негативы для валидации на золоте: 100 зрелых технологий, которых нет в обучении.

В золоте только позитивы, precision и F1 на нём не посчитать. Негативы берутся:
  1. gold_negatives.csv — зрелые «родители» технологий золота (contrastive_negatives.py
     --from-gold): та же задача и лексика, стадия 5 подтверждена по Википедии;
  2. reserve.csv — наши негативы из тех же 6 областей, не вошедшие в обучение.
Квоты по областям повторяют распределение золота. Технология, чей термин есть в
обучающей выборке или в золоте, не берётся. Признаки — тот же featurize() и срез.

Выход: ml/datasets/validation_negatives.csv
Запуск: python ml/scripts/build_validation_set.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_training_set as bts  # noqa: E402  (грузит .env и настройки источников)

from app.ml.features.extract import derive_query, featurize  # noqa: E402
from app.ml.features.schema import FEATURE_NAMES  # noqa: E402

TARGET = 100
OUT = bts.DATASETS / "validation_negatives.csv"


def main() -> int:
    gold = bts.read(bts.DATASETS / "gold_features.csv")
    training = bts.read(bts.DATASETS / "training_merged.csv")
    taken = {bts.norm_term(r["search_term"]) for r in training + gold}
    quota = Counter(g["domain"] for g in gold)

    candidates = []
    for name, tier in (("gold_negatives.csv", 0), ("reserve.csv", 1)):
        for r in bts.read(bts.DATASETS / name):
            if r["label"] == "0":
                candidates.append({"id": r["id"], "source": "validation", "source_file": f"ml/datasets/{name}",
                                   "label": "0", "name_ru": r["name"], "name_original": r["name"],
                                   "domain": r["domain"], "negative_type": r["negative_type"],
                                   "stage_ref": r["stage"], "trend_ref": r["trend"],
                                   "technology_group_id": "", "evidence_urls": r["evidence_urls"], "_tier": tier})
    queries = bts.parallel(lambda r: derive_query(r["name_ru"], r["domain"]), candidates, 16, "термины")

    chosen, per_domain, skipped = [], Counter(), Counter()
    # Сначала пары к золоту, затем резерв; внутри яруса — детерминированный порядок по id.
    order = sorted(zip(candidates, queries), key=lambda t: (t[0]["_tier"], hashlib.sha256(t[0]["id"].encode()).hexdigest()))
    for row, query in order:
        if isinstance(query, Exception) or not query["term"]:
            skipped["нет термина"] += 1
            continue
        key = bts.norm_term(query["term"])
        if key in taken:
            skipped["технология уже в обучении/золоте"] += 1
            continue
        if per_domain[row["domain"]] >= quota[row["domain"]]:
            skipped["квота области"] += 1
            continue
        taken.add(key)
        per_domain[row["domain"]] += 1
        row.update(search_term=query["term"], search_aliases="; ".join(query["aliases"]), _query=query)
        chosen.append(row)
        if len(chosen) == TARGET:
            break
    print(f"отобрано {len(chosen)}: {dict(per_domain)}; пропущено {dict(skipped)}")
    print(f"  из них пар к золоту: {sum(r['_tier'] == 0 for r in chosen)}")

    results = bts.parallel(lambda r: featurize(r["name_ru"], r["domain"], bts.SNAPSHOT, query=r["_query"]),
                           chosen, 16, "признаки")
    for row, res in zip(chosen, results):
        if isinstance(res, Exception):
            raise RuntimeError(f"{row['id']}: {res}")
        row.update(res)
        row["technology_group_id"] = "tech_" + hashlib.sha256(bts.norm_term(row["search_term"]).encode()).hexdigest()[:12]
    reference = __import__("json").loads(bts.REFERENCE.read_text(encoding="utf-8"))
    bts.add_percentiles(chosen, reference)
    bts.write(OUT, chosen, bts.META + bts.SERVICE + FEATURE_NAMES)
    print(f"записано: {OUT.relative_to(bts.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
