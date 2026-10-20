"""Пересчёт признаков существующей выборки полным featurize со всеми источниками.

Набор model_compare/gpt-5.6-luna (обучение logreg_v1) собирался без прямых коннекторов:
денежные признаки там только из чтения новостей, а сервис при оценке подключает
коннекторы. Здесь те же строки (термин и алиасы — из строки, не выводятся заново)
проходят featurize на свой срез со всеми источниками (RADAR_EXTRA_CONNECTORS=all) —
так же, как ретро-строки (retro_dataset.py), только срез 2026-09-22 снимался вживую:
historical=False, GitHub и Hugging Face остаются (их ответы большей частью в кэше).

paper_volume_percentile_domain пересчитывается по новой обучающей выборке; опорное
распределение сервиса (app/ml/features/paper_volume_reference.json) не трогаем — его
обновит обучение v2.

Выход: ml/datasets/enriched/{training,gold,validation}.csv и report.json; строки, где
featurize упал, остаются со старыми признаками и пометкой в enrich_status.
Запуск: python ml/scripts/enrich_training.py [--limit 20] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

os.environ.setdefault("RADAR_EXTRA_CONNECTORS", "all")
os.environ.setdefault("FEATURE_LLM_TRIES", "2")  # сеть до OpenRouter бывает нестабильной
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_training_set as bts  # noqa: E402  (грузит .env и настройки источников)

from app.ml.features.extract import featurize  # noqa: E402
from app.ml.features.schema import FEATURE_NAMES  # noqa: E402

SOURCE = bts.DATASETS / "model_compare" / "gpt-5.6-luna"
OUT = bts.DATASETS / "enriched"
PARTS = ("training", "gold", "validation")
MONEY = ["funding_log_24m", "funding_round_count_24m", "public_grants_log_36m", "procurement_presence",
         "missing_funding", "missing_adoption",
         "source_type_diversity"]


def query(row: dict) -> dict:
    return {"term": row["search_term"], "aliases": [a for a in row["search_aliases"].split("; ") if a],
            "is_technology": row.get("is_technology_flag") == "1"}


def enrich(row: dict) -> dict:
    result = featurize(row["name_ru"], row["domain"], date.fromisoformat(row["snapshot_date"]),
                       query=query(row), historical=False)
    return {**row, **{k: result[k] for k in (*FEATURE_NAMES, "extractor_version", "extraction_status",
                                              "llm_stage", "evidence_digest") if k in result},
            "enrich_status": "ok"}


def filled(rows: list[dict], name: str) -> int:
    """Строки с ненулевым значением признака."""
    def nonzero(value) -> bool:
        try:
            return float(value) != 0
        except (TypeError, ValueError):
            return False
    return sum(1 for r in rows if nonzero(r.get(name)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="пробный прогон: N строк из каждой части")
    args = parser.parse_args()

    data = {part: bts.read(SOURCE / f"{part}.csv") for part in PARTS}
    if args.limit:  # детерминированная смешанная подвыборка
        data = {part: sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[: args.limit]
                for part, rows in data.items()}
    report: dict = {"rows": {p: len(r) for p, r in data.items()}, "fill_before": {}, "fill_after": {}}
    for part, rows in data.items():
        report["fill_before"][part] = {name: filled(rows, name) for name in MONEY}
        results = bts.parallel(enrich, rows, args.workers, part)
        data[part] = [res if not isinstance(res, Exception)
                      else {**row, "enrich_status": f"error: {str(res)[:120]}"}
                      for row, res in zip(rows, results)]
        report["fill_after"][part] = {name: filled(data[part], name) for name in MONEY}

    reference = bts.add_percentiles([r for r in data["training"] if r["enrich_status"] == "ok"])
    for part in ("gold", "validation"):
        bts.add_percentiles(data[part], reference)

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = ".debug" if args.limit else ""
    fields = bts.META + bts.SERVICE + FEATURE_NAMES + ["enrich_status"]
    for part, rows in data.items():
        bts.write(OUT / f"{part}{suffix}.csv", rows, fields)
    report["status"] = {p: Counter(r["enrich_status"].split(":")[0] for r in rows) for p, rows in data.items()}
    before = {r["id"]: r for part in PARTS for r in bts.read(SOURCE / f"{part}.csv")}
    report["changed_features"] = dict(Counter(
        name for part in PARTS for r in data[part] if r["enrich_status"] == "ok"
        for name in FEATURE_NAMES if str(r.get(name, "")) != str(before[r["id"]].get(name, ""))).most_common())
    (OUT / f"report{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "changed_features"}, ensure_ascii=False, indent=1))
    print("изменились признаки:", report["changed_features"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
