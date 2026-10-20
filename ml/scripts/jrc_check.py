"""Внешняя проверка на JRC «Weak signals in Science and Technologies in 2021».

Позитивы — 93 сигнала выпуска 2021 (независимая разметка JRC, вне наших обучающих
выборок). Негативы — 89 зрелых технологий из негативов валидации (отобраны вручную, в
обучении v1 и v2 их нет). Все строки — featurize на AS_OF = 2021-06-30 в историческом
режиме (как ретро-выборка): только документы не позже этой даты, без GitHub/HF.

Перцентиль объёма публикаций — по общему распределению (``__all__``) для всех строк:
кластеры JRC не совпадают с нашими областями, и опора по области была бы только у
негативов. Технологии, чей термин уже есть в обучении v1/v2, считаются отдельно (утечка).

Выход: ml/datasets/external/jrc2021_features.csv, ml/reports/jrc2021_check.json.
Запуск: python ml/scripts/jrc_check.py [--workers 12] [--limit N]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault("RADAR_EXTRA_CONNECTORS", "all")
os.environ.setdefault("FEATURE_LLM_TRIES", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_training_set as bts  # noqa: E402

from app.ml.features.extract import derive_query, featurize  # noqa: E402
from app.ml.features.schema import FEATURE_NAMES  # noqa: E402
from app.ml.scorer import Scorer  # noqa: E402

AS_OF = date(2021, 6, 30)
OUT = bts.DATASETS / "external" / "jrc2021_features.csv"
REPORT = bts.ROOT / "ml" / "reports" / "jrc2021_check.json"
MODELS = {"v1": "app/ml/model/logreg_v1.json", "v2": "app/ml/model/logreg_v2.json"}
# В обучении v2 — без GitHub/HF; v1 обучалась без прямых коннекторов вовсе, но в сервисе
# получает признаки с ними: здесь обе модели видят один и тот же исторический featurize.


def rows() -> list[dict]:
    jrc = [{"id": r["signal_id"], "label": "1", "name": r["title_en"], "domain": r["cluster"], "kind": "jrc"}
           for r in bts.read(bts.DATASETS / "external" / "jrc_weak_signals.csv") if r["report_year"] == "2021"]
    negatives = [{"id": r["id"], "label": "0", "name": r["name_ru"], "domain": r["domain"], "kind": "validation",
                  "query": {"term": r["search_term"], "aliases": [a for a in r["search_aliases"].split("; ") if a],
                            "is_technology": r.get("is_technology_flag") == "1"}}
                 for r in bts.read(bts.DATASETS / "enriched" / "validation.csv")]
    return jrc + negatives


def one(row: dict) -> dict:
    query = row.get("query") or derive_query(row["name"], row["domain"])
    if not query["term"]:
        raise ValueError("нет поискового термина")
    return {**row, **featurize(row["name"], row["domain"], AS_OF, query=query, historical=True)}


def trained_terms() -> set[str]:
    out = set()
    for path in (bts.DATASETS / "enriched" / "training.csv", bts.DATASETS / "retro" / "retro_training.csv"):
        for r in bts.read(path):
            out |= {bts.norm_term(t) for t in [r["search_term"], *r["search_aliases"].split(";")] if t.strip()}
    return out


def evaluate(data: list[dict], scorers: dict[str, Scorer]) -> dict:
    y = np.array([int(r["label"]) for r in data])
    out = {"positives": int(y.sum()), "negatives": int(len(y) - y.sum())}
    if len(set(y)) < 2:
        return out
    for name, scorer in scorers.items():
        p = np.array([scorer.score(r)["probability"] for r in data])
        thr = scorer.artifact["threshold"]
        out[name] = {"roc_auc": round(float(roc_auc_score(y, p)), 3),
                     "pr_auc": round(float(average_precision_score(y, p)), 3),
                     "recall_at_threshold": round(float(np.mean(p[y == 1] >= thr)), 3),
                     "false_positive_rate": round(float(np.mean(p[y == 0] >= thr)), 3),
                     "median_p_positive": round(float(np.median(p[y == 1])), 3),
                     "median_p_negative": round(float(np.median(p[y == 0])), 3)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cached", action="store_true", help="только оценка по готовому jrc2021_features.csv")
    args = parser.parse_args()

    if args.cached:
        data = bts.read(OUT)
    else:
        items = rows()
        if args.limit:
            items = items[: args.limit] + items[-args.limit:]
        results = bts.parallel(one, items, args.workers, "признаки")
        data = [r for r in results if not isinstance(r, Exception)]
        failed = [f"{i['id']}: {str(r)[:80]}" for i, r in zip(items, results) if isinstance(r, Exception)]
        if failed:
            print("ошибки:", failed)
        for r in data:
            r["search_aliases"] = r.get("search_aliases", "")
        reference = json.loads((bts.REFERENCE).read_text(encoding="utf-8"))["__all__"]
        for r in data:
            if r.get("papers_log_3y") not in ("", None):
                r["paper_volume_percentile_domain"] = round(
                    bisect.bisect_right(reference, float(r["papers_log_3y"])) / len(reference), 6)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "label", "kind", "name", "domain", "search_term", "search_aliases", "extraction_status",
                  "llm_stage"] + FEATURE_NAMES
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        data = bts.read(OUT)

    taken = trained_terms()
    for r in data:
        r["leak"] = bool({bts.norm_term(t) for t in [r["search_term"], *r["search_aliases"].split(";")]
                          if t.strip()} & taken)
    scorers = {name: Scorer(json.loads((bts.ROOT / path).read_text(encoding="utf-8"))) for name, path in MODELS.items()}
    ai = {"Artificial Intelligence and Machine Learning", "Information and Communication Technologies",
          "Information & Communication technologies", "Digital Twin"}
    report = {
        "as_of": AS_OF.isoformat(),
        "all": evaluate(data, scorers),
        "without_leaks": evaluate([r for r in data if not r["leak"] or r["label"] == "0"], scorers),
        "jrc_ai_ict_only": evaluate([r for r in data if r["label"] == "0" or r["domain"] in ai], scorers),
        "leaked_positives": [r["search_term"] for r in data if r["leak"] and r["label"] == "1"],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
