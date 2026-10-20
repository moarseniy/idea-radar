"""logreg_v2: пересчитанная выборка (ml/datasets/enriched) + ретро-выборка машины времени.

Варианты на одной валидации (98 золотых позитивов + 89 негативов, пересчитанные признаки):
  v1          — артефакт logreg_v1 как есть;
  v2_old      — только пересчитанные 1327 строк;
  v2_retro    — плюс 684 ретро-строки, bronze с весом 0,3 (кандидат в v2);
  v2_retro_w1 — то же, bronze с весом 1;
  v2_silver   — ретро только silver.
У ретро-строк нет GitHub и Hugging Face (машина времени их отключает), поэтому признаки,
которые от них зависят, исключены во всех вариантах v2, а сервис для v2 берёт признаки
из всех коннекторов кандидата, кроме этих двух.

Разница AUC с v2_old — с бутстрэп-интервалом по валидации: она маленькая, и шум надо видеть.
Запуск: python ml/scripts/train_v2.py [--export]  (--export пишет app/ml/model/logreg_v2.json)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

os.environ.setdefault("RADAR_EXTRA_CONNECTORS", "all")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_model import EXCLUDED, ROOT, fit  # noqa: E402

from app.ml.scorer import Scorer  # noqa: E402
from app.radar.connectors import registry  # noqa: E402

DATA = ROOT / "ml" / "datasets"
OUT = ROOT / "app" / "ml" / "model" / "logreg_v2.json"
REPORT = ROOT / "ml" / "reports" / "logreg_v2.json"
NO_TIME_MACHINE = {"GitHub repositories", "Hugging Face"}
EXCLUDED_V2 = {
    **EXCLUDED,
    "missing_adoption": "зависит от звёзд GitHub и загрузок Hugging Face: у ретро-строк их нет",
    "source_type_diversity": "считает канал внедрения, а он зависит от GitHub и Hugging Face",
}


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def bootstrap_auc_diff(y, p_a, p_b, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    return [round(float(np.percentile(diffs, q)), 3) for q in (2.5, 97.5)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    old = read(DATA / "enriched" / "training.csv")
    retro = read(DATA / "retro" / "retro_training.csv")
    validation = read(DATA / "enriched" / "gold.csv") + read(DATA / "enriched" / "validation.csv")
    y = np.array([int(r["label"]) for r in validation])
    connectors = sorted(name for name, c in registry().items()
                        if "candidate" in c.scopes and name not in NO_TIME_MACHINE)

    def weights(rows, bronze):
        return np.array([1.0 if r.get("sample_weight", "") == "" else
                         (float(r["sample_weight"]) if bronze is None else
                          (bronze if r["tier"] == "bronze" and r["label"] == "1" else 1.0)) for r in rows])

    silver = [r for r in retro if not (r["label"] == "1" and r["tier"] == "bronze")]
    variants = {
        "v2_old": (old, None),
        "v2_retro": (old + retro, None),
        "v2_retro_w1": (old + retro, 1.0),
        "v2_silver": (old + silver, None),
    }
    results, probs = {}, {}
    v1 = Scorer(json.loads((ROOT / "app/ml/model/logreg_v1.json").read_text(encoding="utf-8")))
    probs["v1"] = np.array([v1.score(r)["probability"] for r in validation])
    results["v1"] = {"validation_auc": round(float(roc_auc_score(y, probs["v1"])), 3)}
    artifacts = {}
    for name, (rows, bronze) in variants.items():
        artifact = fit(rows, validation, "ml/datasets/enriched/training.csv"
                       + (" + ml/datasets/retro/retro_training.csv" if rows is not old else ""),
                       name="logreg_v2", excluded=EXCLUDED_V2, sample_weight=weights(rows, bronze),
                       feature_connectors=connectors)
        artifacts[name] = artifact
        scorer = Scorer(json.loads(json.dumps(artifact)))
        probs[name] = np.array([scorer.score(r)["probability"] for r in validation])
        results[name] = {"rows": len(rows), "C": artifact["C"], "threshold": round(artifact["threshold"], 2),
                         "cv": {k: artifact["cv"][k] for k in ("roc_auc", "pr_auc", "f1")},
                         "validation": {k: v for k, v in artifact["validation"].items()
                                        if k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "brier")}}
    for name in ("v1", "v2_retro", "v2_retro_w1", "v2_silver"):
        results[name]["auc_minus_v2_old_ci95"] = bootstrap_auc_diff(y, probs[name], probs["v2_old"])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1))
    if args.export:
        # LLM разметки — из версии извлечения строк ("featurize-v2:<модель>"): сервис
        # сверяет с ней свою FEATURE_MODEL и предупреждает о расхождении.
        llms = {r["extractor_version"].split(":", 1)[-1] for r in old + retro}
        if len(llms) != 1:
            raise SystemExit(f"строки размечены разными LLM: {sorted(llms)}")
        artifacts["v2_retro"]["feature_model"] = llms.pop()
        OUT.write_text(json.dumps(artifacts["v2_retro"], ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"записан {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
