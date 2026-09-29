"""Обучение логрега на training_merged.csv и валидация на золоте + валидационных негативах.

Всё, что настраивается (C, порог), выбирается только на обучающей выборке через
StratifiedGroupKFold по technology_group_id. Валидация (100 позитивов золота +
validation_negatives.csv) используется один раз — для итоговых метрик.

Варианты обучающей выборки сравниваются на одной и той же валидации:
  merged  — наши строки + строки Арсения;
  ours    — только наши;
  arseniy — только Арсения.

Выход: ml/reports/train_eval.json, ml/reports/train_eval.md,
       storage/ml/logreg_merged_v1/model.joblib (вариант merged).
Запуск: python ml/scripts/train_eval.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION  # noqa: E402

DATASETS = ROOT / "ml" / "datasets"
REPORTS = ROOT / "ml" / "reports"
MODEL_DIR = ROOT / "storage" / "ml" / "logreg_merged_v1"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
SEED = 0


def read(name: str) -> list[dict]:
    return list(csv.DictReader((DATASETS / name).open(encoding="utf-8")))


def matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.array([[float(r[f]) if r[f] not in ("", None) else np.nan for f in features] for r in rows])


def model(c: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def oof(rows, features, c):
    y = np.array([int(r["label"]) for r in rows])
    groups = [r["technology_group_id"] for r in rows]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    return y, cross_val_predict(model(c), matrix(rows, features), y, groups=groups, cv=cv,
                                method="predict_proba")[:, 1]


def metrics(y, p, thr) -> dict:
    pred = p >= thr
    return {"precision": round(precision_score(y, pred, zero_division=0), 3),
            "recall": round(recall_score(y, pred), 3), "f1": round(f1_score(y, pred), 3),
            "roc_auc": round(roc_auc_score(y, p), 3), "pr_auc": round(average_precision_score(y, p), 3),
            "threshold": round(float(thr), 3)}


def fit_variant(name: str, rows: list[dict], validation: list[dict]) -> dict:
    features = [f for f in FEATURE_NAMES if len({r[f] for r in rows}) > 1]
    # C и порог — по out-of-fold предсказаниям на обучении.
    scores = {}
    for c in C_GRID:
        y, p = oof(rows, features, c)
        scores[c] = roc_auc_score(y, p)
    best_c = max(C_GRID, key=lambda c: (round(scores[c], 3), -c))
    y, p = oof(rows, features, best_c)
    grid = np.linspace(0.2, 0.8, 61)
    thr = float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])

    final = model(best_c).fit(matrix(rows, features), np.array([int(r["label"]) for r in rows]))
    yv = np.array([int(r["label"]) for r in validation])
    pv = final.predict_proba(matrix(validation, features))[:, 1]

    by_domain = defaultdict(lambda: Counter())
    for r, prob in zip(validation, pv):
        key = "pos" if r["label"] == "1" else "neg"
        by_domain[r["domain"]][f"{key}_total"] += 1
        by_domain[r["domain"]][f"{key}_ok"] += int((prob >= thr) == (r["label"] == "1"))
    neg_sources = Counter()
    for r, prob in zip(validation, pv):
        if r["label"] == "0":
            src = "пары к золоту" if "gold_negatives" in r["source_file"] else "резерв"
            neg_sources[f"{src}_total"] += 1
            neg_sources[f"{src}_ok"] += int(prob < thr)
    coef = final[-1].coef_[0]
    return {
        "variant": name, "train_rows": len(rows), "train_labels": dict(Counter(r["label"] for r in rows)),
        "features": len(features), "C": best_c, "cv_auc_by_C": {str(k): round(v, 4) for k, v in scores.items()},
        "cv": {"at_0.5": metrics(y, p, 0.5), "at_cv_threshold": metrics(y, p, thr)},
        "validation": {"at_0.5": metrics(yv, pv, 0.5), "at_cv_threshold": metrics(yv, pv, thr)},
        "validation_by_domain": {d: dict(v) for d, v in sorted(by_domain.items())},
        "validation_negatives_by_source": dict(neg_sources),
        "top_weights": [(f, round(float(w), 3)) for w, f in sorted(zip(coef, features), key=lambda t: -abs(t[0]))[:15]],
        "_model": final, "_features": features, "_threshold": thr,
        "_errors": sorted(((float(prob), r["label"], r["name_ru"][:90]) for r, prob in zip(validation, pv)
                           if (prob >= thr) != (r["label"] == "1")), key=lambda t: t[0]),
    }


def main() -> int:
    training = read("training_merged.csv")
    gold = read("gold_features.csv")
    negatives = read("validation_negatives.csv")
    for g in gold:
        g["label"] = "1"
    validation = gold + negatives
    print(f"обучение: {len(training)}, валидация: {len(gold)} позитивов + {len(negatives)} негативов")

    variants = {"merged": training,
                "ours": [r for r in training if r["source"] == "ours"],
                "arseniy": [r for r in training if r["source"] == "arseniy"]}
    results = {name: fit_variant(name, rows, validation) for name, rows in variants.items()}

    merged = results["merged"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": merged["_model"], "features": merged["_features"], "threshold": merged["_threshold"],
                 "feature_schema_version": FEATURE_SCHEMA_VERSION}, MODEL_DIR / "model.joblib")

    REPORTS.mkdir(exist_ok=True)
    public = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    (REPORTS / "train_eval.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Обучение и валидация логрега", "",
             f"Валидация: {len(gold)} позитивов золота + {len(negatives)} негативов "
             f"(`validation_negatives.csv`). C и порог выбраны по групповой CV на обучении.", "",
             "| Обучение | Строк | Признаков | C | Порог | CV F1 | Вал. P | Вал. R | Вал. F1 | Вал. ROC-AUC | Вал. PR-AUC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in public.values():
        cv, val = v["cv"]["at_cv_threshold"], v["validation"]["at_cv_threshold"]
        lines.append(f"| {v['variant']} | {v['train_rows']} | {v['features']} | {v['C']} | {cv['threshold']} | "
                     f"{cv['f1']} | {val['precision']} | {val['recall']} | {val['f1']} | {val['roc_auc']} | {val['pr_auc']} |")
    lines += ["", "## merged: по областям (верно / всего)", "", "| Область | Позитивы | Негативы |", "|---|---:|---:|"]
    for d, c in merged["validation_by_domain"].items():
        lines.append(f"| {d} | {c.get('pos_ok', 0)}/{c.get('pos_total', 0)} | {c.get('neg_ok', 0)}/{c.get('neg_total', 0)} |")
    ns = merged["validation_negatives_by_source"]
    lines += ["", f"Негативы — пары к золоту: {ns.get('пары к золоту_ok', 0)}/{ns.get('пары к золоту_total', 0)} "
              f"отклонено, резерв: {ns.get('резерв_ok', 0)}/{ns.get('резерв_total', 0)}.", "",
              "## merged: веса (стандартизованные признаки)", ""]
    lines += [f"- `{f}` {w:+.3f}" for f, w in merged["top_weights"]]
    lines += ["", "## merged: ошибки на валидации", "", "| p | Метка | Название |", "|---:|---:|---|"]
    lines += [f"| {p:.2f} | {lab} | {name} |" for p, lab, name in merged["_errors"]]
    (REPORTS / "train_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12 + len(public)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
