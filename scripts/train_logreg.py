"""Train a leakage-checked logistic baseline on 350 positive + 350 negative rows.

The organizer Excel is never a feature source. Its 100 positive identities are
held out until an independently computed 63-feature CSV is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
POSITIVE = ROOT / "data/training/positive_weak_signals_global.csv"
NEGATIVE = ROOT / "data/training/negative_weak_signals_global.csv"
ORGANIZER_MANIFEST = ROOT / "data/training/organizer_validation_manifest.csv"
ORGANIZER_FEATURES = ROOT / "data/training/organizer_100_features.csv"
EXCEL = ROOT / "data/100_слабых_технологических_сигналов_сентябрь_2026.xlsx"
OUTPUT = ROOT / "storage/ml/logreg_v1"
EXTENSION_TAG = "github_sector_product_name_v1"
MISSINGNESS_FLAGS = {name for name in FEATURE_NAMES if name.startswith("missing_")}
C_GRID = (0.01, 0.1, 1.0, 10.0)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fixed(positive: list[dict], negative: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(positive) != 350 or len(negative) != 450:
        raise ValueError(f"Expected 350 positives and 450 negatives; got {len(positive)}, {len(negative)}")
    if any(row.get("label") != "1" for row in positive):
        raise ValueError("Positive CSV contains a non-positive label")
    if any(row.get("label") != "0" for row in negative):
        raise ValueError("Negative CSV contains a non-negative label")
    if any(row.get("selection_rule", "").startswith(EXTENSION_TAG) for row in negative[:350]):
        raise ValueError("A validation negative appears among first 350")
    if any(not row.get("selection_rule", "").startswith(EXTENSION_TAG) for row in negative[350:]):
        raise ValueError("Last 100 negatives are not the frozen extension")
    train = positive + negative[:350]
    validation_negative = negative[350:]
    if len({row["id"] for row in train + validation_negative}) != 800:
        raise ValueError("Duplicate row IDs across train and validation")
    train_sources = {row["primary_source_url"].casefold() for row in train}
    if train_sources & {row["primary_source_url"].casefold() for row in validation_negative}:
        raise ValueError("Primary source overlaps train and validation")
    if {row.get("technology_group_id") for row in train if row.get("technology_group_id")} & {
        row.get("technology_group_id") for row in validation_negative if row.get("technology_group_id")
    }:
        raise ValueError("Technology group overlaps train and validation")
    return train, validation_negative


def numeric_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(features)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, feature in enumerate(features):
            raw = str(row.get(feature, "") or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"Non-numeric feature {feature} in row {row.get('id', i)}: {raw}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite feature {feature} in row {row.get('id', i)}")
            matrix[i, j] = value
    return matrix


def model_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("vary", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear", max_iter=3000,
                                       random_state=seed)),
    ])


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    predicted = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "count": int(len(y_true)), "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def grouped_nested_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      seed: int) -> tuple[dict, list[dict], np.ndarray]:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = np.full(len(y), np.nan)
    folds = []
    for fold_number, (train_idx, test_idx) in enumerate(outer.split(x, y, groups), 1):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=seed + fold_number)
        search = GridSearchCV(model_pipeline(seed),
                              param_grid={"logreg__C": C_GRID},
                              scoring="balanced_accuracy", cv=inner,
                              error_score="raise", n_jobs=1)
        search.fit(x[train_idx], y[train_idx], groups=groups[train_idx])
        probabilities[test_idx] = search.best_estimator_.predict_proba(x[test_idx])[:, 1]
        folds.append({"fold": fold_number, "train": int(len(train_idx)),
                      "test": int(len(test_idx)), "best_C": float(search.best_params_["logreg__C"]),
                      "test_positives": int(y[test_idx].sum()),
                      "test_negatives": int(len(test_idx) - y[test_idx].sum())})
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Nested CV left an unscored training row")
    return classification_metrics(y, probabilities), folds, probabilities


def organizer_validation_rows(path: Path, manifest: list[dict], features: list[str]) -> tuple[list[dict], dict]:
    if not path.exists():
        return [], {"status": "feature_file_missing", "expected_path": str(path),
                    "covered": 0, "required": 100}
    columns, rows = read_csv(path)
    required = set(FEATURE_NAMES) | {"organizer_id", "feature_schema_version"}
    absent = required - set(columns)
    if absent:
        raise ValueError(f"Organizer feature file missing columns: {sorted(absent)}")
    by_id = {row["organizer_id"]: row for row in rows}
    expected_ids = {row["organizer_id"] for row in manifest}
    if len(rows) != 100 or set(by_id) != expected_ids:
        raise ValueError("Organizer feature file must have exactly the 100 manifest IDs")
    if any(row["feature_schema_version"] != FEATURE_SCHEMA_VERSION for row in rows):
        raise ValueError("Organizer feature schema version mismatch")
    ordered = [by_id[row["organizer_id"]] for row in manifest]
    matrix = numeric_matrix(ordered, features)
    counts = np.isfinite(matrix).sum(axis=1)
    covered = int((counts >= 5).sum())
    if covered < 100:
        return ordered, {"status": "insufficient_feature_coverage", "covered": covered,
                         "required": 100, "minimum_nonmissing_features_per_row": 5,
                         "minimum_found": int(counts.min())}
    return ordered, {"status": "ready", "covered": 100, "required": 100}


def train(args: argparse.Namespace) -> dict:
    positive_cols, positive = read_csv(args.positive)
    negative_cols, negative = read_csv(args.negative)
    if not set(FEATURE_NAMES).issubset(positive_cols) or not set(FEATURE_NAMES).issubset(negative_cols):
        raise ValueError("Both training CSVs must contain all 63 canonical feature columns")
    for row in positive + negative:
        if row.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Feature schema mismatch in {row.get('id')}")
    train_rows, validation_negative = split_fixed(positive, negative)
    features = [name for name in FEATURE_NAMES if args.include_missingness or name not in MISSINGNESS_FLAGS]
    x = numeric_matrix(train_rows, features)
    y = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    groups = np.asarray([row["technology_group_id"] for row in train_rows])
    if len(set(groups)) < 15:
        raise ValueError("Too few technology groups for grouped CV")
    cv_metrics, folds, cv_probabilities = grouped_nested_cv(x, y, groups, args.seed)
    negative_subtype_cv = {}
    for subtype in sorted({row["negative_type"] for row in train_rows if row["label"] == "0"}):
        indices = [i for i, row in enumerate(train_rows)
                   if row["label"] == "0" and row["negative_type"] == subtype]
        negative_subtype_cv[subtype] = {
            "count": len(indices),
            "specificity": float((cv_probabilities[indices] < 0.5).mean()),
        }
    final_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 100)
    final_search = GridSearchCV(model_pipeline(args.seed),
                                param_grid={"logreg__C": C_GRID},
                                scoring="balanced_accuracy", cv=final_inner,
                                error_score="raise", n_jobs=1)
    final_search.fit(x, y, groups=groups)
    estimator = final_search.best_estimator_
    negative_x = numeric_matrix(validation_negative, features)
    negative_prob = estimator.predict_proba(negative_x)[:, 1]
    negative_predictions = (negative_prob >= 0.5).astype(int)
    _, manifest = read_csv(args.organizer_manifest)
    if len(manifest) != 100 or {row["organizer_id"] for row in manifest} != {
        f"organizer_{i:03d}" for i in range(1, 101)
    }:
        raise ValueError("Organizer validation manifest must contain IDs 001–100")
    organizer_rows, organizer_status = organizer_validation_rows(args.organizer_features, manifest, features)
    validation = {"negative_100": {
        "count": 100, "subtype": dict(Counter(row["negative_type"] for row in validation_negative)),
        "specificity": float((negative_predictions == 0).mean()),
        "false_positive_rate": float((negative_predictions == 1).mean()),
        "mean_positive_score": float(negative_prob.mean()),
    }, "organizer_100": organizer_status, "binary_200": {"status": "unavailable"}}
    if organizer_status["status"] == "ready":
        organizer_prob = estimator.predict_proba(numeric_matrix(organizer_rows, features))[:, 1]
        by_query_scope = {}
        for scope in sorted({row.get("crossref_match_scope", "unknown") for row in organizer_rows}):
            indices = [i for i, row in enumerate(organizer_rows)
                       if row.get("crossref_match_scope", "unknown") == scope]
            by_query_scope[scope] = {
                "count": len(indices),
                "recall": float((organizer_prob[indices] >= 0.5).mean()),
                "mean_positive_score": float(organizer_prob[indices].mean()),
            }
        validation["organizer_100"].update({
            "recall": float((organizer_prob >= 0.5).mean()),
            "mean_positive_score": float(organizer_prob.mean()),
            "recall_by_crossref_match_scope": by_query_scope,
        })
        validation["binary_200"] = {"status": "computed",
                                    **classification_metrics(
                                        np.asarray([1] * 100 + [0] * 100),
                                        np.concatenate([organizer_prob, negative_prob]))}
    active = estimator.named_steps["vary"].get_support()
    active_features = [name for name, keep in zip(features, active) if keep]
    coefficients = estimator.named_steps["logreg"].coef_[0]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "L2 logistic regression; median imputation, variance filtering, standard scaling",
        "label_quality": "bronze_provisional_not_expert_adjudicated",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_feature_count": len(features), "active_feature_count": len(active_features),
        "active_features": active_features,
        "standardized_coefficients": {name: float(value)
                                      for name, value in zip(active_features, coefficients)},
        "missingness_flags_included": args.include_missingness,
        "threshold": 0.5, "best_C": float(final_search.best_params_["logreg__C"]),
        "training_protocol": {
            "seed": args.seed,
            "c_grid": list(C_GRID),
            "outer_cv": "StratifiedGroupKFold: 5 folds, shuffle=True",
            "outer_fold_selection_cv": "StratifiedGroupKFold: 3 folds, seed + outer_fold_number",
            "final_selection_cv": "StratifiedGroupKFold: 5 folds, seed + 100",
            "selection_metric": "balanced_accuracy",
            "decision_threshold": 0.5,
            "probability_calibration": "none",
            "group_key": "technology_group_id",
            "include_missingness_flags": args.include_missingness,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__},
        "training_script_sha256": file_sha256(Path(__file__).resolve()),
        "split": {"train_positive": 350, "train_negative": 350,
                  "validation_positive": 100, "validation_negative": 100,
                  "negative_validation_rule": f"last_100_selection_rule_prefix:{EXTENSION_TAG}",
                  "seed": args.seed, "group_key": "technology_group_id"},
        "nested_train_cv": {"metrics": cv_metrics, "folds": folds,
                            "negative_subtype_specificity": negative_subtype_cv,
                            "note": "Diagnostic only: provisional labels and related corpus construction."},
        "validation": validation,
        "dataset_sha256": {"positive": file_sha256(args.positive),
                           "negative": file_sha256(args.negative),
                           "organizer_manifest": file_sha256(args.organizer_manifest),
                           "organizer_features": (file_sha256(args.organizer_features)
                                                  if args.organizer_features.exists() else None),
                           "organizer_excel": file_sha256(EXCEL)},
        "limitations": [
            "The organizer rows have no comparable feature vectors until independently enriched.",
            "The 100 validation negatives are all N3b product-name cases; specificity does not generalize to N1/N2.",
            "Bronze labels are unreviewed and train CV is not a trusted final benchmark.",
            "The logistic scores are not calibrated probabilities.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": estimator, "feature_names": features,
                 "schema_version": FEATURE_SCHEMA_VERSION, "threshold": 0.5},
                args.output_dir / "model.joblib")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                          indent=2), encoding="utf-8")
    with (args.output_dir / "negative_holdout_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "name_original", "label",
                                                       "positive_score", "predicted_label"])
        writer.writeheader()
        for row, score, prediction in zip(validation_negative, negative_prob, negative_predictions):
            writer.writerow({"id": row["id"], "name_original": row["name_original"],
                             "label": 0, "positive_score": round(float(score), 6),
                             "predicted_label": int(prediction)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, default=POSITIVE)
    parser.add_argument("--negative", type=Path, default=NEGATIVE)
    parser.add_argument("--organizer-manifest", type=Path, default=ORGANIZER_MANIFEST)
    parser.add_argument("--organizer-features", type=Path, default=ORGANIZER_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-missingness", action="store_true")
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({"train_rows": 700, "validation_negative": 100,
                      "organizer_validation": report["validation"]["organizer_100"]["status"],
                      "binary_validation": report["validation"]["binary_200"]["status"],
                      "nested_cv_balanced_accuracy": report["nested_train_cv"]["metrics"]["balanced_accuracy"],
                      "negative_holdout_specificity": report["validation"]["negative_100"]["specificity"],
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
