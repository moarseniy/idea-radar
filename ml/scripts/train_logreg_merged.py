"""Train/evaluate logistic regression on the merged weak-signal dataset.

Uses all eligible rows from training_merged.csv except a deterministic,
group-held-out set of 100 negatives. gold_features.csv supplies 100 positive
validation rows only. Labels and features are provisional/bronze.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
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

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "ml/datasets/training_merged.csv"
GOLD = ROOT / "ml/datasets/gold_features.csv"
OUTPUT = ROOT / "storage/ml/merged_logreg_v1"
MISSINGNESS = {name for name in FEATURE_NAMES if name.startswith("missing_")}
C_GRID = (0.01, 0.1, 1.0, 10.0)
# Oversample the small claim/hype subtype for diagnostic visibility. This is a
# stress-test composition, not an estimate of negative-class prevalence.
NEGATIVE_QUOTAS = {"N1": 49, "N3b": 38, "N2": 10, "N3a": 2, "N3d": 1}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_term(term: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in term).split())


def numeric_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(features)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, feature in enumerate(features):
            value = str(row.get(feature, "") or "").strip()
            if not value:
                continue
            try:
                matrix[i, j] = float(value)
            except ValueError as exc:
                raise ValueError(f"Non-numeric {feature} in {row.get('id')}: {value!r}") from exc
            if not np.isfinite(matrix[i, j]):
                raise ValueError(f"Non-finite {feature} in {row.get('id')}")
    return matrix


def add_train_reference_percentiles(train: list[dict], others: list[dict]) -> dict:
    """Recalculate the domain percentile using training rows only."""
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in train:
        raw = row.get("papers_log_3y")
        if raw not in (None, ""):
            by_domain[row["domain"]].append(float(raw))
    reference = {domain: sorted(values) for domain, values in by_domain.items()}
    reference["__all__"] = sorted(value for values in by_domain.values() for value in values)
    if not reference["__all__"]:
        raise ValueError("No measured papers_log_3y values in training rows")
    for row in train + others:
        raw = row.get("papers_log_3y")
        if raw in (None, ""):
            row["paper_volume_percentile_domain"] = ""
            continue
        values = reference.get(row.get("domain", ""), [])
        if len(values) < 20:
            values = reference["__all__"]
        row["paper_volume_percentile_domain"] = round(
            bisect.bisect_right(values, float(raw)) / len(values), 6)
    return reference


def choose_negative_holdout(rows: list[dict], seed: int) -> list[dict]:
    negatives = [r for r in rows if r["label"] == "0"]
    selected: list[dict] = []
    selected_groups: set[str] = set()
    for subtype, quota in NEGATIVE_QUOTAS.items():
        candidates = sorted(
            (r for r in negatives if r["negative_type"] == subtype),
            key=lambda r: hashlib.sha256(f"{seed}|{r['id']}".encode()).hexdigest())
        picked = []
        for row in candidates:
            group = row.get("technology_group_id", "")
            if not group or group in selected_groups:
                continue
            picked.append(row)
            selected_groups.add(group)
            if len(picked) == quota:
                break
        if len(picked) != quota:
            raise ValueError(f"Need {quota} unique groups for {subtype}, found {len(picked)}")
        selected.extend(picked)
    if len(selected) != 100 or len({r["id"] for r in selected}) != 100:
        raise AssertionError("Negative holdout must contain 100 unique rows")
    return selected


def model_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("vary", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear", max_iter=3000,
                                       class_weight="balanced", random_state=seed)),
    ])


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "count": int(len(y_true)), "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def nested_group_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    seed: int) -> tuple[dict, list[dict]]:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = np.full(len(y), np.nan)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups), 1):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + fold)
        search = GridSearchCV(model_pipeline(seed), {"logreg__C": C_GRID},
                              scoring="balanced_accuracy", cv=inner,
                              error_score="raise", n_jobs=1)
        search.fit(x[train_idx], y[train_idx], groups=groups[train_idx])
        probabilities[test_idx] = search.best_estimator_.predict_proba(x[test_idx])[:, 1]
        folds.append({"fold": fold, "n_train": int(len(train_idx)),
                      "n_test": int(len(test_idx)), "best_C": float(search.best_params_["logreg__C"])})
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Grouped CV did not score every training row")
    return metrics(y, probabilities, 0.5), folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    train_columns, all_rows = read_csv(args.training)
    gold_columns, gold_rows = read_csv(args.gold)
    merged_rows_count = len(all_rows)
    required = set(FEATURE_NAMES) | {"id", "label", "feature_schema_version", "technology_group_id", "search_term"}
    for path, columns in ((args.training, train_columns), (args.gold, gold_columns)):
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if any(r["feature_schema_version"] != FEATURE_SCHEMA_VERSION for r in all_rows + gold_rows):
        raise ValueError("Feature schema version mismatch")
    if any(r["label"] not in {"0", "1"} for r in all_rows + gold_rows):
        raise ValueError("Labels must be binary")
    if len(gold_rows) != 100 or any(r["label"] != "1" for r in gold_rows):
        raise ValueError("Gold validation must contain exactly 100 positive examples")

    gold_terms = set()
    for row in gold_rows:
        if row.get("search_term"):
            gold_terms.add(norm_term(row["search_term"]))
        for alias in filter(None, row.get("search_aliases", "").split("; ")):
            gold_terms.add(norm_term(alias))
    overlap = [r for r in all_rows if norm_term(r.get("search_term", "")) in gold_terms]
    overlap_groups = {r.get("technology_group_id") for r in overlap if r.get("technology_group_id")}
    # The committed merger removed most exact gold collisions, but two exact
    # terms and several aliases remain. Exclude their entire groups here.
    all_rows = [r for r in all_rows if norm_term(r.get("search_term", "")) not in gold_terms
                and r.get("technology_group_id") not in overlap_groups]

    negative_holdout = choose_negative_holdout(all_rows, args.seed)
    holdout_ids = {r["id"] for r in negative_holdout}
    holdout_groups = {r["technology_group_id"] for r in negative_holdout}
    training_rows = [r for r in all_rows if r["id"] not in holdout_ids
                     and r.get("technology_group_id") not in holdout_groups]
    removed_for_groups = len(all_rows) - len(holdout_ids) - len(training_rows)
    if {r["technology_group_id"] for r in training_rows} & holdout_groups:
        raise AssertionError("Technology group leaked between train and negative holdout")
    if {r["id"] for r in training_rows} & holdout_ids:
        raise AssertionError("Negative holdout leaked into training")
    if len({r["id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate IDs in merged training data")
    if len({r["id"] for r in gold_rows}) != 100:
        raise ValueError("Duplicate IDs in gold data")

    # Refit the domain reference after holding out negatives; then apply that
    # same train-only reference to the negative holdout and all 100 gold rows.
    reference = add_train_reference_percentiles(training_rows, negative_holdout + gold_rows)
    features = [name for name in FEATURE_NAMES if name not in MISSINGNESS]
    x = numeric_matrix(training_rows, features)
    y = np.asarray([int(r["label"]) for r in training_rows], dtype=int)
    groups = np.asarray([r["technology_group_id"] for r in training_rows])
    if set(y) != {0, 1} or len(set(groups)) < 15:
        raise ValueError("Training split lacks both labels or enough technology groups")

    cv_metrics, cv_folds = nested_group_cv(x, y, groups, args.seed)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 100)
    final_search = GridSearchCV(model_pipeline(args.seed), {"logreg__C": C_GRID},
                                scoring="balanced_accuracy", cv=inner,
                                error_score="raise", n_jobs=1)
    final_search.fit(x, y, groups=groups)
    estimator = final_search.best_estimator_

    neg_prob = estimator.predict_proba(numeric_matrix(negative_holdout, features))[:, 1]
    gold_prob = estimator.predict_proba(numeric_matrix(gold_rows, features))[:, 1]
    y_validation = np.asarray([0] * 100 + [1] * 100)
    p_validation = np.concatenate([neg_prob, gold_prob])
    active_mask = estimator.named_steps["vary"].get_support()
    active_features = [name for name, active in zip(features, active_mask) if active]
    coefficients = estimator.named_steps["logreg"].coef_[0]
    subtype_metrics = {}
    neg_predictions = (neg_prob >= args.threshold).astype(int)
    for subtype in sorted(NEGATIVE_QUOTAS):
        idx = [i for i, row in enumerate(negative_holdout) if row["negative_type"] == subtype]
        if idx:
            subtype_metrics[subtype] = {
                "count": len(idx),
                "specificity": float((neg_predictions[idx] == 0).mean()),
                "mean_positive_score": float(neg_prob[idx].mean()),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": estimator, "feature_names": features,
                 "schema_version": FEATURE_SCHEMA_VERSION, "threshold": args.threshold},
                args.output_dir / "model.joblib")
    write_csv(args.output_dir / "negative_holdout.csv", negative_holdout, train_columns)
    write_csv(args.output_dir / "negative_holdout_predictions.csv", [
        {"id": r["id"], "name_original": r["name_original"],
         "negative_type": r["negative_type"], "positive_score": round(float(prob), 6),
         "predicted_label": int(prob >= args.threshold)}
        for r, prob in zip(negative_holdout, neg_prob)],
        ["id", "name_original", "negative_type", "positive_score", "predicted_label"])
    write_csv(args.output_dir / "gold_predictions.csv", [
        {"id": r["id"], "name_original": r["name_original"],
         "positive_score": round(float(prob), 6),
         "predicted_label": int(prob >= args.threshold)}
        for r, prob in zip(gold_rows, gold_prob)],
        ["id", "name_original", "positive_score", "predicted_label"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "L2 logistic regression; median imputation, variance filter, standard scaling; class_weight=balanced",
        "label_quality": "bronze_provisional_not_expert_adjudicated",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_features": len(features), "active_features": active_features,
        "standardized_coefficients": {name: float(value) for name, value in zip(active_features, coefficients)},
        "threshold": args.threshold, "best_C": float(final_search.best_params_["logreg__C"]),
        "split": {"merged_rows_input": merged_rows_count,
                  "training_rows_after_gold_filter": len(all_rows),
                  "training_rows_used": len(training_rows),
                  "gold_term_or_alias_rows_removed": len(overlap),
                  "gold_overlap_groups_removed": len(overlap_groups),
                  "training_labels": dict(Counter(r["label"] for r in training_rows)),
                  "validation_positive": len(gold_rows), "validation_negative": len(negative_holdout),
                  "negative_subtypes": dict(Counter(r["negative_type"] for r in negative_holdout)),
                  "negative_selection": "deterministic subtype-stratified, group-held-out stress test",
                  "negative_quotas": NEGATIVE_QUOTAS, "groups_removed_with_holdout": removed_for_groups,
                  "seed": args.seed, "group_key": "technology_group_id"},
        "nested_train_cv": {"metrics": cv_metrics, "folds": cv_folds,
                            "negative_subtype_specificity": subtype_metrics},
        "validation": {"negative_100": {"count": 100, "specificity": float((neg_predictions == 0).mean()),
                                          "false_positive_rate": float((neg_predictions == 1).mean()),
                                          "subtypes": subtype_metrics},
                       "gold_positive_100": {"count": 100, "recall": float((gold_prob >= args.threshold).mean()),
                                             "mean_positive_score": float(gold_prob.mean())},
                       "balanced_200": metrics(y_validation, p_validation, args.threshold)},
        "percentile_reference": "recomputed on training rows only, then applied to both validation sets",
        "limitations": [
            "Gold validation contains 100 positive cases from the organizer and has already informed feature design.",
            "Negative validation is a subtype-stratified stress test, oversampling N2 and rare subtypes; overall metrics are not prevalence-calibrated.",
            "Labels are provisional and not expert-adjudicated; this is an internal diagnostic, not an independent benchmark.",
            "Patent features are unavailable/constant in this dataset; logistic scores are not calibrated probabilities.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "paper_volume_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"train_rows": len(training_rows), "train_labels": report["split"]["training_labels"],
                      "validation_positive": len(gold_rows), "validation_negative": len(negative_holdout),
                      "negative_subtypes": report["split"]["negative_subtypes"],
                      "balanced_200": report["validation"]["balanced_200"],
                      "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
