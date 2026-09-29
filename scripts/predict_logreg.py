"""Score rows with the trained 63-feature logistic baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from scripts.train_logreg import ROOT, numeric_matrix, read_csv

DEFAULT_MODEL = ROOT / "storage/ml/logreg_v1/model.joblib"


def score(model_path: Path, input_path: Path, output_path: Path) -> int:
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Model feature schema version mismatch")
    columns, rows = read_csv(input_path)
    if not rows:
        raise ValueError("Input CSV has no rows")
    if not set(FEATURE_NAMES).issubset(columns):
        raise ValueError("Input CSV must contain all 63 canonical feature columns")
    if any(row.get("feature_schema_version") != FEATURE_SCHEMA_VERSION for row in rows):
        raise ValueError("Input row feature schema version mismatch")
    features = bundle["feature_names"]
    matrix = numeric_matrix(rows, features)
    estimator = bundle["estimator"]
    probabilities = estimator.predict_proba(matrix)[:, 1]
    threshold = bundle["threshold"]
    active = estimator.named_steps["vary"].get_support()
    active_names = [name for name, keep in zip(features, active) if keep]
    transformed = estimator.named_steps["scale"].transform(
        estimator.named_steps["vary"].transform(
            estimator.named_steps["impute"].transform(matrix)))
    contributions = transformed * estimator.named_steps["logreg"].coef_[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "name_original", "positive_score",
                                                       "predicted_label", "top_contributions_json"])
        writer.writeheader()
        for row, probability, parts in zip(rows, probabilities, contributions):
            top = sorted(zip(active_names, parts), key=lambda item: abs(item[1]), reverse=True)[:5]
            writer.writerow({
                "id": row.get("id") or row.get("organizer_id") or "",
                "name_original": row.get("name_original") or "",
                "positive_score": round(float(probability), 6),
                "predicted_label": int(probability >= threshold),
                "top_contributions_json": json.dumps(
                    [{"feature": name, "log_odds_contribution": round(float(value), 6)}
                     for name, value in top], ensure_ascii=False),
            })
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "storage/ml/logreg_v1/predictions.csv")
    args = parser.parse_args()
    count = score(args.model, args.input_csv, args.output)
    print(f"scored={count} output={args.output}")


if __name__ == "__main__":
    main()
