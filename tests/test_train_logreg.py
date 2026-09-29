import csv

import pytest

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from scripts import prepare_organizer_validation as organizer
from scripts import train_logreg as training


def test_fixed_split_keeps_last_100_out_and_rejects_reordered_rows():
    positive = [{"id": f"p{i}", "label": "1", "primary_source_url": f"https://p/{i}"}
                for i in range(350)]
    negative = [{"id": f"n{i}", "label": "0", "primary_source_url": f"https://n/{i}",
                 "selection_rule": training.EXTENSION_TAG if i >= 350 else "old"}
                for i in range(450)]

    train, held_out = training.split_fixed(positive, negative)

    assert len(train) == 700
    assert len(held_out) == 100
    assert train[-1]["id"] == "n349"
    assert held_out[0]["id"] == "n350"
    negative[349]["selection_rule"] = training.EXTENSION_TAG
    with pytest.raises(ValueError, match="validation negative"):
        training.split_fixed(positive, negative)


def test_organizer_features_need_real_values(tmp_path):
    manifest = [{"organizer_id": f"organizer_{i:03d}"} for i in range(1, 101)]
    feature_path = tmp_path / "organizer_features.csv"
    fields = ["organizer_id", "feature_schema_version", *FEATURE_NAMES]
    rows = [{"organizer_id": item["organizer_id"],
             "feature_schema_version": FEATURE_SCHEMA_VERSION,
             **{name: "" for name in FEATURE_NAMES}} for item in manifest]

    def write():
        with feature_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write()
    _, status = training.organizer_validation_rows(feature_path, manifest, list(FEATURE_NAMES))
    assert status["status"] == "insufficient_feature_coverage"
    for row in rows:
        for name in FEATURE_NAMES[:5]:
            row[name] = "0"
    write()
    _, status = training.organizer_validation_rows(feature_path, manifest, list(FEATURE_NAMES))
    assert status["status"] == "ready"


def test_manifest_does_not_export_author_scores_or_explanations(tmp_path, monkeypatch):
    records = [{"id": i, "title": f"Технология {i}", "domain": "Область",
                "reference_score": 7, "reference_explanation": "author label hint"}
               for i in range(1, 101)]
    monkeypatch.setattr(organizer, "read_catalog", lambda data_dir: {"records": records})
    path = tmp_path / "manifest.csv"

    organizer.build(path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["organizer_id", "name_original", "domain_original",
                                     "snapshot_date", "label"]
        assert len(list(reader)) == 100
