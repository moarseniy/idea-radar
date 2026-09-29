from __future__ import annotations

import csv
from collections import Counter
from datetime import date
from pathlib import Path

from app.ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from scripts.build_positive_dataset import load_candidates_file, stage_features

DATASET = Path(__file__).resolve().parents[1] / "data/training/positive_weak_signals_global.csv"


def test_dataset_provenance_and_shape():
    with DATASET.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames[-63:] == FEATURE_NAMES
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
    assert 300 <= len(rows) <= 400
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["name_original"] for row in rows}) == len(rows)
    assert len({row["domain_ru"] for row in rows}) >= 10
    assert len({row["original_language"] for row in rows}) >= 3
    assert all(row["label"] == "1" and row["data_tier"] == "bronze" for row in rows)
    assert all(row["review_status"] == "needs_human_review" for row in rows)
    assert all(row["feature_schema_version"] == FEATURE_SCHEMA_VERSION for row in rows)
    assert all(row["technology_group_id"] and row["name_original"] and row["name_ru"] for row in rows)
    assert all(row["primary_source_url"].startswith("https://") for row in rows)
    assert all(row["source_evidence_excerpt_original"] for row in rows)
    assert all(row["primary_source_crossref_id"].startswith("https://api.crossref.org/works/") for row in rows)
    assert all(date.fromisoformat(row["primary_source_date"]) <= date.fromisoformat(row["snapshot_date"]) for row in rows)


def test_missing_channels_are_not_synthetic_zeros():
    with DATASET.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    blocked = (
        "verified_pilots_log_24m", "production_deployments_log", "mass_market_flag",
        "patent_families_log_3y", "media_volume_normalized_12m", "funding_log_24m",
        "independent_high_trust_source_count_log", "unverifiable_source_share",
        "paper_volume_percentile_domain",
    )
    assert all(all(row[name] == "" for name in blocked) for row in rows)
    assert all(all(row[f"missing_{channel}"] == "1" for channel in ("patents", "media", "funding", "adoption")) for row in rows)
    assert all(row["missing_science"] == "0" for row in rows)
    assert all(sum(int(row[name]) for name in (
        "stage_research", "stage_poc", "stage_pilot", "stage_early_adoption", "stage_scaling_or_mature"
    )) == 1 for row in rows)
    assert Counter(row["source_match_method"] for row in rows)["same_abstract_sentence"] > 0
    assert all(row["science_count_scope"] == "relevant_works_in_top_200_crossref_results" for row in rows)


def test_stage_terms_resolve_to_one_stage():
    assert stage_features("A prototype and pilot were deployed in the real world") == [0, 0, 0, 1, 0]
    assert stage_features("Commercially available after a pilot") == [0, 0, 0, 0, 1]


def test_candidate_file_accepts_arbitrary_domain_and_language(tmp_path):
    path = tmp_path / "candidates.csv"
    fields = [
        "domain_original", "domain_ru", "technology_original", "technology_ru",
        "application_original", "application_ru", "name_original", "name_ru",
        "original_language", "search_query",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "domain_original": "Marine biotechnology",
            "domain_ru": "Морские биотехнологии",
            "technology_original": "microalgas editadas",
            "technology_ru": "редактированные микроводоросли",
            "application_original": "purificación de agua",
            "application_ru": "очистки воды",
            "name_original": "microalgas editadas para purificación de agua",
            "name_ru": "редактированные микроводоросли для очистки воды",
            "original_language": "es",
            "search_query": "microalgas editadas purificación de agua",
        })
    records = load_candidates_file(path)
    assert len(records) == 1
    assert records[0]["domain_ru"] == "Морские биотехнологии"
    assert records[0]["original_language"] == "es"
