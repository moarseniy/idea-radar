import csv
import math
from pathlib import Path

from app.ml.features.schema import FEATURE_NAMES
from app.ml.scorer import FEATURE_LABELS, Scorer, display_value, exclusion_reasons

ROOT = Path(__file__).resolve().parents[1]


def gold_rows():
    return list(csv.DictReader((ROOT / "ml/datasets/gold_features.csv").open(encoding="utf-8")))


def test_every_schema_feature_has_russian_label():
    assert set(FEATURE_LABELS) == set(FEATURE_NAMES)


def test_contributions_sum_to_logit_exactly():
    scorer = Scorer.load()
    art = scorer.artifact
    result = scorer.score(gold_rows()[0])
    base = art["calibration"]["a"] * art["intercept"] + art["calibration"]["b"]
    assert math.isclose(base + sum(c["contribution"] for c in result["contributions"]), result["logit"], abs_tol=1e-3)
    assert math.isclose(result["probability"], 1 / (1 + math.exp(-result["logit"])))


def test_missing_values_are_imputed_not_zeroed():
    scorer = Scorer.load()
    result = scorer.score({})
    assert all(c["imputed"] for c in result["contributions"])
    assert 0.0 < result["probability"] < 1.0


def test_model_artifact_matches_schema_and_reports_validation():
    art = Scorer.load().artifact
    assert set(art["features"]) <= set(FEATURE_NAMES)
    assert "rebranding_similarity_to_mature" not in art["features"]
    assert len(art["features"]) == len(art["coef"]) == len(art["mean"]) == len(art["scale"]) == len(art["median"])
    assert art["validation"]["f1"] >= 0.8


def test_live_candidate_extraction_requests_only_the_serving_model_inputs(monkeypatch):
    from datetime import date

    from app.ml import scorer as model
    from app.ml.features import extract

    requested = {}

    def fake_featurize(*_args, **kwargs):
        requested.update(kwargs)
        return {"search_term": "edge sensor", "snapshot_date": "2026-09-22",
                "extraction_status": "ok"}

    monkeypatch.setattr(extract, "featurize", fake_featurize)
    result = model.score_candidate("Edge sensor", "Edge", date(2026, 9, 22))

    assert requested["requested_features"] == model.default_scorer().features
    assert len(result["contributions"]) == len(model.default_scorer().features)


def test_exclusion_reasons_and_display_values():
    reasons = exclusion_reasons({"stage_scaling_or_mature": "1", "is_technology_flag": "0",
                                 "first_evidence_age_years": "30", "paper_growth_2y": "-0.2"})
    assert len(reasons) == 3
    assert exclusion_reasons({"stage_pilot": "1", "is_technology_flag": "1"}) == []
    assert display_value("papers_log_3y", math.log1p(20)) == "20"
    assert display_value("missing_media", 1.0) == "да"
    assert display_value("press_release_share_12m", 0.25) == "25%"
    assert display_value("papers_log_3y", None) == "нет данных"
