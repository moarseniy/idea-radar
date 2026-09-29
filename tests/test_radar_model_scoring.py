from concurrent.futures import Future

import app.radar.model_scoring as model_scoring
import pytest
from app.radar.model_scoring import ModelScorer, apply_model, candidate_name


def model(p, reasons=(), status="patents"):
    return {"probability": p, "is_signal": p >= 0.54, "threshold": 0.54, "confident": p > 0.75,
            "top_for": [], "top_against": [{"label": "Стадия: масштабирование / зрелость", "display_value": "да"}],
            "exclusion_reasons": list(reasons), "warnings": [], "search_term": "x", "extraction_status": status,
            "snapshot_date": "2026-09-22", "model": "logreg_v1", "feature_schema_version": "weak-signals-63-v2"}


def item(verdict="weak_signal", grounded=True):
    return {"verdict": verdict, "reason": "r", "score": 70, "limitations": [],
            "evidence": [{"source_id": "s1"}] if grounded else [],
            "description": {"grounded": grounded}, "predictors": [{"name": "maturity", "explanation": "внедрено в банке"}]}


def test_model_decides_and_score_is_probability():
    out = apply_model(item(), model(0.91))
    assert out["verdict"] == "weak_signal" and out["score"] == 91 and out["score_kind"] == "model"


def test_low_probability_is_excluded_with_model_reasons():
    out = apply_model(item(), model(0.2, ["По публикациям в СМИ технология на стадии масштабирования или зрелости"]))
    assert out["verdict"] == "mature" and "20%" in out["reason"]
    assert apply_model(item(), model(0.2, ["Это продукт, бренд или бизнес-модель, а не технология"]))["verdict"] == "hype_or_noise"


def test_llm_relevance_and_grounding_are_respected():
    assert apply_model(item("irrelevant"), model(0.99))["verdict"] == "irrelevant"
    assert apply_model(item(grounded=False), model(0.99))["verdict"] == "insufficient_evidence"


def test_llm_maturity_disagreement_is_visible():
    out = apply_model(item("mature"), model(0.9))
    assert out["verdict"] == "weak_signal" and out["llm_verdict"] == "mature"
    assert any("внедрено в банке" in x for x in out["limitations"])


def test_missing_model_keeps_llm_decision():
    out = apply_model(item("mature"), None)
    assert out["verdict"] == "mature" and out["model"] is None


def test_missing_logreg_score_cannot_confirm_a_weak_signal():
    out = apply_model(item("weak_signal"), None)
    assert out["verdict"] == "insufficient_evidence"
    assert "Логистическая оценка не завершена" in out["reason"]


def test_fallback_logreg_score_is_present_but_cannot_confirm_signal():
    from datetime import date
    from app.ml.scorer import default_scorer

    scorer = ModelScorer.__new__(ModelScorer)
    scorer.scorer = default_scorer()
    scorer.as_of = date(2026, 9, 29)
    fallback = scorer.fallback_result("candidate-1", "Сбор признаков не уложился в лимит времени.")
    out = apply_model(item(), fallback)

    assert fallback["fallback"] is True
    assert fallback["extraction_status"] == "fallback_median_imputation"
    assert out["score_kind"] == "model_partial" and out["model"] is not None
    assert out["verdict"] == "insufficient_evidence"
    assert any("медианной подстановкой" in text for text in out["limitations"])


def test_fallback_logreg_does_not_replace_existing_non_signal_verdict():
    from datetime import date
    from app.ml.scorer import default_scorer

    scorer = ModelScorer.__new__(ModelScorer)
    scorer.scorer = default_scorer()
    scorer.as_of = date(2026, 9, 29)
    fallback = scorer.fallback_result("candidate-1", "Сбор признаков завершился ошибкой.")

    assert apply_model(item("mature"), fallback)["verdict"] == "mature"


def test_candidate_name_joins_application():
    assert candidate_name({"title": "Нейроморфные чипы", "application": "edge-устройства"}) == "Нейроморфные чипы: edge-устройства"


def test_pending_feature_score_is_not_reported_as_a_failure(monkeypatch):
    scorer = ModelScorer.__new__(ModelScorer)
    scorer.futures = {"candidate-1": Future()}
    monkeypatch.setattr(model_scoring.logger, "warning", lambda *_args, **_kwargs: pytest.fail("unexpected warning"))

    assert scorer.result("candidate-1", 0.01) is None


def test_signal_needs_independent_source():
    blog_only = apply_model(item(), model(0.95), [{"url": "https://medium.com/p"}])
    assert blog_only["verdict"] == "insufficient_evidence" and "первичными индикаторами" in blog_only["reason"]
    good = apply_model(item(), model(0.95), [{"url": "https://techcrunch.com/a"}])
    assert good["verdict"] == "weak_signal"
    assert any("одного независимого" in x for x in good["limitations"])


def test_ranking_keeps_one_place_per_technology():
    from app.radar.analysis import rank_results

    def sig(i, term, score):
        return {"id": f"r_cand_{i}", "title": term, "verdict": "weak_signal", "score": score, "evidence": [],
                "model": {"search_term": term}}
    top, rejected = rank_results([sig(1, "synthetic data generation", 90), sig(2, "privacy-preserving synthetic data", 85),
                                  sig(3, "LLM agent security", 80), sig(4, "synthetic financial data", 70)], 15)
    assert [s["title"] for s in top] == ["synthetic data generation", "LLM agent security"]
    assert [r["title"] for r in top[0]["related"]] == ["privacy-preserving synthetic data", "synthetic financial data"]
    assert [s["rank"] for s in top] == [1, 2] and rejected == []


def test_final_candidate_shortlist_puts_weak_signals_first_then_sorts_by_score():
    from app.radar.analysis import rank_candidates

    assessments = [
        {"id": "a", "title": "low-confidence signal", "verdict": "weak_signal", "score": 61, "evidence": []},
        {"id": "b", "title": "mature technology", "verdict": "mature", "score": 94, "evidence": [1]},
        {"id": "c", "title": "unverified technology", "verdict": "insufficient_evidence", "score": 83, "evidence": []},
    ]
    top = rank_candidates(assessments, 3)
    assert [x["id"] for x in top] == ["a", "b", "c"]
    assert [x["verdict"] for x in top] == ["weak_signal", "mature", "insufficient_evidence"]
    assert [x["rank"] for x in top] == [1, 2, 3]


def test_final_candidate_shortlist_uses_remaining_slots_for_high_confidence_non_signals():
    from app.radar.analysis import rank_candidates

    assessments = [
        {"id": "signal-low", "verdict": "weak_signal", "score": 58, "evidence": []},
        {"id": "noise-high", "verdict": "hype_or_noise", "score": 96, "evidence": [1]},
        {"id": "signal-high", "verdict": "weak_signal", "score": 88, "evidence": [1]},
        {"id": "mature-mid", "verdict": "mature", "score": 78, "evidence": []},
    ]
    top = rank_candidates(assessments, 3)
    assert [item["id"] for item in top] == ["signal-high", "signal-low", "noise-high"]
    assert [item["verdict"] for item in top] == ["weak_signal", "weak_signal", "hype_or_noise"]


def test_final_shortlist_ranks_full_feature_scores_before_median_fallbacks():
    from app.radar.analysis import rank_candidates

    assessments = [
        {"id": "fallback", "verdict": "mature", "score": 54, "evidence": [],
         "model": {"fallback": True}},
        {"id": "full", "verdict": "mature", "score": 20, "evidence": [],
         "model": {"fallback": False}},
    ]

    assert [item["id"] for item in rank_candidates(assessments, 2)] == ["full", "fallback"]
