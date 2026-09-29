from app.radar.models import Assessment, AssessmentBatch, Candidate, CandidateBatch, Finding, Predictor
from app.radar.provider import ResearchProvider


class LengthFinishReasonError(Exception):
    pass


def test_assessment_retries_overflow_with_smaller_context_and_returns_safe_fallback():
    provider = ResearchProvider.__new__(ResearchProvider)
    calls = []

    def structured(role, prompt, schema, max_tokens):
        calls.append((role, prompt, schema, max_tokens))
        raise LengthFinishReasonError()

    provider.structured = structured
    candidates = [
        {"id": "c1", "title": "Technology one", "verification_query": "verify one", "source_ids": ["s1"]},
        {"id": "c2", "title": "Technology two", "verification_query": "verify two", "source_ids": ["s2"]},
    ]
    sources = [
        {"id": "s1", "title": "Evidence one", "published_at": "2026-01-01", "type": "article",
         "trust": "high", "text": "Evidence text one " + "A" * 5000},
        {"id": "s2", "title": "Evidence two", "published_at": "2026-01-01", "type": "article",
         "trust": "high", "text": "Evidence text two " + "B" * 5000},
    ]

    result = provider.assess("topic", "2026-09-28", candidates, sources)

    assert [call[3] for call in calls] == [6000, 4200, 3000, 2000, 4200, 3000, 2000]
    assert "Technology one" in calls[1][1] and "Technology two" not in calls[1][1]
    assert "Evidence one" in calls[1][1] and "Evidence two" not in calls[1][1]
    assert "Technology two" in calls[4][1] and "Technology one" not in calls[4][1]
    assert len(calls[1][1]) > len(calls[2][1]) > len(calls[3][1])
    assert [item.candidate_id for item in result.assessments] == ["c1", "c2"]
    assert all(item.verdict == "insufficient_evidence" and not item.evidence for item in result.assessments)
    assert all("превысил лимит" in item.limitations[0] for item in result.assessments)


def test_assessment_keeps_completed_single_candidate_fallback():
    provider = ResearchProvider.__new__(ResearchProvider)
    calls = []
    finding = Finding(text="Подтверждённый фрагмент.", source_ids=["s1"])

    def structured(role, prompt, _schema, max_tokens):
        calls.append((role, prompt, max_tokens))
        if len(calls) == 1:
            raise LengthFinishReasonError()
        predictors = [Predictor(name=name, value="partial", explanation="Есть подтверждение.", source_ids=["s1"])
                      for name in ("early_stage", "novelty", "momentum", "evidence", "maturity")]
        return AssessmentBatch(assessments=[Assessment(
            candidate_id="c1", verdict="insufficient_evidence", reason="Недостаточно данных.", stage="Pilot",
            description=finding, why_now=finding, why_early=finding, advantage=finding, case=finding,
            case_kind="unknown", business_use="Возможное применение.", limitations=[], predictors=predictors,
            evidence=[],
        )])

    provider.structured = structured
    candidate = {"id": "c1", "title": "Technology one", "verification_query": "verify one", "source_ids": ["s1"]}
    source = {"id": "s1", "title": "Evidence one", "published_at": "2026-01-01", "type": "article",
              "trust": "high", "text": "Evidence text one"}

    result = provider.assess("topic", "2026-09-28", [candidate], [source])

    assert [call[2] for call in calls] == [4200, 3000]
    assert result.assessments[0].candidate_id == "c1"
    assert result.assessments[0].reason == "Недостаточно данных."


def test_consolidation_retries_in_chunks_then_merges_compacted_list():
    provider = ResearchProvider.__new__(ResearchProvider)
    provider.settings = type("Settings", (), {"max_candidates": 10})()
    provider.consolidation_chunked = False
    calls = []
    merged = Candidate(id="cand_merged", title="Merged technology", application="payment settlement",
                       domain="fintech", summary="Short summary", source_ids=["s1"],
                       verification_query="pilot deployment")

    def structured(role, prompt, schema, max_tokens):
        calls.append((role, prompt, schema, max_tokens))
        if len(calls) == 1:
            raise LengthFinishReasonError()
        return CandidateBatch(candidates=[merged])

    provider.structured = structured
    candidates = [{"id": f"c{i}", "title": f"Technology {i}", "application": "payments",
                  "domain": "fintech", "summary": "short", "source_ids": ["s1"],
                  "verification_query": "verify"} for i in range(8)]

    result = provider.consolidate("fintech", candidates)
    assert [call[3] for call in calls] == [9000, 4000, 4000, 9000]
    assert len(result.candidates) == 1 and result.candidates[0].id == "cand_merged"
    assert provider.consolidation_chunked is False
