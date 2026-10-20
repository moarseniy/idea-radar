from types import SimpleNamespace

from app.radar.models import Candidate, CandidateBatch
from app.radar.provider import ResearchProvider


def test_chunked_consolidation_preserves_a_failed_batch_and_other_results():
    length_error = type("LengthFinishReasonError", (Exception,), {})
    provider = object.__new__(ResearchProvider)
    provider.settings = SimpleNamespace(max_candidates=64)
    provider.consolidation_chunked = False
    provider.consolidation_failures = []
    calls = 0

    def structured(role, prompt, schema, max_tokens):
        nonlocal calls
        calls += 1
        if calls in {1, 5}:
            raise length_error()
        if calls == 3:
            raise RuntimeError("detail intentionally not persisted")
        start = (calls - 2) * 6
        expected_count = 1 if calls == 4 else 6
        candidates = [
            Candidate(id=f"c{i}", title=f"Technology {i}", application="Use case",
                      domain="Technology", summary="Evidence", source_ids=[f"s{i}"],
                      verification_query=f"Technology {i} use case")
            for i in range(start, start + expected_count)
        ]
        return CandidateBatch(candidates=candidates)

    provider.structured = structured
    candidates = [
        {"id": f"c{i}", "title": f"Technology {i}", "application": "Use case",
         "domain": "Technology", "summary": "Evidence", "source_ids": [f"s{i}"],
         "verification_query": f"Technology {i} use case"}
        for i in range(13)
    ]

    result = provider.consolidate("technology", candidates)

    assert provider.consolidation_chunked is True
    assert provider.consolidation_failures == ["RuntimeError"]
    assert len(result.candidates) == 13
    assert {candidate.id for candidate in result.candidates} == {f"c{i}" for i in range(13)}
