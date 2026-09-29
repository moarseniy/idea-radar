from app.radar.models import CandidateBatch
from app.radar.provider import ResearchProvider


def test_candidate_extraction_retries_compact_after_output_limit():
    provider = ResearchProvider.__new__(ResearchProvider)
    calls = []

    class LengthFinishReasonError(Exception):
        pass

    def structured(role, prompt, schema, max_tokens):
        calls.append((role, prompt, schema, max_tokens))
        if len(calls) == 1:
            raise LengthFinishReasonError()
        return CandidateBatch(candidates=[])

    provider.structured = structured
    sources = [{"id": f"s{i}", "title": f"title {i}", "published_at": "2026-01-01",
                "url": f"https://example.org/{i}", "text": "x" * 5000} for i in range(4)]

    assert provider.candidates("topic", "2026-09-28", sources).candidates == []
    assert calls[0][0] == calls[1][0] == "extract_candidates"
    assert calls[0][3] == 5000 and calls[1][3] == 3500
    assert "не более 2" in calls[1][1]
    assert len(calls[1][1]) < len(calls[0][1])
