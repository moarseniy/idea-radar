from app.radar.analysis import merge_candidates_with_audit


def candidate(candidate_id, title, application, source_id):
    return {
        "id": candidate_id,
        "title": title,
        "application": application,
        "domain": "materials",
        "summary": "summary",
        "source_ids": [source_id],
    }


def test_near_duplicate_candidates_are_reported_with_the_surviving_candidate():
    rows = [
        candidate("raw-1", "Sodium solid state battery", "for grid storage", "source-1"),
        candidate("raw-2", "Sodium solid-state battery", "for grid storage", "source-2"),
    ]

    merged, filtered = merge_candidates_with_audit(rows, 10)

    assert len(merged) == 1
    assert merged[0]["source_ids"] == ["source-1", "source-2"]
    assert len(filtered) == 1
    assert filtered[0]["action"] == "merged_duplicate"
    assert filtered[0]["candidate"]["id"] == "raw-2"
    assert filtered[0]["merged_into"]["id"] == merged[0]["id"]


def test_candidates_over_limit_are_reported_with_a_reason():
    rows = [
        candidate("raw-1", "Sodium battery", "for storage", "source-1"),
        candidate("raw-2", "Orbital solar manufacturing", "in space", "source-2"),
        candidate("raw-3", "Engineered algae bioreactors", "for fuel", "source-3"),
    ]

    merged, filtered = merge_candidates_with_audit(rows, 2, "candidate_limit")

    assert len(merged) == 2
    assert len(filtered) == 1
    assert filtered[0]["stage"] == "candidate_limit"
    assert filtered[0]["candidate"]["id"] == "raw-3"
    assert "лимит" in filtered[0]["reason"]
