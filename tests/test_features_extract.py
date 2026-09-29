from datetime import date

import pytest

from app.ml.features import evidence as ev
from app.ml.features.extract import compute, extract_events, llm_json, select_headlines
from app.ml.features.schema import FEATURE_NAMES

SNAPSHOT = date(2026, 9, 22)
QUERY = {"term": "in-sensor computing", "aliases": [], "is_technology": True}


def evidence():
    sci = ev.Science(ok=True, total=40, windows=[20, 8, 2], count_3y=24, first_date="2016-05-01",
                     recent=[{"publicationDate": "2026-03-01", "citationCount": 12, "preprint": True},
                             {"publicationDate": "2025-06-01", "citationCount": 3, "preprint": False}],
                     titles=["In-sensor computing with 2D materials"])
    pat = ev.Patents(ok=True, windows=[9, 3, 0], count_3y=11, older_6y=0, first_priority="2021-09-22",
                     titles=["Analog in-sensor computing device"])
    news = {"0-6m": [{"title": "Startup raises $12M for in-sensor AI", "date": "2026-08-01",
                      "publisher": "TechCrunch", "host": "techcrunch.com"},
                     {"title": "Chipmaker pilots in-sensor vision with automaker", "date": "2026-06-10",
                      "publisher": "PR Newswire", "host": "prnewswire.com"}],
            "6-12m": [{"title": "In-sensor computing explained", "date": "2026-01-05",
                       "publisher": "Medium", "host": "medium.com"}],
            "12-24m": [], "24-48m": []}
    return sci, pat, ev.Media(ok=True, windows=news)


EXT = {"stage": 3, "mass_market": False, "rebranding_similarity": 0.1, "events": [
    {"type": "funding", "company": "Startup", "amount_usd_m": 12, "round": "A", "strategic": False,
     "date": "2026-08-01"},
    {"type": "pilot", "company": "Chipmaker", "customer": "Automaker", "date": "2026-06-10"}]}


def test_compute_returns_full_schema_and_is_deterministic():
    sci, pat, med = evidence()
    first = compute(SNAPSHOT, QUERY, sci, pat, med, EXT)
    second = compute(SNAPSHOT, QUERY, sci, pat, med, EXT)
    assert list(first) == FEATURE_NAMES
    assert first == second


def test_compute_values_follow_evidence():
    f = compute(SNAPSHOT, QUERY, *evidence(), EXT)
    assert f["stage_pilot"] == 1 and f["stage_research"] == 0
    assert f["paper_growth_2y"] > 0
    assert f["preprint_share_2y"] == 0.5
    assert f["funding_round_count_24m"] == 1
    assert f["early_stage_funding_share"] == 1.0
    assert f["press_release_share_12m"] == round(1 / 3, 6)
    assert f["missing_funding"] == 0 and f["missing_adoption"] == 0
    assert f["source_type_diversity"] == 5


def test_openalex_affiliation_features_and_official_grant_fill_existing_schema():
    sci, pat, med = evidence()
    sci.institutions_3y = {"I1", "I2"}
    sci.countries_3y = {"GB", "US"}
    sci.industry_affiliation_share_3y = 0.5
    grant = {"url": "https://reporter.nih.gov/project-details/R01-123", "structured_event": {
        "type": "grant", "date": "2026-04-05", "company": "University Lab",
        "amount_usd_m": 2.5, "round": "grant", "strategic": False, "source_id": "R01-123",
    }}
    f = compute(SNAPSHOT, QUERY, sci, pat, med, None, [grant])
    assert f["institution_count_log_3y"] > 0
    assert f["science_country_count_log_3y"] > 0
    assert f["industry_affiliation_share_3y"] == 0.5
    assert f["public_grants_log_36m"] == 0.693147
    assert f["missing_funding"] == 0


def test_official_grant_is_added_without_relying_on_news_headlines(monkeypatch):
    prompt = {}
    monkeypatch.setattr("app.ml.features.extract.llm_json", lambda value:
                        prompt.update(text=value) or {"events": [], "stage": 2, "mass_market": False})
    ref = {"url": "https://reporter.nih.gov/project-details/R01-123", "source_provider": "nih_reporter_project",
           "title": "Novel diagnostic sensor", "observed_at": "2026-04-05",
           "document": {"text": "Federal research grant for a novel diagnostic sensor."},
           "structured_event": {"type": "grant", "date": "2026-04-05", "company": "University Lab",
                                "amount_usd_m": 2.5, "round": "grant", "strategic": False,
                                "source_id": "R01-123"}}
    events = extract_events(QUERY, "Novel diagnostic sensor", SNAPSHOT, [], ev.Science(), ev.Patents(), [ref])
    assert events["events"] == [{**ref["structured_event"], "date": "2026-04-05",
                                 "source_url": ref["url"], "structured": True}]
    assert "Структурированные API-свидетельства" in prompt["text"]
    assert "reporter.nih.gov/project-details/R01-123" in prompt["text"]


def test_failed_channels_stay_empty_not_zero():
    f = compute(SNAPSHOT, QUERY, ev.Science(ok=False), ev.Patents(ok=False), ev.Media(ok=False), None)
    assert f["papers_log_3y"] == "" and f["patent_families_log_3y"] == ""
    assert f["media_volume_normalized_12m"] == "" and f["funding_log_24m"] == ""
    assert f["missing_science"] == 1 and f["missing_media"] == 1


def test_patent_assignee_rollup_populates_patent_diversity_features():
    sci, pat, med = evidence()
    pat.assignee_counts_3y = {"New Labs": 3, "Old Corp": 1}
    pat.assignees_before = ["Old Corp"]
    f = compute(SNAPSHOT, QUERY, sci, pat, med, EXT)
    assert f["patent_new_assignee_share_3y"] == 0.5
    assert f["patent_assignee_hhi_3y"] == 0.625


def test_fpo_result_fetch_uses_httpx_and_parses_patent_links(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ev.LIMITS["patents"], "wait", lambda: None)
    monkeypatch.setattr(ev, "BREAKERS", {**ev.BREAKERS, "patents": ev.Breaker()})

    class Response:
        text = ('<div id="listing_table"><a href="/US20260246810A1.html">'
                'Method for a novel battery separator</a><p>1-1 out of 42</p></div>')

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(ev.httpx, "get", lambda *_args, **_kwargs: Response())
    result = ev._patents(["battery separator"], date(2024, 1, 1), date(2026, 9, 28), tries=1)
    assert result["total"] == 42
    assert result["references"][0]["url"].endswith("/US20260246810A1.html")


def test_live_fpo_search_uses_a_bounded_tls_retry(monkeypatch):
    requested = {}

    def fake_patents(terms, after, before, **kwargs):
        requested.update(terms=terms, after=after, before=before, **kwargs)
        return {"references": []}

    monkeypatch.setattr(ev, "_patents", fake_patents)
    assert ev.search_patents_fpo(["quantum-safe banking"], SNAPSHOT) == []
    assert requested["tries"] == 2


def test_headline_selection_is_stable_and_deduplicated():
    _, _, med = evidence()
    med.windows["6-12m"].append(dict(med.windows["0-6m"][0]))
    picked = select_headlines(med)
    assert [h["title"] for h in picked] == [h["title"] for h in select_headlines(med)]
    assert len({h["title"] for h in picked}) == len(picked)


def test_feature_llm_calls_have_bounded_timeout_and_retry_count(monkeypatch):
    requested = {}
    monkeypatch.setenv("FEATURE_LLM_TIMEOUT", "9999")
    monkeypatch.setenv("FEATURE_LLM_TRIES", "9")
    monkeypatch.setattr(ev, "cached_fetch", lambda *_args, **kwargs: requested.update(kwargs) or {})

    llm_json("test prompt")

    assert requested["timeout"] == 600
    assert requested["tries"] == 2


def test_disabled_openalex_does_not_turn_failed_crossref_into_measured_zero(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    monkeypatch.setenv("FEATURE_STRUCTURED_SOURCES", "1")
    monkeypatch.setattr(ev, "_crossref", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr("app.radar.research_apis.openalex_works", lambda *_args, **_kwargs:
                        pytest.fail("disabled OpenAlex must not be requested"))

    result = ev.science(["edge sensor"], SNAPSHOT)

    assert result.ok is False
    assert result.count_3y == 0
