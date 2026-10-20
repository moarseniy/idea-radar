from datetime import date

import pytest

from app.ml.features import evidence as ev
from app.ml.features.extract import (
    _evidence_timeline_visualization,
    _needs_patent_metrics,
    _publication_visualization,
    compute,
    extract_events,
    llm_json,
    select_headlines,
)
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


def test_live_model_only_fetches_patents_if_patent_features_are_inputs():
    from app.ml.scorer import default_scorer

    assert not _needs_patent_metrics(set(default_scorer().features))
    assert _needs_patent_metrics(None)
    assert _needs_patent_metrics({"patent_growth_2y"})


def test_publication_visualization_distinguishes_observed_zero_from_unavailable():
    available = _publication_visualization(ev.Science(
        ok=True, total=10, count_3y=7, history_complete=True, recent_3y_complete=True,
        year_counts={"2021": 2, "2026": 3}, older_results_capped=True))
    assert available["publication_total_found"] == 10
    assert available["publication_recent_share_found"] == 0.7
    assert available["publication_total_capped"] is True

    empty = _publication_visualization(ev.Science(ok=True))
    unavailable = _publication_visualization(ev.Science(ok=False))
    assert empty["publication_total_found"] == 0
    assert empty["publication_recent_share_found"] is None
    assert unavailable["publication_total_found"] is None
    assert unavailable["science_status"] == "unavailable"


def test_evidence_timeline_uses_channel_search_aggregates_not_card_sources():
    sci = ev.Science(ok=True, year_counts={"2021": 4, "2022": 3, "2023": 2,
                                           "2024": 1, "2025": 0, "2026": 5})
    med = ev.Media(ok=True, windows={
        "0-6m": [{"title": "New report", "date": "2026-08-01", "publisher": "A", "host": "a.test"}],
        "6-12m": [{"title": "Older report", "date": "2025-10-01", "publisher": "B", "host": "b.test"}],
        "12-24m": [], "24-48m": [],
    })
    refs = [
        {"source_provider": "nsf_award", "observed_at": "2014-05-01",
         "structured_event": {"type": "grant", "date": "2014-05-01"}},
        {"source_provider": "google_news", "channel": "market", "url": "https://example.test/story",
         "title": "Archived media mention", "observed_at": "2016-05-01"},
        {"source_provider": "nih_reporter_project", "observed_at": "2026-02-01",
         "structured_event": {"type": "grant", "date": "2026-02-01"}},
        {"source_provider": "github_repository", "observed_at": "2025-01-01"},
        {"source_provider": "clinicaltrials_api", "observed_at": "2024-04-01",
         "structured_event": {"type": "trial", "date": "2024-04-01"}},
    ]

    result = _evidence_timeline_visualization(SNAPSHOT, sci, med, refs)

    assert result["years"] == [str(year) for year in range(2014, 2027)]
    assert result["channels"]["science"]["counts"]["2026"] == 5
    assert result["channels"]["science"]["counts"]["2014"] is None
    assert result["channels"]["news"]["counts"]["2026"] == 1
    assert result["channels"]["news"]["counts"]["2016"] == 1
    assert result["channels"]["news"]["counts"]["2015"] is None
    assert result["channels"]["news"]["counts"]["2014"] is None
    assert result["channels"]["funding"]["counts"]["2014"] == 1
    assert result["channels"]["funding"]["counts"]["2015"] == 0
    assert result["channels"]["funding"]["counts"]["2026"] == 1
    assert result["channels"]["community"]["counts"]["2025"] == 1
    assert result["channels"]["adoption"]["counts"]["2024"] == 1
    assert all(value is None for value in result["channels"]["patents"]["counts"].values())


def test_science_returns_all_available_publication_years_without_extra_queries(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    monkeypatch.setenv("FEATURE_STRUCTURED_SOURCES", "1")
    records = [
        {"title": "Paper 2026", "publicationDate": "2026-02-01", "doi": "10.1/a"},
        {"title": "Paper 2025", "publicationDate": "2025-03-01", "doi": "10.1/b"},
        {"title": "Paper 2023", "publicationDate": "2023-06-01", "doi": "10.1/c"},
        {"title": "Paper 2021", "publicationDate": "2021-08-01", "doi": "10.1/d"},
        {"title": "Old paper", "publicationDate": "2019-07-01", "doi": "10.1/e"},
    ]

    def fake_crossref(_terms, start, end):
        return [row for row in records if start.isoformat() <= row["publicationDate"] <= end.isoformat()]

    monkeypatch.setattr(ev, "_crossref", fake_crossref)
    result = ev.science(["edge sensor"], SNAPSHOT)

    assert result.ok
    assert result.total == 5
    assert result.year_counts == {"2019": 1, "2020": 0, "2021": 1, "2022": 0, "2023": 1,
                                  "2024": 0, "2025": 1, "2026": 1}


def test_google_patents_web_fallback_keeps_only_patent_pages(monkeypatch):
    monkeypatch.setattr(ev.LIMITS["google_patents"], "wait", lambda: None)
    requested = {}

    def fake_search(query):
        requested["query"] = query
        return {"results": [
            {"url": "https://patents.google.com/patent/US123/en", "title": "Patent A"},
            {"url": "https://example.org/patent/US456", "title": "Not a patent result"},
            {"url": "https://patents.google.com/search?q=x", "title": "Search page"},
        ]}

    monkeypatch.setattr("app.search.client.web_search", fake_search)
    refs = ev.search_patents_google_web(["battery separator"], SNAPSHOT)
    assert [ref["title"] for ref in refs] == ["Patent A"]
    assert refs[0]["source_provider"] == "Google Patents web search"
    assert "site:patents.google.com/patent" in requested["query"].text


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


def test_citation_velocity_is_not_computed_in_the_past():
    # Число цитирований Crossref — на сегодня: в историческом срезе это утечка из будущего.
    live = compute(SNAPSHOT, QUERY, *evidence(), EXT)
    past = compute(SNAPSHOT, QUERY, *evidence(), EXT, historical=True)
    assert live["citation_velocity_median"] != ""
    assert past["citation_velocity_median"] == ""
    assert past["papers_log_3y"] == live["papers_log_3y"]


def test_science_counts_historical_years_from_existing_query_windows(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    monkeypatch.setenv("FEATURE_STRUCTURED_SOURCES", "1")
    records = [
        {"title": "Paper 2026", "publicationDate": "2026-02-01", "doi": "10.1000/a"},
        {"title": "Paper 2025", "publicationDate": "2025-03-01", "doi": "10.1000/b"},
        {"title": "Paper 2023", "publicationDate": "2023-06-01", "doi": "10.1000/c"},
        {"title": "Paper 2021", "publicationDate": "2021-08-01", "doi": "10.1000/d"},
        {"title": "Old paper", "publicationDate": "2019-07-01", "doi": "10.1000/e"},
    ]

    def fake_crossref(_terms, start, end):
        return [row for row in records if start.isoformat() <= row["publicationDate"] <= end.isoformat()]

    monkeypatch.setattr(ev, "_crossref", fake_crossref)
    result = ev.science(["edge sensor"], SNAPSHOT)

    assert result.ok is True
    assert result.total == 5
    assert result.history_complete is True
    assert result.year_counts == {"2019": 1, "2020": 0, "2021": 1, "2022": 0, "2023": 1,
                                  "2024": 0, "2025": 1, "2026": 1}


def test_science_marks_years_unknown_when_a_query_window_fails(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    monkeypatch.setenv("FEATURE_STRUCTURED_SOURCES", "1")
    snapshot = date(2026, 9, 22)
    failed_start = ev.years_before(snapshot, 4)

    def partial_crossref(_terms, start, end):
        if start == failed_start:
            raise RuntimeError("temporary Crossref error")
        return [{"title": f"Paper {start.year}", "publicationDate": f"{start.year}-11-01",
                 "doi": f"10.1000/{start.year}"}]

    monkeypatch.setattr(ev, "_crossref", partial_crossref)
    result = ev.science(["edge sensor"], snapshot)

    assert result.ok is True and result.history_complete is False
    assert result.year_counts["2023"] is None
    assert result.year_counts["2026"] is not None
    assert result.recent_3y_complete is False
    assert _publication_visualization(result)["publication_recent_share_found"] is None
