from datetime import date

import httpx
import pytest

from app.radar import research_apis as api

SNAPSHOT = date(2026, 9, 28)


def test_openalex_records_reconstruct_abstract_and_affiliations(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "1")
    requested = {}
    monkeypatch.setattr(api, "_json_request", lambda provider, url, **kwargs: (
        requested.update(provider=provider, url=url, params=kwargs["params"]) or {
            "results": [{
                "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1000/x",
                "title": "In-sensor processor", "publication_date": "2026-01-02",
                "cited_by_count": 9, "type": "article", "language": "en",
                "abstract_inverted_index": {"in-sensor": [0], "processor": [1], "with": [2], "company": [3]},
                "authorships": [{"institutions": [{"id": "I1", "display_name": "Lab", "type": "education",
                                                       "country_code": "GB"}]},
                               {"institutions": [{"id": "I2", "display_name": "Chip Inc", "type": "company",
                                                   "country_code": "US"}]}],
            }]
        }))
    result = api.openalex_works(["in-sensor computing"], date(2025, 1, 1), SNAPSHOT)
    assert requested["provider"] == "openalex"
    assert "api.openalex.org/works" in requested["url"]
    assert result[0]["abstract"] == "in-sensor processor with company"
    assert result[0]["institutions"] == {"I1": "education", "I2": "company"}
    assert result[0]["countries"] == {"GB", "US"}
    assert result[0]["industry_affiliation_share"] == 0.5


def test_openalex_live_inference_can_skip_unused_affiliation_payload(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "1")
    requested = {}
    monkeypatch.setattr(api, "_json_request", lambda _provider, _url, **kwargs: (
        requested.update(params=kwargs["params"]) or {"results": [{
            "id": "https://openalex.org/W2", "title": "Edge sensor", "publication_date": "2026-01-02",
            "abstract_inverted_index": {"Edge": [0], "sensor": [1]},
        }]}))

    result = api.openalex_works(["edge sensor"], date(2025, 1, 1), SNAPSHOT,
                                include_affiliations=False)

    assert "authorships" not in requested["params"]["select"]
    assert result[0]["institutions"] == {}
    assert result[0]["countries"] == set()
    assert result[0]["industry_affiliation_share"] == ""


def test_openalex_429_stops_retries_and_cools_down_shared_requests(monkeypatch, tmp_path):
    requested = []
    monkeypatch.setattr(api, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(api._RATE_LIMITED_UNTIL, "openalex", 0.0)

    def throttled(url, **kwargs):
        requested.append(url)
        return httpx.Response(429, request=httpx.Request("GET", url), headers={"Retry-After": "60"})

    monkeypatch.setattr(api.httpx, "get", throttled)
    with pytest.raises(RuntimeError, match="HTTPStatusError, HTTP 429"):
        api._json_request("openalex", "https://api.openalex.org/works", params={"search": "first"})
    with pytest.raises(RuntimeError, match="rate limited; cooling down"):
        api._json_request("openalex", "https://api.openalex.org/works", params={"search": "second"})

    assert len(requested) == 1


def test_openalex_can_be_disabled_without_disabling_other_direct_sources(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    monkeypatch.setattr(api, "_json_request", lambda *_args, **_kwargs:
                        pytest.fail("disabled OpenAlex must not make a request"))

    assert api.openalex_enabled() is False
    assert api.openalex_works(["edge sensor"], date(2025, 1, 1), SNAPSHOT) == []
    assert api.search_openalex(["edge sensor"], SNAPSHOT) == []


def test_nih_grant_normalizes_official_award_to_structured_event(monkeypatch):
    monkeypatch.setattr(api, "_json_request", lambda *_args, **_kwargs: {"results": [{
        "ProjectTitle": "Novel diagnostic sensor", "ProjectNum": "R01-123", "CoreProjectNum": "R01-123",
        "AbstractText": "Research grant abstract.", "AwardAmount": 2_500_000,
        "AwardNoticeDate": "2026-04-05", "organization": {"OrgName": "University Lab"},
    }]})
    ref = api.search_nih_grants(["diagnostic sensor"], SNAPSHOT)[0]
    assert ref["url"].endswith("/project-details/R01-123")
    assert ref["structured_event"] == {
        "type": "grant", "date": "2026-04-05", "company": "University Lab",
        "amount_usd_m": 2.5, "round": "grant", "strategic": False, "source_id": "R01-123",
    }
    assert "Award amount (USD): 2500000" in ref["document"]["text"]


def test_cordis_result_is_kept_in_eur_without_fabricating_usd(monkeypatch):
    requested = {}
    monkeypatch.setattr(api, "_json_request", lambda provider, url, **kwargs: (
        requested.update(provider=provider, url=url, query=kwargs["params"]["query"]) or {
            "results": {"bindings": [{
                "project_id": {"value": "101234"}, "title": {"value": "Photonics research grant"},
                "abstract": {"value": "Researching new photonic interconnects."},
                "start_date": {"value": "2024-02-01"}, "funding_amount": {"value": "1200000"},
            }]}
        }))
    ref = api.search_cordis_grants(["photonic interconnect"], SNAPSHOT)[0]
    assert "sparql" in requested["url"]
    assert "eurio:Project" in requested["query"]
    assert ref["structured_event"]["currency"] == "EUR"
    assert ref["structured_event"]["amount_usd_m"] is None


def test_github_and_huggingface_records_include_activity_metrics(monkeypatch):
    def fake_request(provider, url, **kwargs):
        if provider == "github":
            return {"items": [{"full_name": "lab/sensor", "html_url": "https://github.com/lab/sensor",
                               "description": "Open sensor stack", "language": "Python", "stargazers_count": 8,
                               "forks_count": 2, "open_issues_count": 1, "pushed_at": "2026-08-02T00:00:00Z",
                               "topics": ["sensor"]}]}
        return [{"modelId": "lab/sensor-model", "downloads": 1250, "likes": 18,
                 "lastModified": "2026-08-01T00:00:00Z", "pipeline_tag": "image-classification",
                 "tags": ["sensor", "edge"]}]

    monkeypatch.setattr(api, "_json_request", fake_request)
    github = api.search_github(["edge sensor"], SNAPSHOT)[0]
    hf = api.search_huggingface(["edge sensor"], SNAPSHOT)[0]
    assert github["adoption_metrics"]["stars"] == 8
    assert "1250" in hf["document"]["text"]
    assert hf["url"] == "https://huggingface.co/lab/sensor-model"


def test_ted_and_health_adapters_return_official_documents(monkeypatch):
    def fake_request(provider, *_args, **_kwargs):
        if provider == "ted":
            return {"notices": [{"publication-number": "123-2026", "notice-title": {"eng": "Pilot sensor tender"},
                                 "publication-date": "2026-03-01", "buyer-name": "City Authority",
                                 "description": "Purchase for a technology pilot."}]}
        if provider == "clinicaltrials":
            return {"studies": [{"protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Sensor clinical study"},
                "statusModule": {"overallStatus": "RECRUITING", "studyFirstPostDateStruct": {"date": "2026-01-02"}},
                "designModule": {"studyType": "INTERVENTIONAL"},
                "descriptionModule": {"briefSummary": "Evaluating the device with patients."},
            }}]}
        return {"results": [{"k_number": "K260001", "device_name": "Sensor diagnostic", "decision_date": "20260102",
                             "decision_description": "Substantially equivalent", "applicant": "Med Labs"}]}

    monkeypatch.setattr(api, "_json_request", fake_request)
    ted = api.search_ted(["sensor pilot"], SNAPSHOT)[0]
    trials = api.search_clinicaltrials(["medical sensor"], SNAPSHOT)[0]
    fda = api.search_openfda(["medical sensor"], SNAPSHOT)[0]
    assert ted["structured_event"]["type"] == "procurement"
    assert trials["structured_event"]["type"] == "trial"
    assert fda["structured_event"]["type"] == "regulation"
    assert not api.search_health(["edge computing"], SNAPSHOT)


def test_openfda_search_maps_510k_decision_without_requiring_a_key(monkeypatch):
    requested = []

    def fake_request(provider, url, **kwargs):
        requested.append((provider, kwargs.get("params", {})))
        if "device/510k" in url:
            return {"results": [{"k_number": "K260001", "device_name": "Insulin pump",
                                 "decision_date": "20260102", "decision_description": "Cleared",
                                 "applicant": "Medical Devices Inc"}]}
        return {"results": []}

    monkeypatch.setattr(api, "_json_request", fake_request)
    monkeypatch.delenv("OPENFDA_API_KEY", raising=False)
    refs = api.search_openfda(["insulin pump"], SNAPSHOT, limit=1)
    assert refs[0]["structured_event"]["source_id"] == "K260001"
    assert refs[0]["observed_at"] == "2026-01-02"
    assert requested[0][1]["search"] == "device_name:insulin OR device_name:pump"
    assert "api_key" not in requested[0][1]
