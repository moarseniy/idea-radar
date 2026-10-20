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
    requested = {}
    monkeypatch.setattr(api, "_json_request", lambda *_args, **kwargs: (requested.update(body=kwargs["body"]) or {
        "results": [{
            "ProjectTitle": "Novel diagnostic sensor", "ProjectNum": "R01-123", "CoreProjectNum": "R01-123",
            "AbstractText": "Research grant abstract.", "AwardAmount": 2_500_000,
            "AwardNoticeDate": "2026-04-05", "organization": {"OrgName": "University Lab"},
        }, {
            "ProjectTitle": "Rural housing outreach", "CoreProjectNum": "R01-999",
            "AbstractText": "Unrelated community program.", "AwardNoticeDate": "2026-04-05",
        }]}))
    refs = api.search_nih_grants(["diagnostic sensor"], SNAPSHOT)
    assert [ref["structured_event"]["source_id"] for ref in refs] == ["R01-123"]  # нерелевантный отсеян
    ref = refs[0]
    assert ref["url"].endswith("/project-details/R01-123")
    event = ref["structured_event"]
    assert (event["type"], event["kind"], event["date"], event["company"]) == (
        "grant", "grant", "2026-04-05", "University Lab")
    assert event["amount_usd_m"] == 2.5 and event["round"] == "grant" and event["strategic"] is False
    assert "Award amount (USD): 2500000" in ref["document"]["text"]
    # слова фразы связаны AND, фразы — OR; дата оценки — конец диапазона
    criteria = requested["body"]["criteria"]
    assert criteria["advanced_text_search"]["search_text"] == "(diagnostic AND sensor)"
    assert criteria["award_notice_date"]["to_date"] == "2026-09-28"


def test_nih_long_title_is_searched_by_short_phrases(monkeypatch):
    requested = {}
    monkeypatch.setattr(api, "_json_request", lambda *_args, **kwargs: (
        requested.update(body=kwargs["body"]) or {"results": []}))
    assert api.search_nih_grants(["agent security posture management platform"], SNAPSHOT) == []
    text = requested["body"]["criteria"]["advanced_text_search"]["search_text"]
    assert text == "(agent AND security AND posture) OR (agent AND security) OR (security AND posture)"
    assert '"' not in text


def test_cordis_result_is_converted_from_eur_to_usd(monkeypatch):
    requested = {}
    monkeypatch.setattr(api, "_json_request", lambda provider, url, **kwargs: (
        requested.update(provider=provider, url=url, query=kwargs["params"]["query"]) or {
            "results": {"bindings": [{
                "project_id": {"value": "101234"}, "title": {"value": "Photonics research grant"},
                "abstract": {"value": "Researching new photonic interconnects."},
                "start_date": {"value": "2024-02-01"}, "funding_amount": {"value": "1200000"},
            }, {
                "project_id": {"value": "101999"}, "title": {"value": "Medieval pottery"},
                "abstract": {"value": "Ceramics of the Middle Ages."},
                "start_date": {"value": "2024-03-01"}, "funding_amount": {"value": "50000"},
            }]}
        }))
    refs = api.search_cordis_grants(["photonic interconnect"], SNAPSHOT)
    ref = refs[0]
    assert [r["structured_event"]["source_id"] for r in refs] == ["101234"]
    assert "sparql" in requested["url"]
    assert "eurio:Project" in requested["query"]
    # слова фразы — через &&, без точной фразы в кавычках
    assert 'CONTAINS(LCASE(CONCAT(STR(?title), " ", COALESCE(STR(?abstract), ""))), "photonic") && ' in requested["query"]
    assert '"photonic interconnect"' not in requested["query"]
    event = ref["structured_event"]
    assert event["currency"] == "EUR" and event["amount_native"] == 1_200_000.0
    assert event["amount_usd_m"] == pytest.approx(1.2 / 0.92, rel=1e-3)  # курс 2024 года


def test_ted_query_has_upper_date_bound_and_short_phrases(monkeypatch):
    requested = {}

    def fake_request(provider, url, **kwargs):
        requested.update(body=kwargs["body"])
        return {"notices": [
            {"publication-number": "1-2026", "notice-title": {"eng": "Sensor pilot tender"},
             "publication-date": "2026-03-01+01:00", "buyer-name": {"eng": ["City"]}},
            {"publication-number": "2-2026", "notice-title": {"eng": "Road repair works"},
             "publication-date": "2026-03-02+01:00"},
            {"publication-number": "3-2026", "notice-title": {"swe": "Sensorer"},
             "publication-date": "2026-03-03+01:00"},
        ]}

    monkeypatch.setattr(api, "_json_request", fake_request)
    refs = api.search_ted(["sensor pilot"], SNAPSHOT)
    query = requested["body"]["query"]
    assert "FT ~ (sensor pilot)" in query and "PD <= 20260928" in query and '"' not in query
    # английский нерелевантный отсеян, неанглийский оставлен на доверии серверу
    assert [r["structured_event"]["source_id"] for r in refs] == ["1-2026", "3-2026"]
    assert refs[0]["structured_event"]["kind"] == "tender"
    assert refs[0]["structured_event"]["customer"] == "City"
    assert refs[0]["observed_at"] == "2026-03-01"


def test_huggingface_skips_non_ml_topics_and_filters_irrelevant_models(monkeypatch):
    calls = []

    def fake_request(provider, url, **kwargs):
        calls.append(kwargs["params"]["search"])
        return [{"modelId": "lab/red-teaming-llm", "downloads": 10, "likes": 1,
                 "lastModified": "2026-08-01T00:00:00Z", "pipeline_tag": "text-generation", "tags": []},
                {"modelId": "lab/cat-classifier", "downloads": 99, "likes": 1,
                 "lastModified": "2026-08-01T00:00:00Z", "pipeline_tag": "image-classification", "tags": []}]

    monkeypatch.setattr(api, "_json_request", fake_request)
    assert api.is_ml_topic(["automated red teaming of LLM"]) is True
    assert api.is_ml_topic(["road construction logistics"]) is False
    assert api.search_huggingface(["road construction logistics"], SNAPSHOT) == []
    assert calls == []  # нетехническая тема: запрос не уходит
    refs = api.search_huggingface(["red teaming llm"], SNAPSHOT)
    assert [r["url"] for r in refs] == ["https://huggingface.co/lab/red-teaming-llm"]
    assert calls == ["red teaming llm", "red teaming"]  # мало записей: пробуем и короткую фразу


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
    hf = api.search_huggingface(["edge sensor model"], SNAPSHOT)[0]
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


def _fixture_contract(monkeypatch, name, fixture_name, terms, *, money, as_of=SNAPSHOT):
    from connector_testkit import assert_connector_contract, fake_json_request, fixture

    from app.radar.connectors import registry

    calls = fake_json_request(monkeypatch, api, fixture(fixture_name) if isinstance(fixture_name, str)
                              else fixture_name)
    connector = registry()[name]
    refs = connector.run(terms, as_of, 5)
    assert_connector_contract(connector, refs, as_of, money=money)
    return refs, calls


def test_nih_recorded_response_drops_grants_with_words_scattered_over_abstract(monkeypatch):
    from connector_testkit import fake_json_request, fixture

    from app.radar.connectors import registry

    calls = fake_json_request(monkeypatch, api, fixture("nih_automated_red_teaming"))
    # Записанный ответ — клеточная биология и микроскопия: "automated", "AI" и "models"
    # разбросаны по аннотации, о red teaming в них нет ничего.
    assert registry()["NIH RePORTER"].run(["automated red teaming of AI models"], SNAPSHOT, 5) == []
    assert calls[0]["body"]["criteria"]["award_notice_date"]["to_date"] == "2026-09-28"


def test_nih_recorded_grant_kept_when_phrase_words_stand_together(monkeypatch):
    from connector_testkit import fixture

    payload = fixture("nih_automated_red_teaming")
    row = payload["results"][0]
    row["abstract_text"] = "We build automated red teaming of clinical AI models. " + (row.get("abstract_text") or "")
    refs, _ = _fixture_contract(monkeypatch, "NIH RePORTER", payload,
                                ["automated red teaming of AI models"], money=True)
    assert len(refs) == 1 and refs[0]["structured_event"]["kind"] == "grant"


def test_cordis_recorded_response_gives_usd_amount_for_eur_grant(monkeypatch):
    refs, _ = _fixture_contract(monkeypatch, "CORDIS grants", "cordis_neuromorphic",
                                ["neuromorphic brain-inspired chips for AI workloads"], money=True)
    events = [ref["structured_event"] for ref in refs]
    assert all(e["currency"] == "EUR" for e in events)
    assert any(e["amount_usd_m"] for e in events)  # раньше было None


def test_ted_recorded_response_follows_strict_contract(monkeypatch):
    refs, calls = _fixture_contract(monkeypatch, "TED procurement", "ted_automated_red_teaming",
                                    ["laboratory optical precision equipment"], money=True)
    # из трёх записанных уведомлений по теме одно: остальные (медтехника) отсеяны
    assert len(refs) == 1 and refs[0]["structured_event"]["kind"] == "tender"
    assert refs[0]["date_support"] == "strict"
    assert calls[0]["body"]["query"].endswith("PD <= 20260928")


def test_huggingface_recorded_response_filters_to_relevant_models(monkeypatch):
    refs, calls = _fixture_contract(monkeypatch, "Hugging Face", "huggingface_red_teaming",
                                    ["red teaming llm"], money=False, as_of=date(2026, 12, 31))
    assert refs and all("red teaming" in r["url"].lower().replace("-", " ").replace("_", " ") or
                        "redteaming" in r["url"].lower() for r in refs)
    assert calls[0]["params"]["search"] == "red teaming llm"
