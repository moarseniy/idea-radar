from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request, fixture

from app.radar.connectors import grants_gov

SNAPSHOT = date(2021, 12, 31)
PREFIX = "https://www.grants.gov/search-results-detail/"


def test_contract_and_date_cut_on_recorded_response(monkeypatch):
    calls = fake_json_request(monkeypatch, grants_gov, fixture("grants_gov_search"))
    connector = grants_gov.CONNECTOR
    refs = connector.run(["artificial intelligence"], SNAPSHOT, 5)
    assert_connector_contract(connector, refs, SNAPSHOT, money=True)
    # 2024 и 2026 открыты после as_of и отсеиваются; порядок ответа сохраняется.
    assert [r["url"] for r in refs] == [PREFIX + "333855", PREFIX + "287284"]
    event = refs[0]["structured_event"]
    assert event["date"] == "2021-05-27" and event["kind"] == "grant_call"
    assert event["source_id"] == "333855" and event["funder"] == "National Institute of Standards and Technology"
    assert event["amount_usd_m"] is None
    assert calls[0]["provider"] == "grants_gov"
    assert calls[0]["body"]["keyword"] == "artificial intelligence"
    assert calls[0]["body"]["rows"] == 25


def test_irrelevant_records_are_dropped(monkeypatch):
    fake_json_request(monkeypatch, grants_gov, fixture("grants_gov_search"))
    assert grants_gov.search_grants_gov(["solid-state battery"], SNAPSHOT) == []


def test_empty_response_gives_no_references(monkeypatch):
    fake_json_request(monkeypatch, grants_gov, {"data": {"hitCount": 0, "oppHits": []}})
    assert grants_gov.search_grants_gov(["artificial intelligence"], SNAPSHOT) == []
