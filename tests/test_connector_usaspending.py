from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request, fixture

from app.radar.connectors import usaspending as usa

SNAPSHOT = date(2021, 12, 31)
CONTRACTS, GRANTS = usa.CONTRACT_SEARCH, usa.GRANT_SEARCH


def _connector(name):
    return next(c for c in usa.CONNECTORS if c.name == name)


def test_contracts_recorded_response_matches_contract(monkeypatch):
    calls = fake_json_request(monkeypatch, usa, fixture("usaspending_contracts"))
    connector = _connector("USAspending contracts")
    refs = connector.run(["quantum sensing"], SNAPSHOT, 5)
    assert_connector_contract(connector, refs, SNAPSHOT, money=True)
    assert calls[0]["body"]["filters"]["time_period"][0]["end_date"] == "2021-12-31"
    assert calls[0]["body"]["filters"]["keywords"] == ["quantum sensing"]
    assert calls[0]["body"]["filters"]["award_type_codes"] == ["A", "B", "C", "D"]
    assert refs[0]["structured_event"]["type"] == "procurement"
    assert refs[0]["structured_event"]["amount_usd_m"] == 0.749936


def test_grants_recorded_response_matches_contract(monkeypatch):
    calls = fake_json_request(monkeypatch, usa, fixture("usaspending_grants"))
    connector = _connector("USAspending grants")
    refs = connector.run(["solid-state battery"], SNAPSHOT, 5)
    assert_connector_contract(connector, refs, SNAPSHOT, money=True)
    assert calls[0]["body"]["filters"]["award_type_codes"] == ["02", "03", "04", "05"]
    assert calls[0]["body"]["filters"]["keywords"] == ["solid-state battery"]
    assert all(ref["structured_event"]["kind"] == "grant" for ref in refs)
    assert all(ref["url"].startswith("https://www.usaspending.gov/award/") for ref in refs)


def test_empty_response_gives_no_refs(monkeypatch):
    fake_json_request(monkeypatch, usa, {"results": []})
    assert CONTRACTS(["quantum sensing"], SNAPSHOT, 5) == []
    assert GRANTS(["solid-state battery"], SNAPSHOT, 5) == []


def test_unrelated_descriptions_are_filtered_out(monkeypatch):
    row = {"Award ID": "W1", "Recipient Name": "CISCO", "Award Amount": 1000.0,
           "Start Date": "2020-01-01", "Description": "Cisco switches for the building",
           "Awarding Agency": "GSA", "generated_internal_id": "CONT_AWD_W1"}
    fake_json_request(monkeypatch, usa, {"results": [row]})
    assert CONTRACTS(["quantum sensing"], SNAPSHOT, 5) == []


def test_rows_without_start_date_or_after_as_of_are_dropped(monkeypatch):
    base = {"Award ID": "W2", "Recipient Name": "ACME", "Award Amount": 5.0,
            "Description": "QUANTUM SENSING PROTOTYPE", "Awarding Agency": "DOE",
            "generated_internal_id": "CONT_AWD_W2"}
    fake_json_request(monkeypatch, usa, {"results": [dict(base, **{"Start Date": None}),
                                                     dict(base, **{"Start Date": "2022-02-01"})]})
    assert CONTRACTS(["quantum sensing"], SNAPSHOT, 5) == []


def test_two_phrases_merge_by_award_id(monkeypatch):
    calls = fake_json_request(monkeypatch, usa, fixture("usaspending_contracts"))
    refs = CONTRACTS(["quantum sensing navigation", "hypersonic weapons guidance"], SNAPSHOT, 10)
    assert len(calls) == 2
    assert all(len(call["body"]["filters"]["keywords"]) == 1 for call in calls)
    assert len({ref["url"] for ref in refs}) == len(refs)
