from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request, fixture

from app.radar.connectors import nsf

SNAPSHOT = date(2021, 12, 31)


def test_nsf_contract_and_date_window_from_recorded_response(monkeypatch):
    calls = fake_json_request(monkeypatch, nsf, fixture("nsf_quantum_sensing"))
    refs = nsf.CONNECTOR.run(["quantum sensing"], SNAPSHOT, 5)
    assert_connector_contract(nsf.CONNECTOR, refs, SNAPSHOT, money=True)
    assert len(calls) == 1
    assert calls[0]["params"]["keyword"] == '"quantum sensing"'
    assert calls[0]["params"]["dateStart"] == "12/31/2016"
    assert calls[0]["params"]["dateEnd"] == "12/31/2021"
    top = next(r for r in refs if r["url"].endswith("AWD_ID=2121044"))
    assert top["observed_at"] == "2021-09-02"
    assert top["structured_event"]["amount_usd_m"] == 26.95
    assert top["structured_event"]["funder"] == "NSF"
    assert top["structured_event"]["source_id"] == "2121044"


def test_nsf_empty_response_gives_no_refs(monkeypatch):
    fake_json_request(monkeypatch, nsf, {"response": {"metadata": {"totalCount": 0}}})
    assert nsf.CONNECTOR.run(["quantum sensing"], SNAPSHOT, 5) == []


def test_nsf_drops_unrelated_and_undated_awards(monkeypatch):
    base = fixture("nsf_quantum_sensing")["response"]["award"]
    extra = [
        {"id": "9000001", "title": "Coral reef fish behavior under warming", "date": "10/01/2021",
         "fundsObligatedAmt": 100000, "awardeeName": "Test University",
         "abstractText": "Field study of reef fish schooling."},
        {"id": "9000002", "title": "Quantum sensing of magnetic fields", "date": "",
         "fundsObligatedAmt": 50000, "awardeeName": "Test University", "abstractText": ""},
    ]
    fake_json_request(monkeypatch, nsf, {"response": {"award": base + extra}})
    refs = nsf.CONNECTOR.run(["quantum sensing"], SNAPSHOT, 10)
    ids = {r["url"].rsplit("=", 1)[-1] for r in refs}
    assert "9000001" not in ids and "9000002" not in ids
    assert ids == {"2121044", "2110052", "2132753"}


def test_nsf_uses_short_phrases_and_deduplicates_by_id(monkeypatch):
    calls = fake_json_request(monkeypatch, nsf, fixture("nsf_quantum_sensing"))
    refs = nsf.search_nsf_awards(["quantum sensing navigation platform"], SNAPSHOT)
    keywords = [c["params"]["keyword"] for c in calls]
    assert keywords == ['"quantum sensing navigation"', '"quantum sensing"']
    urls = [r["url"] for r in refs]
    assert len(urls) == len(set(urls)) == 3
