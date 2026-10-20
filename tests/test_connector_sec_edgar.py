"""Офлайн-тесты коннектора SEC EDGAR: записанная выдача вместо сети.

Фикстура tests/fixtures/connectors/sec_edgar_fts_solid_state.json — живой ответ efts.sec.gov
на "solid-state battery" до 2021-12-31 (17.10.2026), урезанный до трёх документов.
"""
from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request, fixture

from app.radar.connectors import sec_edgar
from app.radar.connectors.sec_edgar import CONNECTOR, form_d_rounds, search_filings

SNAPSHOT = date(2021, 12, 31)
TERMS = ["solid-state battery"]
EMAIL = "radar-tests@example.org"


def _hit(name: str, form: str, description: str, file_date: str, adsh: str, text: str) -> dict:
    cik = adsh.split("-")[0].lstrip("0")
    return {"_id": f"{adsh}:doc.htm",
            "_source": {"ciks": [cik], "display_names": [f"{name}  (TKR)  (CIK {cik})"], "adsh": adsh,
                        "form": form, "file_description": description, "file_date": file_date},
            "highlight": {"file_text": [text]} if text else {}}


def _payload(*hits: dict) -> dict:
    return {"hits": {"total": {"value": len(hits)}, "hits": list(hits)}}


def test_contract_with_recorded_response(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    calls = fake_json_request(monkeypatch, sec_edgar, fixture("sec_edgar_fts_solid_state"))
    refs = CONNECTOR.run(TERMS, SNAPSHOT, 5)
    assert_connector_contract(CONNECTOR, refs, SNAPSHOT, money=True)
    assert calls[0]["params"]["enddt"] == "2021-12-31"
    assert calls[0]["params"]["q"] == '"solid-state battery"'
    assert calls[0]["headers"]["User-Agent"] == f"IDEA-radar/0.1 ({EMAIL})"
    # Записанный живой ответ (17.10.2026): highlight API не отдаёт.
    assert refs[0]["url"] == ("https://www.sec.gov/Archives/edgar/data/1811414/"
                              "000119312521044982/d137857dex991.htm")
    assert refs[0]["title"] == "QuantumScape Corp 8-K EX-99.1"
    assert refs[0]["observed_at"] == "2021-02-16"
    assert refs[0]["structured_event"]["kind"] == "filing_mention"
    assert refs[0]["structured_event"]["company"] == "QuantumScape Corp"


def test_long_term_is_searched_by_short_phrase(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    calls = fake_json_request(monkeypatch, sec_edgar, _payload())
    search_filings(["agent security posture management"], SNAPSHOT, 5)
    assert calls[0]["params"]["q"] == '"agent security posture"'


def test_empty_response_gives_no_refs(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    fake_json_request(monkeypatch, sec_edgar, _payload())
    assert search_filings(TERMS, SNAPSHOT, 5) == []


def test_irrelevant_filing_is_dropped(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    fake_json_request(monkeypatch, sec_edgar, _payload(
        _hit("Dividend Trust", "10-K", "Annual report", "2021-03-01", "0001111111-21-000001",
             "quarterly dividend distribution policy")))
    assert search_filings(TERMS, SNAPSHOT, 5) == []


def test_filing_without_highlight_trusts_server_phrase_match(monkeypatch):
    # Так отвечает живой API: без highlight, в заголовке только компания и форма.
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    payload = _payload(_hit("QuantumScape Corp", "10-K", "Annual report", "2021-02-23",
                            "0001811414-21-000010", "unused"))
    for hit in payload["hits"]["hits"]:
        hit.pop("highlight", None)
    fake_json_request(monkeypatch, sec_edgar, payload)
    refs = search_filings(TERMS, SNAPSHOT, 5)
    assert len(refs) == 1
    assert "contains the phrase" in refs[0]["document"]["text"]


def test_disabled_without_contact_email(monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    calls = fake_json_request(monkeypatch, sec_edgar, _payload())
    assert not CONNECTOR.enabled(TERMS)
    assert search_filings(TERMS, SNAPSHOT, 5) == []
    assert calls == []


def test_form_d_keeps_only_matching_filer(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    calls = fake_json_request(monkeypatch, sec_edgar, _payload(
        _hit("Example Power Corp", "D", "Form D", "2021-05-10", "0001234567-21-000777", ""),
        _hit("Unrelated Holdings", "D", "Form D", "2021-05-11", "0002222222-21-000001", "")))
    refs = form_d_rounds("Example Power Corp", SNAPSHOT)
    assert calls[0]["params"]["forms"] == "D"
    assert calls[0]["params"]["enddt"] == "2021-12-31"
    assert len(refs) == 1
    assert refs[0]["observed_at"] == "2021-05-10"
    event = refs[0]["structured_event"]
    assert event["kind"] == "funding_round" and event["date"] == "2021-05-10"
    assert event["amount_usd_m"] is None
