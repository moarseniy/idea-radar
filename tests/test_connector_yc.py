from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request, fixture

from app.radar.connectors import yc

AS_OF = date(2026, 10, 17)
SNAPSHOT = date(2021, 12, 31)


def test_batch_date_maps_seasons_to_start_dates():
    assert yc.batch_date("W21") == date(2021, 1, 1)
    assert yc.batch_date("S21") == date(2021, 6, 1)
    assert yc.batch_date("F24") == date(2024, 9, 1)
    assert yc.batch_date("X25") == date(2025, 4, 1)
    assert yc.batch_date("s21") == date(2021, 6, 1)
    for bad in (None, "", "Z21", "W2021", "W"):
        assert yc.batch_date(bad) is None


def test_contract_on_recorded_yc_response(monkeypatch):
    calls = fake_json_request(monkeypatch, yc, fixture("yc_battery"))
    refs = yc.CONNECTOR.run(["battery"], AS_OF, 5)
    assert_connector_contract(yc.CONNECTOR, refs, AS_OF, money=True)
    assert calls[0]["url"] == yc.API
    assert calls[0]["params"] == {"q": "battery", "page": 1}
    gbatteries = next(r for r in refs if r["url"].endswith("/gbatteries"))
    assert gbatteries["title"].startswith("GBatteries — ")
    assert gbatteries["observed_at"] == "2014-01-01"
    assert gbatteries["structured_event"]["type"] == "funding"
    assert gbatteries["structured_event"]["kind"] == "accelerator"
    assert gbatteries["structured_event"]["date"] == "2014-01-01"


def test_companies_after_as_of_are_dropped(monkeypatch):
    fake_json_request(monkeypatch, yc, fixture("yc_battery"))
    refs = yc.CONNECTOR.run(["battery"], SNAPSHOT, 5)
    assert {r["url"] for r in refs} == {
        "https://www.ycombinator.com/companies/gbatteries",
        "https://www.ycombinator.com/companies/zitara-technologies-inc"}


def test_status_is_kept_outside_document_text(monkeypatch):
    fake_json_request(monkeypatch, yc, fixture("yc_battery"))
    refs = yc.CONNECTOR.run(["battery"], AS_OF, 5)
    kyten = next(r for r in refs if r["url"].endswith("/kyten-technologies"))
    assert kyten["current_status"] == "Inactive"
    assert "Inactive" not in kyten["title"]
    assert "Inactive" not in kyten["document"]["text"]


def test_unrelated_startups_are_filtered(monkeypatch):
    fake_json_request(monkeypatch, yc, fixture("yc_battery"))
    refs = yc.CONNECTOR.run(["battery"], AS_OF, 5)
    assert not any(r["url"].endswith("/portal-entryways") for r in refs)


def test_query_uses_short_phrase_and_follows_next_page(monkeypatch):
    empty_page = {"companies": [], "nextPage": "next", "page": 1, "totalPages": 2}
    calls = fake_json_request(monkeypatch, yc, empty_page, fixture("yc_battery"))
    yc.CONNECTOR.run(["solid-state battery technology"], AS_OF, 5)
    assert calls[0]["params"] == {"q": "solid-state battery", "page": 1}
    assert calls[1]["params"] == {"q": "solid-state battery", "page": 2}


def test_empty_response_returns_no_references(monkeypatch):
    fake_json_request(monkeypatch, yc, {"companies": [], "nextPage": None, "page": 1, "totalPages": 0})
    assert yc.CONNECTOR.run(["battery"], AS_OF, 5) == []


def test_connector_stays_off_by_default():
    assert yc.CONNECTOR.default_on is False
    assert yc.CONNECTOR.date_support == "post"
    assert yc.CONNECTOR.channel == "money"


def test_phrase_words_scattered_over_description_do_not_match(monkeypatch):
    def company(slug, one_liner, description):
        return {"name": slug.title(), "slug": slug, "url": f"https://www.ycombinator.com/companies/{slug}",
                "batch": "W24", "oneLiner": one_liner, "longDescription": description, "tags": []}

    filler = " ".join(["We help people get everyday things done faster."] * 4)
    fake_json_request(monkeypatch, yc, {"companies": [
        company("identityco", "Identity layer for AI agents", "Every agent gets its own identity."),
        company("errands", "Personal AI agent", f"An agent for errands. {filler} Later it may need your identity."),
    ]})
    refs = yc.search(["agent identity"], AS_OF, 5)
    assert [r["url"].rsplit("/", 1)[1] for r in refs] == ["identityco"]
