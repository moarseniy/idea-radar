from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request, fixture

from app.radar.connectors import ukri
from app.radar.connectors.events import to_usd_m

SNAPSHOT = date(2021, 12, 31)


def test_ukri_connector_is_money_post_and_off_by_default():
    assert ukri.CONNECTOR.channel == "money"
    assert ukri.CONNECTOR.date_support == "post"
    assert ukri.CONNECTOR.default_on is False


def test_ukri_contract_with_recorded_response(monkeypatch):
    # Записанный ответ: поиск, фонд и организация; первый проект начат в 2024 и отсекается датой.
    calls = fake_json_request(monkeypatch, ukri, fixture("ukri_search"), fixture("ukri_fund"),
                              fixture("ukri_org"))
    refs = ukri.CONNECTOR.run(["quantum sensing"], SNAPSHOT, 1)
    assert_connector_contract(ukri.CONNECTOR, refs, SNAPSHOT, money=True)
    ref = refs[0]
    assert ref["url"] == "https://gtr.ukri.org/projects?ref=103434"
    assert ref["title"] == "Diamond Quantum Magnetic Sensing"
    assert ref["observed_at"] == "2017-10-01"
    event = ref["structured_event"]
    assert event["date"] == "2017-10-01"
    assert event["amount_native"] == 286604 and event["currency"] == "GBP"
    assert event["amount_usd_m"] == to_usd_m(286604, "GBP", 2017)
    assert event["company"] == "ELEMENT SIX (UK) LIMITED."
    assert event["funder"] == "ISCF"
    assert [c["url"] for c in calls] == [
        "https://gtr.ukri.org/gtr/api/projects",
        "https://gtr.ukri.org/gtr/api/funds/F117477E-1890-4801-AD96-1DB5327C669E",
        "https://gtr.ukri.org/gtr/api/organisations/A6099C1C-9770-427D-BA9E-D9E5D9CF385D",
    ]
    assert calls[0]["params"] == {"q": "quantum sensing", "s": 25, "p": 1}
    assert calls[0]["headers"]["Accept"] == "application/vnd.rcuk.gtr.json-v7"


def test_ukri_drops_projects_started_after_as_of(monkeypatch):
    calls = fake_json_request(monkeypatch, ukri, {"project": [fixture("ukri_search")["project"][0]]})
    assert ukri.CONNECTOR.run(["quantum sensing"], SNAPSHOT, 5) == []
    assert len(calls) == 1  # фонд и организация не запрашиваются для записи вне даты


def test_ukri_empty_response_gives_empty_list(monkeypatch):
    fake_json_request(monkeypatch, ukri, {"project": []})
    assert ukri.search_ukri(["quantum sensing"], SNAPSHOT, 5) == []


def test_ukri_rejects_unrelated_projects(monkeypatch):
    project = dict(fixture("ukri_search")["project"][1])
    project.update(title="Motorcycle carrier attachment", abstractText="Mechanical design of a rack.")
    calls = fake_json_request(monkeypatch, ukri, {"project": [project]})
    assert ukri.search_ukri(["quantum sensing"], SNAPSHOT, 5) == []
    assert len(calls) == 1


def test_ukri_query_uses_short_phrases_and_no_date_filter(monkeypatch):
    calls = fake_json_request(monkeypatch, ukri, {"project": []})
    ukri.search_ukri(["quantum sensing for next-generation navigation"], SNAPSHOT, 5)
    assert [c["params"]["q"] for c in calls] == ["quantum sensing navigation", "quantum sensing",
                                               "sensing navigation"]
    assert all("as_of" not in c["params"] for c in calls)
