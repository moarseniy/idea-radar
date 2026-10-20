from datetime import date

import pytest

from app.radar.connectors import registry
from app.radar.connectors.venture_news import _clean, parse_round

SNAPSHOT = date(2021, 12, 31)


def test_parse_round_reads_funding_headlines():
    assert parse_round("Zapata Computing Raises $38M As Quantum Computing Nears") == (
        "Zapata Computing", 38_000_000, "USD", None)
    assert parse_round("Acme Robotics raises $20M Series A") == ("Acme Robotics", 20_000_000, "USD", "Series A")
    assert parse_round("Nordic Batteries secures €10M") == ("Nordic Batteries", 10_000_000, "EUR", None)
    assert parse_round("Cellforge lands $5 million seed") == ("Cellforge", 5_000_000, "USD", "Seed")
    assert parse_round("Helio Grid closes £2.5M pre-seed round") == (
        "Helio Grid", 2_500_000, "GBP", "Pre-seed")


@pytest.mark.parametrize("title", [
    "Why solid-state batteries matter for EVs",
    "Solid Power ramps up solid-state battery race with public market debut",
    "Nissan to invest $17.6 billion in EV development over the next five years",
    "Stellantis, Mercedes-Benz invest in solid-state battery developer Factorial Energy",
    "The Briefing: IonQ Going Public Via SPAC, Starling Bank Lands $376M, And More",
    "Funding Pours Into Cybersecurity As Mid-Year 2021 Numbers Eclipse Last Year's Total",
])
def test_parse_round_ignores_headlines_without_a_round(title):
    assert parse_round(title) is None


def test_clean_unescapes_entities_and_strips_tags():
    assert _clean("<p>Solid &#8217;Power&#8221; &amp; more</p>\n<img src='x'>") == "Solid ’Power” & more"


def test_techcrunch_request_uses_four_year_window_and_short_phrase(monkeypatch):
    from connector_testkit import assert_connector_contract, fake_json_request, fixture

    from app.radar.connectors import venture_news as module

    calls = fake_json_request(monkeypatch, module, fixture("venture_news_techcrunch"))
    connector = registry()["TechCrunch"]
    refs = connector.run(["solid-state battery"], SNAPSHOT, 5)
    assert_connector_contract(connector, refs, SNAPSHOT)
    params = calls[0]["params"]
    assert params["after"] == "2017-12-31T00:00:00"
    assert params["before"] == "2021-12-31T23:59:59"
    assert all(len(call["params"]["search"].split()) <= 3 for call in calls)
    assert calls[0]["provider"] == "techcrunch"
    assert {r["title"] for r in refs} == {
        "Solid Power ramps up solid-state battery race with public market debut",
        "Stellantis, Mercedes-Benz invest in solid-state battery developer Factorial Energy",
    }
    assert all("structured_event" not in r for r in refs)


def test_crunchbase_recorded_round_becomes_funding_event(monkeypatch):
    from connector_testkit import assert_connector_contract, fake_json_request, fixture

    from app.radar.connectors import venture_news as module

    fake_json_request(monkeypatch, module, fixture("venture_news_crunchbase"))
    connector = registry()["Crunchbase News"]
    refs = connector.run(["quantum computing"], SNAPSHOT, 5)
    rounds = [r for r in refs if "structured_event" in r]
    assert_connector_contract(connector, rounds, SNAPSHOT, money=True)
    event = rounds[0]["structured_event"]
    assert (event["kind"], event["company"], event["amount_usd_m"], event["date"]) == (
        "funding_round", "Zapata Computing", 38.0, "2020-11-19")
    # Новость без раунда про кибербезопасность не относится к запросу и отсеивается.
    assert "Funding Pours" not in " ".join(r["title"] for r in refs)


def test_empty_response_gives_no_references(monkeypatch):
    from connector_testkit import fake_json_request

    from app.radar.connectors import venture_news as module

    fake_json_request(monkeypatch, module, [])
    assert registry()["Crunchbase News"].run(["quantum computing"], SNAPSHOT, 5) == []


def test_http_error_propagates_to_pipeline(monkeypatch):
    from app.radar.connectors import venture_news as module

    def blocked(provider, url, **kwargs):
        raise RuntimeError("403 Forbidden: Cloudflare challenge")

    monkeypatch.setattr(module, "_json_request", blocked)
    with pytest.raises(RuntimeError, match="403"):
        registry()["TechCrunch"].run(["solid-state battery"], SNAPSHOT, 5)


def test_venture_news_connectors_are_off_by_default():
    from app.radar.connectors import connectors

    assert {"TechCrunch", "Crunchbase News"} <= set(registry())
    assert not registry()["TechCrunch"].default_on and not registry()["Crunchbase News"].default_on
    assert not {"TechCrunch", "Crunchbase News"} & {c.name for c in connectors(["solid-state battery"])}
