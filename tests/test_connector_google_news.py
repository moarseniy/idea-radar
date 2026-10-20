"""Офлайн-тесты Google News RSS: записанные ответы и синтетические записи для денег."""
from datetime import date
from pathlib import Path

from connector_testkit import assert_connector_contract, fake_json_request

from app.radar.connectors import google_news, registry
from app.radar.connectors._terms import ReadyPhrases

SNAPSHOT = date(2021, 12, 31)
FIXTURES = Path(__file__).parent / "fixtures" / "connectors"

# Точные строки запросов, с которыми записаны фикстуры (as_of 2021-12-31).
VENTURE_QUERY = (
    '("solid-state battery") (raises OR funding OR seed OR "series a" OR "series b" OR acquires '
    'OR acquisition OR invests) after:2017-12-31 before:2022-01-01'
)
RU_QUERY = (
    'квантовые вычисления (site:vc.ru OR site:rbc.ru OR site:cnews.ru OR site:tadviser.ru '
    "OR site:kommersant.ru OR site:rb.ru) after:2017-12-31 before:2022-01-01"
)


def _fixture(name: str) -> str:
    return (FIXTURES / f"google_news_{name}.xml").read_text(encoding="utf-8")


def _rss(*items: tuple[str, str, str, str]) -> str:
    """Синтетический RSS: (заголовок, pubDate, домен, название издания)."""
    body = "".join(
        f"<item><title>{title}</title><link>https://news.google.com/rss/articles/{i}</link>"
        f"<guid>g{i}</guid><pubDate>{pub}</pubDate><description>{title}</description>"
        f'<source url="https://{host}">{name}</source></item>'
        for i, (title, pub, host, name) in enumerate(items))
    return f"<rss><channel>{body}</channel></rss>"


def test_venture_recorded_response_contract_and_query(monkeypatch):
    calls = fake_json_request(monkeypatch, google_news, _fixture("venture"))
    connector = registry()["Google News venture"]
    refs = connector.run(["solid-state battery technology"], SNAPSHOT, 5)

    assert calls[0]["params"]["q"] == VENTURE_QUERY
    assert calls[0]["params"]["hl"] == "en-US" and calls[0]["params"]["ceid"] == "US:en"
    assert len(calls) == 1  # фраза из двух слов: ядра не запрашиваются
    # BMW (не про батареи), Smart Materials (2025) и Mullen (2022) отсеяны; осталась одна запись.
    assert [r["title"] for r in refs] == [
        "Stellantis, Mercedes-Benz invest in solid-state battery developer Factorial Energy"]
    assert refs[0]["publisher"] == "TechCrunch"
    assert refs[0]["url"].startswith("https://news.google.com/rss/articles/")
    assert "structured_event" not in refs[0]   # в заголовке нет суммы: запись без события
    assert_connector_contract(connector, refs, SNAPSHOT)


def test_ru_recorded_response_contract_and_query(monkeypatch):
    calls = fake_json_request(monkeypatch, google_news, _fixture("ru"))
    connector = registry()["Google News RU business"]
    refs = connector.run(["квантовые вычисления"], SNAPSHOT, 5)

    assert calls[0]["params"]["q"] == RU_QUERY
    assert calls[0]["params"]["hl"] == "ru" and calls[0]["params"]["ceid"] == "RU:ru"
    assert len(calls) == 1  # кириллица одна, латинских фраз нет
    # Запись 2023 года и нерелевантный Broadcom отсеяны.
    assert len(refs) == 4
    assert {r["publisher"] for r in refs} == {"RB.RU", "РБК Тренды", "ZOOM.CNews.ru"}
    assert all("2023" not in str(r["observed_at"]) for r in refs)
    assert_connector_contract(connector, refs, SNAPSHOT)


def test_market_connector_has_no_site_filter_and_no_events(monkeypatch):
    calls = fake_json_request(monkeypatch, google_news, _fixture("venture"))
    connector = registry()["Google News"]
    refs = connector.run(["solid-state battery technology"], SNAPSHOT, 5)

    assert connector.channel == "market"
    assert "site:" not in calls[0]["params"]["q"]
    assert calls[0]["params"]["q"].endswith("after:2017-12-31 before:2022-01-01")
    assert len(refs) == 1 and "structured_event" not in refs[0]
    assert_connector_contract(connector, refs, SNAPSHOT)


def test_venture_amount_in_title_becomes_funding_event(monkeypatch):
    xml = _rss(("Solid-state battery startup Acme raises $40 million Series A", "Tue, 02 Mar 2021 08:00:00 GMT",
                "techcrunch.com", "TechCrunch"))
    fake_json_request(monkeypatch, google_news, xml)
    connector = registry()["Google News venture"]
    refs = connector.run(["solid-state battery"], SNAPSHOT, 5)

    assert len(refs) == 1
    event = refs[0]["structured_event"]
    assert event["kind"] == "funding_round" and event["type"] == "funding"
    assert event["date"] == "2021-03-02" and event["company"] == "Solid-state battery startup Acme"
    assert event["amount_usd_m"] == 40.0
    assert_connector_contract(connector, refs, SNAPSHOT, money=True)


def test_ru_amounts_convert_by_currency_and_skip_unknown_currency(monkeypatch):
    xml = _rss(
        ("Квантовый стартап привлек 500 млн рублей", "Mon, 10 May 2021 08:00:00 GMT", "rb.ru", "RB.RU"),
        ("Квантовый стартап привлек $2,5 млн", "Tue, 11 May 2021 08:00:00 GMT", "vc.ru", "VC.ru"),
        ("Квантовый стартап привлек 500 млн", "Wed, 12 May 2021 08:00:00 GMT", "rbc.ru", "РБК"),
    )
    fake_json_request(monkeypatch, google_news, xml)
    connector = registry()["Google News RU business"]
    refs = connector.run(["квантовые вычисления"], SNAPSHOT, 5)

    by_currency = {r["structured_event"]["currency"]: r["structured_event"]
                   for r in refs if "structured_event" in r}
    assert by_currency["RUB"]["amount_native"] == 500_000_000
    assert by_currency["USD"]["amount_native"] == 2_500_000
    assert by_currency["USD"]["company"] == "Квантовый стартап"
    # Без валюты запись остаётся новостью, но событие не создаётся.
    assert len(refs) == 3 and len(by_currency) == 2
    assert_connector_contract(connector, [r for r in refs if "structured_event" in r], SNAPSHOT, money=True)


def test_any_publisher_kept_undated_and_later_records_dropped(monkeypatch):
    xml = _rss(
        ("Solid-state battery maker raises $10 million", "Fri, 01 Oct 2021 08:00:00 GMT",
         "example.com", "Example"),
        ("Solid-state battery maker raises $11 million", "", "techcrunch.com", "TechCrunch"),
        ("Solid-state battery maker raises $12 million", "Mon, 03 Jan 2022 08:00:00 GMT",
         "techcrunch.com", "TechCrunch"),
        ("Solid-state battery maker raises $13 million", "Thu, 02 Dec 2021 08:00:00 GMT",
         "news.crunchbase.com", "Crunchbase News"),
    )
    fake_json_request(monkeypatch, google_news, xml)
    refs = registry()["Google News venture"].run(["solid-state battery"], SNAPSHOT, 5)

    assert [r["structured_event"]["amount_usd_m"] for r in refs] == [10.0, 13.0]


def test_venture_keeps_only_deal_headlines_and_falls_back_to_phrase_cores(monkeypatch):
    calls = []

    def fake(_provider, _url, *, params, **_kwargs):
        calls.append(params["q"])
        if '"prompt injection detection"' in params["q"]:
            return "<rss><channel></channel></rss>"
        return _rss(
            ("Prompt injection explained", "Mon, 01 Mar 2021 08:00:00 GMT", "example.com", "Example"),
            ("Lakera raises $20M to stop prompt injection attacks", "Tue, 02 Mar 2021 08:00:00 GMT",
             "techcrunch.com", "TechCrunch"),
            ("UK NCSC raises alarms over prompt injection", "Wed, 03 Mar 2021 08:00:00 GMT",
             "infosecurity-magazine.com", "Infosecurity"),
            ("Acme acquires prompt injection startup Guard", "Thu, 04 Mar 2021 08:00:00 GMT",
             "siliconangle.com", "SiliconANGLE"))

    monkeypatch.setattr(google_news, "_json_request", fake)
    refs = google_news._venture(ReadyPhrases(["prompt injection detection"]), SNAPSHOT, 5)

    # Статья без сделки и «raises alarms» отсеяны; двух сделок по ядру хватило, хвост не нужен.
    assert [r["title"] for r in refs] == ["Lakera raises $20M to stop prompt injection attacks",
                                          "Acme acquires prompt injection startup Guard"]
    assert [q.split(")")[0] for q in calls] == ['("prompt injection detection"', '("prompt injection"']


def test_empty_response_returns_no_records(monkeypatch):
    fake_json_request(monkeypatch, google_news, "<rss><channel></channel></rss>")
    assert registry()["Google News venture"].run(["solid-state battery"], SNAPSHOT, 5) == []


def test_window_is_four_years_back_and_handles_leap_day():
    assert google_news._window(SNAPSHOT) == ("2017-12-31", "2022-01-01")
    # 2100 не високосный: 29 февраля 2104 года переносится на 28 февраля.
    assert google_news._window(date(2104, 2, 29)) == ("2100-02-28", "2104-03-01")
