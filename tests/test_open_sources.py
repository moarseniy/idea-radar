import hashlib
import json
import os
import subprocess
import urllib.parse
from datetime import date

import httpx
import pytest

from app.radar import open_sources
from app.radar import sources as source_module
from app.radar.open_sources import _get, _keywords, branch_tasks, gdelt_news, run_task
from app.radar.resilience import CircuitBreaker, CircuitOpenError
from app.radar.sources import SourceFetcher


@pytest.fixture(autouse=True)
def isolated_research_cache(tmp_path, monkeypatch):
    from app.radar import research_apis

    monkeypatch.setattr(research_apis, "CACHE_DIR", tmp_path / "research_apis")
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)


def test_failed_connector_returns_empty_result_not_exception():
    def boom():
        raise TimeoutError
    result = run_task("arXiv", boom, "ветка", "q")
    assert result["references"] == [] and result["error"] == "TimeoutError"


def test_http_status_is_visible_in_connector_audit():
    response = httpx.Response(400, request=httpx.Request("GET", "https://api.example.test"))
    failure = httpx.HTTPStatusError("bad request", request=response.request, response=response)
    result = run_task("Crossref", lambda: (_ for _ in ()).throw(failure), "branch", "query")
    assert result["error"] == "HTTP 400"


def test_references_are_tagged_with_connector():
    result = run_task("Хабр", lambda: [{"url": "https://habr.com/ru/articles/1/", "title": "t"}], "ветка", "q")
    assert result["references"][0]["branch"] == "ветка · Хабр"


def test_api_abstract_is_read_without_network(tmp_path):
    def no_network(*_):
        raise AssertionError("страница не должна загружаться")
    fetcher = SourceFetcher(tmp_path, fetch=no_network)
    text = "In-sensor computing for wearables. " + "We demonstrate an analog in-sensor processor. " * 5
    source = fetcher.load({"url": "https://doi.org/10.1/x", "title": "t", "branch": "b",
                           "document": {"title": "t", "text": text, "date": "2026-03-01", "language": "en",
                                        "extraction": "crossref_abstract"}}, date(2026, 9, 22))
    assert source["status"] == "read" and source["type"] == "publication" and source["text"] == text


def test_keywords_drop_stopwords():
    assert _keywords("Emerging technologies for AI agent identity in banking") == ["agent", "identity", "banking"]


def test_branch_connectors_are_generated_for_localized_queries(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "1")
    tasks = branch_tasks("AI security", "защита ИИ", "AI security", date.today(), [
        {"language": "ru", "query": "защита ИИ"},
        {"language": "en", "query": "AI security"},
        {"language": "zh", "query": "人工智能安全"},
    ])
    names = {name for name, _, _, _ in tasks}
    assert "Bing News (Simplified Chinese)" in names
    assert "Crossref (Simplified Chinese)" in names
    assert "GDELT Global News" in names
    assert "Patents (FreePatentsOnline)" in names
    assert {"OpenAlex", "NIH RePORTER", "CORDIS grants", "GitHub repositories",
            "Hugging Face", "TED procurement"} <= names
    assert "arXiv (EN)" in names and "Хабр (RU)" in names


def test_historical_branch_drops_sources_without_reliable_dates(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "1")
    monkeypatch.setenv("RADAR_EXTRA_CONNECTORS", "all")
    tasks = branch_tasks("AI security", "защита ИИ", "AI security", date(2021, 12, 31), [
        {"language": "ru", "query": "защита ИИ"},
        {"language": "en", "query": "AI security"},
    ])
    names = {name for name, _, _, _ in tasks}
    # Bing и Хабр — без фильтра по дате, GitHub и Hugging Face — текущие звёзды и даты.
    assert not any(name.startswith(("Bing News", "Хабр")) for name in names)
    assert not names & {"GitHub repositories", "Hugging Face"}
    # Рынок в прошлом — Google News (before:) на каждом языке вместо Bing и GDELT.
    assert {"Google News (English)", "Google News (Russian)", "Google News", "GDELT Global News",
            "Crossref (English)", "OpenAlex", "NIH RePORTER"} <= names


def test_verification_news_uses_dated_google_news_in_the_past():
    from app.radar.pipeline import verification_news

    assert verification_news(date.today())[0] == "Bing News (EN)"
    assert verification_news(date(2021, 12, 31))[0] == "Google News"


def test_openalex_can_be_disabled_without_disabling_other_structured_connectors(monkeypatch):
    monkeypatch.setenv("FEATURE_STRUCTURED_SOURCES", "1")
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    tasks = branch_tasks("AI security", "защита ИИ", "AI security", date.today(), [
        {"language": "ru", "query": "защита ИИ"},
        {"language": "en", "query": "AI security"},
    ])
    names = {name for name, *_ in tasks}

    assert "OpenAlex" not in names
    assert {"NIH RePORTER", "CORDIS grants", "GitHub repositories", "TED procurement"} <= names


def test_gdelt_429_is_not_retried(monkeypatch):
    requested = []
    monkeypatch.setattr(open_sources.PACE["gdelt"], "wait", lambda: None)
    monkeypatch.setitem(open_sources.BREAKERS, "gdelt", CircuitBreaker())

    def throttled(url, **_kwargs):
        requested.append(url)
        response = httpx.Response(429, request=httpx.Request("GET", url), headers={"Retry-After": "60"})
        response.raise_for_status()
        return response

    monkeypatch.setattr(open_sources.httpx, "get", throttled)
    with pytest.raises(httpx.HTTPStatusError):
        _get("gdelt", "https://api.gdeltproject.org/api/v2/doc/doc")

    assert len(requested) == 1


def test_arxiv_api_failure_uses_openalex_indexed_preprint(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "1")
    monkeypatch.setattr(open_sources, "_keywords", lambda *_args, **_kwargs: ["quantum", "battery"])
    monkeypatch.setattr(open_sources, "_get", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("HTTP 406")))
    monkeypatch.setattr("app.radar.research_apis.openalex_works", lambda *_args, **_kwargs: [{
        "title": "Quantum battery", "abstract": "A preprint abstract.", "publicationDate": "2026-01-01",
        "language": "en", "primary_location_url": "https://arxiv.org/abs/2601.00001",
        "primary_source": "arXiv",
    }])
    refs = open_sources.arxiv("quantum battery", date(2026, 9, 28), limit=2)
    assert refs[0]["url"] == "https://arxiv.org/abs/2601.00001"
    assert refs[0]["document"]["extraction"] == "openalex_arxiv_abstract"


def test_arxiv_does_not_fall_back_to_openalex_when_it_is_disabled(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    monkeypatch.setattr(open_sources, "_keywords", lambda *_args, **_kwargs: ["quantum", "battery"])
    monkeypatch.setattr(open_sources, "_get", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("HTTP 406")))
    monkeypatch.setattr("app.radar.research_apis.openalex_works", lambda *_args, **_kwargs:
                        pytest.fail("disabled OpenAlex must not be called"))

    assert open_sources.arxiv("quantum battery", date(2026, 9, 28), limit=2) == []


def test_crossref_live_connector_requests_only_supported_fields(monkeypatch):
    request = {}
    abstract = "A detailed abstract about technology and validated performance. " * 4

    def fake_get(_kind, url):
        request["url"] = url
        return ('{"message":{"items":[{"DOI":"10.1000/x","title":["Fintech payments"],'
                '"abstract":"' + abstract + '","published":{"date-parts":[[2026,4,1]]}}]}}')

    monkeypatch.setattr(open_sources, "_get", fake_get)
    refs = open_sources.crossref("fintech payments", date(2026, 9, 28), limit=2)
    assert len(refs) == 1 and refs[0]["document"]["extraction"] == "crossref_abstract"
    assert "language" not in request["url"].split("&select=")[-1].split("&")[0]


def test_crossref_drops_off_topic_records_and_records_created_after_as_of(monkeypatch):
    request = {}
    abstract = " Detailed abstract with methods, evaluation and validated performance results." * 3

    def item(doi, title):
        return {"DOI": doi, "title": [title], "abstract": title + abstract, "published": {"date-parts": [[2022, 3, 1]]}}

    def fake_get(_kind, url):
        request["url"] = urllib.parse.unquote(url)
        return json.dumps({"message": {"items": [
            item("10.1/a", "Conversational chatbots in retail banking customer service"),
            item("10.1/b", "Wedding banquet service quality and customer satisfaction")]}})

    monkeypatch.setattr(open_sources, "_get", fake_get)
    refs = open_sources.crossref("chatbots in banking customer service", date(2022, 6, 30))
    assert [r["url"] for r in refs] == ["https://doi.org/10.1/a"]
    assert "until-created-date:2022-06-30" in request["url"]


def test_crossref_skips_scripts_without_checkable_words(monkeypatch):
    monkeypatch.setattr(open_sources, "_get", lambda *_: (_ for _ in ()).throw(AssertionError("no request")))
    assert open_sources.crossref("银行客户服务中的大型语言模型", date(2022, 6, 30), language="zh") == []


def test_localized_google_news_keeps_period_and_relevant_items(monkeypatch):
    from app.radar.connectors import google_news

    calls = []

    def fake(_provider, _url, *, params, **_kwargs):
        calls.append(params)
        items = [("Чат-боты в банках: как нейросети отвечают клиентам", "Mon, 10 Jan 2022 08:00:00 GMT"),
                 ("Футбольный клуб сменил тренера", "Tue, 11 Jan 2022 08:00:00 GMT"),
                 ("Чат-боты в банках после запуска ChatGPT", "Thu, 01 Dec 2022 08:00:00 GMT")]
        return "<rss><channel>" + "".join(
            f"<item><title>{t}</title><link>https://news.google.com/rss/articles/{i}</link><guid>g{i}</guid>"
            f"<pubDate>{d}</pubDate><description>{t}</description><source url=\"https://vc.ru\">VC</source></item>"
            for i, (t, d) in enumerate(items)) + "</channel></rss>"

    monkeypatch.setattr(google_news, "_json_request", fake)
    refs = open_sources.google_news_localized("чат-боты в банках", "ru", date(2022, 6, 30))
    assert [r["title"] for r in refs] == ["Чат-боты в банках: как нейросети отвечают клиентам"]
    assert calls[0]["hl"] == "ru-RU" and calls[0]["ceid"] == "RU:ru"
    assert calls[0]["q"].startswith("чат-боты в банках ") and "before:2022-07-01" in calls[0]["q"]


def test_openalex_drops_works_whose_crossref_record_appeared_after_as_of(monkeypatch):
    from app.radar import research_apis as api

    monkeypatch.setenv("OPENALEX_ENABLED", "1")
    works = [{"id": "W1", "doi": "10.1/old", "title": "Chatbots in banking", "abstract": "abstract",
              "publicationDate": "2022-03-01", "language": "en", "countries": set()},
             {"id": "W2", "doi": "10.1/BACKDATED", "title": "Generative AI chatbots", "abstract": "abstract",
              "publicationDate": "2022-06-30", "language": "en", "countries": set()}]
    monkeypatch.setattr(api, "openalex_works", lambda *_a, **_k: works)
    calls = []

    def fake_request(provider, url, *, params, **_kwargs):
        calls.append(params["filter"])
        return {"message": {"items": [{"DOI": "10.1/backdated"}]}}

    monkeypatch.setattr(api, "_json_request", fake_request)
    refs = api.search_openalex(["chatbots banking"], date(2022, 6, 30))
    assert [r["url"] for r in refs] == ["https://doi.org/10.1/old"]
    assert "doi:10.1/backdated" in calls[0] and "from-created-date:2022-07-01" in calls[0]
    # В живом режиме проверка не нужна и не вызывается.
    calls.clear()
    api.search_openalex(["chatbots banking"], date.today())  # noqa: DTZ011
    assert calls == []


def test_patent_search_uses_fpo_as_the_only_patent_source(monkeypatch):
    calls = []
    monkeypatch.setattr("app.ml.features.evidence.search_patents_fpo",
                        lambda *_args, **_kwargs: calls.append("fpo") or [{"url": "https://fpo.example/patent"}])
    assert open_sources._patent_search(["AI safety"], date(2026, 9, 22))[0]["url"] == \
        "https://fpo.example/patent"
    assert calls == ["fpo"]


def test_patent_search_reports_failure_when_both_sources_fail(monkeypatch):
    monkeypatch.setenv("RADAR_WEB_SEARCH", "service")
    def fail(*_args, **_kwargs):
        raise RuntimeError("temporary FPO error")

    monkeypatch.setattr("app.ml.features.evidence.search_patents_fpo", fail)
    monkeypatch.setattr("app.ml.features.evidence.search_patents_google_web", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="Google Patents web fallback returned no results.*temporary FPO error"):
        open_sources._patent_search(["AI safety"], date(2026, 9, 22))


def test_patent_search_falls_back_to_google_patents_web(monkeypatch):
    monkeypatch.setenv("RADAR_WEB_SEARCH", "service")
    def fail(*_args, **_kwargs):
        raise RuntimeError("temporary FPO error")

    monkeypatch.setattr("app.ml.features.evidence.search_patents_fpo", fail)
    monkeypatch.setattr("app.ml.features.evidence.search_patents_google_web",
                        lambda *_args, **_kwargs: [{"url": "https://patents.google.com/patent/US123/en",
                                                    "title": "Patent", "source_provider":
                                                    "Google Patents web search"}])
    result = open_sources._patent_search(["battery separator", "защита батареи", "energy storage"],
                                         date(2026, 9, 22))
    assert result[0]["source_provider"] == "Google Patents web search"
    assert open_sources.run_task("Patents (FreePatentsOnline)", lambda: result, "topic", "query")["connector"] == \
        "Patents (Google Patents web fallback)"


def test_gdelt_uses_global_english_index_and_preserves_article_locale(monkeypatch):
    monkeypatch.setattr(open_sources.PACE["gdelt"], "wait", lambda: None)
    requested = {}

    def fake_get(_kind, url):
        requested["url"] = url
        return ('{"articles":[{"url":"https://example.jp/article",'
                '"title":"新しい技術","seendate":"20260926093000",'
                '"language":"Japanese","sourcecountry":"Japan"}]}')

    monkeypatch.setattr(open_sources, "_get", fake_get)
    refs = gdelt_news("AI security", date(2026, 9, 26))
    assert "AI+security" in requested["url"]
    assert refs == [{"url": "https://example.jp/article", "title": "新しい技術",
                     "observed_at": "2026-09-26", "source_language": "japanese",
                     "source_country": "JAPAN"}]


def test_gdelt_non_json_response_has_actionable_connector_error(monkeypatch):
    monkeypatch.setattr(open_sources.PACE["gdelt"], "wait", lambda: None)
    monkeypatch.setattr(open_sources, "_get", lambda *_args: "upstream temporarily unavailable")
    with pytest.raises(RuntimeError, match="GDELT returned a non-JSON response"):
        gdelt_news("AI security", date(2026, 9, 26))


def test_gdelt_does_not_request_data_outside_its_rolling_window(monkeypatch):
    monkeypatch.setattr(open_sources, "_get", lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert gdelt_news("AI security", date(2020, 1, 1)) == []


def test_connector_retries_transient_network_failure(monkeypatch):
    monkeypatch.setattr(open_sources.PACE["crossref"], "wait", lambda: None)
    monkeypatch.setattr(open_sources, "BREAKERS", {**open_sources.BREAKERS,
                                                     "crossref": CircuitBreaker()})
    monkeypatch.setattr(open_sources.time, "sleep", lambda _: None)
    attempts = iter([httpx.ConnectError("temporary", request=httpx.Request("GET", "https://example.org")),
                     httpx.Response(200, text="ok", request=httpx.Request("GET", "https://example.org"))])
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(open_sources.httpx, "get", fake_get)
    assert _get("crossref", "https://example.org") == "ok"
    assert len(calls) == 2


def test_circuit_breaker_pauses_connector_after_repeated_failures():
    breaker = CircuitBreaker(threshold=2, reset_after=30)
    breaker.failure()
    breaker.failure()
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_source_fetch_retries_once_and_records_input_hash(tmp_path, monkeypatch):
    body = b"html response bytes"
    attempts = []

    def fetch(url, _timeout):
        attempts.append(url)
        if len(attempts) == 1:
            raise TimeoutError("temporary")
        return body, "text/html", url

    text = "technical evidence " * 20
    monkeypatch.setattr(source_module, "extract_document", lambda *_: {
        "text": text, "title": "Example", "date": None, "language": "en", "extraction": "html"})
    monkeypatch.setattr(source_module.time, "sleep", lambda _: None)
    fetcher = SourceFetcher(tmp_path, timeout=2, fetch=fetch)
    source = fetcher.load({"url": "https://example.org/report", "title": "Example"}, date.today())
    assert source["status"] == "read"
    assert source["parse_status"] == "parsed"
    assert source["ingest_sha256"] == hashlib.sha256(body).hexdigest()
    assert source["parser_version"] == source_module.PARSER_VERSION
    assert len(attempts) == 2


def test_dns_resolution_is_bounded(tmp_path, monkeypatch):
    def slow_dns(*_args, **kwargs):
        raise subprocess.TimeoutExpired("getent", kwargs["timeout"])

    monkeypatch.setattr(source_module.subprocess, "run", slow_dns)
    with pytest.raises(TimeoutError, match="DNS-разрешения"):
        source_module.public_ip("slow.example", 443, timeout=0.2)


def test_dns_resolution_rejects_private_addresses(monkeypatch):
    monkeypatch.setattr(source_module.subprocess, "run", lambda *_args, **_kwargs:
                        type("Result", (), {"returncode": 0, "stdout": "127.0.0.1 STREAM\n"})())
    with pytest.raises(ValueError, match="публичным"):
        source_module.public_ip("localhost", 443)


def test_dns_check_works_without_getent(monkeypatch):
    """macOS: getent нет — адрес разрешается системным резолвером, проверка публичности остаётся."""
    import socket
    import subprocess

    import pytest

    from app.radar import sources

    def no_getent(*args, **kwargs):
        raise FileNotFoundError("getent")
    monkeypatch.setattr(subprocess, "run", no_getent)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.215.14", 0))])
    assert sources.public_ip("example.org", 443) == "93.184.215.14"
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("127.0.0.1", 0))])
    with pytest.raises(ValueError):
        sources.public_ip("localhost.example", 443)


def test_gdelt_pace_respects_one_request_per_five_seconds():
    assert open_sources.PACE["gdelt"].interval == 5.5


def test_gdelt_response_is_cached_on_disk_for_six_hours(monkeypatch):
    calls = []

    def fake_get(_kind, url):
        calls.append(url)
        return '{"articles":[{"url":"https://example.org/a","title":"AI security","seendate":"20260926093000"}]}'

    monkeypatch.setattr(open_sources, "_get", fake_get)
    first = gdelt_news("AI security", date(2026, 9, 26))
    second = gdelt_news("AI security", date(2026, 9, 26))
    assert first == second and len(calls) == 1


def test_gdelt_cache_expires_after_six_hours(monkeypatch):
    from app.radar import research_apis

    calls = []

    def fake_get(_kind, url):
        calls.append(url)
        return '{"articles":[]}'

    monkeypatch.setattr(open_sources, "_get", fake_get)
    gdelt_news("AI security", date(2026, 9, 26))
    cached = next(research_apis.CACHE_DIR.glob("gdelt/*.json"))
    old = cached.stat().st_mtime - research_apis.CACHE_TTL - 1
    os.utime(cached, (old, old))
    gdelt_news("AI security", date(2026, 9, 26))
    assert len(calls) == 2


def test_gdelt_throttled_response_is_not_cached(monkeypatch):
    from app.radar import research_apis

    monkeypatch.setattr(open_sources, "_get", lambda *_args: "<html>Too many requests</html>")
    with pytest.raises(RuntimeError, match="non-JSON"):
        gdelt_news("AI security", date(2026, 9, 26))
    assert not list(research_apis.CACHE_DIR.glob("gdelt/*.json"))


def test_bing_zh_does_not_send_cn_locale_parameters(monkeypatch):
    requested = []
    monkeypatch.setattr(open_sources, "_get", lambda _kind, url: requested.append(url) or "")
    open_sources.bing_news("人工智能安全", "zh")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requested[0]).query)
    assert query == {"q": ["人工智能安全"], "format": ["rss"]}


def test_bing_other_locales_keep_their_market_parameters(monkeypatch):
    requested = []
    monkeypatch.setattr(open_sources, "_get", lambda _kind, url: requested.append(url) or "")
    open_sources.bing_news("защита ИИ", "ru")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requested[0]).query)
    assert query["setlang"] == ["ru-RU"] and query["cc"] == ["RU"]


def test_crossref_sends_contact_from_environment(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "team@example.com")
    request = {}

    def fake_get(_kind, url):
        request["url"] = url
        return '{"message":{"items":[]}}'

    monkeypatch.setattr(open_sources, "_get", fake_get)
    open_sources.crossref("fintech payments", date(2026, 9, 28))
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request["url"]).query)["mailto"] == ["team@example.com"]


def test_crossref_omits_mailto_when_no_contact_is_configured(monkeypatch):
    request = {}

    def fake_get(_kind, url):
        request["url"] = url
        return '{"message":{"items":[]}}'

    monkeypatch.setattr(open_sources, "_get", fake_get)
    open_sources.crossref("fintech payments", date(2026, 9, 28))
    assert "mailto" not in request["url"]


def test_habr_takes_observed_date_from_rss_pubdate(monkeypatch):
    rss = ("<rss><channel>"
           "<item><title>Статья</title><link>https://habr.com/ru/articles/42/?utm=1</link>"
           "<pubDate>Sat, 17 Oct 2026 10:30:00 +0300</pubDate></item>"
           "<item><title>Без даты</title><link>https://habr.com/ru/articles/43/</link></item>"
           "</channel></rss>")
    monkeypatch.setattr(open_sources, "_get", lambda *_args: rss)
    refs = open_sources.habr("AI security")
    assert refs[0] == {"url": "https://habr.com/ru/articles/42/", "title": "Статья", "observed_at": "2026-10-17"}
    assert refs[1]["observed_at"] is None


def test_historical_run_is_marked_and_explains_limits():
    from app.radar.config import RadarSettings
    from app.radar.models import SearchRequest
    from app.radar.pipeline import new_run

    past = new_run(SearchRequest(query="AI security", as_of=date(2021, 12, 31)), RadarSettings())
    assert past["mode"] == "historical"
    assert "31.12.2021" in past["warnings"][0] and "Bing News" in past["warnings"][0]
    live = new_run(SearchRequest(query="AI security"), RadarSettings())
    assert live["mode"] == "live_search" and live["warnings"] == []
