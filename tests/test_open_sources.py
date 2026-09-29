import hashlib
import subprocess
from datetime import date

import httpx
import pytest

from app.radar import open_sources
from app.radar import sources as source_module
from app.radar.open_sources import _get, _keywords, branch_tasks, gdelt_news, run_task
from app.radar.resilience import CircuitBreaker, CircuitOpenError
from app.radar.sources import SourceFetcher


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
    tasks = branch_tasks("AI security", "защита ИИ", "AI security", date(2026, 9, 22), [
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


def test_openalex_can_be_disabled_without_disabling_other_structured_connectors(monkeypatch):
    monkeypatch.setenv("FEATURE_STRUCTURED_SOURCES", "1")
    monkeypatch.setenv("OPENALEX_ENABLED", "0")
    tasks = branch_tasks("AI security", "защита ИИ", "AI security", date(2026, 9, 22), [
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


def test_patent_search_uses_fpo_as_the_only_patent_source(monkeypatch):
    calls = []
    monkeypatch.setattr("app.ml.features.evidence.search_patents_fpo",
                        lambda *_args, **_kwargs: calls.append("fpo") or [{"url": "https://fpo.example/patent"}])
    assert open_sources._patent_search(["AI safety"], date(2026, 9, 22))[0]["url"] == \
        "https://fpo.example/patent"
    assert calls == ["fpo"]


def test_patent_search_reports_fpo_failure_without_fallback(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("temporary FPO error")

    monkeypatch.setattr("app.ml.features.evidence.search_patents_fpo", fail)
    with pytest.raises(RuntimeError, match="FreePatentsOnline: temporary FPO error"):
        open_sources._patent_search(["AI safety"], date(2026, 9, 22))


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
