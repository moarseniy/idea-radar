import base64
import json
from datetime import date

import httpx
import pytest

from app.search.base import SearchQuery, iso_date
from app.search.config import SearchSettings, load_search_settings
from app.search.providers import REGISTRY
from app.search.service import SearchService

Q = SearchQuery("in-sensor computing", as_of=date(2026, 9, 22), language="en", limit=5)
KEYS = {"searxng_url": "http://searxng", "tavily": "t", "exa": "e", "brave": "b", "yandex": "y",
        "yandex_folder": "f", "openserp_url": "http://openserp", "openserp_engines": "bing",
        "openrouter": "o", "openrouter_model": "openai/gpt-4.1"}

YANDEX_XML = """<?xml version="1.0" encoding="utf-8"?><yandexsearch><response><results><grouping>
<group><doc><url>https://example.ru/a</url><title>Датчики с <hlword>вычислениями</hlword></title>
<modtime>20260310T120000</modtime><passages><passage>Пилот в банке</passage></passages></doc></group>
</grouping></results></response></yandexsearch>"""

PAYLOADS = {
    "searxng": {"results": [{"url": "https://a.org/1", "title": "A", "content": "snip",
                             "publishedDate": "2025-11-15T00:00:00", "engine": "brave"}]},
    "tavily": {"results": [{"url": "https://a.org/1", "title": "A", "content": "snip", "published_date": "2025-11-15"}]},
    "exa": {"results": [{"url": "https://a.org/1", "title": "A", "publishedDate": "2025-11-15T00:00:00.000Z",
                         "text": "Full page text " * 40}]},
    "brave": {"web": {"results": [{"url": "https://a.org/1", "title": "A", "description": "snip",
                                   "page_age": "2025-11-15T00:00:00"}]}},
    "yandex": {"rawData": base64.b64encode(YANDEX_XML.encode()).decode()},
    # живой OpenSERP отдаёт объект с results (проверено на контейнере karust/openserp)
    "openserp": {"query": {}, "results": [
        {"url": "https://a.org/1", "title": "A", "snippet": "snip", "engine": "bing", "type": "organic"},
        {"url": "https://ads.org", "title": "Ad", "type": "ad"}]},
    "openrouter": {"choices": [{"message": {"content": "text", "annotations": [
        {"type": "url_citation", "url_citation": {"url": "https://a.org/1", "title": "A"}}]}}]},
}


def settings(providers=("searxng",), mode="fallback", tmp_path=None, cache_hours=0):
    return SearchSettings(providers=tuple(providers), mode=mode, cache_hours=cache_hours,
                          cache_dir=tmp_path or SearchSettings().cache_dir, keys=dict(KEYS))


def client_for(payloads: dict, seen: list | None = None, status: dict | None = None) -> httpx.Client:
    hosts = {"searxng": "searxng", "api.tavily.com": "tavily", "api.exa.ai": "exa", "api.search.brave.com": "brave",
             "searchapi.api.cloud.yandex.net": "yandex", "openserp": "openserp", "openrouter.ai": "openrouter"}

    def handler(request: httpx.Request) -> httpx.Response:
        name = hosts[request.url.host]
        if seen is not None:
            seen.append((name, request))
        code = (status or {}).get(name, 200)
        return httpx.Response(code, json=payloads.get(name, {}) if code == 200 else {"error": "x"})
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("name", list(PAYLOADS))
def test_every_provider_returns_the_common_format(name):
    s = settings([name])
    provider = REGISTRY[name](s, client_for(PAYLOADS))
    out = provider.search(Q if name != "yandex" else SearchQuery("датчики", as_of=Q.as_of, language="ru"))
    assert out and {"url", "title", "snippet", "published_at", "provider", "content"} <= out[0].keys()
    assert out[0]["provider"] == name and out[0]["url"].startswith("https://")
    if name == "yandex":
        assert out[0]["title"] == "Датчики с вычислениями" and out[0]["published_at"] == "2026-03-10"
    if name == "exa":
        assert out[0]["content"].startswith("Full page text")  # текст страницы — радару как документ
    if name == "openserp":
        assert [r["url"] for r in out] == ["https://a.org/1"]  # реклама отброшена


def test_providers_pass_date_window_and_keys():
    seen = []
    for name in ("tavily", "exa", "brave"):
        REGISTRY[name](settings([name]), client_for(PAYLOADS, seen)).search(Q)
    by = {name: req for name, req in seen}
    assert json.loads(by["tavily"].content)["end_date"] == "2026-09-22"
    assert by["tavily"].headers["authorization"] == "Bearer t"
    assert json.loads(by["exa"].content)["endPublishedDate"].startswith("2026-09-22")
    assert by["exa"].headers["x-api-key"] == "e"
    assert "to2026-09-22" in by["brave"].url.params["freshness"]


def test_fallback_skips_unconfigured_and_failed_providers(tmp_path):
    s = settings(["tavily", "brave", "searxng"], tmp_path=tmp_path)
    s.keys["tavily"] = ""                                    # нет ключа
    service = SearchService(s, client_for(PAYLOADS, status={"brave": 503}))
    out = service.search(Q)
    assert [p["status"] for p in out["providers"]] == ["not_configured", "error", "ok"]
    assert out["results"][0]["provider"] == "searxng"


def test_fanout_merges_without_duplicate_urls(tmp_path):
    payloads = {**PAYLOADS, "brave": {"web": {"results": [{"url": "https://www.a.org/1/", "title": "dup"},
                                                         {"url": "https://b.org/2", "title": "B"}]}}}
    service = SearchService(settings(["searxng", "brave"], mode="fanout", tmp_path=tmp_path), client_for(payloads))
    urls = [r["url"] for r in service.search(Q)["results"]]
    assert urls == ["https://a.org/1", "https://b.org/2"]


def test_results_after_as_of_are_dropped(tmp_path):
    payloads = {"searxng": {"results": [{"url": "https://a.org/new", "title": "new", "publishedDate": "2026-12-01"},
                                        {"url": "https://a.org/old", "title": "old", "publishedDate": "2026-01-01"},
                                        {"url": "https://a.org/nodate", "title": "?"}]}}
    out = SearchService(settings(tmp_path=tmp_path), client_for(payloads)).search(Q)
    assert [r["url"] for r in out["results"]] == ["https://a.org/old", "https://a.org/nodate"]


def test_cache_saves_paid_requests(tmp_path):
    seen = []
    service = SearchService(settings(["tavily"], tmp_path=tmp_path, cache_hours=24), client_for(PAYLOADS, seen))
    service.search(Q)
    again = service.search(Q)
    assert len(seen) == 1 and again["providers"][0]["cached"] is True


def test_default_provider_is_searxng_and_unknown_is_rejected(monkeypatch):
    monkeypatch.setattr("app.search.config.dotenv_values", lambda _: {})
    monkeypatch.delenv("SEARCH_PROVIDERS", raising=False)
    assert load_search_settings().providers == ("searxng",)
    monkeypatch.setenv("SEARCH_PROVIDERS", "google")
    with pytest.raises(ValueError):
        load_search_settings()


def test_iso_date_formats():
    assert iso_date("2025-11-15T00:00:00.000Z") == "2025-11-15"
    assert iso_date("20260310T120000") == "2026-03-10"
    assert iso_date("Tue, 10 Mar 2026 12:00:00 GMT") == "2026-03-10"
    assert iso_date("вчера") is None


def test_radar_task_splits_languages_and_passes_documents(monkeypatch):
    from app.radar import web_search as ws
    calls = []

    def fake(query):
        calls.append((query.language, query.text))
        return {"results": [{"url": f"https://{query.language}.org/x", "title": "T", "content": "c" * 400,
                             "provider": "exa", "published_at": "2026-01-01", "language": query.language}],
                "providers": [{"name": "exa", "count": 1, "error": None}]}
    monkeypatch.setattr(ws, "web_search", fake)
    out = ws.search_task({"branch": "b", "query": "Русский: токенизация активов\nEnglish: asset tokenization"},
                         "2026-09-22")
    assert calls == [("ru", "токенизация активов"), ("en", "asset tokenization")]
    assert out["connector"] == "Веб-поиск (exa)" and out["references"][0]["document"]["extraction"] == "exa_content"


def test_search_service_http_api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.search import server
    monkeypatch.setattr(server, "_service", SearchService(settings(tmp_path=tmp_path), client_for(PAYLOADS)))
    api = TestClient(server.app)
    assert api.get("/health").json()["providers"][0]["name"] == "searxng"
    body = api.post("/search", json={"text": "in-sensor computing", "as_of": "2026-09-22"}).json()
    assert body["results"][0]["url"] == "https://a.org/1"


def test_search_service_token(monkeypatch, tmp_path):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from app.search import server
    s = replace(settings(tmp_path=tmp_path), service_token="secret")
    monkeypatch.setattr(server, "_service", SearchService(s, client_for(PAYLOADS)))
    api = TestClient(server.app)
    assert api.post("/search", json={"text": "query"}).status_code == 401
    assert api.post("/search", json={"text": "query"}, headers={"X-Search-Token": "secret"}).status_code == 200


def test_task_queries_keep_three_languages_but_all_candidate_queries():
    from app.radar.web_search import task_queries
    branch = {"branch": "b", "query": "", "queries": [{"language": code, "query": f"q {code}"}
                                                      for code in ("zh", "en", "de", "ru", "ja", "ar")]}
    assert [q["language"] for q in task_queries(branch, 3)] == ["ru", "en", "zh"]
    verify = {"branch": "v", "query": "\n".join(f"candidate {i} pilot deployment" for i in range(6))}
    assert len(task_queries(verify, 3)) == 6          # один язык — все проверочные запросы


def test_self_hosted_provider_requests_are_spaced(monkeypatch, tmp_path):
    from dataclasses import replace
    s = replace(settings(tmp_path=tmp_path), min_interval=0.2)
    service = SearchService(s, client_for(PAYLOADS))
    slept = []
    monkeypatch.setattr("app.search.service.time.sleep", lambda x: slept.append(x))
    for i in range(3):
        service.search(SearchQuery(f"query {i}", as_of=Q.as_of))
    assert len([x for x in slept if x > 0.1]) >= 2  # второй и третий запрос ждут своей очереди


def test_searxng_uses_configured_engines():
    seen = []
    REGISTRY["searxng"](settings(), client_for(PAYLOADS, seen)).search(Q)
    s = settings()
    s.keys["searxng_engines"] = "yahoo,bing news"
    REGISTRY["searxng"](s, client_for(PAYLOADS, seen)).search(Q)
    assert "categories" in seen[0][1].url.params and "engines" not in seen[0][1].url.params
    assert seen[1][1].url.params["engines"] == "yahoo,bing news"
