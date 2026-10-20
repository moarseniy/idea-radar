import urllib.parse
from datetime import date

import pytest
from connector_testkit import assert_connector_contract, fake_json_request, fixture

from app.radar.connectors import connectors, registry
from app.radar.connectors import patents as mod

SNAPSHOT = date(2021, 12, 31)
CONNECTOR = mod.CONNECTOR


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    monkeypatch.delenv("EPO_OPS_KEY", raising=False)
    monkeypatch.delenv("EPO_OPS_SECRET", raising=False)
    monkeypatch.setattr(mod, "_TOKEN", {"value": "", "expires": 0.0})


def _inner(call: dict) -> dict:
    return dict(urllib.parse.parse_qsl(call["params"]["url"]))


def test_registered_off_by_default():
    assert registry()[CONNECTOR.name] is CONNECTOR
    assert CONNECTOR.default_on is False and CONNECTOR.date_support == "strict"
    assert CONNECTOR.channel == "science"
    assert CONNECTOR.name not in {c.name for c in connectors(["agent security"])}


def test_google_contract_and_query(monkeypatch):
    calls = fake_json_request(monkeypatch, mod, fixture("patents_google"))
    refs = CONNECTOR.run(["AI agent security posture management"], SNAPSHOT, 5)
    assert_connector_contract(CONNECTOR, refs, SNAPSHOT)
    assert calls[0]["provider"] == "google_patents"
    inner = _inner(calls[0])
    assert inner["before"] == "publication:20211231"          # as_of — конец диапазона
    assert inner["after"] == "priority:20120101"
    assert inner["q"].count('("') >= 1 and "management" not in inner["q"]  # короткие фразы
    assert "Mozilla" in calls[0]["headers"]["User-Agent"]
    ref = refs[0]
    assert ref["source_provider"] == "google_patents_xhr"
    assert ref["observed_at"] == ref["priority_date"] <= ref["publication_date"]
    assert ref["url"] == f"https://patents.google.com/patent/{ref['patent_number']}/en"


def test_irrelevant_patents_are_dropped(monkeypatch):
    fake_json_request(monkeypatch, mod, fixture("patents_google"))
    refs = mod.search(["agent security"], SNAPSHOT, 10)
    numbers = {r["patent_number"] for r in refs}
    assert "US10938837B2" in numbers          # в реферате «security agent»
    assert "US10841420B2" not in numbers      # «Virtual communications assessment…» (B3)
    fake_json_request(monkeypatch, mod, fixture("patents_google"))
    assert mod.search(["motorcycle carrier"], SNAPSHOT, 10) == []


def test_publication_after_as_of_is_dropped(monkeypatch):
    data = fixture("patents_google")
    for item in data["results"]["cluster"][0]["result"]:
        item["patent"]["publication_date"] = "2022-02-01"
    fake_json_request(monkeypatch, mod, data)
    assert mod._search_google(["agent security"], SNAPSHOT, 5) == []


def test_empty_response_gives_nothing(monkeypatch):
    fake_json_request(monkeypatch, mod, {"results": {"total_num_results": 0, "cluster": []}})
    monkeypatch.setattr(mod, "_search_fpo", lambda *_: [])
    assert CONNECTOR.run(["agent security"], SNAPSHOT, 5) == []


def test_falls_back_to_fpo_when_google_fails(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("HTTP 429")
    monkeypatch.setattr(mod, "_json_request", broken)
    monkeypatch.setattr(mod, "_search_fpo", lambda *_: [{"url": "https://fpo/x", "title": "Agent security"}])
    assert mod.search(["agent security"], SNAPSHOT, 5)[0]["url"] == "https://fpo/x"
    monkeypatch.setattr(mod, "_search_fpo", broken)
    with pytest.raises(RuntimeError, match="недоступны"):
        mod.search(["agent security"], SNAPSHOT, 5)


def test_cyrillic_only_terms_do_not_query(monkeypatch):
    calls = fake_json_request(monkeypatch, mod, fixture("patents_google"))
    monkeypatch.setattr(mod, "_search_fpo", lambda *_: [])
    assert mod.search(["защита агентов"], SNAPSHOT, 5) == []
    assert calls == []


# ---- EPO OPS: фикстуры в формате живых ответов

def test_ops_branch_used_only_with_keys_and_parses_biblio(monkeypatch):
    monkeypatch.setenv("EPO_OPS_KEY", "k")
    monkeypatch.setenv("EPO_OPS_SECRET", "s")
    monkeypatch.setattr(mod, "_ops_token", lambda *_: "TOKEN")
    calls = fake_json_request(monkeypatch, mod, fixture("patents_epo_search"), fixture("patents_epo_biblio"))
    refs = CONNECTOR.run(["agent security"], SNAPSHOT, 5)
    assert_connector_contract(CONNECTOR, refs, SNAPSHOT)
    assert [r["patent_number"] for r in refs] == ["EP3975021B1"]  # «Heat storage device» отсеян
    assert refs[0]["observed_at"] == "2020-09-29" and refs[0]["publication_date"] == "2021-09-15"
    assert refs[0]["assignee"] == "Dynatrace LLC" and refs[0]["source_provider"] == "epo_ops"
    cql = calls[0]["params"]["q"]
    assert 'ta="agent security"' in cql and cql.endswith('pd within "20120101 20211231"')
    assert calls[0]["headers"]["Authorization"] == "Bearer TOKEN"
    assert calls[1]["url"].endswith("/published-data/publication/docdb/EP.3975021.B1,US.11882128.B2/biblio")


def test_ops_failure_falls_through_to_google(monkeypatch):
    monkeypatch.setenv("EPO_OPS_KEY", "k")
    monkeypatch.setenv("EPO_OPS_SECRET", "s")
    monkeypatch.setattr(mod, "_ops_token", lambda *_: (_ for _ in ()).throw(RuntimeError("403")))
    calls = fake_json_request(monkeypatch, mod, fixture("patents_google"))
    assert mod.search(["agent security"], SNAPSHOT, 5)
    assert calls[0]["provider"] == "google_patents"


def test_ops_token_is_requested_with_basic_auth(monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self): ...
        def json(self): return fixture("patents_epo_token")

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return Resp()
    monkeypatch.setattr(mod.httpx, "post", post)
    monkeypatch.setattr(mod.limiter("epo_ops"), "wait", lambda: None)
    assert mod._ops_token("key", "secret") == "TEST_TOKEN_NOT_REAL"
    assert seen["auth"] == ("key", "secret") and seen["data"] == {"grant_type": "client_credentials"}
    assert mod._ops_token("key", "secret") == "TEST_TOKEN_NOT_REAL"  # из кэша


# ---- patent_counts

def _window_response(total, items):
    return {"results": {"total_num_results": total, "cluster": [{"result": [{"patent": p} for p in items]}]}}


def test_patent_counts_google_fills_patents_structure(monkeypatch):
    good = [{"publication_number": f"US{i}B2", "title": "Agent security monitor",
             "snippet": "security agent", "assignee": "Acme" if i < 2 else "Beta",
             "priority_date": "2020-01-01", "publication_date": "2021-05-01"} for i in range(4)]
    noise = [{"publication_number": "US9B2", "title": "Heat storage device", "snippet": "", "assignee": "X"}]
    # половина выборки нерелевантна: счётчик окна уменьшается вдвое
    calls = fake_json_request(monkeypatch, mod, _window_response(100, good[:2] + noise * 2),
                              _window_response(40, good), _window_response(10, good), _window_response(60, good),
                              _window_response(8, good), _window_response(2, good))
    result = mod.patent_counts(["agent security"], SNAPSHOT)
    assert result.ok and result.source == "google_patents_xhr"
    assert result.windows == [50, 40, 10] and result.count_3y == 60
    assert result.older_6y == 8 and result.older_10y == 2
    assert result.first_priority and result.titles == ["Agent security monitor"] * 4
    assert result.assignee_counts_3y == {"Acme": 2, "Beta": 2}
    kinds = [_inner(c)["before"].split(":")[0] for c in calls]
    assert kinds == ["priority", "priority", "priority", "publication", "priority", "priority"]
    assert _inner(calls[3])["before"] == "publication:20211231"


def test_patent_counts_failure_is_not_exception(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("down")
    monkeypatch.setattr(mod, "_json_request", broken)
    import app.ml.features.evidence as ev
    monkeypatch.setattr(ev, "_patents_from_fpo", lambda *_: ev.Patents(warning="FPO 403"))
    result = mod.patent_counts(["agent security"], SNAPSHOT)
    assert not result.ok and result.source == "unavailable" and "down" in result.warning


def test_patent_counts_ops_branch(monkeypatch):
    monkeypatch.setenv("EPO_OPS_KEY", "k")
    monkeypatch.setenv("EPO_OPS_SECRET", "s")
    monkeypatch.setattr(mod, "_ops_token", lambda *_: "T")
    search_page = fixture("patents_epo_search")
    fake_json_request(monkeypatch, mod, *[search_page] * 6, search_page, fixture("patents_epo_biblio"))
    result = mod.patent_counts(["agent security"], SNAPSHOT)
    assert result.source == "epo_ops" and result.ok
    assert result.count_3y == 37 and result.titles == ["Agent based security monitoring of application data flow"]


def test_ops_cql_searches_phrases_without_hyphens():
    cql = mod._ops_cql(["retrieval-augmented", "prompt injection"], None, SNAPSHOT)
    assert cql.startswith('(ta="retrieval augmented" or ta="prompt injection") and pd within "19000101 ')


def test_ops_empty_search_is_zero_not_failure(monkeypatch):
    monkeypatch.setenv("EPO_OPS_KEY", "k")
    monkeypatch.setenv("EPO_OPS_SECRET", "s")
    monkeypatch.setattr(mod, "_ops_token", lambda *_: "TOKEN")
    calls = fake_json_request(monkeypatch, mod, {})  # _json_request(allow_404=True) на 404 отдаёт {}
    assert mod._search_ops(["agent security"], SNAPSHOT, 5) == []
    assert calls[0]["allow_404"] is True
