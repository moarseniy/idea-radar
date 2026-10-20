from datetime import date

import pytest

from app.radar import connectors as reg
from app.radar import research_apis as api
from app.radar.connectors import Connector, connectors, describe, registry
from app.radar.connectors.events import make_event, to_usd_m

SNAPSHOT = date(2021, 12, 31)


def _ref(url: str, observed_at: str | None) -> dict:
    return {"url": url, "title": url, "observed_at": observed_at}


def test_registry_lists_existing_structured_sources_in_call_order():
    assert list(registry())[:8] == ["OpenAlex", "NIH RePORTER", "CORDIS grants", "GitHub repositories",
                                    "Hugging Face", "TED procurement", "ClinicalTrials.gov", "FDA openFDA"]
    assert {c.channel for c in registry().values()} <= set(reg.CHANNELS)


def test_connector_rejects_unknown_channel_and_date_support():
    with pytest.raises(ValueError, match="канал"):
        Connector("X", "news", "strict", lambda *_: [])
    with pytest.raises(ValueError, match="даты"):
        Connector("X", "money", "sometimes", lambda *_: [])


def test_run_tags_channel_and_drops_records_after_as_of():
    connector = Connector("Grants", "money", "post", lambda *_: [
        _ref("https://a", "2021-06-01"), _ref("https://b", "2022-03-01"), _ref("https://c", None)])
    refs = connector.run(["x"], SNAPSHOT, 5)
    assert [r["url"] for r in refs] == ["https://a", "https://c"]
    assert {r["channel"] for r in refs} == {"money"}
    assert {r["date_support"] for r in refs} == {"post"}


def test_historical_mode_skips_sources_without_date_filter(monkeypatch):
    monkeypatch.setenv("OPENALEX_ENABLED", "1")
    live = {c.name for c in connectors(["edge AI"])}
    past = {c.name for c in connectors(["edge AI"], historical=True)}
    assert {"GitHub repositories", "Hugging Face"} <= live
    assert not past & {"GitHub repositories", "Hugging Face"}
    assert {"OpenAlex", "NIH RePORTER", "TED procurement"} <= past


def test_health_connectors_only_for_health_topics():
    assert "ClinicalTrials.gov" not in {c.name for c in connectors(["edge computing"])}
    assert "ClinicalTrials.gov" in {c.name for c in connectors(["insulin patch"])}


def test_describe_covers_registry_and_query_sources():
    assert describe("NIH RePORTER") == ("money", "strict")
    assert describe("Bing News (Russian)") == ("market", "none")
    assert describe("Patents (Google Patents web fallback)") == ("science", "strict")
    assert describe("Неизвестный") == (None, None)


def test_structured_sources_use_registry_and_isolate_failures(monkeypatch):
    def broken(*_):
        raise RuntimeError("down")
    monkeypatch.setattr(reg, "_REGISTRY", {
        "A": Connector("A", "money", "strict", lambda *_: [_ref("https://a", "2021-01-01")]),
        "B": Connector("B", "science", "strict", broken),
        "OpenAlex": Connector("OpenAlex", "science", "strict", lambda *_: [_ref("https://oa", "2021-01-01")]),
    })
    refs = api.search_structured_sources(["x"], SNAPSHOT, include_openalex=False)
    assert [r["url"] for r in refs] == ["https://a"]
    assert refs[0]["channel"] == "money"


def test_make_event_maps_kind_to_feature_type_and_converts_currency():
    event = make_event("contract", "2021-05-04", "CONT_1", amount=2_500_000, company="Acme",
                       customer="DoD")
    assert event["type"] == "procurement" and event["kind"] == "contract"
    assert event["amount_usd_m"] == 2.5
    assert make_event("funding_round", date(2021, 1, 2), "r1", round_name="A")["type"] == "funding"
    assert make_event("grant_call", "2021-01-02", "c1")["type"] == "grant_call"
    assert to_usd_m(850_000, "EUR", 2021) == 1.0
    assert to_usd_m(1, "XYZ", 2021) is None
    with pytest.raises(ValueError):
        make_event("contract", None, "x")
    with pytest.raises(ValueError):
        make_event("donation", "2021-01-01", "x")


def test_testkit_example_with_recorded_nih_response(monkeypatch):
    # Образец для новых коннекторов: записанный ответ, контракт ссылки, проверка запроса.
    from connector_testkit import assert_connector_contract, fake_json_request, fixture

    calls = fake_json_request(monkeypatch, api, fixture("nih_reporter"))
    connector = registry()["NIH RePORTER"]
    refs = connector.run(["sweat sensor"], SNAPSHOT, 5)
    assert_connector_contract(connector, refs, SNAPSHOT)
    assert calls[0]["body"]["criteria"]["award_notice_date"]["to_date"] == "2021-12-31"
    assert refs[0]["structured_event"]["amount_usd_m"] == 0.412


def test_open_source_tasks_tag_channel_for_quota_accounting():
    from app.radar.open_sources import run_task

    result = run_task("Bing News (Russian)", lambda: [{"url": "https://vc.ru/x", "title": "x"}],
                      "Тема", "запрос", "ru")
    assert result["references"][0]["channel"] == "market"
    assert result["references"][0]["date_support"] == "none"


def test_short_phrases_drop_generic_words_and_cyrillic():
    from app.radar.connectors._terms import short_phrases

    assert short_phrases(["agent security posture management", "защита агентов", "AI agent security"]) == [
        "agent security posture", "agent security", "security posture"]
    assert short_phrases(["automated red teaming"]) == ["automated red teaming", "automated red", "red teaming"]
    assert short_phrases(["solid-state battery technology"]) == ["solid-state battery"]


def test_relevance_check_rejects_unrelated_patent_titles():
    from app.radar.connectors._terms import relevant

    terms = ["AI agent security"]
    assert relevant("Securing autonomous AI agents with runtime monitors", terms)
    assert relevant("Solid-state batteries for EVs", ["solid-state battery"])
    assert not relevant("Motorcycle carrier attachment", terms)
    assert not relevant("Heat storage device", terms)
    assert relevant("Что надо знать о квантовых вычислениях", ["квантовые вычисления"])
    assert not relevant("Сколько стоит аренда офиса", ["квантовые вычисления"])


def test_new_connectors_stay_off_until_enabled(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {
        "Old": Connector("Old", "money", "strict", lambda *_: []),
        "New": Connector("New", "money", "strict", lambda *_: [], default_on=False),
    })
    monkeypatch.delenv("RADAR_EXTRA_CONNECTORS", raising=False)
    assert [c.name for c in connectors()] == ["Old"]
    monkeypatch.setenv("RADAR_EXTRA_CONNECTORS", "New")
    assert [c.name for c in connectors()] == ["Old", "New"]
    monkeypatch.setenv("RADAR_EXTRA_CONNECTORS", "all")
    monkeypatch.setenv("RADAR_DISABLED_CONNECTORS", "Old")
    assert [c.name for c in connectors()] == ["New"]


def test_reserve_channel_reads_money_on_top_of_the_budget():
    from app.radar.pipeline import reserve_channel

    refs = [{"url": f"m{i}", "channel": "market"} for i in range(8)]
    refs[1]["channel"] = refs[6]["channel"] = refs[7]["channel"] = "money"
    # Рынку — все 4 места бюджета, деньгам — 2 сверх них; порядок прежний.
    assert [r["url"] for r in reserve_channel(refs, 4, "money", 0.5)] == ["m0", "m1", "m2", "m3", "m4", "m6"]
    # Денег нет — читаются те же 4 ссылки, что и без резерва.
    plain = [{"url": f"s{i}", "channel": "science"} for i in range(5)]
    assert reserve_channel(plain, 4, "money", 0.5) == plain[:4]
    # Остальных ссылок мало: свободные места бюджета тоже уходят деньгам.
    rich = [{"url": f"x{i}", "channel": "money"} for i in range(6)] + [{"url": "n", "channel": "market"}]
    assert [r["url"] for r in reserve_channel(rich, 4, "money", 0.25)] == ["x0", "x1", "x2", "x3", "n"]


def test_channel_mix_counts_read_unique_sources_and_web_hosts():
    from app.radar.connectors import channel_for_url
    from app.radar.pipeline import channel_mix

    assert channel_for_url("https://techcrunch.com/2021/x") == "money"
    assert channel_for_url("https://www.nature.com/articles/x") == "science"
    assert channel_for_url("https://rbc.ru/x") == "market"
    assert channel_mix([{"status": "read", "channel": "money"}, {"status": "read", "channel": None},
                        {"status": "failed", "channel": "money"},
                        {"status": "read", "channel": "money", "duplicate_of": "a"}]) == {"money": 1, "other": 1}


def test_model_sees_only_trained_connectors_but_card_gets_all_money(monkeypatch):
    from app.ml.features import extract

    refs = [{"url": "https://a", "connector": "NIH RePORTER", "channel": "money", "title": "old grant",
             "structured_event": {"type": "grant", "kind": "grant", "date": "2021-01-01"}},
            {"url": "https://b", "connector": "USAspending grants", "channel": "money", "title": "new grant",
             "structured_event": {"type": "grant", "kind": "grant", "date": "2021-02-01",
                                  "amount_usd_m": 1.5}}]
    seen = {}

    def fake_search(*_args, only=None, skip=frozenset(), **_kwargs):
        return [r for r in refs if (only is None or r["connector"] in only) and r["connector"] not in skip]
    monkeypatch.setattr("app.radar.research_apis.search_structured_sources", fake_search)
    monkeypatch.setattr(extract.ev, "science", lambda *a, **k: extract.ev.Science())
    monkeypatch.setattr(extract.ev, "media", lambda *a, **k: extract.ev.Media())
    monkeypatch.setattr(extract, "extract_events", lambda *a, **k: None)
    monkeypatch.setattr(extract, "compute", lambda *a, **k: seen.update(refs=a[-1]) or {})
    monkeypatch.setattr(extract, "llm_json", lambda _prompt: {"keep": [1, 2]})
    row = extract.featurize("X", "Финтех", SNAPSHOT, query={"term": "x", "aliases": []},
                            requested_features=["funding_log_24m"],
                            feature_connectors=frozenset({"NIH RePORTER"}), with_money_events=True)
    assert [r["url"] for r in seen["refs"]] == ["https://a"]
    assert [e["title"] for e in row["money_events"]] == ["new grant", "old grant"]
    assert row["money_events"][0]["amount_usd_m"] == 1.5


def _money_ref(url: str, title: str, text: str = "") -> dict:
    return {"url": url, "title": title, "channel": "money", "connector": "NIH RePORTER",
            "document": {"text": f"Federal research grant. Award/project: {title}.\n\n{text}"},
            "structured_event": {"kind": "grant", "date": "2021-01-01"}}


def test_card_money_events_keep_only_what_llm_confirms(monkeypatch):
    from app.ml.features import extract

    prompts = []
    monkeypatch.setattr(extract, "llm_json", lambda prompt: prompts.append(prompt) or {"keep": [2]})
    refs = [_money_ref("https://a", "Single-cell transcriptomics", "immune variation"),
            _money_ref("https://b", "Securing agentic AI in banking", "agent authorization")]
    events = extract.money_events(refs, SNAPSHOT, name="Авторизация агентов", domain="Банки",
                                  terms=["agent authorization"])
    assert [e["url"] for e in events] == ["https://b"]
    # В промпт идёт аннотация без служебной шапки коннектора.
    assert "2. [grant] Securing agentic AI in banking — agent authorization" in prompts[0]


def test_card_money_events_fall_back_to_strict_title_match(monkeypatch):
    from app.ml.features import extract

    def broken(_prompt):
        raise RuntimeError("LLM недоступна")
    monkeypatch.setattr(extract, "llm_json", broken)
    refs = [_money_ref("https://a", "Testing of military aircraft", "adversarial testing of agents"),
            _money_ref("https://b", "Adversarial testing for banking agents")]
    events = extract.money_events(refs, SNAPSHOT, terms=["adversarial testing"])
    assert [e["url"] for e in events] == ["https://b"]


def test_slow_card_sources_are_dropped_at_deadline(monkeypatch):
    import time as _time

    monkeypatch.setattr(reg, "_REGISTRY", {
        "Fast": Connector("Fast", "money", "strict", lambda *_: [_ref("https://fast", "2021-01-01")]),
        "Slow": Connector("Slow", "money", "strict",
                          lambda *_: _time.sleep(1.5) or [_ref("https://slow", "2021-01-01")]),
    })
    started = _time.monotonic()
    refs = api.search_structured_sources(["x"], SNAPSHOT, deadline=0.5)
    assert [r["url"] for r in refs] == ["https://fast"]
    assert _time.monotonic() - started < 1.2


def test_run_money_sources_reach_the_card_and_duplicates_merge(monkeypatch):
    from app.ml.features import extract
    from app.radar.model_scoring import money_refs

    sources = [{"url": "https://techcrunch.com/reco", "title": "Reco raises $55M for AI agent security",
                "channel": "money", "status": "read", "published_at": "2026-05-01", "publisher": "TechCrunch",
                "text": "Reco secures AI agents"},
               {"url": "https://nature.com/x", "title": "Paper", "channel": "science", "status": "read",
                "published_at": "2026-05-01"},
               {"url": "https://nsf.gov/1", "title": "Collaborative Research: Securing Agentic AI in Banks",
                "channel": "money", "status": "read", "connector": "NSF awards", "structured_event": {"kind": "grant", "date": "2026-03-01"}}]
    refs = money_refs(sources)
    assert [r["structured_event"]["kind"] for r in refs] == ["news", "grant"]
    usaspending = {**refs[1], "url": "https://usaspending.gov/1",
                   "title": "COLLABORATIVE RESEARCH: SECURING AGENTIC AI IN BANKS -THIS AWARD FUNDS"}
    monkeypatch.setattr(extract, "llm_json", lambda _prompt: {"keep": [1, 2]})
    events = extract.money_events([*refs, usaspending], date(2026, 10, 18), terms=["agent security"])
    assert [e["url"] for e in events] == ["https://techcrunch.com/reco", "https://nsf.gov/1"]


def test_money_connectors_search_plan_phrases_others_branch_query(monkeypatch):
    from app.radar.open_sources import branch_tasks

    seen: dict[str, list] = {}

    def spy(name):
        return lambda terms, _as_of, limit: seen.setdefault(name, []).append((list(terms), limit)) or [
            _ref(f"https://{name}/{terms[0]}/{i}", "2021-01-01") for i in range(limit)]
    monkeypatch.setattr(reg, "_REGISTRY", {
        "Money": Connector("Money", "money", "strict", spy("Money")),
        "Science": Connector("Science", "science", "strict", spy("Science")),
    })
    tasks = branch_tasks("AI security", "защита ИИ", "AI agent security in banking", SNAPSHOT,
                         [{"language": "en", "query": "AI agent security in banking"}],
                         ["agent identity", "prompt injection"])
    results = {name: action() for name, action, *_ in tasks if name in {"Money", "Science"}}
    # Каждая фраза плана — отдельный вызов по 2 записи, записи чередуются.
    assert seen["Money"] == [(["agent identity"], 2), (["prompt injection"], 2)]
    assert [r["url"] for r in results["Money"]] == [
        "https://Money/agent identity/0", "https://Money/prompt injection/0",
        "https://Money/agent identity/1", "https://Money/prompt injection/1"]
    # Фразы плана не разворачиваются в варианты.
    from app.radar.connectors._terms import ReadyPhrases, short_phrases
    assert short_phrases(ReadyPhrases(["agent identity", "prompt injection"]), limit=2) == [
        "agent identity", "prompt injection"]
    assert short_phrases(["prompt injection detection"]) == ["prompt injection detection",
                                                             "prompt injection", "injection detection"]
    assert seen["Science"] == [(["AI agent security in banking", "AI security"], 4)]


def test_plan_money_phrases_are_cleaned():
    from app.radar.pipeline import money_phrases

    assert money_phrases(["agent identity", "AI", "Prompt injection detection.", "agent identity",
                          "агентская идентичность", "a b c d e"]) == ["agent identity",
                                                                       "Prompt injection detection"]


def test_assessment_gets_money_documents_close_to_the_candidate():
    from app.radar.pipeline import money_for

    batch = [{"title": "Делегированная авторизация агентов",
              "verification_query": "delegated authorization AI agents banking pilot"}]
    sources = [{"id": "m1", "channel": "money", "title": "Keycard raises $38M for AI agent authorization",
                "text": "delegated access for agents", "published_at": "2026-05-01"},
               {"id": "m2", "channel": "money", "title": "Single-cell transcriptomics grant", "text": "immune"},
               {"id": "s1", "channel": "science", "title": "Delegated authorization for AI agents", "text": ""},
               {"id": "m3", "channel": "money", "title": "Agents raise funding", "text": "", "published_at": "x"}]
    assert [s["id"] for s in money_for(batch, sources, set())] == ["m1"]
    assert money_for(batch, sources, {"m1"}) == []


def test_model_connector_list_includes_off_by_default_sources(monkeypatch):
    from app.radar.connectors import connectors

    monkeypatch.delenv("RADAR_EXTRA_CONNECTORS", raising=False)
    monkeypatch.setenv("RADAR_DISABLED_CONNECTORS", "TechCrunch")
    names = {c.name for c in connectors(only=frozenset({"NSF awards", "TechCrunch", "OpenAlex"}))}
    assert names == {"NSF awards", "OpenAlex"}  # NSF выключен по умолчанию, TechCrunch — вручную
    assert "NSF awards" not in {c.name for c in connectors()}
