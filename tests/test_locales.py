from app.radar.locales import LANGUAGES, clean_search_query, normalize_queries, training_language_batch
from app.radar.models import Branch, LocalizedQuery, LocalizedQueryBatch, SearchPlan
from app.radar.provider import ResearchProvider


def test_live_queries_cover_priority_locales_and_keep_original_prompt_language():
    queries = normalize_queries("защита ИИ", "AI security", [
        {"language": "zh", "query": "人工智能安全"},
        {"language": "th", "query": "ความปลอดภัยของ AI"},
        {"language": "xx-long", "query": "ignored"},
    ])
    by_language = {q["language"]: q for q in queries}
    assert {"ru", "en"}.issubset(by_language)
    assert by_language["zh"]["query"] == "人工智能安全"
    assert by_language["th"]["language_name"] == "TH"
    assert "xx-long" not in by_language


def test_search_query_removes_weak_signal_classifier_language():
    assert clean_search_query("Слабые сигналы в финтехе", "ru") == "финтехе"
    assert clean_search_query("weak signals in fintech") == "fintech"
    assert clean_search_query("weak-signal detection for banking") == "detection for banking"
    assert "слабые сигналы" not in normalize_queries(
        "Слабые сигналы в финтехе", "weak signals in fintech", None
    )[0]["query"].lower()


def test_training_locale_rotation_is_stable_and_covers_priority_language_set():
    domains = ["Индустриальный ИИ", "Роботы", "Инфраструктура ИИ", "Финтех", "Защита ИИ", "Edge"]
    batches = [training_language_batch(domain) for domain in domains]
    assert all(batch == training_language_batch(domain) for batch, domain in zip(batches, domains))
    assert all({"en", "ru"}.issubset(batch) for batch in batches)
    assert {language for batch in batches for language in batch} == {item[0] for item in LANGUAGES}


def test_search_plan_fills_missing_priority_languages():
    provider = ResearchProvider.__new__(ResearchProvider)
    provider.settings = type("Settings", (), {"branches": 2})()
    provider.structured_calls = []

    def structured(role, prompt, schema, _tokens):
        provider.structured_calls.append(role)
        if schema is SearchPlan:
            assert "Оставшиеся 1" in prompt
            return SearchPlan(interpretation="topic", query_language="th", branches=[Branch(
                topic="AI security pilots", query_ru="пилоты защиты ИИ", query_en="AI security pilots",
                localized_queries=[LocalizedQuery(language="en", query="AI security"),
                                    LocalizedQuery(language="ru", query="защита ИИ")])])
        assert schema is LocalizedQueryBatch
        return LocalizedQueryBatch(queries=[
            LocalizedQuery(language=code, query=f"native {code}")
            for code, _, _, _ in LANGUAGES
        ] + [LocalizedQuery(language="th", query="ความปลอดภัย AI")])

    provider.structured = structured
    plan = provider.plan("AI security", "2026-09-26")
    assert len(plan.branches) == 2
    original = plan.branches[0]
    assert original.topic == "AI security"
    assert original.query_en == "native en"
    assert {q.language for q in original.localized_queries} == {
        *(item[0] for item in LANGUAGES), "th"
    }
    assert next(q.query for q in original.localized_queries if q.language == "th") == "AI security"
    assert {q.language for q in plan.branches[1].localized_queries} == {
        *(item[0] for item in LANGUAGES), "th"
    }
    assert provider.structured_calls == ["search_plan", "localized_original_query", "localized_search_queries"]


def test_original_query_localization_retries_in_smaller_groups_after_length_limit():
    provider = ResearchProvider.__new__(ResearchProvider)
    provider.settings = type("Settings", (), {"branches": 1})()
    calls = []

    class LengthFinishReasonError(Exception):
        pass

    def structured(role, _prompt, schema, max_tokens):
        calls.append((role, schema, max_tokens))
        if schema is SearchPlan:
            return SearchPlan(interpretation="topic", query_language="ru", branches=[])
        if len(calls) == 2:
            raise LengthFinishReasonError()
        return LocalizedQueryBatch(queries=[
            LocalizedQuery(language=code, query=f"native {code}")
            for code, _, _, _ in LANGUAGES
        ])

    provider.structured = structured
    plan = provider.plan("Слабые сигналы в финтехе", "2026-09-28")

    assert len(plan.branches) == 1
    assert plan.branches[0].topic == "финтехе"
    assert plan.branches[0].query_ru == "финтехе"
    assert len(plan.branches[0].localized_queries) == len(LANGUAGES)
    assert calls[0][0] == "search_plan"
    assert calls[1:] == [("localized_original_query", LocalizedQueryBatch, 4500),
                         ("localized_original_query", LocalizedQueryBatch, 850),
                         ("localized_original_query", LocalizedQueryBatch, 850)]


def test_empty_model_plan_preserves_user_directions_and_fills_them():
    provider = ResearchProvider.__new__(ResearchProvider)
    provider.settings = type("Settings", (), {"branches": 3})()
    calls = []

    def structured(role, prompt, schema, _tokens):
        calls.append((role, prompt))
        if schema is SearchPlan:
            if role == "search_plan":
                assert '"AI agents in finance", "post-quantum payments"' in prompt
                return SearchPlan(interpretation="Broad topic", query_language="ru", branches=[])
            assert role == "search_expansions"
            assert "ровно 2 недостающих веток" in prompt
            return SearchPlan(interpretation="Expanded", query_language="ru", branches=[
                Branch(topic="AI agents in financial services", query_ru="ИИ-агенты в финансах",
                       query_en="AI agents in finance"),
                Branch(topic="Post-quantum finance security", query_ru="Постквантовая безопасность финансов",
                       query_en="Post-quantum finance security"),
            ])
        assert schema is LocalizedQueryBatch
        return LocalizedQueryBatch(queries=[
            LocalizedQuery(language=code, query=f"native {code}")
            for code, _, _, _ in LANGUAGES
        ])

    provider.structured = structured
    plan = provider.plan("Слабые сигналы в финансовых технологиях", "2026-09-28",
                         ["AI agents in finance", "post-quantum payments"])

    assert [branch.topic for branch in plan.branches] == [
        "финансовых технологиях",
        "AI agents in finance",
        "post-quantum payments",
    ]
    assert [role for role, _ in calls].count("search_expansions") == 1
    assert len(plan.branches[0].localized_queries) == len(LANGUAGES)
    assert all(len(branch.localized_queries) == len(LANGUAGES) + 1 for branch in plan.branches[1:])
    assert [q.query for q in plan.branches[1].localized_queries if q.language == "und"] == ["AI agents in finance"]


def test_branch_count_is_filled_with_same_scope_facets_if_both_model_attempts_underfill():
    provider = ResearchProvider.__new__(ResearchProvider)
    provider.settings = type("Settings", (), {"branches": 6})()
    calls = []

    def structured(role, prompt, schema, _tokens):
        calls.append(role)
        if schema is SearchPlan:
            return SearchPlan(interpretation="fintech", query_language="ru", branches=[])
        assert schema is LocalizedQueryBatch
        return LocalizedQueryBatch(queries=[
            LocalizedQuery(language=code, query=f"local {code}")
            for code, _, _, _ in LANGUAGES
        ])

    provider.structured = structured
    query = "технологии в финтехе"
    plan = provider.plan(query, "2026-09-28")

    assert len(plan.branches) == 6
    assert plan.branches[0].topic == query
    assert provider.plan_paraphrase_fallback_count == 5
    assert all(query in branch.topic for branch in plan.branches[1:])
    assert "search_expansions" in calls
