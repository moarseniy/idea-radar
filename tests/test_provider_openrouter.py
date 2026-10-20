from types import SimpleNamespace

from app.radar.provider_openrouter import OpenRouterProvider


def test_low_reasoning_is_only_sent_for_deepseek_candidate_assessments():
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed="ok"))])

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.model = "deepseek/deepseek-v4.1-flash"
    provider.client = SimpleNamespace(beta=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    provider._call = lambda _role, _model, invoke: invoke(5)

    assert provider.structured("critical_assessment", "prompt", object) == "ok"
    assert provider.structured("consolidate", "prompt", object) == "ok"

    assert calls[0]["extra_body"] == {"reasoning_effort": "low"}
    assert "extra_body" not in calls[1]


def test_low_reasoning_is_not_sent_to_non_deepseek_models():
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed="ok"))])

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.model = "openai/gpt-4.1"
    provider.client = SimpleNamespace(beta=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    provider._call = lambda _role, _model, invoke: invoke(5)

    assert provider.structured("critical_assessment", "prompt", object) == "ok"
    assert "extra_body" not in calls[0]


def test_reasoning_effort_setting_applies_to_every_call_of_any_model():
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed="ok"))])

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.model = "deepseek/deepseek-v4.1-flash-1017"
    provider.settings = SimpleNamespace(reasoning_effort="none")
    provider.client = SimpleNamespace(beta=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    provider._call = lambda _role, _model, invoke: invoke(5)

    provider.structured("consolidate", "prompt", object)
    provider.settings = SimpleNamespace(reasoning_effort="low")
    provider.structured("search_plan", "prompt", object)

    assert calls[0]["extra_body"] == {"reasoning": {"enabled": False}}
    assert calls[1]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_reasoning_can_stay_on_for_one_role_with_doubled_token_budget():
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed="ok"))])

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.model = "deepseek/deepseek-v4.1-flash-1017"
    provider.settings = SimpleNamespace(reasoning_effort="none",
                                        reasoning_by_role=(("critical_assessment", "default"),))
    provider.client = SimpleNamespace(beta=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    provider._call = lambda _role, _model, invoke: invoke(5)

    provider.structured("critical_assessment", "prompt", object, 3000)
    provider.structured("extract_candidates", "prompt", object, 3000)

    assert "extra_body" not in calls[0] and calls[0]["max_tokens"] == 6000
    assert calls[1]["extra_body"] == {"reasoning": {"enabled": False}} and calls[1]["max_tokens"] == 3000


def test_historical_web_search_asks_for_period_sources_only():
    from datetime import date

    from app.radar.provider_openrouter import search_prompt

    past = search_prompt("AI security", "2021-12-31")
    assert "before:2021-12-31" in past and "2019–2021" in past
    assert "историческая оценка" not in search_prompt("AI security", date.today().isoformat())  # noqa: DTZ011
