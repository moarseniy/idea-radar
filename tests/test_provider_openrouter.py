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
