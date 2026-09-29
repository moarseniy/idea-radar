import time
from types import SimpleNamespace

import pytest

from app.radar.config import RadarSettings
from app.radar.models import SearchPlan
from app.radar.pipeline import make_provider
from app.radar.provider_local import LocalProvider

PLAN = '{"interpretation": "i", "branches": [{"topic": "t", "query_ru": "р", "query_en": "e"}]}'


class FakeCompletions:
    def __init__(self, content):
        self.content, self.kwargs = content, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(id="r1", usage=None, choices=[SimpleNamespace(message=message)])


def local(content: str) -> tuple[LocalProvider, FakeCompletions]:
    provider = LocalProvider(RadarSettings(provider="local"), time.monotonic() + 60)
    completions = FakeCompletions(content)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=completions), close=lambda: None)
    return provider, completions


def test_local_provider_needs_no_api_keys():
    settings = RadarSettings(provider="local")
    assert settings.configuration_error() is None
    assert settings.active_model == "qwen3.6:35b-a3b" and settings.provider_label == "локальный сервер"
    assert isinstance(make_provider(settings, time.monotonic() + 60), LocalProvider)


def test_local_provider_requires_open_sources_and_valid_url():
    assert RadarSettings(provider="local", open_sources=False).configuration_error() is None  # есть сервис поиска
    assert "RADAR_WEB_SEARCH" in RadarSettings(provider="local", open_sources=False, web_search="off").configuration_error()
    assert "LOCAL_LLM_URL" in RadarSettings(provider="local", local_url="localhost").configuration_error()


def test_structured_answer_is_validated_and_thinking_disabled():
    provider, completions = local("Вот план:\n```json\n" + PLAN + "\n```")
    plan = provider.structured("search_plan", "p", SearchPlan)
    assert plan.branches[0].query_en == "e"
    assert completions.kwargs["response_format"]["json_schema"]["name"] == "SearchPlan"
    assert completions.kwargs["extra_body"] == {"reasoning_effort": "none"}
    assert provider.audit()[0]["provider"] == "локальный сервер"


def test_broken_structured_answer_raises():
    provider, _ = local("не JSON")
    with pytest.raises(ValueError):
        provider.structured("search_plan", "p", SearchPlan)


def test_local_provider_has_no_web_search():
    provider, _ = local(PLAN)
    assert provider.web_search is False
    with pytest.raises(NotImplementedError):
        provider.search("q", "2026-09-22", "b")


def clear_llm_env(monkeypatch):
    monkeypatch.setattr("app.ml.env._DOTENV", {})  # тест не зависит от локального .env
    for name in ("RADAR_PROVIDER", "LOCAL_MODEL", "LOCAL_LLM_URL", "LOCAL_LLM_KEY", "LOCAL_LLM_BACKEND",
                 "LOCAL_LLM_EXTRA", "FEATURE_MODEL",
                 "FEATURE_LLM_URL", "FEATURE_LLM_KEY", "FEATURE_LLM_EXTRA", "SIGNAL_MODEL_PATH"):
        monkeypatch.setenv(name, "")


def test_scorer_artifact_follows_feature_llm(monkeypatch):
    from app.ml import scorer
    clear_llm_env(monkeypatch)
    assert scorer.model_path().name == "logreg_v1.json"                      # облако, luna
    monkeypatch.setenv("RADAR_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_MODEL", "qwen3.8:27b")
    assert scorer.model_path().name == "logreg_v1_qwen3.8-27b.json"          # всё локально
    monkeypatch.setenv("FEATURE_MODEL", "openai/gpt-5.6-luna")
    monkeypatch.setenv("FEATURE_LLM_URL", "https://openrouter.ai/api/v1")
    assert scorer.model_path().name == "logreg_v1.json"                      # поиск локально, признаки в облаке
    monkeypatch.setenv("LOCAL_MODEL", "unknown:7b")
    monkeypatch.setenv("FEATURE_MODEL", "")
    monkeypatch.setenv("FEATURE_LLM_URL", "")
    assert scorer.model_path().name == "logreg_v1.json"                      # нет артефакта — основная


def test_cloud_search_with_local_features(monkeypatch):
    from app.ml.env import feature_llm
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("RADAR_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("FEATURE_MODEL", "qwen3.6:35b-a3b")
    monkeypatch.setenv("FEATURE_LLM_URL", "http://localhost:11434/v1")
    llm = feature_llm()
    assert llm["url"] == "http://localhost:11434/v1" and llm["extra"] == {"reasoning_effort": "none"}
    assert llm["key"] == ""                                                   # ключ OpenRouter не уходит на свой сервер
    from app.ml import scorer
    assert scorer.model_path().name == "logreg_v1_qwen3.6-35b-a3b.json"


def test_local_key_never_goes_to_openrouter(monkeypatch):
    from app.ml.env import feature_llm
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("RADAR_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_KEY", "local-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert feature_llm()["key"] == "local-secret"
    monkeypatch.setenv("FEATURE_MODEL", "openai/gpt-5.6-luna")
    monkeypatch.setenv("FEATURE_LLM_URL", "https://openrouter.ai/api/v1")
    llm = feature_llm()
    assert llm["key"] == "or-key" and llm["extra"] == {}


def test_garbled_assessment_ids_are_matched_by_order():
    from app.radar.pipeline import match_assessments
    batch = [{"id": "cand_aaa"}, {"id": "cand_bbb"}, {"id": "cand_ccc"}]
    got = [SimpleNamespace(candidate_id=i) for i in ("cand_bbb", "cand_a", "cand_c")]
    by_id = match_assessments(batch, got)
    assert by_id["cand_bbb"] is got[0] and by_id["cand_aaa"] is got[1] and by_id["cand_ccc"] is got[2]
    assert match_assessments(batch, got[:2]).keys() == {"cand_bbb"}  # число не сходится — не угадываем


@pytest.mark.parametrize("backend,extra", [
    ("ollama", {"reasoning_effort": "none"}),
    ("vllm", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("sglang", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("lmstudio", {}),  # неизвестный сервер — без спец. полей
])
def test_thinking_is_disabled_per_backend(monkeypatch, backend, extra):
    from app.ml.env import feature_llm
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("RADAR_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_BACKEND", backend)
    assert feature_llm()["extra"] == extra
    provider = LocalProvider(RadarSettings(provider="local"), time.monotonic() + 60)
    assert provider.extra == extra
    monkeypatch.setenv("LOCAL_LLM_EXTRA", '{"top_k": 20}')                   # явная настройка важнее
    assert feature_llm()["extra"] == {"top_k": 20}


def test_think_block_is_stripped_before_json():
    from app.ml.env import strip_thinking
    assert strip_thinking('<think>maybe {"a": 1}?</think>\n{"b": 2}') == '{"b": 2}'
    assert strip_thinking('{"b": 2}<think>обрезано {"a"') == '{"b": 2}'
    provider, _ = local("<think>{\"interpretation\": \"ложный\"}</think>" + PLAN)
    assert provider.structured("search_plan", "p", SearchPlan).interpretation == "i"


def test_hf_model_ids_select_the_same_scorer(monkeypatch):
    from app.ml import scorer
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("RADAR_PROVIDER", "local")
    for name in ("Qwen/Qwen3.6-35B-A3B", "qwen3.6:35b-a3b", "qwen/qwen3.6-35b-a3b"):
        monkeypatch.setenv("LOCAL_MODEL", name)
        assert scorer.model_path().name == "logreg_v1_qwen3.6-35b-a3b.json"
