from dataclasses import replace
from types import SimpleNamespace

import openai

from app.config import load_settings
from app.openrouter_service import OPENROUTER_API_URL, OpenRouterService


def test_legacy_lab_model_requests_are_pinned_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    settings = replace(load_settings(), openrouter_api_key="test-key",
                       openrouter_model="openai/gpt-4.1")
    recorded = {}

    class FakeClient:
        def __init__(self, **kwargs):
            recorded["client"] = kwargs
            self.responses = self

        def create(self, **kwargs):
            recorded["request"] = kwargs
            return SimpleNamespace(output_text="ok")

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    service = OpenRouterService(settings)
    response = service._call_response("instructions", "input", 100)

    assert response.output_text == "ok"
    assert recorded["client"]["api_key"] == "test-key"
    assert recorded["client"]["base_url"] == OPENROUTER_API_URL
    assert recorded["request"]["model"] == "openai/gpt-4.1"
