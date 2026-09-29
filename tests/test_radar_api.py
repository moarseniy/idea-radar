from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.radar.api import (
    create_search,
    delete_search,
    markdown_report,
    search_scope_chat,
)
from app.radar.models import (
    ScopeChatRequest,
    ScopeChatResponse,
    ScopeSuggestion,
    SearchRequest,
)


def test_markdown_report_lists_ranked_candidates_with_verdicts():
    weak = {"id": "w", "rank": 1, "title": "Early technology", "verdict": "weak_signal",
            "score": 86, "score_kind": "model", "stage": "pilot", "evidence": [], "limitations": []}
    mature = {"id": "m", "rank": 2, "title": "Mature technology", "verdict": "mature",
              "score": 91, "score_kind": "model", "stage": "mature", "reason": "Широко внедрена",
              "evidence": [], "limitations": []}
    report = markdown_report({"query": "topic", "as_of": "2026-09-27", "status": "completed", "limit": 2,
                              "signals": [weak], "top_candidates": [mature, weak], "rejected": [mature],
                              "sources": [], "warnings": [], "score_note": "model"})
    assert "Кандидатов в ТОП: 2/2" in report
    assert "Подтверждено слабых сигналов: 1" in report
    assert "Зрелая технология" in report
    assert "Причина статуса: Широко внедрена" in report


def test_delete_search_removes_a_completed_run():
    class Store:
        def __init__(self):
            self.deleted = []

        def get(self, _run_id):
            return {"status": "completed"}

        def delete_run(self, run_id):
            self.deleted.append(run_id)
            return True

    store = Store()
    result = delete_search("run-1", SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(radar_store=store))))
    assert result == {"deleted": True}
    assert store.deleted == ["run-1"]


def test_delete_search_rejects_running_run():
    class Store:
        def __init__(self):
            self.deleted = False

        def get(self, _run_id):
            return {"status": "running"}

        def delete_run(self, _run_id):
            self.deleted = True

    store = Store()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(radar_store=store)))
    with pytest.raises(HTTPException) as error:
        delete_search("run-active", request)
    assert error.value.status_code == 409
    assert not store.deleted


def test_scope_chat_returns_optional_suggestions_without_starting_a_search(monkeypatch):
    captured = {}

    class Settings:
        branches = 4

        @staticmethod
        def configuration_error():
            return None

    class Provider:
        def structured(self, role, prompt, schema, _tokens):
            captured.setdefault("role", role)
            captured.update(prompt=prompt, schema=schema)
            return ScopeChatResponse(reply="Уточним фокус?", suggestions=[
                ScopeSuggestion(title="Платежи", description="Новые технологии расчётов.")
            ])

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr("app.radar.api.load_radar_settings", lambda: Settings())
    monkeypatch.setattr("app.radar.api.make_provider", lambda *_args: Provider())
    request = ScopeChatRequest(query="технологии в финтехе", message="Интересуют платежи",
                               history=[], selected_directions=["Кредитный скоринг"])

    response = search_scope_chat(request, None)
    assert response.reply == "Уточним фокус?"
    assert len(response.suggestions) == 4
    assert response.suggestions[0].title == "Платежи"
    assert captured["role"] == "scope_clarification"
    assert '"Кредитный скоринг"' in captured["prompt"]
    assert captured["closed"] is True


def test_scope_chat_fills_empty_llm_suggestions_with_same_scope_variants(monkeypatch):
    class Settings:
        branches = 8

        @staticmethod
        def configuration_error():
            return None

    class Provider:
        def structured(self, _role, _prompt, _schema, _tokens):
            return ScopeChatResponse(reply="Тему можно искать сразу.", suggestions=[])

        def close(self):
            pass

    monkeypatch.setattr("app.radar.api.load_radar_settings", lambda: Settings())
    monkeypatch.setattr("app.radar.api.make_provider", lambda *_args: Provider())
    payload = ScopeChatRequest(query="технологии в финтехе", message="во втором уточнении")

    response = search_scope_chat(payload, None)

    assert len(response.suggestions) == 4
    assert all("технологии в финтехе" in item.title for item in response.suggestions)
    assert len({item.title for item in response.suggestions}) == 4
    assert any("синонимы" in item.description for item in response.suggestions)
    assert any("задачи" in item.description for item in response.suggestions)
    assert any("альтернативные" in item.description for item in response.suggestions)


def test_create_search_rejects_more_confirmed_directions_than_branch_capacity(monkeypatch):
    from app.radar import api

    class Settings:
        branches = 2

        @staticmethod
        def configuration_error():
            return None

    monkeypatch.setattr(api, "load_radar_settings", lambda: Settings())
    request = SearchRequest(query="технологии в финтехе", directions=["ИИ в финансах", "блокчейн"])
    with pytest.raises(HTTPException) as error:
        create_search(request, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())))
    assert error.value.status_code == 422
