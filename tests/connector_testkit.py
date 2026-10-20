"""Шаблон офлайн-тестов коннектора: записанный ответ API вместо сети.

Записанные ответы лежат в tests/fixtures/connectors/<имя>.json (сырое тело ответа API,
сокращённое до 2–5 записей). Тест подменяет сетевой вызов модуля и проверяет контракт:

    from connector_testkit import assert_connector_contract, fixture, fake_json_request

    def test_usaspending_contract(monkeypatch):
        calls = fake_json_request(monkeypatch, usaspending, fixture("usaspending"))
        refs = usaspending.CONNECTOR.run(["solid-state battery"], SNAPSHOT, 5)
        assert_connector_contract(usaspending.CONNECTOR, refs, SNAPSHOT, money=True)
        assert calls[0]["body"]["filters"]["time_period"][0]["end_date"] == "2021-12-31"
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.radar.connectors import Connector

FIXTURES = Path(__file__).parent / "fixtures" / "connectors"


def fixture(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def fake_json_request(monkeypatch, module, *responses) -> list[dict]:
    """Подменяет module._json_request: отдаёт ответы по очереди и записывает вызовы."""
    calls: list[dict] = []
    queue = list(responses)

    def fake(provider, url, **kwargs):
        calls.append({"provider": provider, "url": url, **kwargs})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(module, "_json_request", fake)
    return calls


def assert_connector_contract(connector: Connector, refs: list[dict], as_of: date, *,
                              money: bool = False) -> None:
    """Общий контракт ссылки: URL, заголовок, документ, канал, дата не позже as_of."""
    assert refs, f"{connector.name}: записанный ответ должен давать хотя бы одну ссылку"
    for ref in refs:
        assert ref["url"].startswith("https://"), ref["url"]
        assert ref["title"].strip()
        assert ref["channel"] == connector.channel
        assert ref["date_support"] == connector.date_support
        document = ref.get("document") or {}
        assert document.get("text", "").strip(), f"{connector.name}: пустой текст документа"
        assert ref.get("source_provider"), "нужен source_provider для аудита"
        if connector.date_support != "none":
            assert ref.get("observed_at"), f"{connector.name}: у {connector.date_support}-источника нужна дата"
            assert str(ref["observed_at"])[:10] <= as_of.isoformat()
        if money:
            event = ref.get("structured_event")
            assert isinstance(event, dict), f"{connector.name}: денежная запись без structured_event"
            assert event["date"] <= as_of.isoformat()
            assert event.get("kind") and event.get("source_id")
