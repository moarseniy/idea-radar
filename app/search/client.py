"""Клиент сервиса поиска: по SEARCH_URL — HTTP, без него — тот же код в процессе."""
from __future__ import annotations

import threading

import httpx

from app.search.base import SearchQuery
from app.search.config import load_search_settings
from app.search.service import SearchService

_local: SearchService | None = None
_lock = threading.Lock()


def web_search(query: SearchQuery) -> dict:
    settings = load_search_settings()
    if settings.service_url:
        headers = {"X-Search-Token": settings.service_token} if settings.service_token else {}
        body = {"text": query.text, "as_of": query.as_of.isoformat() if query.as_of else None,
                "language": query.language, "limit": query.limit, "lookback_days": query.lookback_days}
        response = httpx.post(f"{settings.service_url}/search", json=body, headers=headers,
                              timeout=settings.timeout * max(2, len(settings.providers)) + 10)
        response.raise_for_status()
        return response.json()
    global _local
    with _lock:
        if _local is None or _local.settings != settings:
            _local = SearchService(settings)
    return _local.search(query)
