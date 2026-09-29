"""Цепочка провайдеров, кэш и фильтр по дате оценки."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.search.base import NotConfigured, SearchError, SearchQuery
from app.search.config import SearchSettings, load_search_settings
from app.search.providers import REGISTRY, Provider


def url_key(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), (parts.hostname or "").lower().removeprefix("www."),
                       parts.path.rstrip("/"), parts.query, ""))


class SearchService:
    def __init__(self, settings: SearchSettings | None = None, client: httpx.Client | None = None,
                 providers: list[Provider] | None = None) -> None:
        self.settings = settings or load_search_settings()
        self.client = client or httpx.Client(follow_redirects=True)
        self.providers = providers or [REGISTRY[name](self.settings, self.client) for name in self.settings.providers]
        self._pace = {p.name: [threading.Lock(), 0.0] for p in self.providers}

    def _wait_turn(self, provider: Provider) -> None:
        """Разносит запросы к своему SearXNG/OpenSERP: подряд идущие ~100 запросов ловят капчу."""
        if not provider.self_hosted or not self.settings.min_interval:
            return
        lock, _ = self._pace[provider.name]
        with lock:
            slot = max(time.monotonic(), self._pace[provider.name][1])
            self._pace[provider.name][1] = slot + self.settings.min_interval
        time.sleep(max(0.0, slot - time.monotonic()))

    def status(self) -> list[dict]:
        return [{"name": p.name, "configured": p.configuration_error() is None, "error": p.configuration_error()}
                for p in self.providers]

    def search(self, query: SearchQuery) -> dict:
        """{"results": [...], "providers": [аудит по каждому провайдеру], "mode": ...}."""
        audit: list[dict] = []
        if self.settings.mode == "fanout":
            with ThreadPoolExecutor(len(self.providers)) as pool:
                runs = list(pool.map(lambda p: self._run(p, query), self.providers))
            audit = [a for a, _ in runs]
            results = interleave([r for _, r in runs], query.limit)
        else:
            results = []
            for provider in self.providers:
                entry, found = self._run(provider, query)
                audit.append(entry)
                if found:
                    results = found[: query.limit]
                    break
        return {"results": results, "providers": audit, "mode": self.settings.mode, "query": query.to_dict()}

    def _run(self, provider: Provider, query: SearchQuery) -> tuple[dict, list[dict]]:
        entry = {"name": provider.name, "status": "ok", "count": 0, "error": None, "cached": False, "seconds": 0.0}
        error = provider.configuration_error()
        if error:
            entry.update(status="not_configured", error=error)
            return entry, []
        started = time.monotonic()
        cached = self._cache_get(provider.name, query)
        if cached is None:
            self._wait_turn(provider)
        try:
            found = cached if cached is not None else provider.search(query)
        except NotConfigured as exc:
            entry.update(status="not_configured", error=str(exc))
            found = []
        except SearchError as exc:
            entry.update(status="error", error=str(exc))
            found = []
        except (httpx.HTTPError, ValueError, KeyError) as exc:  # битый ответ провайдера
            entry.update(status="error", error=f"{provider.name}: {type(exc).__name__}")
            found = []
        else:
            if cached is None:
                self._cache_put(provider.name, query, found)
        found = within(found, query)
        entry.update(count=len(found), cached=cached is not None, seconds=round(time.monotonic() - started, 2))
        if entry["status"] == "ok" and not found:
            entry["status"] = "empty"
        return entry, found

    def _cache_path(self, name: str, query: SearchQuery) -> Path:
        key = json.dumps([name, query.to_dict()], sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.settings.cache_dir / name / f"{digest}.json"

    def _cache_get(self, name: str, query: SearchQuery) -> list[dict] | None:
        path = self._cache_path(name, query)
        if not self.settings.cache_hours or not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.settings.cache_hours * 3600:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _cache_put(self, name: str, query: SearchQuery, found: list[dict]) -> None:
        if not self.settings.cache_hours or not found:  # пустую выдачу не кэшируем: могла быть временной
            return
        path = self._cache_path(name, query)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def within(found: list[dict], query: SearchQuery) -> list[dict]:
    """Отбрасывает результаты, опубликованные позже даты оценки (дата известна не всегда)."""
    if not query.as_of:
        return found
    limit = query.as_of.isoformat()
    return [r for r in found if not r.get("published_at") or r["published_at"] <= limit]


def interleave(lists: list[list[dict]], limit: int) -> list[dict]:
    """Выдачи провайдеров по очереди, без повторов URL: ни один провайдер не забирает всё."""
    out, seen = [], set()
    for index in range(max((len(x) for x in lists), default=0)):
        for items in lists:
            if index < len(items):
                key = url_key(items[index]["url"])
                if key not in seen:
                    seen.add(key)
                    out.append(items[index])
    return out[: max(limit, 1) * max(1, len([x for x in lists if x]))]
