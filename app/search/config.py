"""Настройки сервиса поиска: окружение процесса, затем .env — как у радара."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ("searxng", "tavily", "exa", "brave", "yandex", "openserp", "openrouter")


def _values() -> dict:
    return {**dotenv_values(ROOT / ".env"), **os.environ}


@dataclass(frozen=True)
class SearchSettings:
    providers: tuple[str, ...] = ("searxng",)
    mode: str = "fallback"                 # fallback | fanout
    timeout: float = 20.0
    cache_hours: int = 24
    cache_dir: Path = ROOT / "storage" / "search_cache"
    service_url: str = ""                  # SEARCH_URL — внешний сервис; пусто — в процессе
    service_token: str = ""                # SEARCH_SERVICE_TOKEN — заголовок X-Search-Token
    # Обычный лимит веб-поиска на одну задачу (ru, en, затем другие локали).
    # Широкий первый проход по исходной теме может явно задать отдельный per-task cap.
    max_queries: int = 3
    # Пауза между запросами к самостоятельно поднятым метапоисковикам (SearXNG, OpenSERP),
    # чтобы их движки не упирались в капчу и лимиты.
    min_interval: float = 1.0
    keys: dict = field(default_factory=dict)


def load_search_settings() -> SearchSettings:
    values = _values()

    def val(key: str, default: str = "") -> str:
        return str(values.get(key) or default).strip()

    names = tuple(p.strip().lower() for p in val("SEARCH_PROVIDERS", "searxng").split(",") if p.strip())
    unknown = [p for p in names if p not in PROVIDERS]
    if unknown:
        raise ValueError(f"Неизвестные провайдеры поиска: {', '.join(unknown)}. Доступны: {', '.join(PROVIDERS)}.")
    try:
        timeout = float(val("SEARCH_TIMEOUT", "20"))
        cache_hours = int(val("SEARCH_CACHE_HOURS", "24"))
        max_queries = int(val("SEARCH_MAX_QUERIES", "3"))
        min_interval = float(val("SEARCH_MIN_INTERVAL", "1.0"))
    except ValueError:
        timeout, cache_hours, max_queries, min_interval = 20.0, 24, 3, 1.0
    storage = Path(val("STORAGE_DIR", str(ROOT / "storage")))
    keys = {
        "searxng_url": val("SEARXNG_URL", "http://localhost:8888").rstrip("/"),
        # Движки SearXNG. С одного IP Brave и Google CSE быстро упираются в лимиты, DuckDuckGo и
        # Qwant — в капчу; Yahoo, Bing News и Google News держатся (проверено живым прогоном).
        "searxng_engines": val("SEARXNG_ENGINES", "yahoo,bing news,google news,brave,google cse,duckduckgo"),
        "tavily": val("TAVILY_API_KEY"),
        "exa": val("EXA_API_KEY"),
        "brave": val("BRAVE_API_KEY"),
        "yandex": val("YANDEX_SEARCH_API_KEY"),
        "yandex_folder": val("YANDEX_FOLDER_ID"),
        "openserp_url": val("OPENSERP_URL", "http://localhost:7000").rstrip("/"),
        "openserp_engines": val("OPENSERP_ENGINES", "bing,duckduckgo,yandex"),
        "openrouter": val("OPENROUTER_API_KEY"),
        "openrouter_model": val("SEARCH_OPENROUTER_MODEL", val("OPENROUTER_RESEARCH_MODEL", "openai/gpt-4.1")),
    }
    return SearchSettings(
        providers=names or ("searxng",), mode="fanout" if val("SEARCH_MODE").lower() == "fanout" else "fallback",
        timeout=max(3.0, min(120.0, timeout)), cache_hours=max(0, min(168, cache_hours)),
        cache_dir=Path(val("SEARCH_CACHE_DIR", str(storage / "search_cache"))),
        service_url=val("SEARCH_URL").rstrip("/"), service_token=val("SEARCH_SERVICE_TOKEN"),
        max_queries=max(1, min(15, max_queries)), min_interval=max(0.0, min(10.0, min_interval)), keys=keys)
