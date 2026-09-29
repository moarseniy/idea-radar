from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ALLOWED_MODELS = {"openai/gpt-4.1", "openai/gpt-4.1-2025-04-14", "openai/gpt-5.6-luna",
                  "deepseek/deepseek-v4.1-flash"}
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RadarSettings:
    model: str = "openai/gpt-4.1"
    research_model: str = "openai/gpt-4.1"
    database_url: str = ""
    storage_dir: Path = ROOT / "storage"
    data_dir: Path = ROOT / "data"
    branches: int = 4
    max_sources: int = 110
    max_candidates: int = 60
    search_workers: int = 3
    max_active_runs: int = 1
    deadline_seconds: int = 1080
    fetch_timeout: int = 12
    cache_hours: int = 24
    api_timeout: int = 90
    retries: int = 1
    # openrouter — облачная модель через OpenRouter; local — свой OpenAI-совместимый сервер.
    provider: str = "openrouter"
    openrouter_api_key: str = ""
    use_model: bool = True
    open_sources: bool = True
    # Веб-поиск: service — сервис поиска app/search (SEARCH_PROVIDERS, по умолчанию SearXNG);
    # llm — веб-плагин LLM-провайдера (только openrouter); off — только открытые коннекторы.
    web_search: str = "service"
    # RADAR_PROVIDER=local: OpenAI-совместимый сервер на своём железе (Ollama, vLLM, LM Studio).
    local_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen3.6:35b-a3b"
    local_key: str = ""
    local_timeout: int = 600
    audio_transcription_url: str = ""
    audio_transcription_token: str = ""
    audio_transcription_model: str = "whisper-1"
    audio_transcription_allowlist: tuple[str, ...] = ()
    audio_transcription_timeout: int = 180
    audio_transcription_max_mb: int = 20

    @property
    def active_model(self) -> str:
        return self.local_model if self.provider == "local" else self.model

    @property
    def active_research_model(self) -> str:
        return self.local_model if self.provider == "local" else self.research_model

    @property
    def provider_label(self) -> str:
        return "локальный сервер" if self.provider == "local" else "OpenRouter"

    def configuration_error(self) -> str | None:
        if self.provider == "local":
            # Открытая модель на своём сервере. Из списка ТЗ локально — Qwen3.6 35B-A3B и Qwen3 235B;
            # веб-поиска у локальной модели нет, ссылки дают открытые коннекторы (open_sources.py).
            parsed = urlsplit(self.local_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                return "LOCAL_LLM_URL должен быть адресом OpenAI-совместимого сервера, например http://localhost:11434/v1."
            if not self.local_model:
                return "Укажите LOCAL_MODEL — имя модели на локальном сервере."
            if not self.open_sources and self.web_search != "service":
                return ("Для RADAR_PROVIDER=local нужен сервис поиска (RADAR_WEB_SEARCH=service) или открытые "
                        "источники (RADAR_OPEN_SOURCES=1): веб-плагина у локальной модели нет.")
            return None
        if self.model not in ALLOWED_MODELS or self.research_model not in ALLOWED_MODELS:
            return "Выбранная конфигурация LLM не поддерживается приложением."
        if not self.openrouter_api_key:
            return "Добавьте OPENROUTER_API_KEY в .env. Каталог примеров доступен без ключа."
        return None


def load_radar_settings() -> RadarSettings:
    values = {**dotenv_values(ROOT / ".env"), **os.environ}

    def val(key: str, default: str = "") -> str:
        return str(values.get(key) or default).strip()

    def num(key: str, default: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(val(key, str(default)))))
        except ValueError:
            return default

    model = val("OPENROUTER_MODEL", "openai/gpt-4.1")
    provider = "local" if val("RADAR_PROVIDER", "openrouter").lower() == "local" else "openrouter"
    deadline_limit = 3600 if provider == "local" else 1200
    audio_hosts = tuple(dict.fromkeys(
        host.strip().lower().lstrip(".") for host in val("RADAR_AUDIO_TRANSCRIPTION_ALLOWLIST").split(",")
        if host.strip()
    ))
    return RadarSettings(
        model=model,
        research_model=val("OPENROUTER_RESEARCH_MODEL", model),
        database_url=val("RADAR_DATABASE_URL"),
        storage_dir=Path(val("STORAGE_DIR", str(ROOT / "storage"))),
        data_dir=Path(val("SAMPLE_DATA_DIR", str(ROOT / "data"))),
        branches=num("RADAR_SEARCH_BRANCHES", 4, 1, 8),
        max_sources=num("RADAR_MAX_SOURCES", 110, 8, 128),
        max_candidates=num("RADAR_MAX_CANDIDATES", 60, 15, 64),
        search_workers=num("RADAR_SEARCH_WORKERS", 3, 1, 4),
        max_active_runs=num("RADAR_MAX_ACTIVE_RUNS", 1, 1, 3),
        # Облачный профиль целится в 18 минут и жёстко ограничен 20; локальной модели — до часа.
        deadline_seconds=num("RADAR_DEADLINE_SECONDS", 1080, 60, deadline_limit),
        fetch_timeout=num("RADAR_FETCH_TIMEOUT", 12, 3, 30),
        cache_hours=num("RADAR_CACHE_HOURS", 24, 0, 168),
        api_timeout=num("OPENROUTER_TIMEOUT", 90, 10, 180),
        retries=num("OPENROUTER_MAX_RETRIES", 1, 0, 2),
        provider=provider,
        openrouter_api_key=val("OPENROUTER_API_KEY"),
        use_model=val("RADAR_USE_MODEL", "1") != "0",
        open_sources=val("RADAR_OPEN_SOURCES", "1") != "0",
        web_search=(val("RADAR_WEB_SEARCH", "service").lower()
                    if val("RADAR_WEB_SEARCH", "service").lower() in ("service", "llm", "off") else "service"),
        local_url=val("LOCAL_LLM_URL", "http://localhost:11434/v1").rstrip("/"),
        local_model=val("LOCAL_MODEL", "qwen3.6:35b-a3b"),
        local_key=val("LOCAL_LLM_KEY"),
        local_timeout=num("LOCAL_TIMEOUT", 600, 30, 1800),
        audio_transcription_url=val("RADAR_AUDIO_TRANSCRIPTION_URL").rstrip("/"),
        audio_transcription_token=val("RADAR_AUDIO_TRANSCRIPTION_TOKEN"),
        audio_transcription_model=val("RADAR_AUDIO_TRANSCRIPTION_MODEL", "whisper-1"),
        audio_transcription_allowlist=audio_hosts,
        audio_transcription_timeout=num("RADAR_AUDIO_TRANSCRIPTION_TIMEOUT", 180, 15, 600),
        audio_transcription_max_mb=num("RADAR_AUDIO_TRANSCRIPTION_MAX_MB", 20, 1, 25),
    )
