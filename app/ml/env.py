"""Настройки ML-слоя: окружение процесса, затем .env — как у радара (app/radar/config.py).

LLM поиска (RADAR_PROVIDER) и LLM разметки признаков настраиваются независимо:
  RADAR_PROVIDER=local           — по умолчанию и то и другое на локальном сервере;
  FEATURE_MODEL / FEATURE_LLM_URL — переопределяют только разметку, например облачный
                                   поиск и локальная разметка или наоборот.
Модель скоринга обучена на признаках конкретной LLM, поэтому выбирается по LLM разметки
(app/ml/scorer.py: model_path).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
_DOTENV = dotenv_values(ROOT / ".env")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_FEATURE_MODEL = "openai/gpt-5.6-luna"


def env(name: str, default: str = "") -> str:
    return str(os.environ.get(name) or _DOTENV.get(name) or default).strip()


def local_mode() -> bool:
    """RADAR_PROVIDER=local — радар (и по умолчанию разметка) на локальном сервере."""
    return env("RADAR_PROVIDER").lower() == "local"


def local_model() -> str:
    return env("LOCAL_MODEL", "qwen3.6:35b-a3b")


def local_url() -> str:
    return env("LOCAL_LLM_URL", "http://localhost:11434/v1").rstrip("/")


# Как выключить рассуждения на разных OpenAI-совместимых серверах. Ollama понимает
# reasoning_effort; vLLM и SGLang для Qwen3/DeepSeek — флаг шаблона чата enable_thinking
# (reasoning_effort они для этих моделей молча игнорируют).
NO_THINKING = {
    "ollama": {"reasoning_effort": "none"},
    "vllm": {"chat_template_kwargs": {"enable_thinking": False}},
    "sglang": {"chat_template_kwargs": {"enable_thinking": False}},
    "openai-compatible": {},
}


def local_backend() -> str:
    """LOCAL_LLM_BACKEND: ollama | vllm | sglang | openai-compatible (по умолчанию ollama)."""
    backend = env("LOCAL_LLM_BACKEND", "ollama").lower()
    return backend if backend in NO_THINKING else "openai-compatible"


def local_extra() -> dict:
    """Доп. поля запроса к локальному серверу: LOCAL_LLM_EXTRA или «без рассуждений» для бэкенда."""
    raw = env("LOCAL_LLM_EXTRA")
    return json.loads(raw) if raw else dict(NO_THINKING[local_backend()])


def strip_thinking(content: str) -> str:
    """Убирает <think>…</think>: без reasoning-парсера vLLM/SGLang кладут рассуждения в content,
    и в них тоже бывают фигурные скобки. Незакрытый блок (обрезанный ответ) — всё до конца."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return re.sub(r"<think>.*\Z", "", content, flags=re.DOTALL).strip()


def feature_llm() -> dict:
    """Модель, адрес, доп. поля запроса и ключ для LLM разметки признаков."""
    local = local_mode()
    model = env("FEATURE_MODEL", local_model() if local else DEFAULT_FEATURE_MODEL)
    url = env("FEATURE_LLM_URL", local_url() if local else OPENROUTER_URL).rstrip("/")
    on_openrouter = url.startswith(OPENROUTER_URL)
    # Не OpenRouter — значит свой сервер: по умолчанию без рассуждений (параметры — по
    # LOCAL_LLM_BACKEND), иначе на ноутбуке тысячи токенов «размышлений» превращают вызов в минуты.
    raw_extra = env("FEATURE_LLM_EXTRA")
    extra = json.loads(raw_extra) if raw_extra else ({} if on_openrouter else local_extra())
    # Каждый ключ уходит только на свой адрес: OpenRouter — в OpenRouter, локальный — на LOCAL_LLM_URL.
    key = env("FEATURE_LLM_KEY") or (env("OPENROUTER_API_KEY") if on_openrouter else
                                     env("LOCAL_LLM_KEY") if url == local_url() else "")
    return {"model": model, "url": url, "extra": extra, "key": key}


def model_slug(model: str) -> str:
    """«qwen/qwen3.6-35b-a3b» и «qwen3.6:35b-a3b» → «qwen3.6-35b-a3b» — имя артефакта скоринга."""
    return re.sub(r"[^a-z0-9.\-]+", "-", model.split("/")[-1].lower().replace(":", "-")).strip("-")
