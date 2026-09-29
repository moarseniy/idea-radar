"""Провайдер радара на локальной открытой модели — без внешнего API.

Включается `RADAR_PROVIDER=local`. Любой OpenAI-совместимый сервер: Ollama
(`http://localhost:11434/v1`), vLLM (`:8000/v1`), SGLang (`:30000/v1`), LM Studio;
`LOCAL_LLM_BACKEND` выбирает, как у сервера выключаются рассуждения. Модель — `LOCAL_MODEL`; из списка ТЗ локально
запускаются Qwen3.6 35B-A3B и Qwen3 235B. Сравнение открытых моделей, скорость на ноутбуке и
качество — docs/10-model-comparison.md.

Отличия от облака:
  веб-плагина у локальной модели нет — `web_search = False`; общий веб даёт сервис поиска
  (app/search, по умолчанию SearXNG), специализированные источники — открытые коннекторы;
  структурный ответ — JSON-схема в response_format и проверка pydantic на нашей стороне
  (строгий режим OpenAI поддерживают не все серверы);
  рассуждения выключены (параметры — по `LOCAL_LLM_BACKEND`), блок <think> из ответа
  вырезается: тысячи токенов «размышлений» на ноутбуке превращают вызов в минуты.

Промпты и аудит вызовов — общие с ResearchProvider.
"""
from __future__ import annotations

import re

from openai import OpenAI

from app.ml.env import local_extra, strip_thinking
from app.radar.config import RadarSettings
from app.radar.provider import SYSTEM, ResearchProvider, T


class LocalProvider(ResearchProvider):
    web_search = False
    provider_name = "локальный сервер"

    def __init__(self, settings: RadarSettings, deadline: float):
        super().__init__(settings, deadline)
        self.client = OpenAI(api_key=settings.local_key or "local", base_url=settings.local_url,
                             max_retries=0, timeout=settings.local_timeout)
        self.model = settings.local_model
        self.call_timeout = settings.local_timeout  # ноутбук отвечает медленнее облака
        self.extra = local_extra()  # выключение рассуждений — по LOCAL_LLM_BACKEND

    def structured(self, role: str, prompt: str, schema: type[T], max_tokens: int = 6000) -> T:
        response = self._call(role, self.model, lambda timeout: self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0, seed=7,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema()}},
            extra_body=self.extra, timeout=timeout))
        content = strip_thinking(response.choices[0].message.content or "")
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("Модель не вернула структурированный результат")
        return schema.model_validate_json(match.group(0))

    def search(self, query: str, as_of: str, branch: str) -> dict:
        raise NotImplementedError("У локальной модели нет веб-поиска; ссылки дают открытые коннекторы")
