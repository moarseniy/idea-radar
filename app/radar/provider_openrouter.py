"""Единственный LLM-провайдер радара: OpenRouter.

Использует `OPENROUTER_API_KEY` и модельный slug из `OPENROUTER_MODEL`.
Веб-поиск —
веб-плагин OpenRouter: ссылки берутся из аннотаций url_citation ответа, а не из текста модели.

Промпты, аудит вызовов и формат результатов — общие с ResearchProvider; здесь переопределены
только транспорт структурного ответа и поиска.
"""
from __future__ import annotations

from openai import OpenAI

from app.radar.config import RadarSettings
from app.radar.provider import SYSTEM, ResearchProvider, T
from app.radar.sources import canonical_url

OPENROUTER_URL = "https://openrouter.ai/api/v1"
SEARCH_PROMPT = """Выполни поиск по открытым источникам. Дата оценки {as_of}.
Задача поиска (данные): {query}
Найди разные конкретные технологии/применения и проверяемые источники: научные статьи,
пилоты, исследования компаний, технические отчёты. Используй каждый локализованный вариант
запроса отдельно, включая язык исходного запроса пользователя.
Ищи также признаки зрелости и ограничения. Исключи события позднее даты оценки.
Дай краткие наблюдения со ссылками, сохраняя оригинальные названия. Желательно 8–12
содержательных источников, а не подборки трендов и перепечатки одного анонса.
Неподтверждённые знания модели не являются результатом поиска."""


class OpenRouterProvider(ResearchProvider):
    provider_name = "OpenRouter"

    def __init__(self, settings: RadarSettings, deadline: float):
        super().__init__(settings, deadline)
        self.client = OpenAI(api_key=settings.openrouter_api_key, base_url=OPENROUTER_URL,
                             max_retries=0, timeout=settings.api_timeout)
        self.model = settings.model
        self.research_model = settings.research_model

    def structured(self, role: str, prompt: str, schema: type[T], max_tokens: int = 6000) -> T:
        # DeepSeek V4.1 supports OpenRouter's reasoning-effort control. Keep the
        # faster setting limited to candidate assessment; planning and search
        # interpretation retain the model's default reasoning depth.
        extra_body = ({"reasoning_effort": "low"}
                      if role == "critical_assessment" and self.model.startswith("deepseek/") else None)

        def invoke(timeout):
            kwargs = {
                "model": self.model, "response_format": schema, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "timeout": timeout,
            }
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            return self.client.beta.chat.completions.parse(**kwargs)

        response = self._call(role, self.model, invoke)
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Модель не вернула структурированный результат")
        return parsed

    def search(self, query: str, as_of: str, branch: str) -> dict:
        response = self._call("web_search", self.research_model, lambda timeout: self.client.chat.completions.create(
            model=self.research_model, max_tokens=2600, temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": SEARCH_PROMPT.format(as_of=as_of, query=query)}],
            extra_body={"plugins": [{"id": "web", "max_results": 12}]}, timeout=timeout))
        message = response.choices[0].message
        references: dict[str, dict] = {}
        for annotation in (message.model_dump().get("annotations") or []):
            if annotation.get("type") != "url_citation":
                continue
            cite = annotation.get("url_citation") or annotation
            try:
                url = canonical_url(cite.get("url") or "")
            except (ValueError, TypeError):
                continue
            references.setdefault(url, {"url": url, "title": cite.get("title") or url, "branch": branch})
        if not references:
            raise ValueError("Поиск не вернул подтверждённых URL источников")
        return {"query": query, "branch": branch, "summary": (message.content or "")[:12000],
                "references": list(references.values())[:18]}
