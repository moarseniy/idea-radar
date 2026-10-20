"""Единственный LLM-провайдер радара: OpenRouter.

Использует `OPENROUTER_API_KEY` и модельный slug из `OPENROUTER_MODEL`.
Веб-поиск —
веб-плагин OpenRouter: ссылки берутся из аннотаций url_citation ответа, а не из текста модели.

Промпты, аудит вызовов и формат результатов — общие с ResearchProvider; здесь переопределены
только транспорт структурного ответа и поиска.
"""
from __future__ import annotations

from datetime import date

from openai import OpenAI

from app.radar.config import RadarSettings
from app.radar.connectors import is_historical
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

# Машина времени: веб-поиск идёт по сегодняшнему интернету, а документы позже даты
# оценки потом всё равно отсекаются при чтении. Без этой вставки до половины ссылок
# оказываются свежими и только занимают места чтения.
HISTORICAL_SEARCH = """
Это историческая оценка: нужны только материалы, опубликованные не позже {as_of}.
Добавляй к запросам годы {years} или оператор before:{as_of}. Не приводи более поздние
источники, даже если они полнее, и не опирайся на то, что стало известно после {as_of}."""


def search_prompt(query: str, as_of: str) -> str:
    prompt = SEARCH_PROMPT.format(as_of=as_of, query=query)
    day = date.fromisoformat(str(as_of)[:10])
    if is_historical(day):
        prompt += HISTORICAL_SEARCH.format(as_of=day.isoformat(), years=f"{day.year - 2}–{day.year}")
    return prompt


class OpenRouterProvider(ResearchProvider):
    provider_name = "OpenRouter"

    def __init__(self, settings: RadarSettings, deadline: float):
        super().__init__(settings, deadline)
        self.client = OpenAI(api_key=settings.openrouter_api_key, base_url=OPENROUTER_URL,
                             max_retries=0, timeout=settings.api_timeout)
        self.model = settings.model
        self.research_model = settings.research_model

    def _reasoning(self, role: str) -> dict:
        """OpenRouter reasoning control: RADAR_REASONING_EFFORT for every call, if set.

        RADAR_REASONING_BY_ROLE overrides it per call role ("default" = the model's own
        depth). Without either setting DeepSeek keeps the previous behaviour: low effort
        only for candidate assessment, the model's default depth elsewhere.
        """
        settings = getattr(self, "settings", None)
        effort = getattr(settings, "reasoning_effort", "")
        effort = dict(getattr(settings, "reasoning_by_role", ())).get(role, effort)
        if effort == "default":
            return {}
        if effort == "none":
            return {"reasoning": {"enabled": False}}
        if effort:
            return {"reasoning": {"effort": effort}}
        if role == "critical_assessment" and self.model.startswith("deepseek/"):
            return {"reasoning_effort": "low"}
        return {}

    def structured(self, role: str, prompt: str, schema: type[T], max_tokens: int = 6000) -> T:
        extra_body = self._reasoning(role) or None
        settings = getattr(self, "settings", None)
        if getattr(settings, "reasoning_effort", "") == "none" and \
                dict(getattr(settings, "reasoning_by_role", ())).get(role, "none") != "none":
            # Рассуждения входят в max_tokens: роль, где они оставлены при общем «none»,
            # получает двойной запас, иначе ответ обрезается (LengthFinishReasonError).
            max_tokens *= 2

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
                      {"role": "user", "content": search_prompt(query, as_of)}],
            extra_body={"plugins": [{"id": "web", "max_results": 12}], **self._reasoning("web_search")},
            timeout=timeout))
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
