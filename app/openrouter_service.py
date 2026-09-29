from __future__ import annotations

import base64
import json
import mimetypes
import re
from typing import Any

from app.config import Settings
from app.knowledge import (
    edge_id,
    is_low_signal_numeric_label,
    score_hypothesis,
    stable_id,
)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


class OpenRouterServiceError(RuntimeError):
    """Raised when an OpenRouter-backed action cannot be completed."""


class OpenRouterService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.openrouter_enabled

    def generate_hypotheses(
        self,
        project: dict[str, Any],
        documents: list[dict[str, Any]],
        graph: dict[str, list[dict[str, Any]]],
        feedback: list[dict[str, Any]],
        count: int,
        weights: dict[str, float],
        exclusions: list[str],
        include_roadmap: bool,
        research_context: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.enabled:
            raise OpenRouterServiceError("OPENROUTER_API_KEY не задан. Генерация гипотез требует OpenRouter.")

        prompt = _build_hypothesis_prompt(
            project,
            documents,
            graph,
            feedback,
            count,
            weights,
            exclusions,
            include_roadmap,
            self.settings.max_context_chars,
            research_context=research_context,
        )
        instructions = (
            "Ты исследовательский AI-agent для генерации научно-исследовательских гипотез в материаловедении, "
            "обогащении и металлургии. Формируй конкретные проверяемые гипотезы, не выдумывай источники, "
            "явно отделяй факты из контекста от допущений. Источники могут быть на русском, английском, "
            "китайском и других языках; корректно интерпретируй их, сохраняя исходные термины там, где они важны. "
            "Если во входе есть блок web research, используй только реальные ссылки из него и не выдумывай URL. "
            "Ответ верни строго валидным JSON без markdown."
        )
        try:
            text = self._call_text(
                instructions=instructions,
                prompt=prompt,
                max_output_tokens=12000,
                text_format=_hypotheses_text_format(),
            )
            payload = _extract_json(text)
            hypotheses = payload.get("hypotheses", []) if isinstance(payload, dict) else []
            normalized = [_normalize_hypothesis(item, weights) for item in hypotheses[:count] if isinstance(item, dict)]
            if not normalized:
                raise ValueError("OpenRouter response did not contain hypotheses")
            return normalized, {"mode": "openrouter", "model": self.settings.openrouter_model}
        except Exception as exc:  # noqa: BLE001
            raise OpenRouterServiceError(f"OpenRouter generation failed ({exc.__class__.__name__}): {exc}") from exc

    def research_topic(
        self,
        project: dict[str, Any],
        documents: list[dict[str, Any]],
        graph: dict[str, list[dict[str, Any]]],
        query: str,
        max_sources: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.enabled:
            raise OpenRouterServiceError("OPENROUTER_API_KEY не задан. Research требует OpenRouter.")

        source_limit = max(1, min(max_sources or self.settings.openrouter_research_max_sources, 12))
        research_query = (query or _default_research_query(project, graph)).strip()
        prompt = _build_research_prompt(project, documents, graph, research_query, self.settings.max_context_chars)
        instructions = (
            "Ты research-agent для R&D проекта в обогащении, металлургии и материаловедении. "
            "Используй web search для поиска актуальных научных статей, патентов, промышленных примеров и релевантных обзоров. "
            "Пиши по-русски. Не выдавай ссылку как источник, если не нашел ее через web search. "
            "Отделяй подтвержденные сведения от инженерных допущений."
        )
        try:
            response = self._call_response(
                instructions=instructions,
                input_payload=prompt,
                max_output_tokens=4500,
                model=self.settings.openrouter_research_model,
                extra_body={"plugins": [{"id": "web", "max_results": source_limit}]},
            )
            mode = "openrouter_web_search"
            text = (getattr(response, "output_text", None) or _response_to_text(response)).strip()
            if not text:
                raise ValueError("OpenRouter research returned empty text")
            sources = _response_citations(response)[:source_limit]
            return (
                {
                    "query": research_query,
                    "summary": text[:14000],
                    "sources": sources,
                },
                {
                    "mode": mode,
                    "model": self.settings.openrouter_research_model,
                    "sources": len(sources),
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise OpenRouterServiceError(f"OpenRouter research failed ({exc.__class__.__name__}): {exc}") from exc

    def extract_graph(self, document_text: str, source_id: str, source_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if not self.enabled or not self.settings.openrouter_graph_extraction or len(document_text.strip()) < 80:
            return [], [], {"mode": "skipped"}

        prompt = f"""
Источник: {source_name}

Извлеки из текста компактный граф знаний для исследовательской системы. Нужны только предметные сущности:
материалы, процессы, реагенты, свойства, метрики, группы параметров, оборудование, риски, наблюдения.
Не создавай отдельные узлы, label которых состоит только из числа и единицы измерения: "1 мм", "50 г/т", "465 мм".
Если численные значения важны, объединяй их в узел type=parameter с предметным названием: "Крупность / размер частиц", "Расход реагента / дозировка", "pH / кислотность".

Верни JSON:
{{
  "nodes": [{{"label": "...", "type": "material|process|reagent|property|metric|parameter|equipment|risk|observation", "summary": "..."}}],
  "edges": [{{"source": "label узла", "target": "label узла", "relation": "influences|used_in|processed_by|measured_by|constrained_by|associated_with", "evidence": "короткий фрагмент", "confidence": 0.0}}]
}}

Текст:
{document_text[: self.settings.max_context_chars]}
""".strip()
        instructions = (
            "Извлекай только явно поддержанные текстом связи. Источник может быть multilingual: русский, английский, "
            "китайский или другой язык. Не теряй оригинальные технические термины, но summary пиши по-русски. "
            "Ответ строго JSON без markdown."
        )
        try:
            text = self._call_text(instructions=instructions, prompt=prompt, max_output_tokens=3000)
            payload = _extract_json(text)
            raw_nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
            raw_edges = payload.get("edges", []) if isinstance(payload, dict) else []
            label_to_id: dict[str, str] = {}
            nodes: list[dict[str, Any]] = []
            for raw in raw_nodes[:48]:
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("label") or "").strip()
                kind = str(raw.get("type") or "concept").strip().lower()
                if not label:
                    continue
                if is_low_signal_numeric_label(label):
                    continue
                nid = stable_id(kind, label)
                label_to_id[label.lower()] = nid
                nodes.append(
                    {
                        "id": nid,
                        "label": label[:100],
                        "type": kind,
                        "summary": str(raw.get("summary") or "")[:400],
                        "weight": 2.8,
                        "source_ids": [source_id],
                    }
                )
            edges: list[dict[str, Any]] = []
            for raw in raw_edges[:80]:
                if not isinstance(raw, dict):
                    continue
                source_label = str(raw.get("source") or "").strip().lower()
                target_label = str(raw.get("target") or "").strip().lower()
                source = label_to_id.get(source_label)
                target = label_to_id.get(target_label)
                if not source or not target or source == target:
                    continue
                relation = str(raw.get("relation") or "associated_with").strip()
                confidence = _as_float(raw.get("confidence"), 0.65)
                edges.append(
                    {
                        "id": edge_id(source, target, relation),
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "evidence": str(raw.get("evidence") or "")[:240],
                        "weight": 1.5 + confidence,
                        "source_ids": [source_id],
                    }
                )
            return nodes, edges, {"mode": "openrouter", "model": self.settings.openrouter_model}
        except Exception as exc:  # noqa: BLE001
            return [], [], {"mode": "failed", "reason": str(exc)}

    def describe_image(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        ocr_languages: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            return "", {"mode": "skipped", "reason": "OPENROUTER_API_KEY не задан"}

        mime_type = _image_mime_type(filename, content_type)
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        prompt = f"""
Проанализируй изображение как научно-технический источник для R&D проекта по обогащению и металлургии.

Файл: {filename}
Ожидаемые языки/письменности источника: {_vision_language_hint(ocr_languages)}

Нужно извлечь максимум полезного текста и предметного смысла для дальнейшей генерации гипотез и графа знаний:
- видимый текст может быть на русском, английском, китайском или другом языке;
- тип изображения: схема, регламент, список оборудования, таблица, скриншот или другое;
- видимые подписи, параметры, численные значения, названия стадий, реагентов, оборудования и потоков;
- связи между объектами: что подается, куда идет, чем измеряется, что влияет на результат;
- риски, ограничения, наблюдения и возможные исследовательские зацепки;
- если часть текста нечитабельна, явно отметь это как неопределенность.

Верни компактный структурированный текст на русском языке без JSON и без markdown-таблиц.
Иностранные технические термины, китайские названия и маркировки сохрани в оригинале рядом с русской интерпретацией.
""".strip()
        instructions = (
            "Ты анализируешь промышленные схемы, регламенты и научно-технические изображения. "
            "Поддерживай multilingual OCR/vision: русский, английский, китайский и другие языки. "
            "Не придумывай невидимые подписи. Если элемент неразборчив, отмечай неопределенность."
        )
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded_image}",
                        "detail": self.settings.openrouter_vision_detail,
                    },
                ],
            }
        ]
        try:
            text = self._call_response_text(
                instructions=instructions,
                input_payload=input_payload,
                max_output_tokens=2200,
            ).strip()
            if not text:
                raise ValueError("OpenRouter image analysis returned empty text")
            return text, {
                "mode": "openrouter_vision",
                "model": self.settings.openrouter_model,
                "content_type": mime_type,
                "ocr_languages": ocr_languages or self.settings.pdf_ocr_languages,
            }
        except Exception as exc:  # noqa: BLE001
            return "", {"mode": "failed", "model": self.settings.openrouter_model, "reason": str(exc)}

    def chat(
        self,
        project: dict[str, Any],
        hypotheses: list[dict[str, Any]],
        graph: dict[str, list[dict[str, Any]]],
        feedback: list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        message: str,
    ) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            raise OpenRouterServiceError("OPENROUTER_API_KEY не задан. Чат требует OpenRouter.")

        prompt = _build_chat_prompt(project, hypotheses, graph, feedback, chat_history, message, self.settings.max_context_chars)
        instructions = (
            "Ты интерактивный научный ассистент проекта. Отвечай по-русски, кратко, предметно. "
            "Единственный источник правды о существующих гипотезах — блок «Гипотезы проекта» во входных данных. "
            "Он отражает актуальное состояние и имеет приоритет над всем, что упоминалось в истории чата ранее. "
            "Не выдумывай гипотезы, не восстанавливай их из прошлых сообщений и не выдавай узлы графа или идеи из документов за существующие гипотезы. "
            "Если в блоке указано, что гипотез нет, честно сообщи, что в проекте пока нет ни одной гипотезы, "
            "и предложи запустить генерацию — но не перечисляй никаких гипотез. "
            "Если эксперт корректирует систему, явно сформулируй, как это должно повлиять на следующие гипотезы, критерии или граф знаний. "
            "Не притворяйся, что провел лабораторные опыты."
        )
        try:
            text = self._call_text(instructions=instructions, prompt=prompt, max_output_tokens=2200)
            return text.strip(), {"mode": "openrouter", "model": self.settings.openrouter_model}
        except Exception as exc:  # noqa: BLE001
            raise OpenRouterServiceError(f"OpenRouter chat failed ({exc.__class__.__name__}): {exc}") from exc

    def check_connection(self) -> dict[str, Any]:
        if not self.enabled:
            raise OpenRouterServiceError("OPENROUTER_API_KEY не задан. Проверка требует API-ключ.")
        try:
            text = self._call_text(
                instructions="Answer with one short word: ok.",
                prompt="Connectivity check.",
                max_output_tokens=16,
            )
            return {
                "ok": True,
                "model": self.settings.openrouter_model,
                "base_url": OPENROUTER_API_URL,
                "output": text[:80],
            }
        except Exception as exc:  # noqa: BLE001
            raise OpenRouterServiceError(f"OpenRouter connectivity check failed ({exc.__class__.__name__}): {exc}") from exc

    def _call_text(
        self,
        instructions: str,
        prompt: str,
        max_output_tokens: int,
        text_format: dict[str, Any] | None = None,
    ) -> str:
        return self._call_response_text(
            instructions=instructions,
            input_payload=prompt,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )

    def _call_response_text(
        self,
        instructions: str,
        input_payload: Any,
        max_output_tokens: int,
        text_format: dict[str, Any] | None = None,
    ) -> str:
        response = self._call_response(
            instructions=instructions,
            input_payload=input_payload,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return _response_to_text(response)

    def _call_response(
        self,
        instructions: str,
        input_payload: Any,
        max_output_tokens: int,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        include: list[str] | None = None,
        text_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        # OpenRouter exposes an OpenAI-compatible client surface. All traffic is
        # explicitly pinned to OpenRouter's endpoint and authenticated with its key.
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": self.settings.openrouter_api_key,
            "base_url": OPENROUTER_API_URL,
            "timeout": self.settings.openrouter_timeout,
            "max_retries": self.settings.openrouter_max_retries,
        }
        client = OpenAI(**kwargs)
        request: dict[str, Any] = {
            "model": model or self.settings.openrouter_model,
            "instructions": instructions,
            "input": input_payload,
            "max_output_tokens": max_output_tokens,
        }
        if tools:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if include:
            request["include"] = include
        if text_format:
            request["text"] = {"format": text_format}
        if extra_body:
            request["extra_body"] = extra_body
        return client.responses.create(**request)


def _build_hypothesis_prompt(
    project: dict[str, Any],
    documents: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
    feedback: list[dict[str, Any]],
    count: int,
    weights: dict[str, float],
    exclusions: list[str],
    include_roadmap: bool,
    max_chars: int,
    research_context: dict[str, Any] | None = None,
) -> str:
    docs_context = _documents_context(documents, max_chars)
    graph_context = _graph_context(graph)
    feedback_context = _feedback_context(feedback)
    research_text = _research_context_text(research_context)
    return f"""
Проект: {project.get("name")}
Домен: {project.get("domain")}
Цель/KPI: {project.get("goal") or "не задано"}
Ограничения: {project.get("constraints") or "не заданы"}
Веса ранжирования: {weights}
Исключить направления: {", ".join(exclusions) if exclusions else "нет"}
Нужна дорожная карта: {include_roadmap}

Контекст источников:
{docs_context}

Сводка графа знаний:
{graph_context}

Актуальный web research:
{research_text}

Экспертный фидбэк:
{feedback_context}

Сгенерируй {count} гипотез как структурированные мини-отчеты для экспертного чтения.
Каждая гипотеза должна быть проверяемой лабораторно, с механизмом, рисками, ожидаемой ценностью и источниками из контекста.
Используй экспертный фидбэк как механизм самоулучшения: усиливай паттерны из liked/высоко оцененных гипотез,
а признаки disliked/низко оцененных гипотез избегай или явно исправляй в новых формулировках.
Если web research включен, в описании гипотез подчеркни актуальную новизну относительно найденных статей,
патентов или промышленных практик и добавь реальные URL в evidence.url.
Отдельно выдели план внедрения в roadmap и возможные экономические расчеты в economics.
Не оставляй roadmap и economics пустыми: если точных чисел нет, дай порядок оценки, формулу, допущения и какие данные нужно собрать.
Ссылки без контекста запрещены: для каждого URL объясни, какой факт/патент/исследование он подтверждает и как это влияет на гипотезу.

Критически важно для прозрачного ранжирования:
- novelty, feasibility, impact и risk верни целыми числами от 0 до 100;
- novelty: 0 = тривиально, 100 = принципиально новое направление;
- feasibility: 0 = практически нереализуемо, 100 = легко проверить имеющимися средствами;
- impact: 0 = нет ожидаемого эффекта, 100 = высокий технологический/экономический эффект;
- risk: 0 = низкий риск, 100 = высокий риск провала, безопасности, CAPEX/OPEX или внедрения;
- score не возвращай: система пересчитает его сама по указанным весам.

JSON-схема ответа:
{{
  "hypotheses": [
    {{
      "title": "короткое название",
      "statement": "проверяемое утверждение в формате если/то",
      "rationale": "структурированное резюме: проблема, идея решения, ожидаемый технологический эффект, почему это не очевидный baseline",
      "mechanism": "ожидаемый механизм влияния: физико-химическая, технологическая или организационная причинность",
      "novelty": 0,
      "feasibility": 0,
      "impact": 0,
      "risk": 0,
      "uncertainty": "ключевые неизвестные, риски интерпретации и какие измерения их снимут",
      "evidence": [{{"source": "имя файла, статья или патент", "quote": "короткая цитата или пересказ", "why": "что именно подтверждает источник и почему это важно для гипотезы", "url": "https://..."}}],
      "roadmap": [{{"step": 1, "title": "шаг проверки или внедрения", "output": "артефакт, критерий или gate", "owner": "лаборатория|технолог|экономист|производство"}}],
      "economics": [{{"item": "показатель", "assumption": "допущение", "calculation": "формула или порядок оценки", "expected_effect": "диапазон эффекта", "confidence": "low|medium|high", "data_needed": "какие данные нужны для уточнения"}}]
    }}
  ]
}}
""".strip()


def _hypotheses_text_format() -> dict[str, Any]:
    evidence_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source": {"type": "string"},
            "quote": {"type": "string"},
            "why": {"type": "string"},
            "url": {"type": "string"},
            "kind": {"type": "string"},
        },
        "required": ["source", "quote", "why", "url", "kind"],
    }
    roadmap_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "step": {"type": "integer"},
            "title": {"type": "string"},
            "output": {"type": "string"},
            "owner": {"type": "string"},
        },
        "required": ["step", "title", "output", "owner"],
    }
    economics_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "item": {"type": "string"},
            "assumption": {"type": "string"},
            "calculation": {"type": "string"},
            "expected_effect": {"type": "string"},
            "confidence": {"type": "string"},
            "data_needed": {"type": "string"},
        },
        "required": ["item", "assumption", "calculation", "expected_effect", "confidence", "data_needed"],
    }
    hypothesis_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "statement": {"type": "string"},
            "rationale": {"type": "string"},
            "mechanism": {"type": "string"},
            "novelty": {"type": "integer"},
            "feasibility": {"type": "integer"},
            "impact": {"type": "integer"},
            "risk": {"type": "integer"},
            "uncertainty": {"type": "string"},
            "evidence": {"type": "array", "items": evidence_schema},
            "roadmap": {"type": "array", "items": roadmap_schema},
            "economics": {"type": "array", "items": economics_schema},
        },
        "required": [
            "title",
            "statement",
            "rationale",
            "mechanism",
            "novelty",
            "feasibility",
            "impact",
            "risk",
            "uncertainty",
            "evidence",
            "roadmap",
            "economics",
        ],
    }
    return {
        "type": "json_schema",
        "name": "hypothesis_report_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": hypothesis_schema,
                }
            },
            "required": ["hypotheses"],
        },
    }


def _build_research_prompt(
    project: dict[str, Any],
    documents: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
    query: str,
    max_chars: int,
) -> str:
    return f"""
Тема research:
{query}

Проект:
- название: {project.get("name")}
- домен: {project.get("domain")}
- цель/KPI: {project.get("goal") or "не задано"}
- ограничения: {project.get("constraints") or "не заданы"}

Локальный контекст источников:
{_documents_context(documents, max(4000, max_chars // 2))}

Сводка локального графа знаний:
{_graph_context(graph)}

Задача:
1. Найди актуальные внешние источники по теме: статьи, патенты, обзоры, промышленные кейсы, стандарты или данные производителей.
2. Ищи по русским, английским и при необходимости китайским терминам.
3. Сфокусируйся на том, что может усилить гипотезы: новизна, аналоги, ограничения, параметры внедрения, CAPEX/OPEX, ожидаемый эффект.
4. Верни компактный research brief на русском языке с разделами:
   - краткая картина области;
   - что выглядит новым или мало проверенным;
   - патенты/исследования/кейсы со ссылками;
   - что встроить в гипотезы;
   - возможные экономические вводные и ограничения.
5. Для каждого важного источника укажи URL и почему он релевантен.
""".strip()


def _build_chat_prompt(
    project: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
    feedback: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    message: str,
    max_chars: int,
) -> str:
    total_hypotheses = len(hypotheses)
    if total_hypotheses:
        top_hypotheses = "\n".join(
            f"- #{idx}: {item.get('title')} | score={item.get('score')} | status={item.get('status')}\n  {item.get('statement')}"
            for idx, item in enumerate(hypotheses[:8], start=1)
        )
        hypotheses_block = f"Всего гипотез в системе: {total_hypotheses}.\n{top_hypotheses}"
    else:
        hypotheses_block = "СПИСОК ПУСТ: в проекте сейчас 0 гипотез. Никаких существующих гипотез перечислять нельзя."
    history = "\n".join(f"{item.get('role')}({item.get('actor')}): {item.get('content')}" for item in chat_history[-10:])
    return f"""
Проект: {project.get("name")}
Цель: {project.get("goal") or "не задано"}
Ограничения: {project.get("constraints") or "не заданы"}

Гипотезы проекта (единственный источник правды, актуальное состояние):
{hypotheses_block}

Граф:
{_graph_context(graph)}

Фидбэк:
{_feedback_context(feedback)}

История чата:
{history[-max_chars // 3:]}

Сообщение пользователя:
{message}
""".strip()


def _documents_context(documents: list[dict[str, Any]], max_chars: int) -> str:
    chunks: list[str] = []
    budget = max_chars
    for doc in documents[:10]:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        excerpt = text[: min(3500, budget)]
        chunks.append(f"### {doc.get('filename')}\n{excerpt}")
        budget -= len(excerpt)
        if budget <= 1000:
            break
    return "\n\n".join(chunks) or "Источники пока не загружены."


def _graph_context(graph: dict[str, list[dict[str, Any]]]) -> str:
    nodes = graph.get("nodes", [])[:28]
    edges = graph.get("edges", [])[:36]
    node_lines = [f"{node.get('label')} ({node.get('type')}, w={node.get('weight')})" for node in nodes]
    id_to_label = {node.get("id"): node.get("label") for node in graph.get("nodes", [])}
    edge_lines = [
        f"{id_to_label.get(edge.get('source'), edge.get('source'))} -[{edge.get('relation')}]-> {id_to_label.get(edge.get('target'), edge.get('target'))}"
        for edge in edges
    ]
    return "Узлы: " + "; ".join(node_lines) + "\nСвязи: " + "; ".join(edge_lines)


def _feedback_context(feedback: list[dict[str, Any]]) -> str:
    if not feedback:
        return "Нет."
    latest_reactions: dict[tuple[str, str], dict[str, Any]] = {}
    notes: list[str] = []
    for item in feedback[:80]:
        actor = str(item.get("actor") or "expert")
        hypothesis_id = str(item.get("hypothesis_id") or "project")
        outcome = str(item.get("outcome") or "").strip().lower()
        title = str(item.get("hypothesis_title") or hypothesis_id or "проект")
        comment = str(item.get("comment") or "").strip()
        rating = item.get("rating")

        if outcome in {"liked", "like", "disliked", "dislike", "neutral", "reaction_removed"}:
            key = (hypothesis_id, actor)
            if key not in latest_reactions:
                latest_reactions[key] = {"outcome": outcome, "title": title, "actor": actor}
            continue

        if comment.startswith("quick_reaction:"):
            continue
        notes.append(f"- {actor}: hypothesis={title[:120]}, rating={rating}, outcome={outcome or '-'}, comment={comment[:500] or '-'}")
        if len(notes) >= 12:
            break

    reaction_summary: dict[str, dict[str, Any]] = {}
    for reaction in latest_reactions.values():
        outcome = reaction["outcome"]
        if outcome in {"neutral", "reaction_removed"}:
            continue
        title = reaction["title"]
        bucket = reaction_summary.setdefault(title, {"liked": 0, "disliked": 0})
        if outcome in {"liked", "like"}:
            bucket["liked"] += 1
        elif outcome in {"disliked", "dislike"}:
            bucket["disliked"] += 1

    lines: list[str] = []
    if reaction_summary:
        lines.append("Быстрые реакции пользователей (учитывать как обучающий сигнал):")
        for title, counts in list(reaction_summary.items())[:12]:
            lines.append(f"- {title[:140]}: liked={counts['liked']}, disliked={counts['disliked']}")
    if notes:
        lines.append("Развернутые экспертные замечания:")
        lines.extend(notes)
    return "\n".join(lines) if lines else "Нет содержательного фидбэка."


def _default_research_query(project: dict[str, Any], graph: dict[str, list[dict[str, Any]]]) -> str:
    labels = [str(node.get("label") or "") for node in graph.get("nodes", [])[:14] if node.get("label")]
    parts = [
        str(project.get("domain") or "обогащение и металлургия"),
        str(project.get("goal") or ""),
        ", ".join(labels),
    ]
    return " ".join(part for part in parts if part).strip() or "актуальные исследования и патенты для R&D гипотез"


def _research_context_text(research_context: dict[str, Any] | None) -> str:
    if not research_context:
        return "Не включен."
    lines = [
        f"Запрос: {research_context.get('query') or '-'}",
        "",
        str(research_context.get("summary") or "").strip(),
    ]
    sources = research_context.get("sources") if isinstance(research_context.get("sources"), list) else []
    if sources:
        lines.extend(["", "Найденные URL-источники:"])
        for source in sources[:12]:
            if not isinstance(source, dict):
                continue
            title = source.get("title") or source.get("url") or "source"
            url = source.get("url") or ""
            lines.append(f"- {title}: {url}")
    return "\n".join(line for line in lines if line is not None).strip()


def _normalize_hypothesis(item: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    scale_factor = _metric_scale_factor(
        [item.get("novelty"), item.get("feasibility"), item.get("impact"), item.get("risk")]
    )
    novelty = _as_score_scale(item.get("novelty"), 50, scale_factor)
    feasibility = _as_score_scale(item.get("feasibility"), 50, scale_factor)
    impact = _as_score_scale(item.get("impact"), 50, scale_factor)
    risk = _as_score_scale(item.get("risk"), 50, scale_factor)
    score = score_hypothesis(novelty, feasibility, impact, risk, weights)
    return {
        "title": str(item.get("title") or "Гипотеза")[:180],
        "statement": str(item.get("statement") or "")[:1600],
        "rationale": str(item.get("rationale") or "")[:2200],
        "mechanism": str(item.get("mechanism") or "")[:1600],
        "novelty": novelty,
        "feasibility": feasibility,
        "impact": impact,
        "risk": risk,
        "score": score,
        "uncertainty": str(item.get("uncertainty") or "")[:1200],
        "evidence": _normalize_evidence(item.get("evidence")),
        "roadmap": _normalize_roadmap(item.get("roadmap")),
        "economics": _normalize_economics(item.get("economics")),
    }


def _normalize_evidence(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for raw in value[:8]:
        if isinstance(raw, dict):
            normalized.append(
                {
                    "source": str(raw.get("source") or raw.get("title") or "source")[:240],
                    "quote": str(raw.get("quote") or "")[:700],
                    "why": str(raw.get("why") or "")[:700],
                    "url": str(raw.get("url") or "")[:1000],
                    "kind": str(raw.get("kind") or raw.get("type") or "")[:80],
                }
            )
        elif raw:
            normalized.append({"source": "source", "quote": str(raw)[:700], "why": "", "url": ""})
    return normalized


def _normalize_roadmap(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value[:8], start=1):
        if isinstance(raw, dict):
            normalized.append(
                {
                    "step": raw.get("step") or index,
                    "title": str(raw.get("title") or "")[:300],
                    "output": str(raw.get("output") or "")[:500],
                    "owner": str(raw.get("owner") or "")[:120],
                }
            )
        elif raw:
            normalized.append({"step": index, "title": str(raw)[:300], "output": ""})
    return normalized


def _normalize_economics(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for raw in value[:6]:
        if isinstance(raw, dict):
            normalized.append(
                {
                    "item": str(raw.get("item") or raw.get("metric") or "оценка")[:240],
                    "assumption": str(raw.get("assumption") or "")[:600],
                    "calculation": str(raw.get("calculation") or "")[:700],
                    "expected_effect": str(raw.get("expected_effect") or raw.get("value") or "")[:500],
                    "confidence": str(raw.get("confidence") or "")[:80],
                    "data_needed": str(raw.get("data_needed") or "")[:500],
                }
            )
        elif raw:
            normalized.append({"item": "оценка", "assumption": "", "calculation": str(raw)[:700], "expected_effect": "", "confidence": ""})
    return normalized


def _extract_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?", "", clean).strip()
    clean = re.sub(r"```$", "", clean).strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(clean[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


def _response_to_text(response: Any) -> str:
    try:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
    except Exception:  # noqa: BLE001
        pass
    return str(response)


def _response_citations(response: Any) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []

    def add(url: Any, title: Any = "", text: Any = "") -> None:
        url_text = str(url or "").strip()
        if not url_text:
            return
        citations.append(
            {
                "url": url_text[:1000],
                "title": str(title or url_text)[:300],
                "text": str(text or "")[:500],
            }
        )

    for item in _as_list(_get(response, "output")):
        item_type = _get(item, "type")
        if item_type == "web_search_call":
            action = _get(item, "action")
            for source in _as_list(_get(action, "sources")):
                add(_get(source, "url"), _get(source, "title"))
        for content in _as_list(_get(item, "content")):
            for annotation in _as_list(_get(content, "annotations")):
                if _get(annotation, "type") == "url_citation":
                    add(_get(annotation, "url"), _get(annotation, "title"), _get(annotation, "text"))

    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for citation in citations:
        key = citation["url"].split("#", 1)[0].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _metric_scale_factor(values: list[Any]) -> float:
    numeric_values: list[float] = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if numeric_values and all(0 <= value <= 1 for value in numeric_values):
        return 100.0
    if numeric_values and all(0 <= value <= 10 for value in numeric_values):
        return 10.0
    return 1.0


def _as_score_scale(value: Any, default: float, scale_factor: float) -> float:
    if value is None or value == "":
        return float(default)
    try:
        result = float(value) * scale_factor
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(100.0, result))


def _as_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return default


def _image_mime_type(filename: str, content_type: str | None) -> str:
    if content_type and content_type.startswith("image/"):
        return content_type
    guessed = mimetypes.guess_type(filename)[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


def _vision_language_hint(languages: list[str] | None) -> str:
    labels = {
        "ru": "русский/кириллица",
        "en": "английский/латиница",
        "ch_sim": "китайский упрощенный",
        "ch_tra": "китайский традиционный",
        "de": "немецкий",
        "fr": "французский",
        "es": "испанский",
    }
    selected = [labels.get(language, language) for language in languages or []]
    return ", ".join(selected) if selected else "авто: русский, английский, китайский; при необходимости другие языки"
