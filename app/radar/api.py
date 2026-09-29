from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.radar.analysis import RUBRIC_VERSION, WEIGHTS
from app.radar.config import ALLOWED_MODELS, load_radar_settings
from app.radar.dataset import read_catalog
from app.radar.models import (
    RegistryDomainInput,
    ScopeChatRequest,
    ScopeChatResponse,
    ScopeSuggestion,
    SearchRequest,
)
from app.radar.pipeline import make_provider

logger = logging.getLogger("idea.radar.api")


router = APIRouter(prefix="/api", tags=["Технологический радар"])


def require_run(request: Request, run_id: str):
    run = request.app.state.radar_store.get(run_id)
    if not run:
        raise HTTPException(404, "Запуск не найден")
    return run


def model_info(settings) -> dict:
    if not settings.use_model:
        return {"kind": "rubric"}
    try:
        from app.ml.scorer import default_scorer
        art = default_scorer().artifact
    except Exception as exc:  # noqa: BLE001
        return {"kind": "rubric", "error": type(exc).__name__}
    from app.ml.env import feature_llm, model_slug
    from app.ml.scorer import MODEL_PATH
    llm = feature_llm()["model"]
    info = {"kind": "model", "name": art["name"], "features": len(art["features"]),
            "threshold": art["threshold"], "validation": art.get("validation", {}),
            "artifact": MODEL_PATH.name, "feature_model": llm}
    if model_slug(llm) not in art.get("training_data", ""):
        # Признаки размечает одна LLM, а модель обучена на разметке другой — вероятности смещены.
        info["warning"] = (f"Модель скоринга обучена не на разметке {llm}; добавьте артефакт "
                           f"logreg_v1_{model_slug(llm)}.json или задайте SIGNAL_MODEL_PATH.")
    return info


def web_search_info(settings) -> dict:
    """Какой веб-поиск работает: сервис поиска (провайдеры цепочки), плагин LLM или никакой."""
    if settings.web_search != "service":
        return {"mode": settings.web_search}
    try:
        from app.search.config import load_search_settings
        s = load_search_settings()
        return {"mode": "service", "providers": list(s.providers), "chain": s.mode,
                "remote": bool(s.service_url)}
    except ValueError as exc:
        return {"mode": "service", "error": str(exc)}


@router.get("/sources/registry")
def sources_registry():
    """Реестр доверенных источников: типы, уровни доверия, правила и домены."""
    from app.registry import registry_table
    return registry_table()


@router.put("/sources/registry/domains")
def save_registry_domain(payload: RegistryDomainInput):
    """Create a domain or edit its display name, source type, and language."""
    from app.registry import save_domain
    try:
        return save_domain(payload.domain, payload.name, payload.type, payload.lang)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/sources/registry/domains/{domain}")
def remove_registry_domain(domain: str):
    """Remove a custom entry or hide a bundled entry from the effective registry."""
    from app.registry import delete_domain
    try:
        removed = delete_domain(domain)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not removed:
        raise HTTPException(404, "Домен не найден в реестре.")
    from app.registry import registry_table
    return registry_table()


@router.get("/radar/config")
def configuration(request: Request):
    settings = load_radar_settings()
    return {"ready": settings.configuration_error() is None, "message": settings.configuration_error(),
            "search_limits": {"branches": settings.branches, "sources": settings.max_sources,
                              "candidates": settings.max_candidates},
            "model": settings.active_model, "research_model": settings.active_research_model,
            "provider": settings.provider_label, "web_search": web_search_info(settings),
            "storage": request.app.state.radar_store.backend, "allowed_models": sorted(ALLOWED_MODELS),
            "rubric_version": RUBRIC_VERSION, "weights": WEIGHTS, "scoring": model_info(settings),
            "score_note": ("Скоринг — калиброванная вероятность обученной логистической регрессии."
                           if settings.use_model else
                           "Баллы отражают признаки по фиксированной рубрике, а не калиброванную вероятность."),
            "catalog_note": "Примеры организаторов не используются как готовые ответы живого поиска."}


@router.post("/search-scope/chat", response_model=ScopeChatResponse)
def search_scope_chat(payload: ScopeChatRequest, request: Request):
    settings = load_radar_settings()
    error = settings.configuration_error()
    if error:
        raise HTTPException(503, error)
    capacity = max(0, settings.branches - 1)
    selected = payload.selected_directions[:capacity]
    remaining = max(0, capacity - len(selected))
    history = [{"role": item.role, "content": item.content} for item in payload.history[-10:]]
    prompt = f"""Исходная тема исследования (данные): {json.dumps(payload.query, ensure_ascii=False)}
Подтверждённые пользователем направления: {json.dumps(selected, ensure_ascii=False)}
Свободно мест для дополнительных направлений: {remaining}.
История уточнения: {json.dumps(history, ensure_ascii=False)}
Новое сообщение пользователя: {json.dumps(payload.message, ensure_ascii=False)}

Помоги человеку уточнить область технологического поиска. Отвечай по-русски кратко и понятно.
Предлагай ровно 4 конкретных возможных фокуса только внутри исходной темы. Это варианты для выбора,
а не факты и не команды для поиска. Не добавляй новые технологии, компании или подотрасли без
основания в контексте. Не снимай и не изменяй уже подтверждённые направления.
Если тема и так понятна, скажи, что можно искать сразу, и предложи только действительно разные
фокусы. Варианты должны различаться по смыслу: где уместно, предлагай синоним или альтернативное
название, конкретное применение, механизм, задачу или стадию от исследований до внедрений.
Можно слегка расширить или сузить формулировку, но оставайся в исходной области и не выдумывай
несвязанные технологии. Не добавляй слова «слабые сигналы» или "weak signals" в поисковые строки.
Если пользователь просит не менять область, предложи разные переформулировки, а не четыре копии
одной темы с формальными суффиксами.
Текст каждой suggestion должен быть коротким названием/направлением; description — одна фраза,
поясняющая, что именно попадёт в поиск. Не возвращай пустой список.
"""
    provider = make_provider(settings, time.monotonic() + 60)
    try:
        result = provider.structured("scope_clarification", prompt, ScopeChatResponse, 2200)
        suggestions = list(result.suggestions[:4])
        seen = {item.title.casefold().strip() for item in suggestions}
        if len(suggestions) < 4:
            missing = 4 - len(suggestions)
            fallback_prompt = f"""Продолжи уточнение области поиска.
Исходный запрос: {json.dumps(payload.query, ensure_ascii=False)}
Последнее сообщение пользователя: {json.dumps(payload.message, ensure_ascii=False)}
Уже выбранные направления: {json.dumps(selected, ensure_ascii=False)}
Уже предложенные варианты: {json.dumps([item.title for item in suggestions], ensure_ascii=False)}
Верни ровно {missing} дополнительных коротких поисковых формулировок, которые сохраняют тот же
смысл исходного запроса, но могут немного обобщать или конкретизировать его. Используй разные
синонимы, применения, механизмы и стадии, а не одинаковый шаблон. Не добавляй несвязанные темы.
Не используй в поисковых формулировках «слабые сигналы» или "weak signals". Не возвращай пустой suggestions."""
            try:
                more = provider.structured("scope_suggestion_fallback", fallback_prompt,
                                           ScopeChatResponse, 1800)
                for item in more.suggestions:
                    key = item.title.casefold().strip()
                    if key not in seen:
                        suggestions.append(item)
                        seen.add(key)
                    if len(suggestions) == 4:
                        break
            except Exception as exc:  # noqa: BLE001 — deterministic same-scope variants are the last fallback
                logger.warning("Scope suggestion fallback failed: %s", type(exc).__name__)
                pass
        focus = payload.query.strip()[:110]
        template_items = [
            (f"Альтернативные названия и термины: {focus}", "Ищет синонимы и профессиональные формулировки той же области."),
            (f"Применения и решаемые задачи: {focus}", "Слегка конкретизирует область через реальные задачи и сценарии использования."),
            (f"Разработки, прототипы и пилоты: {focus}", "Охватывает технические работы и проверку решений на практике."),
            (f"Другие подходы и сравнения: {focus}", "Ищет близкие альтернативные и традиционные подходы в рамках той же задачи."),
        ]
        for title, description in template_items:
            key = title.casefold().strip()
            if key not in seen:
                suggestions.append(ScopeSuggestion(title=title, description=description))
                seen.add(key)
            if len(suggestions) == 4:
                break
        return ScopeChatResponse(reply=result.reply, suggestions=suggestions[:4])
    except Exception as exc:
        raise HTTPException(502, f"Не удалось уточнить тему через помощника ({type(exc).__name__}).") from exc
    finally:
        provider.close()


@router.get("/searches")
def list_searches(request: Request):
    fields = ["id", "query", "created_at", "updated_at", "status", "stage", "as_of", "elapsed_seconds", "counters"]
    return {"runs": [{**{k: r.get(k) for k in fields}, "signal_count": len(r["signals"])}
                     for r in request.app.state.radar_store.list_runs()]}


@router.post("/searches", status_code=202)
def create_search(payload: SearchRequest, request: Request):
    settings = load_radar_settings()
    error = settings.configuration_error()
    if error:
        raise HTTPException(503, error)
    if len(payload.directions) > max(0, settings.branches - 1):
        raise HTTPException(422, f"Выберите не больше {max(0, settings.branches - 1)} дополнительных направлений.")
    try:
        return request.app.state.radar_manager.start(payload, settings)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/searches/{run_id}")
def get_search(run_id: str, request: Request):
    return require_run(request, run_id)


@router.delete("/searches/{run_id}")
def delete_search(run_id: str, request: Request):
    run = require_run(request, run_id)
    if run.get("status") in {"queued", "running"}:
        raise HTTPException(409, "Дождитесь завершения поиска, прежде чем удалять исследование.")
    request.app.state.radar_store.delete_run(run_id)
    return {"deleted": True}


@router.post("/searches/{run_id}/cancel")
def cancel_search(run_id: str, request: Request):
    require_run(request, run_id)
    return {"cancellation_requested": request.app.state.radar_manager.cancel(run_id)}


@router.get("/searches/{run_id}/signals")
def signals(run_id: str, request: Request):
    run = require_run(request, run_id)
    return {"signals": run["signals"], "top_candidates": run.get("top_candidates", run["signals"]),
            "status": run["status"], "warnings": run["warnings"]}


@router.get("/searches/{run_id}/rejected")
def rejected(run_id: str, request: Request):
    return {"rejected": require_run(request, run_id)["rejected"]}


@router.get("/signals/{signal_id}")
def signal(signal_id: str, request: Request):
    run_id = signal_id.partition("_cand_")[0]
    run = require_run(request, run_id)
    item = next((s for s in run["assessments"] if s["id"] == signal_id), None)
    if not item:
        raise HTTPException(404, "Сигнал не найден")
    return {"signal": item, "sources": [s for s in run["sources"] if s["id"] in item["source_ids"]],
            "query": run["query"], "as_of": run["as_of"]}


@router.get("/searches/{run_id}/sources/{source_id}")
def source(run_id: str, source_id: str, request: Request):
    require_run(request, run_id)
    result = request.app.state.radar_store.source(run_id, source_id)
    if not result:
        raise HTTPException(404, "Источник не найден")
    return result


@router.get("/catalog")
def catalog():
    return read_catalog(load_radar_settings().data_dir)


def markdown_report(run: dict) -> str:
    top_candidates = run.get("top_candidates") or run["signals"]
    verdict_names = {"weak_signal": "Слабый сигнал", "mature": "Зрелая технология",
                     "hype_or_noise": "Шум или не технология", "irrelevant": "Не соответствует запросу",
                     "insufficient_evidence": "Недостаточно доказательств"}
    lines = ["# IDEA — технологический радар", "", f"Запрос: {run['query']}", f"Дата оценки: {run['as_of']}",
             f"Статус запуска: {run['status']}. Кандидатов в ТОП: {len(top_candidates)}/{run['limit']}. "
             f"Подтверждено слабых сигналов: {len(run['signals'])}.", "", run["score_note"], ""]
    source_map = {s["id"]: s for s in run["sources"]}
    for item in top_candidates:
        model = item.get("model")
        score = (f"Предварительная оценка логистической регрессии: {item['score']}%"
                 if item.get("score_kind") == "model_partial" else
                 f"Уверенность модели: {item['score']}%" if item.get("score_kind") == "model" else
                 f"Оценка: {item['score']}/100")
        verdict = item.get("verdict", "insufficient_evidence")
        lines.extend([f"## {item.get('rank', '-')}. {item['title']}", "",
                      f"Статус: {verdict_names.get(verdict, verdict)}. {score}. "
                      f"Стадия: {item.get('stage') or 'не установлена'}.", ""])
        if verdict != "weak_signal" and item.get("reason"):
            lines.extend([f"Причина статуса: {item['reason']}", ""])
        if model:
            lines.append("Ключевые признаки модели: " + "; ".join(
                f"{c['label']} = {c['display_value']} ({c['contribution']:+.2f})"
                for c in model["top_for"][:3] + model["top_against"][:2]))
            lines.append("")
        if item.get("related"):
            lines.extend(["Другие применения: " + "; ".join(f"{r['title']} ({r['score']}%)" for r in item["related"]), ""])
        for key, title in [("description", "Технология"), ("why_now", "Почему сейчас"), ("why_early", "Почему ранняя стадия"),
                           ("advantage", "Преимущество"), ("case", "Кейс")]:
            finding = item.get(key)
            if finding and finding.get("text"):
                lines.extend([f"### {title}", finding["text"], ""])
        lines.extend([f"Тип кейса: {item.get('case_kind', 'unknown')}", "", "### Основания"])
        for evidence in item.get("evidence", []):
            s = source_map.get(evidence["source_id"])
            if not s:
                continue
            lines.extend([f"- {evidence['claim']}", f"> {evidence['quote']}",
                          f"[{s['title']}]({s['url']}) — {s['published_at'] or 'дата не установлена'}, {s['language']}, {s['type']}", ""])
        limitations = item.get("limitations", [])
        if limitations:
            lines.extend(["### Ограничения", *[f"- {x}" for x in limitations], ""])
        lines.extend(["Русское резюме сформировано ИИ по указанным источникам.", ""])
    top_ids = {x["id"] for x in top_candidates}
    other_rejected = [x for x in run["rejected"] if x["id"] not in top_ids]
    lines.extend(["## Остальные исключённые кандидаты",
                  *[f"- {x['title']}: {x.get('reason', 'не прошёл критерии слабого сигнала')}"
                    for x in other_rejected], ""])
    lines.extend(["## Замечания к запуску", *[f"- {x}" for x in run["warnings"]]])
    return "\n".join(lines)


@router.get("/searches/{run_id}/export.{format}")
def export(run_id: str, format: str, request: Request):
    run = require_run(request, run_id)
    if format == "json":
        content, mime = json.dumps(run, ensure_ascii=False, indent=2), "application/json"
    elif format == "md":
        content, mime = markdown_report(run), "text/markdown"
    else:
        raise HTTPException(404, "Доступны форматы md и json")
    return Response(content, media_type=mime, headers={"Content-Disposition": f'attachment; filename="idea-{run_id}.{format}"'})
