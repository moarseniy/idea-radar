"""Веб-поиск радара через сервис поиска (app/search) вместо веб-плагина LLM.

Задача ветки из пайплайна → отдельные запросы по языкам → сервис поиска (провайдер из
SEARCH_PROVIDERS, по умолчанию SearXNG) → ссылки в формате ResearchProvider.search().
Если провайдер отдал текст страницы (Exa, OpenSERP с extract), он передаётся как документ,
и радар не скачивает страницу повторно; цитаты сверяются с этим текстом.
"""
from __future__ import annotations

import re
from datetime import date

from app.search.base import SearchQuery, language_of
from app.search.client import web_search
from app.search.config import load_search_settings
from app.radar.spoken_media import is_youtube_url

MAX_REFS = 18


def task_queries(task: dict, limit: int | None = None) -> list[dict]:
    """Запросы задачи: явный список queries или строки текста («Русский: …» → ru).

    Не больше limit языков (SEARCH_MAX_QUERIES): сначала русский и английский, затем остальные
    по порядку плана — их и так покрывают коннекторы Bing News и Crossref. Разные запросы одного
    языка (проверочные запросы по кандидатам) сохраняются все.
    """
    if task.get("queries"):
        out = [{"language": q.get("language") or language_of(q["query"]), "query": q["query"]}
               for q in task["queries"] if q.get("query", "").strip()]
    else:
        out = []
        for line in task["query"].splitlines():
            text = re.sub(r"^[^:\n]{2,24}:\s+", "", line.strip())  # префикс языка из плана
            if len(text) >= 3:
                out.append({"language": language_of(text), "query": text})
    priority_codes = list(dict.fromkeys(["ru", "en", *task.get("priority_languages", [])]))
    priority = {code: index for index, code in enumerate(priority_codes)}
    ordered = sorted(out, key=lambda q: priority.get(q["language"], len(priority)))  # стабильно для остальных
    if not limit:
        return ordered
    languages = list(dict.fromkeys(q["language"] for q in ordered))[:limit]
    return [q for q in ordered if q["language"] in languages]


def search_task(task: dict, as_of: str, per_query: int = 8) -> dict:
    """Результат в формате ResearchProvider.search(), с аудитом провайдеров."""
    refs, seen, audit, errors = [], set(), [], []
    max_queries = task.get("max_queries") or load_search_settings().max_queries
    queries = task_queries(task, max_queries)
    for q in queries:
        try:
            response = web_search(SearchQuery(text=q["query"][:400], as_of=date.fromisoformat(as_of),
                                              language=q["language"], limit=task.get("per_query", per_query)))
        except Exception as exc:  # noqa: BLE001 — сервис недоступен: ветка пустая, поиск продолжается
            errors.append(f"сервис поиска: {type(exc).__name__}")
            continue
        audit.extend(response.get("providers", []))
        for r in response.get("results", []):
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            youtube_video = is_youtube_url(r["url"])
            ref = {"url": r["url"], "title": r.get("title") or r["url"], "branch": task["branch"],
                   "query_language": q["language"],
                   "published_at": r.get("published_at"), "source_language": r.get("language"),
                   "source_provider": r.get("provider"),
                   "media_kind": "spoken" if youtube_video else task.get("media_kind")}
            if r.get("content") and len(r["content"]) > 300 and not youtube_video:
                ref["document"] = {"title": r.get("title") or r["url"], "text": f"{r.get('title', '')}\n\n{r['content']}",
                                   "date": r.get("published_at"), "language": r.get("language"),
                                   "extraction": f"{r.get('provider')}_content"}
            refs.append(ref)
    used = sorted({a["name"] for a in audit if a.get("count")})
    failed = [a["error"] for a in audit if a.get("error")]
    connector = task.get("connector_label") or f"Веб-поиск ({', '.join(used) or 'нет выдачи'})"
    ref_limit = max(1, min(40, int(task.get("max_refs", MAX_REFS))))
    return {"query": task["query"], "branch": task["branch"], "summary": "", "references": refs[:ref_limit],
            "connector": connector, "providers": audit, "query_count": len(queries),
            "query_languages": list(dict.fromkeys(q["language"] for q in queries)),
            "error": "; ".join(dict.fromkeys(errors + failed)) or None}
