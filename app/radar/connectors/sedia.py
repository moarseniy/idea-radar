"""EU Funding & Tenders (SEDIA): конкурсы и тендеры ЕС.

Ответ поиска смешивает FAQ, объявления, конкурсы и тендеры; нужные отбираются по
``metadata.type``. Сверено с живым ответом (октябрь 2026): конкурс (topic) — type "1",
поля metadata.identifier, metadata.title, metadata.startDate; тендер — type "0". Каждый
конкурс приходит отдельной записью на каждом языке портала, поэтому склеиваем по identifier.
Текстовый поиск без кавычек ищет любое слово (сотни тысяч записей) — фраза в кавычках;
тип и язык фильтруются на сервере полем multipart ``query``: без него страница из 50 записей
весит 1,8 МБ и отвечает ~40 с, с ним — ~10 с.
Фильтр по дате на сервере не проверен, поэтому date_support = "post".
"""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

from app.radar.connectors import Connector
from app.radar.connectors._terms import near, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import _date_prefix, _document, _json_request, limiter

URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
CALL_TYPES = {"1", "2", "8"}  # конкурсы: topic, грант без темы, каскадное финансирование
TENDER_TYPES = {"0"}
QUERY = json.dumps({"bool": {"must": [{"terms": {"type": sorted(CALL_TYPES | TENDER_TYPES)}},
                                      {"terms": {"language": ["en"]}}]}})
PORTAL = "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/"

limiter("sedia", 1.0)


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _types(metadata: dict) -> set[str]:
    raw = metadata.get("type") or []
    return {str(value) for value in (raw if isinstance(raw, list) else [raw])}


def _day(metadata: dict) -> str | None:
    for field in ("startDate", "esST_publicationDate"):
        day = _date_prefix(_first(metadata.get(field)))
        if day:
            return day
    try:
        millis = int(_first(metadata.get("es_SortDate")))
        return datetime.fromtimestamp(millis / 1000, UTC).date().isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def search_sedia(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    refs: list[dict] = []
    seen: set[str] = set()
    for phrase in short_phrases(terms):
        payload = _json_request("sedia", URL, params={"apiKey": "SEDIA", "text": f'"{phrase}"',
                                                      "pageSize": 20, "pageNumber": 1},
                                files={"query": QUERY, "languages": '["en"]'}, timeout=30,
                                # Пустой ответ со сжатием сервер обрывает: Python видит
                                # «[SSL] record layer failure» (ReadError), curl — нет.
                                headers={"Accept-Encoding": "identity"})
        for item in (payload or {}).get("results", []):
            metadata = item.get("metadata") or {}
            types = _types(metadata)
            if types & CALL_TYPES:
                kind = "grant_call"
            elif types & TENDER_TYPES:
                kind = "tender"
            else:
                continue
            source_id = str(_first(metadata.get("identifier")) or item.get("reference") or "").strip()
            title = str(_first(metadata.get("title")) or item.get("title") or "").strip()
            day = _day(metadata)
            if not source_id or not title or not day or day > as_of.isoformat() or source_id in seen:
                continue
            summary = str(_first(metadata.get("callTitle")) or item.get("summary") or "").strip()
            if not near(f"{title} {summary}", terms):
                continue
            seen.add(source_id)
            label = "EU funding call" if kind == "grant_call" else "EU tender"
            text = f"{label}: {title}. Published: {day}. Reference: {source_id}.\n\n{summary}"
            event = (make_event(kind, day, source_id, funder="European Commission")
                     if kind == "grant_call" else
                     make_event(kind, day, source_id, customer="European Commission"))
            refs.append(_document(PORTAL + source_id, title, text, day, "en", "sedia_opportunity",
                                  structured_event=event, source_country="EU"))
    return refs[:limit]


CONNECTOR = Connector("EU Funding & Tenders", "money", "post",
                      lambda terms, as_of, limit: search_sedia(terms, as_of, limit=limit),
                      default_on=False, scopes=("branch",))
