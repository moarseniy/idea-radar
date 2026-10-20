"""Y Combinator: каталог стартапов по батчам (неофициальный API сайта, без ключа).

Дата записи — начало батча (W24 → 2024-01-01). Статус компании («Acquired», «Inactive»…)
текущий, поэтому в текст документа он не попадает: лежит отдельно в ``current_status``.
"""
from __future__ import annotations

import re
from datetime import date

from app.radar.connectors import Connector
from app.radar.connectors._terms import NEAR, relevant, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import _document, _json_request, limiter

API = "https://api.ycombinator.com/v0.1/companies"
MAX_PAGES = 2
_BATCH = re.compile(r"([WSFX])(\d{2})")
# Сезон батча → месяц и день начала: W — зима, S — лето, F — осень, X — весна (новое расписание).
_SEASON_START = {"W": (1, 1), "X": (4, 1), "S": (6, 1), "F": (9, 1)}

limiter("yc", 1.0)


def batch_date(batch: str | None) -> date | None:
    """Дата начала батча: ``W21`` → 2021-01-01; без распознанного кода — None."""
    match = _BATCH.fullmatch(str(batch or "").strip().upper())
    if not match:
        return None
    season, year = match.groups()
    month, day = _SEASON_START[season]
    return date(2000 + int(year), month, day)


def _ref(company: dict, as_of: date) -> dict | None:
    name = str(company.get("name") or "").strip()
    slug = str(company.get("slug") or "").strip()
    url = str(company.get("url") or "").strip()
    started = batch_date(company.get("batch"))
    if not (name and slug and url) or started is None or started > as_of:
        return None
    one_liner = str(company.get("oneLiner") or "").strip()
    title = f"{name} — {one_liner}" if one_liner else name
    description = str(company.get("longDescription") or "").strip()
    tags = ", ".join(str(tag) for tag in company.get("tags") or [])
    text = (f"Y Combinator startup {name}, batch {company['batch']}.\n\n{description}\n\n"
            f"Tags: {tags or 'none'}. Team size: {company.get('teamSize') or 'unknown'}.")
    return _document(url, title, text, started.isoformat(), "en", "yc_company",
                     current_status=company.get("status"),
                     structured_event=make_event("accelerator", started, slug, company=name,
                                                 funder="Y Combinator", round_name="accelerator"))


def search(terms: list[str], as_of: date, limit: int) -> list[dict]:
    refs: list[dict] = []
    seen: set[str] = set()
    for phrase in short_phrases(terms):
        page = 1
        while page <= MAX_PAGES and len(refs) < limit:
            data = _json_request("yc", API, params={"q": phrase, "page": page}) or {}
            for company in data.get("companies") or []:
                ref = _ref(company, as_of)
                if ref is None or ref["url"] in seen:
                    continue
                # Поиск YC смысловой и отдаёт просто похожие компании ("персональный
                # ИИ-агент для Кореи" на "agent identity"): слова полной фразы должны стоять
                # рядом. Без ядер из двух слов, как у грантов: "ai agent" подходит любой
                # агентной компании.
                if not relevant(f"{ref['title']}. {ref['document']['text']}", terms, window=NEAR):
                    continue
                seen.add(ref["url"])
                refs.append(ref)
            if not data.get("nextPage"):
                break
            page += 1
    return refs[:limit]


CONNECTOR = Connector("Y Combinator", "money", "post", search, default_on=False)
