"""Grants.gov: объявления федеральных грантов США (search2, без ключа).

Поиск не фильтрует по дате открытия, а параметр dateRange даёт только окна «за последние
N дней». Поэтому openDate (MM/DD/YYYY) отсекаем сами: date_support = "post".
Суммы в поиске нет (нужен fetchOpportunity), структурное событие без amount.
"""
from __future__ import annotations

from datetime import date

from app.radar.connectors import Connector
from app.radar.connectors._terms import relevant, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import _document, _json_request, limiter

URL = "https://api.grants.gov/v1/api/search2"
STATUSES = "forecasted|posted|closed|archived"

limiter("grants_gov", 1.0)


def _day(value) -> str | None:
    """MM/DD/YYYY из поиска -> YYYY-MM-DD; None для пустой или кривой даты."""
    parts = str(value or "").strip().split("/")
    try:
        month, day, year = (int(part) for part in parts)
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def search_grants_gov(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    refs: list[dict] = []
    seen: set[str] = set()
    for phrase in short_phrases(terms):
        payload = _json_request("grants_gov", URL, body={"keyword": phrase, "rows": 25,
                                                         "oppStatuses": STATUSES})
        for hit in ((payload or {}).get("data") or {}).get("oppHits", []):
            opp_id = str(hit.get("id") or "").strip()
            title = str(hit.get("title") or "").strip()
            opened = _day(hit.get("openDate"))
            if not opp_id or not title or not opened or opened > as_of.isoformat() or opp_id in seen:
                continue
            agency = str(hit.get("agency") or "").strip()
            if not relevant(f"{title} {agency}", terms):
                continue
            seen.add(opp_id)
            number = str(hit.get("number") or "").strip()
            closes = _day(hit.get("closeDate")) or "unknown"
            text = (f"US federal funding opportunity. Title: {title}. Agency: {agency or 'unknown'}. "
                    f"Number: {number}. Status: {hit.get('oppStatus') or 'unknown'}. "
                    f"Opened: {opened}. Closes: {closes}.")
            refs.append(_document(
                f"https://www.grants.gov/search-results-detail/{opp_id}", title, text, opened, "en",
                "grants_gov_opportunity",
                structured_event=make_event("grant_call", opened, opp_id, funder=agency or None),
                source_country="US"))
    return refs[:limit]


CONNECTOR = Connector("Grants.gov", "money", "post",
                      lambda terms, as_of, limit: search_grants_gov(terms, as_of, limit=limit),
                      default_on=False)
