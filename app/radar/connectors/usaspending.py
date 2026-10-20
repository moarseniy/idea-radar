"""USAspending: федеральные контракты и гранты США (ключ API не нужен).

Поиск по короткой фразе из названия кандидата. Дата отсекается на стороне API: конец
диапазона — as_of, поэтому date_support="strict". Описания в USAspending часто общие
("Cisco switches"), поэтому каждую запись дополнительно проверяем relevant() по тексту.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from app.radar.connectors import Connector
from app.radar.connectors._terms import near, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import _date_prefix, _document, _json_request, limiter

URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
CONTRACT_CODES = ["A", "B", "C", "D"]
GRANT_CODES = ["02", "03", "04", "05"]
FIELDS = ["Award ID", "Recipient Name", "Award Amount", "Start Date", "Description",
          "Awarding Agency", "generated_internal_id"]
TITLE_LIMIT = 140
limiter("usaspending", 0.5)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _unhyphen(text: str) -> str:
    # relevant() не склеивает дефис: "SOLID STATE" и "solid-state" иначе не совпадут.
    return text.replace("-", " ")


def _rows(phrases: list[str], as_of: date, limit: int, codes: list[str]) -> list[dict]:
    """По одному запросу на фразу; записи объединяются по Award ID."""
    start = as_of - timedelta(days=365 * 5)
    found: dict[str, dict] = {}
    for phrase in phrases:
        body = {"filters": {"keywords": [phrase], "award_type_codes": codes,
                            "time_period": [{"start_date": start.isoformat(),
                                             "end_date": as_of.isoformat()}]},
                "fields": FIELDS, "limit": max(1, min(limit, 10)), "page": 1,
                "sort": "Award Amount", "order": "desc"}
        response = _json_request("usaspending", URL, body=body)
        for row in response.get("results", []):
            key = _clean(row.get("Award ID")) or _clean(row.get("generated_internal_id"))
            if key:
                found.setdefault(key, row)
    return list(found.values())


def _search(kind: str, codes: list[str]):
    def search(terms: list[str], as_of: date, limit: int) -> list[dict]:
        phrases = short_phrases(terms, limit=2)
        if not phrases:
            return []
        needles = [_unhyphen(phrase) for phrase in phrases]
        refs = []
        for row in _rows(phrases, as_of, limit, codes):
            award_id = _clean(row.get("Award ID"))
            internal_id = _clean(row.get("generated_internal_id"))
            description = _clean(row.get("Description"))
            recipient = _clean(row.get("Recipient Name"))
            agency = _clean(row.get("Awarding Agency"))
            awarded = _date_prefix(row.get("Start Date"))
            if not (award_id and internal_id and description and awarded):
                continue
            if awarded > as_of.isoformat() or not near(_unhyphen(description), needles):
                continue
            amount = row.get("Award Amount")
            url = f"https://www.usaspending.gov/award/{internal_id}"
            title = description[:TITLE_LIMIT]
            text = (f"US federal {'contract' if kind == 'contract' else 'grant'}. Recipient: {recipient}. "
                    f"Awarding agency: {agency or 'unknown'}. Start date: {awarded}. "
                    f"Award amount (USD): {amount if amount is not None else 'unknown'}. "
                    f"Description: {description}")
            if kind == "contract":
                event = make_event("contract", awarded, award_id, amount=amount, company=recipient,
                                   customer=agency or None)
            else:
                event = make_event("grant", awarded, award_id, amount=amount, company=recipient,
                                   funder=agency or None)
            refs.append(_document(url, title, text, awarded, "en", "usaspending_award",
                                  structured_event=event, source_country="US"))
            if len(refs) >= limit:
                break
        return refs
    return search


CONTRACT_SEARCH = _search("contract", CONTRACT_CODES)
GRANT_SEARCH = _search("grant", GRANT_CODES)

CONNECTORS = (
    Connector("USAspending contracts", "money", "strict", CONTRACT_SEARCH, default_on=False),
    Connector("USAspending grants", "money", "strict", GRANT_SEARCH, default_on=False),
)
