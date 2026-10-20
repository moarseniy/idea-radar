"""NSF Awards: выданные гранты Национального научного фонда США (USA, открытый API без ключа)."""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.radar.connectors import Connector
from app.radar.connectors._terms import near, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import _document, _json_request, limiter

API = "https://api.nsf.gov/services/v1/awards.json"
FIELDS = "id,title,date,startDate,fundsObligatedAmt,awardeeName,abstractText"
YEARS = 5

limiter("nsf", 0.5)


def _us_date(value: Any) -> str | None:
    """Дата NSF в формате MM/DD/YYYY → ISO; None для пустой или кривой даты."""
    match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", str(value or "").strip())
    if not match:
        return None
    month, day, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _years_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 февраля
        return day.replace(year=day.year - years, day=28)


def _awards(phrase: str, as_of: date) -> list[dict]:
    params = {"keyword": f'"{phrase}"', "dateStart": _years_before(as_of, YEARS).strftime("%m/%d/%Y"),
              "dateEnd": as_of.strftime("%m/%d/%Y"), "printFields": FIELDS, "rpp": 25}
    response = _json_request("nsf", API, params=params, timeout=25) or {}
    return (response.get("response") or {}).get("award") or []


def search_nsf_awards(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    seen: dict[str, dict] = {}
    for phrase in short_phrases(terms, limit=2):
        for award in _awards(phrase, as_of):
            seen.setdefault(str(award.get("id") or ""), award)
    refs = []
    for award_id, award in seen.items():
        title = str(award.get("title") or "").strip()
        abstract = str(award.get("abstractText") or "").strip()
        awarded = _us_date(award.get("date"))
        org = str(award.get("awardeeName") or "").strip()
        if not award_id or not title or not awarded:
            continue
        if not near(f"{title} {abstract}", terms):
            continue
        amount = award.get("fundsObligatedAmt")
        text = (f"Federal research award. Award: {title}. Awardee: {org or 'unknown'}. "
                f"Award date: {awarded}. Obligated amount (USD): {amount or 'unknown'}.\n\n{abstract}")
        refs.append(_document(
            f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={award_id}", title, text, awarded, "en",
            "nsf_award",
            structured_event=make_event("grant", awarded, award_id, amount=amount, company=org,
                                        funder="NSF"),
            source_country="US"))
    return refs[:limit]


CONNECTOR = Connector("NSF awards", "money", "strict",
                      lambda terms, as_of, limit: search_nsf_awards(terms, as_of, limit=limit),
                      default_on=False)
