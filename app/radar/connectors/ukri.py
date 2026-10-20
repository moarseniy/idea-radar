"""UKRI Gateway to Research: гранты британских исследовательских советов (GBP).

Серверного фильтра по дате у API нет, поэтому источник с date_support="post": проекты,
начатые после as_of, отсекает Connector.run. Дата начала и сумма лежат не в проекте
(у него ``start`` пустой), а в связанном фонде: дата в ссылке FUND, сумма в ``valuePounds``.
Ведущая организация — по ссылке LEAD_ORG. Поэтому на каждый отобранный проект
делаются ещё два GET; до них отсекаем нерелевантные и вне даты записи.
"""
from __future__ import annotations

import html
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

from app.radar.connectors import Connector
from app.radar.connectors._terms import near, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import _document, _json_request, limiter

PROVIDER = "ukri"
API = "https://gtr.ukri.org/gtr/api"
HEADERS = {"Accept": "application/vnd.rcuk.gtr.json-v7"}
LONDON = ZoneInfo("Europe/London")  # даты фондов в API — миллисекунды; день берём по Лондону
limiter(PROVIDER, 0.5)


def _link(project: dict, rel: str) -> dict | None:
    links = (project.get("links") or {}).get("link") or []
    return next((link for link in links if link.get("rel") == rel), None)


def _https(href: str) -> str:
    return href.replace("http://", "https://", 1)


def _london_date(millis) -> str | None:
    try:
        return datetime.fromtimestamp(int(millis) / 1000, LONDON).date().isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _grant_reference(project: dict) -> str:
    identifiers = (project.get("identifiers") or {}).get("identifier") or []
    return next((str(item["value"]).strip() for item in identifiers
                 if item.get("type") == "RCUK" and item.get("value")), "")


def _project_ref(project: dict, phrases: list[str], as_of) -> dict | None:
    title = str(project.get("title") or "").strip()
    abstract = html.unescape(str(project.get("abstractText") or "")).strip()
    if not title or not near(f"{title} {abstract}", phrases):
        return None
    reference = _grant_reference(project)
    fund_link = _link(project, "FUND")
    awarded = _london_date(fund_link.get("start")) if fund_link else None
    if not reference or not awarded or awarded > as_of.isoformat():
        return None
    fund = _json_request(PROVIDER, _https(fund_link["href"]), headers=HEADERS) or {}
    value = fund.get("valuePounds") or {}
    org_link = _link(project, "LEAD_ORG")
    org = _json_request(PROVIDER, _https(org_link["href"]), headers=HEADERS) if org_link else {}
    company = str((org or {}).get("name") or "").strip()
    funder = str(project.get("leadFunder") or "").strip() or None
    url = f"https://gtr.ukri.org/projects?ref={quote(reference, safe='/')}"
    text = (f"UKRI research grant funded by {funder or 'UKRI'}. Project: {title}. "
            f"Lead organisation: {company or 'unknown'}. Start: {awarded}. "
            f"Value (GBP): {value.get('amount') or 'unknown'}.\n\n{abstract}")
    return _document(url, title, text, awarded, "en", "ukri_gtr_project",
                     structured_event=make_event("grant", awarded, reference,
                                                 amount=value.get("amount"), currency="GBP",
                                                 company=company, funder=funder),
                     source_country="GB")


def search_ukri(terms: list[str], as_of, limit: int) -> list[dict]:
    phrases = short_phrases(terms)
    refs: list[dict] = []
    seen: set[str] = set()
    for phrase in phrases:
        data = _json_request(PROVIDER, f"{API}/projects",
                             params={"q": phrase, "s": 25, "p": 1}, headers=HEADERS) or {}
        for project in data.get("project") or []:
            if len(refs) >= limit:
                return refs
            ref = _project_ref(project, phrases, as_of)
            if ref and ref["url"] not in seen:
                seen.add(ref["url"])
                refs.append(ref)
    return refs


CONNECTOR = Connector("UKRI GtR", "money", "post", search_ukri, default_on=False, scopes=("branch",))
