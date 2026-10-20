"""SEC EDGAR: полнотекстовый поиск по отчётности и Form D.

EDGAR FTS ищет фразу в тексте 10-K, 10-Q и 8-K. Публичные компании, которые упоминают
технологию, дают сырьё для признака «упоминают публичные компании» (kind filing_mention,
признаки его пока не считают). Form D — уведомление о частном размещении: берём только
факт и дату, сумму не тянем.

SEC требует User-Agent с контактом и не больше 10 запросов в секунду; держим 2 в секунду.
Без CONTACT_EMAIL SEC отвечает 403, поэтому коннектор выключен без него.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from app.radar.connectors import Connector
from app.radar.connectors._terms import relevant, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import (
    _date_prefix,
    _document,
    _json_request,
    contact_email,
    limiter,
)

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"
FILING_FORMS = "10-K,10-Q,8-K"
WINDOW_DAYS = 365  # отчёты за год до as_of

limiter("sec_edgar", 0.5)


def _headers() -> dict[str, str]:
    return {"User-Agent": f"IDEA-radar/0.1 ({contact_email()})"}


def _params(query: str, as_of: date, forms: str) -> dict[str, Any]:
    return {"q": f'"{query}"', "forms": forms, "dateRange": "custom",
            "startdt": (as_of - timedelta(days=WINDOW_DAYS)).isoformat(),
            "enddt": as_of.isoformat()}


def _rows(payload: Any) -> list[dict]:
    """Строки выдачи: _source плюс _id и highlight, нужные для ссылки и текста."""
    if not isinstance(payload, dict):
        return []
    rows = []
    for hit in (payload.get("hits") or {}).get("hits") or []:
        row = dict(hit.get("_source") or {})
        row["_id"] = str(hit.get("_id") or "")
        row["_highlight"] = hit.get("highlight") or {}
        rows.append(row)
    return rows


def _company(names: Any) -> str:
    """Название без тикера и CIK: "Acme Corp  (ACME)  (CIK 0001234)" → "Acme Corp"."""
    first = names[0] if isinstance(names, list) and names else names
    return " ".join(re.sub(r"\([^)]*\)", " ", str(first or "")).split())


def _cik(row: dict) -> str:
    if row.get("ciks"):
        return str(row["ciks"][0]).lstrip("0")
    match = re.search(r"CIK\s*0*(\d+)", str(row.get("display_names") or ""))
    return match.group(1) if match else ""


def _filing_url(row: dict) -> str:
    """Документ в архиве: имя файла — часть _id после двоеточия, adsh без дефисов."""
    cik, adsh = _cik(row), str(row.get("adsh") or "").replace("-", "")
    _, _, filename = row["_id"].partition(":")
    if not (cik and adsh and filename):
        return ""
    return f"{ARCHIVE_URL}/{cik}/{adsh}/{filename}"


def _text(row: dict, title: str) -> str:
    """Фрагменты из highlight, если API их отдал; иначе заголовок и метаданные."""
    fragments: list[str] = []
    for value in row["_highlight"].values():
        fragments.extend(value if isinstance(value, list) else [value])
    text = " ".join(re.sub(r"<[^>]+>", "", str(fragment)) for fragment in fragments)
    if text.strip():
        return " ".join(text.split())
    places = ", ".join(str(place) for place in row.get("biz_locations") or [])
    return " ".join(f"{title}. {row.get('file_description') or ''}. {places}".split())


def _filing_ref(row: dict, terms: list[str], phrase: str) -> dict | None:
    company = _company(row.get("display_names"))
    form = str(row.get("form") or "")
    title = " ".join(f"{company} {form} {row.get('file_description') or ''}".split())
    day = _date_prefix(row.get("file_date"))
    url = _filing_url(row)
    if not (title and day and url):
        return None
    text = _text(row, title)
    if row["_highlight"]:
        if not relevant(f"{title} {text}", terms):
            return None
    else:
        # Живой API фрагментов не отдаёт, а фразу в кавычках ищет по полному тексту
        # документа: совпадение уже гарантировано сервером, заголовок его не содержит.
        text = f'{text} The filing text contains the phrase "{phrase}".'
    
    ref = _document(url, title, text, day, "en", "sec_edgar_filing", form=form)
    ref["structured_event"] = make_event("filing_mention", day, row.get("adsh") or row["_id"],
                                         company=company)
    return ref


def search_filings(terms: list[str], as_of: date, limit: int) -> list[dict]:
    if not contact_email():
        return []
    refs: list[dict] = []
    seen: set[str] = set()
    for phrase in short_phrases(terms):
        payload = _json_request("sec_edgar", SEARCH_URL, params=_params(phrase, as_of, FILING_FORMS),
                                headers=_headers())
        for row in _rows(payload):
            ref = _filing_ref(row, terms, phrase)
            if ref and ref["url"] not in seen:
                seen.add(ref["url"])
                refs.append(ref)
        if len(refs) >= limit:
            break
    return refs[:limit]


def form_d_rounds(company: str, as_of: date, *, limit: int = 4) -> list[dict]:
    """Form D компании до as_of: факт и дата частного размещения, сумма пока не берётся."""
    name = " ".join(company.split())
    if not name or not contact_email():
        return []
    payload = _json_request("sec_edgar", SEARCH_URL, params=_params(name, as_of, "D"), headers=_headers())
    refs: list[dict] = []
    for row in _rows(payload):
        filer = _company(row.get("display_names"))
        day = _date_prefix(row.get("file_date"))
        url = _filing_url(row)
        if not (filer and day and url) or not relevant(filer, [name]):
            continue
        text = (f"{filer} filed Form D (notice of exempt securities offering) on {day}; "
                "the offering amount is not extracted.")
        ref = _document(url, f"{filer} Form D", text, day, "en", "sec_edgar_form_d")
        ref["structured_event"] = make_event("funding_round", day, row.get("adsh") or row["_id"],
                                             company=filer, round_name="Form D")
        refs.append(ref)
    return refs[:limit]


CONNECTOR = Connector("SEC EDGAR filings", "money", "strict", search_filings,
                      enabled=lambda _terms: bool(contact_email()), default_on=False)
