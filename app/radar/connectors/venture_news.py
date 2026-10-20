"""TechCrunch и Crunchbase News через WordPress REST: новости о венчурных раундах.

Поиск идёт по короткой фразе (``search=``) в окне четыре года до даты оценки. Раунд
распознаётся по заголовку (``parse_round``): «X raises $20M Series A», «X lands $5 million
seed», «X secures €10M». Новость без распознанного раунда остаётся ссылкой без события.
"""
from __future__ import annotations

import html
import re
from datetime import date

from app.radar.connectors import Connector
from app.radar.connectors._terms import relevant, short_phrases
from app.radar.connectors.events import make_event
from app.radar.research_apis import _date_prefix, _document, _json_request, limiter

limiter("techcrunch", 1.0)
limiter("crunchbase_news", 1.0)

TECHCRUNCH_POSTS = "https://techcrunch.com/wp-json/wp/v2/posts"
CRUNCHBASE_POSTS = "https://news.crunchbase.com/wp-json/wp/v2/posts"
FIELDS = "date,link,title,excerpt"

CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP"}
MULTIPLIER = {"m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
ROUND_RE = re.compile(
    r"^(?P<company>[^:,|]+?)\s+(?:raises|lands|secures|closes|bags|nabs|snags|gets|scores)\s+"
    r"(?P<cur>[$€£])\s?(?P<num>\d+(?:\.\d+)?)\s?(?P<unit>million|billion|mn|bn|[mb])?\b"
    r"(?:\s+(?:in|of)\s+)?(?:\s*(?P<round>pre-seed|seed|series\s+[a-z]\d?))?",
    re.IGNORECASE,
)


def _clean(raw: str | None) -> str:
    """Текст из HTML WordPress: сущности раскрываются, теги убираются."""
    text = re.sub(r"<[^>]+>", " ", str(raw or ""))
    return " ".join(html.unescape(text).split())


def _round_name(raw: str | None) -> str | None:
    if not raw:
        return None
    word = " ".join(raw.lower().split())
    if word.startswith("series"):
        return "Series " + word.split()[1].upper()
    return word.capitalize()  # Seed, Pre-seed


def parse_round(title: str) -> tuple[str, int, str, str | None] | None:
    """(компания, сумма в валюте, код валюты, раунд) из заголовка; None, если раунда нет."""
    match = ROUND_RE.search(title.strip())
    if not match:
        return None
    unit = (match.group("unit") or "").lower()
    amount = round(float(match.group("num")) * MULTIPLIER.get(unit, 1))
    return (match.group("company").strip(), int(amount), CURRENCY[match.group("cur")],
            _round_name(match.group("round")))


def _search(url: str, provider: str, extraction: str, terms: list[str], as_of: date,
            limit: int) -> list[dict]:
    phrases = short_phrases(terms, limit=2)
    if not phrases:
        return []
    after = as_of.replace(year=as_of.year - 4)
    refs: dict[str, dict] = {}
    for phrase in phrases:
        params = {"search": phrase, "after": f"{after.isoformat()}T00:00:00",
                  "before": f"{as_of.isoformat()}T23:59:59", "per_page": 10, "_fields": FIELDS}
        response = _json_request(provider, url, params=params)
        for item in response if isinstance(response, list) else []:
            title = _clean((item.get("title") or {}).get("rendered"))
            text = _clean((item.get("excerpt") or {}).get("rendered"))
            link = str(item.get("link") or "")
            published = _date_prefix(item.get("date"))
            if not title or not link or not published or link in refs:
                continue
            if not relevant(f"{title} {text}", terms):
                continue
            meta = {}
            parsed = parse_round(title)
            if parsed:
                company, amount, currency, round_name = parsed
                meta["structured_event"] = make_event("funding_round", published, link, amount=amount,
                                                      currency=currency, company=company,
                                                      round_name=round_name)
            refs[link] = _document(link, title, f"{title}\n\n{text}", published, "en", extraction,
                                   source_country="US", **meta)
    return list(refs.values())[:limit]


def _techcrunch(terms: list[str], as_of: date, limit: int) -> list[dict]:
    return _search(TECHCRUNCH_POSTS, "techcrunch", "techcrunch_news", terms, as_of, limit)


def _crunchbase(terms: list[str], as_of: date, limit: int) -> list[dict]:
    return _search(CRUNCHBASE_POSTS, "crunchbase_news", "crunchbase_news", terms, as_of, limit)


CONNECTORS = (
    Connector("TechCrunch", "money", "strict", _techcrunch, default_on=False),
    Connector("Crunchbase News", "money", "strict", _crunchbase, default_on=False),
)
