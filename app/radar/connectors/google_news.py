"""Google News RSS: венчурные и деловые новости с суммами раундов (W1-2).

Три коннектора: венчурный — сделки в любом издании (событие ``funding_round`` при сумме
в заголовке), деловые издания РФ (суммы в млн/млрд, ₽ → RUB, $ → USD) и общий новостной поиск
(без событий, будущая замена Bing для машины времени).

Операторы ``site:`` и ``after:``/``before:`` Google применяет нестабильно: в живых
ответах попадались записи вне окна и с чужих доменов. Поэтому домен и дата дополнительно
проверяются здесь; ``date_support`` — ``post``, а ``Connector.run`` отсекает записи после as_of.
"""
from __future__ import annotations

import email.utils
import html
import re
from dataclasses import dataclass
from datetime import date, timedelta
from urllib.parse import urlparse

from app.radar.connectors import Connector
from app.radar.connectors._terms import (
    ReadyPhrases,
    key_tokens,
    relevant,
    short_phrases,
)
from app.radar.connectors.events import make_event
from app.radar.locales import LANGUAGE_MARKETS
from app.radar.research_apis import _document, _json_request, limiter

limiter("google_news", 1.0)

RSS_URL = "https://news.google.com/rss/search"
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
FUNDING_WORDS = 'raises OR funding OR seed OR "series a" OR "series b" OR acquires OR acquisition OR invests'
# Сделка в заголовке: раунд, покупка, инвестиция или сумма. Без списка изданий: о раундах
# узкой технологии пишут отраслевые издания (Help Net Security, Biometric Update, Finovate),
# а запрос с site: по семи венчурным изданиям и точной фразой был пуст на половине фраз.
DEAL = re.compile(r"\b(?:raise[sd]?|raising|funding|funded|seed|pre-seed|series [a-e]|acquir\w*"
                  r"|acquisition|buys|invest\w*|backed|valuation|round)\b|\$\s?\d", re.IGNORECASE)
NOT_DEAL = re.compile(r"\braise[sd]?\s+(?:the\s+)?(?:alarms?|concerns?|questions?|eyebrows|awareness"
                      r"|doubts?|fears?|bar|stakes|red flags?|prices?|rates?)\b", re.IGNORECASE)
RU_SITES = ("vc.ru", "rbc.ru", "cnews.ru", "tadviser.ru", "kommersant.ru", "rb.ru")

AMOUNT_USD = re.compile(r"\$\s?([\d.,]+)\s?(M|B|million|billion)\b", re.IGNORECASE)
AMOUNT_RU = re.compile(r"([$₽])?\s?(\d+(?:[.,]\d+)?)\s?(млн|млрд)\.?\s*(₽|руб\w*|долл\w*)?", re.IGNORECASE)
COMPANY = re.compile(
    r"^(.{2,60}?)\s+(?:raises|secures|closes|gets|lands|nabs|bags|receives|announces"
    r"|привлек\w*|получил\w*|собрал\w*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class _Feed:
    slug: str                      # source_provider и extraction записи
    hl: str
    gl: str
    ceid: str
    language: str
    sites: tuple[str, ...] = ()    # домены: попадают в запрос и проверяются по <source>
    extra: str = ""                # дополнительная группа слов в запросе (раунды)
    ru: bool = False               # русские термины и суммы млн/млрд
    events: bool = False           # есть ли structured_event при сумме в заголовке
    deals: bool = False            # только сделки: фраза и глагол сделки в заголовке
    raw: bool = False              # запрос ветки как есть, на любом языке (машина времени)


VENTURE = _Feed("google_news_venture", "en-US", "US", "US:en", "en",
                extra=FUNDING_WORDS, events=True, deals=True)
RU_BUSINESS = _Feed("google_news_ru", "ru", "RU", "RU:ru", "multi",
                    sites=RU_SITES, ru=True, events=True)
GENERAL = _Feed("google_news", "en-US", "US", "US:en", "en")


def _window(as_of: date) -> tuple[str, str]:
    """after = as_of − 4 года, before = as_of + 1 день (у Google before исключающий)."""
    try:
        after = as_of.replace(year=as_of.year - 4)
    except ValueError:  # 29 февраля
        after = as_of.replace(year=as_of.year - 4, day=28)
    return after.isoformat(), (as_of + timedelta(days=1)).isoformat()


def _phrases(feed: _Feed, terms: list[str]) -> list[str]:
    if feed.raw:
        return terms[:1]
    latin = short_phrases(terms, limit=2)
    ru = [term for term in terms if CYRILLIC.search(term)][:2] if feed.ru else []
    return latin + ru


def _variants(phrase: str) -> list[str]:
    """Фраза и её ядра из двух слов: "prompt injection detection" → "prompt injection"…

    Пресса пишет о сделке короче, чем называет подход план радара: раунд Lakera — "prompt
    injection", а не "prompt injection detection".
    """
    words = phrase.split()
    if CYRILLIC.search(phrase) or len(words) < 3:
        return [phrase]
    return list(dict.fromkeys([phrase, " ".join(words[:2]), " ".join(words[-2:])]))


def _is_deal(title: str, phrase: str) -> bool:
    return (relevant(title, [phrase], min_share=1.0) and bool(DEAL.search(title))
            and not NOT_DEAL.search(title))


def _query(feed: _Feed, phrase: str, after: str, before: str) -> str:
    # Русский запрос ветки длиннее английских фраз плана: точной фразой он пуст, словами —
    # находит; лишнее отсекает _ru_match по основам слов.
    parts = [phrase if feed.raw or CYRILLIC.search(phrase) else f'("{phrase}")']
    if feed.extra:
        parts.append(f"({feed.extra})")
    if feed.sites:
        parts.append("(" + " OR ".join(f"site:{site}" for site in feed.sites) + ")")
    parts += [f"after:{after}", f"before:{before}"]
    return " ".join(parts)


def _text(block: str, tag: str) -> str:
    match = re.search(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", block, re.DOTALL)
    return html.unescape(match.group(1)).strip() if match else ""


def _items(xml: str) -> list[dict]:
    """Разбор RSS: записи с датой; без даты запись не возвращаем."""
    rows = []
    for block in re.findall(r"<item>(.*?)</item>", xml, re.DOTALL):
        source = re.search(r'<source url="([^"]*)"[^>]*>(.*?)</source>', block, re.DOTALL)
        try:
            day = email.utils.parsedate_to_datetime(_text(block, "pubDate")).date().isoformat()
        except (TypeError, ValueError):
            continue
        publisher = html.unescape(source.group(2)).strip() if source else ""
        title = _text(block, "title")
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(publisher) - 3]
        # Описание приходит HTML-разметкой (ссылка и издание): оставляем только текст.
        plain = html.unescape(re.sub(r"<[^>]+>", " ", _text(block, "description")))
        rows.append({"title": title, "link": _text(block, "link"), "guid": _text(block, "guid"),
                     "date": day, "publisher": publisher, "text": re.sub(r"\s+", " ", plain).strip(),
                     "source_url": html.unescape(source.group(1)) if source else ""})
    return rows


def _on_sites(source_url: str, sites: tuple[str, ...]) -> bool:
    if not sites:
        return True
    host = urlparse(source_url).netloc.lower().removeprefix("www.")
    return any(host == domain or host.endswith("." + domain)
               for domain in (site.split("/")[0] for site in sites))


def _ru_match(text: str, phrase: str) -> bool:
    """Кириллица: половина слов фразы встречается по основе (первые 5 букв)."""
    low = text.lower()
    words = phrase.lower().split()
    return bool(words) and sum(word[:5] in low for word in words) / len(words) >= 0.5


def _matches(text: str, phrases: list[str]) -> bool:
    return relevant(text, phrases) or any(_ru_match(text, p) for p in phrases if CYRILLIC.search(p))


def _amount(title: str, ru: bool) -> tuple[float, str] | None:
    """Сумма в единицах валюты и код валюты; без валюты или суммы — None."""
    if ru:
        match = AMOUNT_RU.search(title)
        if not match:
            return None
        sign = match.group(1) or match.group(4) or ""
        if "$" in sign or sign.lower().startswith("долл"):
            currency = "USD"
        elif "₽" in sign or sign.lower().startswith("руб"):
            currency = "RUB"
        else:
            return None
        value = float(match.group(2).replace(",", "."))
        return value * (1e6 if match.group(3).lower() == "млн" else 1e9), currency
    match = AMOUNT_USD.search(title)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", "").strip(".,"))
    except ValueError:
        return None
    return value * (1e6 if match.group(2).upper() in {"M", "MILLION"} else 1e9), "USD"


def _event(feed: _Feed, item: dict) -> dict | None:
    amount = _amount(item["title"], feed.ru)
    if not amount:
        return None
    company = COMPANY.match(item["title"])
    return make_event("funding_round", item["date"], item["guid"] or item["link"],
                      amount=amount[0], currency=amount[1],
                      company=company.group(1) if company else "")


def _search(feed: _Feed, terms: list[str], as_of: date, limit: int) -> list[dict]:
    after, before = _window(as_of)
    phrases = _phrases(feed, terms)
    seen: set[str] = set()
    refs: list[dict] = []
    for phrase in phrases:
        found = 0
        # Ядра фразы запрашиваем, только если по полной фразе сделок не хватило.
        for variant in _variants(phrase) if feed.deals else [phrase]:
            if feed.deals and found >= 2:
                break
            xml = _json_request("google_news", RSS_URL,
                                params={"q": _query(feed, variant, after, before),
                                        "hl": feed.hl, "gl": feed.gl, "ceid": feed.ceid},
                                headers={"Accept": "application/rss+xml"}, as_text=True)
            for item in _items(xml):
                if item["link"] in seen or not _on_sites(item["source_url"], feed.sites):
                    continue
                seen.add(item["link"])
                if feed.deals:
                    if not _is_deal(item["title"], variant):
                        continue
                elif feed.raw:
                    # Китайский, японский…: слов не выделить — полагаемся на ранжирование Google.
                    if key_tokens(variant) and not relevant(f'{item["title"]} {item["text"]}', [variant],
                                                            min_share=0.5):
                        continue
                elif not _matches(f'{item["title"]} {item["text"]}', phrases):
                    continue
                found += 1
                meta: dict = {"publisher": item["publisher"]}
                if feed.events:
                    event = _event(feed, item)
                    if event:
                        meta["structured_event"] = event
                refs.append(_document(item["link"], item["title"], f'{item["title"]}. {item["text"]}',
                                      item["date"], feed.language, feed.slug, **meta))
    return refs[:limit]


def localized(query: str, language: str, as_of: date, limit: int) -> list[dict]:
    """Новости на языке запроса с before: — замена Bing News в историческом прогоне."""
    market, country = LANGUAGE_MARKETS.get(language, LANGUAGE_MARKETS["en"])
    feed = _Feed("google_news", market, country, f"{country}:{language}", language, raw=True)
    cutoff = as_of.isoformat()
    refs = [ref for ref in _search(feed, [query], as_of, limit) if str(ref["observed_at"])[:10] <= cutoff]
    for ref in refs:
        ref.update(channel="market", date_support="post")
    return refs


def _venture(terms: list[str], as_of: date, limit: int) -> list[dict]:
    return _search(VENTURE, terms, as_of, limit)


def _ru_business(terms: list[str], as_of: date, limit: int) -> list[dict]:
    # Английские фразы плана в русских изданиях пусты (0 из 11 на замере), а запросы делят
    # с венчурной лентой паузу Google News: ищем только по русскому запросу ветки.
    if isinstance(terms, ReadyPhrases) and not any(CYRILLIC.search(term) for term in terms):
        return []
    return _search(RU_BUSINESS, terms, as_of, limit)


def _general(terms: list[str], as_of: date, limit: int) -> list[dict]:
    return _search(GENERAL, terms, as_of, limit)


# Все три делят паузу Google News (1 запрос в секунду), поэтому на каждого кандидата
# ходит только венчурный: общие новости кандидата уже собирает evidence.media, а
# русские деловые издания по латинскому названию кандидата почти всегда пусты.
CONNECTORS = (
    Connector("Google News venture", "money", "post", _venture, language="en", default_on=False),
    Connector("Google News RU business", "money", "post", _ru_business, language="multi",
              default_on=False, scopes=("branch",)),
    Connector("Google News", "market", "post", _general, language="en", default_on=False,
              scopes=("branch",)),
)
