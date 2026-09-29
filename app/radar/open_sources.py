"""Открытые источники для поиска кандидатов — в дополнение к веб-поиску LLM.

Веб-поиск модели — один канал: вся выдача зависела от его поисковика, прямых научных
и русскоязычных источников не было. Коннекторы ниже бесплатны и возвращают прямые ссылки:

  bing_news — Bing News RSS: локальные выдачи на 14 приоритетных языках;
  GDELT DOC — глобальная новостная индексация: поиск по английскому переводу публикаций на десятках языков;
  patents — FreePatentsOnline, плюс OpenAlex, NIH/CORDIS, GitHub/Hugging Face, TED и медицинские реестры;
  arxiv    — arXiv API: препринты с аннотациями, фильтр по дате оценки;
  crossref — Crossref: научные работы с аннотациями (вежливый пул);
  habr     — поиск Хабра (RSS): русскоязычное техническое сообщество.

Для научных работ документом служит аннотация из API (поле "document" у ссылки):
страница издателя за DOI часто закрыта, а цитата сверяется с тем текстом, что прочитан.
Каждый коннектор возвращает результат в формате ResearchProvider.search(), поэтому
пайплайн перемешивает их со ссылками веб-поиска по очереди.
"""
from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import urllib.parse
from collections.abc import Callable
from datetime import date, timedelta
from email.utils import parsedate_to_datetime

import httpx

from app.radar.locales import LANGUAGE_MARKETS, LANGUAGE_NAMES, normalize_queries
from app.radar.resilience import CircuitBreaker

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36", "Accept": "*/*"}
CONTACT = os.getenv("OPENALEX_MAILTO", "").strip() or "hackathon@example.org"
STOP = {"the", "and", "for", "with", "from", "into", "new", "novel", "based", "using", "technology",
        "technologies", "early", "emerging", "approaches", "applications", "application", "systems"}


class _Pace:
    def __init__(self, interval: float) -> None:
        self.interval, self.lock, self.next = interval, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.interval
        time.sleep(max(0.0, slot - now))


PACE = {"bing": _Pace(0.6), "gdelt": _Pace(1.5), "arxiv": _Pace(3.1),
        "crossref": _Pace(0.2), "habr": _Pace(0.6)}


BREAKERS = {kind: CircuitBreaker() for kind in PACE}
CONNECTOR_RETRIES = 2


def _retry_after(response: httpx.Response, attempt: int) -> float:
    value = response.headers.get("Retry-After", "").strip()
    try:
        return min(20.0, max(0.0, float(value)))
    except ValueError:
        try:
            target = parsedate_to_datetime(value).timestamp()
            return min(20.0, max(0.0, target - time.time()))
        except (TypeError, ValueError, OverflowError):
            return min(10.0, 1.0 * (2 ** attempt))


def _get(kind: str, url: str, timeout: float = 20) -> str:
    breaker = BREAKERS[kind]
    breaker.before_call()
    for attempt in range(CONNECTOR_RETRIES + 1):
        PACE[kind].wait()
        try:
            response = httpx.get(url, headers=UA, timeout=timeout, follow_redirects=True)
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            response.raise_for_status()
            breaker.success()
            return response.text
        except httpx.HTTPStatusError as exc:
            transient = exc.response.status_code in {429, 500, 502, 503, 504}
            if not transient:
                # A non-retryable HTTP response proves the service is reachable.
                breaker.success()
                raise
            if kind == "gdelt" and exc.response.status_code == 429:
                # A public GDELT throttle is shared for the service IP; retries
                # only prolong this optional connector and do not improve recall.
                breaker.failure()
                raise
            if attempt >= CONNECTOR_RETRIES:
                breaker.failure()
                raise
            time.sleep(_retry_after(exc.response, attempt))
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt >= CONNECTOR_RETRIES:
                breaker.failure()
                raise
            time.sleep(min(1.0, 0.2 * (2 ** attempt)))
        except Exception:
            # Invalid requests and local bugs should fail visibly without tripping
            # the transport circuit or consuming retry budget.
            breaker.success()
            raise
    raise RuntimeError("Исчерпаны повторные попытки коннектора")


def _tag(block: str, name: str) -> str:
    match = re.search(rf"<{name}[^>]*>(.*?)</{name}>", block, re.DOTALL)
    return html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", match.group(1))).strip() if match else ""


def _keywords(query: str, limit: int = 5) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query) if w.lower() not in STOP]
    return list(dict.fromkeys(w.lower() for w in words))[:limit]


# ------------------------------------------------------------------------ коннекторы

def bing_news(query: str, market: str, limit: int = 8) -> list[dict]:
    refs = _bing(query, market, limit)
    words = query.split()
    if not refs and len(words) > 3:  # длинная фраза в падеже часто не находит ничего
        refs = _bing(" ".join(words[:3]), market, limit)
    return refs


def gdelt_news(query: str, as_of: date, limit: int = 12) -> list[dict]:
    """Search GDELT's multilingual news index using the English branch query.

    DOC searches English translations of monitored coverage (65 source languages)
    and has a rolling three-month horizon. Respect that horizon and the run's as-of
    date; historical runs outside it simply receive no GDELT results.
    """
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc).date()
    end = min(as_of, today)
    earliest = today - timedelta(days=88)  # stay safely inside the API's 3-month window
    if end < earliest:
        return []
    start = max(end - timedelta(days=87), earliest)
    params = {
        "query": query[:400], "mode": "artlist", "format": "json",
        "maxrecords": max(1, min(limit, 25)), "sort": "HybridRel",
        "startdatetime": f"{start:%Y%m%d}000000", "enddatetime": f"{end:%Y%m%d}235959",
    }
    body = _get("gdelt", "https://api.gdeltproject.org/api/v2/doc/doc?"
                         + urllib.parse.urlencode(params))
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GDELT returned a non-JSON response (temporary throttle or upstream error)") from exc
    refs = []
    for item in payload.get("articles", []):
        url = str(item.get("url", ""))
        title = re.sub(r"\s+", " ", str(item.get("title", ""))).strip()
        if not url.startswith(("https://", "http://")) or not title:
            continue
        raw_date = str(item.get("seendate", ""))
        observed_at = (f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                       if re.fullmatch(r"\d{14}", raw_date) else None)
        language = str(item.get("language", "")).lower() or None
        country = str(item.get("sourcecountry", "")).upper() or None
        refs.append({"url": url, "title": title, "observed_at": observed_at,
                     "source_language": language, "source_country": country})
    return refs[:limit]


def _bing(query: str, market: str, limit: int) -> list[dict]:
    params = {"q": query, "format": "rss"}
    if market in LANGUAGE_MARKETS:
        locale, country = LANGUAGE_MARKETS[market]
        params.update(setlang=locale, cc=country)
    text = _get("bing", "https://www.bing.com/news/search?" + urllib.parse.urlencode(params))
    refs = []
    for block in re.findall(r"<item>(.*?)</item>", text, re.DOTALL):
        link = _tag(block, "link")
        target = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query).get("url", [link])[0]
        if target.startswith("http") and "bing.com" not in urllib.parse.urlsplit(target).netloc:
            refs.append({"url": target, "title": _tag(block, "title")})
    return refs[:limit]


def arxiv(query: str, as_of: date, limit: int = 6) -> list[dict]:
    words = _keywords(query, 4)
    if not words:
        return []
    start = (as_of - timedelta(days=730)).strftime("%Y%m%d0000")
    search = " AND ".join(f"all:{w}" for w in words) + f" AND submittedDate:[{start} TO {as_of:%Y%m%d}2359]"
    from app.radar.research_apis import openalex_enabled

    try:
        text = _get("arxiv", "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": search, "max_results": limit, "sortBy": "relevance"}))
    except Exception:  # noqa: BLE001 — OpenAlex indexes arXiv preprints and is a safe read-only fallback
        return _arxiv_openalex(query, as_of, limit) if openalex_enabled() else []
    refs = []
    for entry in re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL):
        url = _tag(entry, "id").replace("http://", "https://")
        title = re.sub(r"\s+", " ", _tag(entry, "title"))
        abstract = re.sub(r"\s+", " ", _tag(entry, "summary"))
        published = _tag(entry, "published")[:10] or None
        if url and abstract:
            refs.append({"url": url, "title": title, "document": {
                "title": title, "text": f"{title}\n\n{abstract}", "date": published, "language": "en",
                "extraction": "arxiv_api_abstract"}})
    return refs or (_arxiv_openalex(query, as_of, limit) if openalex_enabled() else [])


def _arxiv_openalex(query: str, as_of: date, limit: int) -> list[dict]:
    from app.radar.research_apis import openalex_enabled, openalex_works

    if not openalex_enabled():
        return []

    try:
        works = openalex_works([query], as_of - timedelta(days=730), as_of, limit=max(12, limit * 3))
    except RuntimeError as exc:
        raise RuntimeError(f"OpenAlex arXiv fallback failed: {exc}") from exc
    refs = []
    for work in works:
        url = str(work.get("primary_location_url") or "")
        if "arxiv.org/" not in url or not work.get("abstract"):
            continue
        title = work["title"]
        refs.append({"url": url, "title": title,
                     "source_language": work.get("language") or "en",
                     "document": {"title": title, "text": f"{title}\n\n{work['abstract']}",
                                  "date": work.get("publicationDate"),
                                  "language": work.get("language") or "en",
                                  "extraction": "openalex_arxiv_abstract"},
                     "source_provider": "openalex_arxiv_fallback"})
        if len(refs) >= limit:
            break
    return refs


def crossref(query: str, as_of: date, limit: int = 6, language: str = "en") -> list[dict]:
    params = {"query.bibliographic": query, "rows": limit, "select": "DOI,title,abstract,published,URL,type",
              "filter": f"from-pub-date:{as_of - timedelta(days=1095):%Y-%m-%d},until-pub-date:{as_of:%Y-%m-%d},has-abstract:true",
              "mailto": CONTACT}
    items = json.loads(_get("crossref", "https://api.crossref.org/works?" + urllib.parse.urlencode(params)))["message"]["items"]
    refs = []
    for item in items:
        title = re.sub(r"\s+", " ", " ".join(item.get("title") or [])).strip()
        abstract = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", item.get("abstract") or "")).strip()
        parts = ((item.get("published") or {}).get("date-parts") or [[]])[0]
        published = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts)) if parts else None
        if item.get("DOI") and title and len(abstract) > 150:
            refs.append({"url": f"https://doi.org/{item['DOI']}", "title": title, "document": {
                "title": title, "text": f"{title}\n\n{abstract}",
                "date": published, "language": item.get("language") or language,
                "extraction": "crossref_abstract"}})
    return refs


def habr(query: str, limit: int = 4) -> list[dict]:
    text = _get("habr", "https://habr.com/ru/rss/search/?" + urllib.parse.urlencode(
        {"q": query, "target_type": "posts", "order": "relevance"}))
    refs = []
    for block in re.findall(r"<item>(.*?)</item>", text, re.DOTALL):
        link = _tag(block, "link").split("?")[0]
        if "/articles/" in link or "/companies/" in link:
            refs.append({"url": link, "title": _tag(block, "title")})
    return refs[:limit]


# ------------------------------------------------------------------------ задания

def branch_tasks(topic: str, query_ru: str, query_en: str, as_of: date,
                 localized: list[dict] | None = None) -> list[tuple[str, Callable[[], list[dict]], str, str]]:
    """Run localized searches plus public structured research and deployment APIs per branch."""
    queries = normalize_queries(query_ru, query_en, localized)
    tasks: list[tuple[str, Callable[[], list[dict]], str, str]] = []
    for item in queries:
        language, query = item["language"], item["query"]
        language_name = LANGUAGE_NAMES.get(language, language.upper())
        tasks.append((f"Bing News ({language_name})",
                      lambda query=query, language=language: bing_news(query, language), query, language))
        # Crossref indexes multilingual abstracts/titles; native phrases supplement EN terms.
        tasks.append((f"Crossref ({language_name})",
                      lambda query=query, language=language: crossref(query, as_of, language=language), query, language))
    if os.getenv("FEATURE_PATENTS", "1") != "0":
        patent_queries = [item["query"] for item in queries if item["query"].strip()]
        tasks.append(("Patents (FreePatentsOnline)", lambda: _patent_search(patent_queries, as_of),
                      " / ".join(patent_queries[:2]), "multi"))
    if os.getenv("FEATURE_STRUCTURED_SOURCES", "1") != "0":
        api_queries = [query_en, topic]
        from app.radar import research_apis as apis

        if apis.openalex_enabled():
            tasks.append(("OpenAlex", lambda: apis.search_openalex(api_queries, as_of, limit=6), query_en, "multi"))
        tasks.extend([
            ("NIH RePORTER", lambda: apis.search_nih_grants(api_queries, as_of, limit=4), query_en, "en"),
            ("CORDIS grants", lambda: apis.search_cordis_grants(api_queries, as_of, limit=4), query_en, "multi"),
            ("GitHub repositories", lambda: apis.search_github(api_queries, as_of, limit=4), query_en, "en"),
            ("Hugging Face", lambda: apis.search_huggingface(api_queries, as_of, limit=4), query_en, "en"),
            ("TED procurement", lambda: apis.search_ted(api_queries, as_of, limit=4), query_en, "multi"),
        ])
        if apis.is_health_topic(api_queries):
            tasks.extend([
                ("ClinicalTrials.gov", lambda: apis.search_clinicaltrials(api_queries, as_of, limit=4),
                 query_en, "multi"),
                ("FDA openFDA", lambda: apis.search_openfda(api_queries, as_of, limit=4), query_en, "en"),
            ])
    tasks.extend([
        ("GDELT Global News", lambda: gdelt_news(query_en, as_of), query_en, "multi"),
        ("arXiv (EN)", lambda: arxiv(query_en, as_of), query_en, "en"),
        ("Хабр (RU)", lambda: habr(query_ru), query_ru, "ru"),
    ])
    return tasks


def _patent_search(queries: list[str], as_of: date) -> list[dict]:
    """Search patent records in FreePatentsOnline."""
    from app.ml.features.evidence import search_patents_fpo
    try:
        return search_patents_fpo(queries, as_of, limit=12, years_back=5)
    except Exception as exc:  # Report the single patent connector's failure.
        detail = str(exc).replace("\n", " ")[:180] or type(exc).__name__
        raise RuntimeError(f"FreePatentsOnline: {detail}") from exc


def run_task(name: str, action: Callable[[], list[dict]], branch: str, query: str,
             language: str | None = None) -> dict:
    """Результат в формате ResearchProvider.search(); ошибка коннектора — пустой список."""
    started = time.monotonic()
    try:
        refs, error = action(), None
    except Exception as exc:  # noqa: BLE001 — один недоступный источник не останавливает поиск
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is None and getattr(exc, "__cause__", None) is not None:
            status = getattr(getattr(exc.__cause__, "response", None), "status_code", None)
        detail = f"HTTP {status}" if status else (str(exc)[:140] if isinstance(exc, RuntimeError)
                                                   else type(exc).__name__)
        refs, error = [], detail
    for ref in refs:
        ref["branch"] = f"{branch} · {name}"
        if language:
            ref["query_language"] = language
    return {"query": query, "branch": f"{branch} · {name}", "connector": name, "summary": "",
            "references": refs, "error": error, "seconds": round(time.monotonic() - started, 2)}
