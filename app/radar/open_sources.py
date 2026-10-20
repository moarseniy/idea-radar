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
from datetime import date, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx

from app.radar.connectors._terms import key_tokens, relevant
from app.radar.locales import LANGUAGE_MARKETS, LANGUAGE_NAMES, normalize_queries
from app.radar.resilience import CircuitBreaker

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36", "Accept": "*/*"}
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


# GDELT DOC: один запрос в 5 с с одного IP (B8).
PACE = {"bing": _Pace(0.6), "gdelt": _Pace(5.5), "arxiv": _Pace(3.1),
        "crossref": _Pace(0.2), "habr": _Pace(0.6)}


BREAKERS = {kind: CircuitBreaker() for kind in PACE}
CONNECTOR_RETRIES = 2
# Для этих языков Bing не получает setlang/cc: с zh-CN он отвечает редиректом без выдачи (B10).
BING_UNLOCALIZED = {"zh"}


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


def google_news_localized(query: str, language: str, as_of: date, limit: int = 8) -> list[dict]:
    from app.radar.connectors.google_news import localized

    refs = localized(query, language, as_of, limit)
    words = query.split()
    if not refs and len(words) > 3:  # как у Bing: длинная фраза в падеже часто пуста
        refs = localized(" ".join(words[:3]), language, as_of, limit)
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
    payload = _gdelt_payload("https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params))
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


def _gdelt_payload(url: str) -> dict:
    """Ответ GDELT с дисковым кэшем на 6 часов; кэшируется только валидный JSON (429 и HTML не сохраняются)."""
    from app.radar import research_apis

    cache = research_apis.CACHE_DIR / "gdelt" / f"{research_apis._cache_key('gdelt', 'GET', url, None, None)}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < research_apis.CACHE_TTL:
        return json.loads(cache.read_text(encoding="utf-8"))
    body = _get("gdelt", url)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GDELT returned a non-JSON response (temporary throttle or upstream error)") from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(cache)
    return payload


def _bing(query: str, market: str, limit: int) -> list[dict]:
    params = {"q": query, "format": "rss"}
    if market in LANGUAGE_MARKETS and market not in BING_UNLOCALIZED:
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
    from app.radar.research_apis import contact_email

    words = [word for word in key_tokens(query) if len(word) > 3]
    if not words:
        # Китайский, арабский, хинди…: слов не выделить и релевантность не проверить, а без
        # проверки Crossref на этих языках приносил только шум (охрана среды, исламские контракты).
        return []
    params = {"query.bibliographic": query, "rows": limit, "select": "DOI,title,abstract,published,URL,type",
              # until-created-date: журналы задним числом кладут статьи 2024 года в выпуски 2022-го
              # (запись Crossref создана позже даты оценки) — в прошлом такой записи ещё не было.
              "filter": f"from-pub-date:{as_of - timedelta(days=1095):%Y-%m-%d},until-pub-date:{as_of:%Y-%m-%d},"
                        f"until-created-date:{as_of:%Y-%m-%d},has-abstract:true"}
    contact = contact_email()
    if contact:  # вежливый пул; без контакта запрос идёт без mailto
        params["mailto"] = contact
    items = json.loads(_get("crossref", "https://api.crossref.org/works?" + urllib.parse.urlencode(params)))["message"]["items"]
    refs = []
    for item in items:
        title = re.sub(r"\s+", " ", " ".join(item.get("title") or [])).strip()
        abstract = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", item.get("abstract") or "")).strip()
        parts = ((item.get("published") or {}).get("date-parts") or [[]])[0]
        published = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts)) if parts else None
        # Локализованный запрос Crossref ищет по любому слову: на "обслуживание клиентов
        # банков" приходили качество свадебных банкетов и словари. Слова короче четырёх букв
        # не считаются: стоп-слова есть только английские, а "de", "la", "al" совпадают всегда.
        if item.get("DOI") and title and len(abstract) > 150 and relevant(f"{title} {abstract}", [" ".join(words)]):
            refs.append({"url": f"https://doi.org/{item['DOI']}", "title": title, "document": {
                "title": title, "text": f"{title}\n\n{abstract}",
                "date": published, "language": item.get("language") or language,
                "extraction": "crossref_abstract"}})
    return refs


def _rss_date(block: str) -> str | None:
    """RFC 822 pubDate из RSS в дату UTC (YYYY-MM-DD); без даты или с мусором — None."""
    try:
        moment = parsedate_to_datetime(_tag(block, "pubDate"))
    except (TypeError, ValueError, IndexError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date().isoformat()


def habr(query: str, limit: int = 4) -> list[dict]:
    text = _get("habr", "https://habr.com/ru/rss/search/?" + urllib.parse.urlencode(
        {"q": query, "target_type": "posts", "order": "relevance"}))
    refs = []
    for block in re.findall(r"<item>(.*?)</item>", text, re.DOTALL):
        link = _tag(block, "link").split("?")[0]
        if "/articles/" in link or "/companies/" in link:
            refs.append({"url": link, "title": _tag(block, "title"), "observed_at": _rss_date(block)})
    return refs[:limit]


# ------------------------------------------------------------------------ задания

def branch_tasks(topic: str, query_ru: str, query_en: str, as_of: date,
                 localized: list[dict] | None = None,
                 money_terms: list[str] | None = None) -> list[tuple[str, Callable[[], list[dict]], str, str]]:
    """Run localized searches plus public structured research and deployment APIs per branch.

    money_terms — короткие фразы подходов для денежных коннекторов (их фразовый поиск по
    длинной формулировке ветки почти всегда пуст); остальные коннекторы ищут по ветке.
    """
    from app.radar.connectors import connectors, is_historical

    historical = is_historical(as_of)
    queries = normalize_queries(query_ru, query_en, localized)
    tasks: list[tuple[str, Callable[[], list[dict]], str, str]] = []
    for item in queries:
        language, query = item["language"], item["query"]
        language_name = LANGUAGE_NAMES.get(language, language.upper())
        # Bing News RSS не фильтрует по дате: в прошлом его место на каждом языке занимает
        # Google News с before: (тот же рынок, иначе рыночных документов почти нет).
        if historical:
            tasks.append((f"Google News ({language_name})",
                          lambda query=query, language=language: google_news_localized(query, language, as_of),
                          query, language))
        else:
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
        for connector in connectors(api_queries, scope="branch", historical=historical):
            if money_terms and connector.channel == "money":
                # Многоязычные (русские деловые издания) ищут ещё и по русскому запросу ветки:
                # английские фразы плана в них почти всегда пусты.
                phrases = money_terms + ([query_ru] if connector.language == "multi"
                                         and re.search(r"[Ѐ-ӿ]", query_ru) else [])
                tasks.append((connector.name,
                              lambda connector=connector, phrases=phrases: _by_phrase(connector, phrases, as_of),
                              " / ".join(phrases), connector.language))
                continue
            tasks.append((connector.name,
                          lambda connector=connector: connector.run(api_queries, as_of, connector.radar_limit),
                          query_en, connector.language))
    tasks.extend([
        ("GDELT Global News", lambda: gdelt_news(query_en, as_of), query_en, "multi"),
        ("arXiv (EN)", lambda: arxiv(query_en, as_of), query_en, "en"),
    ])
    if not historical:  # RSS Хабра — только свежие статьи, без поиска по дате
        tasks.append(("Хабр (RU)", lambda: habr(query_ru), query_ru, "ru"))
    return tasks


def _by_phrase(connector, phrases: list[str], as_of: date, per_phrase: int = 2,
               budget: float = 60.0) -> list[dict]:
    """Денежный коннектор по каждой фразе плана отдельно, записи чередуются.

    Одним вызовом первая фраза занимает весь radar_limit, и до остальных дело не доходит.
    budget — секунд на все фразы: поиск ждёт каждую задачу, а UKRI отвечает до двух минут.
    """
    from itertools import zip_longest

    from app.radar.connectors._terms import ReadyPhrases

    groups = []
    started = time.monotonic()
    for phrase in phrases:
        if groups and time.monotonic() - started > budget:
            break
        try:
            groups.append(connector.run(ReadyPhrases([phrase]), as_of, per_phrase))
        except Exception:  # noqa: BLE001 — одна фраза не должна обнулять остальные
            groups.append([])
    seen: set[str] = set()
    refs = []
    for row in zip_longest(*groups):
        for ref in row:
            if ref and ref["url"] not in seen:
                seen.add(ref["url"])
                refs.append(ref)
    return refs


def _patent_search(queries: list[str], as_of: date) -> list[dict]:
    """Search FPO first; use Google Patents web results for discovery if FPO fails or is empty."""
    from app.ml.features.evidence import search_patents_fpo, search_patents_google_web

    patent_terms = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    # Patent indexes are strongest in English; short OR expressions also avoid
    # oversized FPO URLs from passing every localized query in a broad search.
    english_terms = [term for term in patent_terms if not re.search(r"[\u0400-\u04ff]", term)]
    patent_terms = (english_terms or patent_terms)[:3]
    fpo_error = None
    fallback_error = None
    try:
        refs = search_patents_fpo(patent_terms, as_of, limit=12, years_back=5)
        if refs:
            return refs
    except Exception as exc:  # noqa: BLE001 — continue with the alternate discovery path.
        fpo_error = exc
    if os.getenv("RADAR_WEB_SEARCH", "service").strip().lower() == "service":
        try:
            refs = search_patents_google_web(patent_terms, as_of, limit=12)
            if refs:
                return refs
        except Exception as exc:  # noqa: BLE001 — the reserve is best-effort.
            fallback_error = exc
    if fpo_error:
        detail = str(fpo_error).replace("\n", " ")[:180] or type(fpo_error).__name__
        fallback = (f"; Google Patents fallback failed: {type(fallback_error).__name__}"
                    if fallback_error else "")
        raise RuntimeError(f"FPO unavailable; Google Patents web fallback returned no results: {detail}{fallback}") \
            from fpo_error
    return []


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
    connector = name
    from app.radar.connectors import describe

    channel, date_support = describe(name)
    if name.startswith("Patents") and refs:
        providers = {ref.get("source_provider") for ref in refs if ref.get("source_provider")}
        if providers == {"Google Patents web search"}:
            connector = "Patents (Google Patents web fallback)"
        elif providers == {"FreePatentsOnline"}:
            connector = "Patents (FreePatentsOnline)"
    for ref in refs:
        ref["branch"] = f"{branch} · {connector}"
        if channel:
            ref.setdefault("channel", channel)
            ref.setdefault("date_support", date_support)
        if language:
            ref["query_language"] = language
    return {"query": query, "branch": f"{branch} · {connector}", "connector": connector, "summary": "",
            "references": refs, "error": error, "seconds": round(time.monotonic() - started, 2)}
