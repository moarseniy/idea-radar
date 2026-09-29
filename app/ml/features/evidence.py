"""Сбор сырых свидетельств по технологии из бесплатных открытых источников.

Каналы и источники:
  наука   — Crossref, вежливый пул (топ-1000 по окну, фильтр по фразе, цитирования, препринты);
  патенты — FreePatentsOnline по свежему окну;
  медиа   — Google News RSS с операторами after:/before: (заголовки, издания, даты).

Все запросы ограничены датой среза (snapshot), поэтому повторный сбор на тот же срез
возвращает те же окна. Каждый ответ кэшируется на диске по хэшу запроса: повторный
прогон не ходит в сеть и даёт байт-в-байт те же признаки. Это и есть детерминизм
разметки: признаки — чистая функция от закэшированных ответов.

Недоступность источника не превращается в ноль: функция возвращает ok=False, и
вычисление признаков оставляет соответствующие поля пустыми.
"""
from __future__ import annotations

import email.utils
import hashlib
import html
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = Path(os.getenv("FEATURE_CACHE_DIR", str(ROOT / "ml" / ".cache" / "features")))
CONTACT = os.getenv("OPENALEX_MAILTO", "").strip() or "hackathon@example.org"
UA = {"User-Agent": f"Mozilla/5.0 (compatible; weak-signal-radar/0.2; {CONTACT})"}


class HostLimiter:
    """Минимальный интервал между запросами к одному сервису, безопасно для потоков."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))


LIMITS = {"science": HostLimiter(0.15), "patents": HostLimiter(0.5),
          "google_patents": HostLimiter(0.5), "news": HostLimiter(0.6), "llm": HostLimiter(0.0)}


class Breaker:
    """После серии отказов подряд источник отключается на время остывания.

    Google News и другие источники при частых запросах временно отвечают 503/429 на весь IP;
    без предохранителя каждая строка тратила бы минуты на ретраи. В сервисе источник
    возвращается через FEATURE_BREAKER_COOLDOWN секунд (по умолчанию 300). Отказы не
    кэшируются, поэтому повторная разметка дозаполнит пропуски.
    """

    def __init__(self, threshold: int = int(os.getenv("FEATURE_BREAKER_THRESHOLD", "8")),
                 cooldown: float = float(os.getenv("FEATURE_BREAKER_COOLDOWN", "300"))) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def open(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return False
            if time.monotonic() - self.opened_at >= self.cooldown:
                self.opened_at, self.failures = None, 0  # пробуем источник снова
                return False
            return True

    def record(self, ok: bool) -> None:
        with self._lock:
            self.failures = 0 if ok else self.failures + 1
            if self.failures >= self.threshold and self.opened_at is None:
                self.opened_at = time.monotonic()


BREAKERS = {name: Breaker() for name in LIMITS}


def _cache_path(source: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / source / digest[:2] / f"{digest}.json"


def cached_fetch(source: str, url: str, *, parse=json.loads,
                 tries: int | None = None,
                 timeout: float | None = None,
                 body: dict | None = None, headers: dict | None = None,
                 cache_key: str | None = None):
    """GET/POST с дисковым кэшем. Кэшируются только успешные ответы."""
    if tries is None:
        try:
            tries = int(os.getenv("FEATURE_FETCH_TRIES", "2"))
        except ValueError:
            tries = 2
        tries = max(1, min(4, tries))
    if timeout is None:
        try:
            timeout = float(os.getenv("FEATURE_FETCH_TIMEOUT", "20"))
        except ValueError:
            timeout = 20
        timeout = max(5, min(90, timeout))
    path = _cache_path(source, cache_key or url)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if BREAKERS[source].open:
        raise RuntimeError(f"{source}: отключён после серии отказов")
    last: Exception | None = None
    for attempt in range(tries):
        LIMITS[source].wait()
        try:
            data = json.dumps(body).encode() if body is not None else None
            request = urllib.request.Request(
                url, data=data, headers={**UA, **({"Content-Type": "application/json"} if data else {}),
                                         **(headers or {})})
            if source == "patents" and data is None:
                response = httpx.get(url, headers={**UA, **(headers or {})}, timeout=timeout,
                                     follow_redirects=True)
                response.raise_for_status()
                value = parse(response.text)
            else:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    value = parse(response.read().decode("utf-8", "replace"))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            BREAKERS[source].record(True)
            return value
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                break
            retry_after = exc.headers.get("Retry-After", "")
            pause = float(retry_after) if retry_after.isdigit() and int(retry_after) < 60 else 0
            time.sleep(min(60.0, max(pause, 2 ** attempt + random.random())))
        except Exception as exc:  # noqa: BLE001 — сеть, таймаут, битый ответ
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in (408, 425, 429, 500, 502, 503, 504):
                break
            time.sleep(min(60.0, 2 ** attempt + random.random()))
    BREAKERS[source].record(False)
    raise RuntimeError(f"{source}: {last}")


def years_before(day: date, years: float) -> date:
    return day - timedelta(days=round(365.25 * years))


# --------------------------------------------------------------------------- наука

CROSSREF_SELECT = "title,published,is-referenced-by-count,type,abstract,DOI"


def _crossref(terms: list[str], start: date, end: date) -> list[dict]:
    """Топ-1000 работ окна по релевантности; оставляем те, где фраза есть в заголовке/аннотации.

    Поиск Crossref словесный, а не фразовый, поэтому без фильтра «in-sensor computing»
    совпадал бы с любыми работами про сенсоры. Счёт по окну — число совпадений фразы
    среди первых 1000 результатов: ограничен сверху, но одинаков для всех строк.
    """
    params = {"query.bibliographic": " ".join(terms), "rows": 1000, "select": CROSSREF_SELECT,
              "filter": f"from-pub-date:{start.isoformat()},until-pub-date:{end.isoformat()}",
              "mailto": CONTACT}
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)

    def parse(text: str) -> list[dict]:
        items = json.loads(text)["message"]["items"]
        phrases = [t.lower() for t in terms]
        out = []
        for item in items:
            title = re.sub(r"\s+", " ", " ".join(item.get("title") or [])).strip()
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
            text_low = re.sub(r"\s+", " ", f"{title} {abstract}").lower()
            parts = ((item.get("published") or {}).get("date-parts") or [[]])[0]
            if not parts or not any(p in text_low for p in phrases):
                continue
            published = f"{parts[0]:04d}-{(parts[1] if len(parts) > 1 else 7):02d}-{(parts[2] if len(parts) > 2 else 1):02d}"
            out.append({"title": title, "publicationDate": published,
                        "citationCount": item.get("is-referenced-by-count") or 0,
                        "preprint": item.get("type") == "posted-content", "doi": item.get("DOI")})
        return out

    return cached_fetch("science", url, parse=parse)


@dataclass
class Science:
    ok: bool = False
    total: int = 0
    windows: list[int] = field(default_factory=lambda: [0, 0, 0])  # [0-2 лет, 2-4, 4-6]
    count_3y: int = 0
    first_date: str | None = None
    recent: list[dict] = field(default_factory=list)  # работы последних 2 лет
    titles: list[str] = field(default_factory=list)
    institutions_3y: set[str] = field(default_factory=set)
    countries_3y: set[str] = field(default_factory=set)
    industry_affiliation_share_3y: float | str = ""


def science(terms: list[str], snapshot: date, *, include_affiliations: bool = True) -> Science:
    out = Science()
    from app.radar.research_apis import openalex_enabled, openalex_works

    day = timedelta(days=1)
    b2, b3, b4, b6 = (years_before(snapshot, y) for y in (2, 3, 4, 6))
    windows = ((b2, snapshot), (b4, b2 - day), (b6, b4 - day))
    crossref_windows, openalex_windows = [], []
    any_ok = False
    for start, end in windows:
        try:
            crossref_windows.append(_crossref(terms, start, end))
            any_ok = True
        except RuntimeError:
            crossref_windows.append([])
        if os.getenv("FEATURE_STRUCTURED_SOURCES", "1") == "0" or not openalex_enabled():
            openalex_windows.append([])
        else:
            try:
                openalex_windows.append(openalex_works(terms, start, end, limit=100,
                                                       include_affiliations=include_affiliations))
                any_ok = True
            except RuntimeError:
                openalex_windows.append([])
    try:
        older = _crossref(terms, date(1900, 1, 1), b6 - day)
        any_ok = True
    except RuntimeError:
        older = []

    def merge(primary: list[dict], additional: list[dict]) -> list[dict]:
        output, seen = [], set()
        for item in [*primary, *additional]:
            doi = re.sub(r"^https?://doi.org/", "", str(item.get("doi") or "")).lower()
            title = re.sub(r"\W+", " ", str(item.get("title") or "").casefold()).strip()
            key = f"doi:{doi}" if doi else f"title:{title}"
            if not title or key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    merged = [merge(cr, oa) for cr, oa in zip(crossref_windows, openalex_windows)]
    w0, w1, w2 = merged
    all_recent = [item for window in merged for item in window]
    out.windows = [len(w0), len(w1), len(w2)]
    out.count_3y = len(w0) + sum(w["publicationDate"] >= b3.isoformat() for w in w1)
    out.total = len({str(item.get("doi") or item.get("title", "")).casefold()
                     for item in [*all_recent, *older]})
    out.recent = w0
    out.titles = [w["title"] for w in sorted(w0, key=lambda w: w["publicationDate"], reverse=True)[:40]]
    recent_3y = [item for item in [*w0, *w1] if item.get("publicationDate") and
                 item["publicationDate"] >= b3.isoformat()]
    out.institutions_3y = {institution for item in recent_3y
                           for institution in (item.get("institutions") or {})}
    out.countries_3y = {country for item in recent_3y for country in (item.get("countries") or set())}
    industry_shares = [item.get("industry_affiliation_share") for item in recent_3y
                       if isinstance(item.get("industry_affiliation_share"), (int, float))]
    out.industry_affiliation_share_3y = (round(sum(industry_shares) / len(industry_shares), 6)
                                         if industry_shares else "")
    for window in (older, w2, w1, w0):
        if window:
            out.first_date = min(w["publicationDate"] for w in window if w.get("publicationDate"))
            break
    out.ok = any_ok
    return out


# -------------------------------------------------------------------------- патенты

FPO_COLLECTIONS = {"patents_us": "on", "patents_eu": "on", "patents_wo": "on", "apps": "on"}


def _fpo_parse(text: str) -> dict:
    if "returned no results" in text:
        return {"total": 0, "titles": [], "references": []}
    match = re.search(r"out of ([0-9,]+)", text)
    if not match:
        raise ValueError("FreePatentsOnline: не удалось разобрать страницу выдачи")
    table = text[text.find("listing_table"):]
    links = re.findall(r'<a href="(/[^"]+\.html)"[^>]*>([^<]{8,300})</a>', table)
    references = [{"url": "https://www.freepatentsonline.com" + href,
                   "title": html.unescape(re.sub(r"<[^>]+>", "", title)).strip()}
                  for href, title in links]
    references = [item for item in references if item["title"]]
    return {"total": int(match.group(1).replace(",", "")),
            "titles": [item["title"] for item in references[:50]],
            "references": references[:50]}


def _patents(terms: list[str], after: date | None, before: date, *, tries: int = 6,
             timeout: float = 90) -> dict:
    """Число документов (патенты и заявки US, EP, WO) по дате подачи в окне [after, before]."""
    phrase = " OR ".join(f'"{t}"' for t in terms)
    query = f"({phrase})" if len(terms) > 1 else phrase
    if after:
        query += f" AND APD/{after.month}/{after.day}/{after.year}->{before.month}/{before.day}/{before.year}"
    else:
        query += f" AND APD/1/1/1900->{before.month}/{before.day}/{before.year}"
    url = "https://www.freepatentsonline.com/result.html?" + urllib.parse.urlencode(
        {"sort": "relevance", "srch": "top", "query_txt": query, "submit": "", **FPO_COLLECTIONS})
    return cached_fetch("patents", url, parse=_fpo_parse, tries=tries, timeout=timeout)


@dataclass
class Patents:
    """Patent evidence normalized from FreePatentsOnline."""
    ok: bool = False
    windows: list[int] = field(default_factory=lambda: [0, 0, 0])  # [1,5-3,5 года, 3,5-5,5, 5,5-7,5]
    count_3y: int = 0
    older_6y: int = 0
    older_10y: int = 0
    first_priority: str | None = None  # оценка по самому старому непустому окну
    assignees_3y: list[str] = field(default_factory=list)
    assignees_before: list[str] = field(default_factory=list)
    assignee_counts_3y: dict[str, int] = field(default_factory=dict)
    titles: list[str] = field(default_factory=list)
    source: str = "unavailable"
    warning: str = ""


PATENTS_ENABLED = os.getenv("FEATURE_PATENTS", "1") != "0"


def _patents_from_fpo(terms: list[str], snapshot: date) -> Patents:
    out = Patents()
    try:
        # Заявки публикуются примерно через 18 месяцев после подачи, поэтому окна роста
        # сдвинуты на 1,5 года назад: иначе рост «за последние 2 года» отрицателен у всех.
        lag = years_before(snapshot, 1.5)
        b2, b4, b6 = (years_before(lag, y) for y in (2, 4, 6))
        b3, b10 = years_before(snapshot, 3), years_before(snapshot, 10)
        def fpo(after, before):
            return _patents(terms, after, before, tries=2, timeout=15)

        w0 = fpo(b2, lag)
        w1 = fpo(b4, b2 - timedelta(days=1))
        w2 = fpo(b6, b4 - timedelta(days=1))
        three = fpo(b3, snapshot)
        older = fpo(None, b6 - timedelta(days=1))
        out.windows = [w0["total"], w1["total"], w2["total"]]
        out.count_3y = three["total"]
        out.older_6y = older["total"]
        if older["total"]:
            out.older_10y = fpo(None, b10 - timedelta(days=1))["total"]
        # Середина самого старого непустого окна (лет назад): 15 — старше 10, 9 — 7,5-10,
        # 6,5 / 4,5 / 2,5 — окна роста, 1 — только последние 1,5 года.
        age = next((a for a, n in ((15, out.older_10y), (9, out.older_6y - out.older_10y),
                                   (6.5, w2["total"]), (4.5, w1["total"]), (2.5, w0["total"]),
                                   (1, three["total"])) if n > 0), None)
        out.first_priority = years_before(snapshot, age).isoformat() if age else None
        out.titles = three["titles"][:15]
        out.source = "freepatentsonline"
        out.ok = True
    except Exception as exc:  # noqa: BLE001 — endpoint, timeout, or changed HTML response
        out.ok = False
        out.warning = f"FreePatentsOnline: {str(exc)[:180]}"
    return out


def search_patents_fpo(terms: list[str], snapshot: date, *, limit: int = 12,
                       years_back: float = 5) -> list[dict]:
    """Find recent FPO patent documents for live search (FPO filing-date filter)."""
    after = years_before(snapshot, max(1.5, min(15, years_back)))
    # FPO sometimes closes TLS connections mid-response. One retry is useful;
    # repeated retries on a failing endpoint stall source collection for minutes.
    data = _patents(terms, after, snapshot, tries=2, timeout=15)
    return [{**item, "source_country": None, "observed_at": None, "document": None}
            for item in data.get("references", [])[:limit]]


def patents(terms: list[str], snapshot: date) -> Patents:
    if not PATENTS_ENABLED:
        return Patents(warning="Патентный канал отключён настройкой FEATURE_PATENTS=0.")
    result = _patents_from_fpo(terms, snapshot)
    if not result.ok:
        result.source = "freepatentsonline_unavailable"
    return result


# ---------------------------------------------------------------------------- медиа

def _rss(text: str) -> list[dict]:
    items = []
    for block in re.findall(r"<item>(.*?)</item>", text, re.S):
        def tag(name):
            m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", block, re.S)
            return html.unescape(m.group(1)).strip() if m else ""
        source = re.search(r'<source url="([^"]*)"[^>]*>(.*?)</source>', block, re.S)
        try:
            published = email.utils.parsedate_to_datetime(tag("pubDate")).date().isoformat()
        except (TypeError, ValueError):
            published = ""
        title = tag("title")
        publisher = html.unescape(source.group(2)).strip() if source else ""
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(publisher) - 3]
        items.append({"title": title, "date": published, "publisher": publisher,
                      "host": urllib.parse.urlparse(source.group(1)).netloc.lower().removeprefix("www.")
                      if source else ""})
    return items


def _news(terms: list[str], after: date, before: date) -> list[dict]:
    q = " OR ".join(f'"{t}"' for t in terms)
    query = f"({q}) after:{after.isoformat()} before:{before.isoformat()}"
    url = ("https://news.google.com/rss/search?" +
           urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}))
    items = cached_fetch("news", url, parse=_rss)
    # Google News иногда отдаёт записи вне окна — отрезаем по дате.
    return [i for i in items if i["date"] and after.isoformat() <= i["date"] <= before.isoformat()]


@dataclass
class Media:
    ok: bool = False
    windows: dict[str, list[dict]] = field(default_factory=dict)  # 0-6m, 6-12m, 12-24m, 24-48m


def media(terms: list[str], snapshot: date) -> Media:
    out = Media()
    spans = {"0-6m": (0, 0.5), "6-12m": (0.5, 1), "12-24m": (1, 2), "24-48m": (2, 4)}
    try:
        for name, (lo, hi) in spans.items():
            start, end = years_before(snapshot, hi), years_before(snapshot, lo)
            out.windows[name] = _news(terms, start, end if lo == 0 else end - timedelta(days=1))
        out.ok = True
    except RuntimeError:
        out.ok = False
    return out
