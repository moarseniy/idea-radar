"""Патенты: источник, не зависящий от режима веб-поиска и от FreePatentsOnline (B1, B3).

Порядок источников, первый непустой выигрывает:
  1. EPO OPS — только если заданы EPO_OPS_KEY и EPO_OPS_SECRET (OAuth2 client credentials).
     Поиск точной фразы в названии и реферате (ta=); ответы сверены с живым OPS 3.2.
  2. Google Patents xhr (основной, пока нет ключа OPS): поиск по короткой фразе, фильтр
     before=publication: (на дату оценки патент уже опубликован) и after=priority:.
  3. FreePatentsOnline через app.ml.features.evidence.search_patents_fpo.

Любой ответ фильтруется через relevant() по названию и реферату (B3): Google и FPO ищут
по словам, а не по фразе, и без фильтра приносят «Heat storage device».

Дата записи — дата приоритета (иначе подачи): когда идея была заявлена. Публикация всегда
не позже as_of, иначе патент в тот момент нельзя было увидеть.

patent_counts() возвращает ту же структуру Patents, что evidence.patents(): этап 5 вернёт
по ней признаки patent_* (B2).
"""
from __future__ import annotations

import html
import re
import time
import urllib.parse
from collections import Counter
from datetime import date, timedelta
from typing import Any

import httpx

from app.ml.env import env
from app.ml.features.evidence import Patents, years_before
from app.radar.connectors import Connector
from app.radar.connectors._terms import relevant, short_phrases
from app.radar.research_apis import _date_prefix, _document, _json_request, limiter

GOOGLE_XHR = "https://patents.google.com/xhr/query"
OPS_BASE = "https://ops.epo.org/3.2"
OPS_TOKEN_URL = f"{OPS_BASE}/auth/accesstoken"
# Обычный браузерный User-Agent: xhr-эндпоинт Google не отдаёт ответ «ботам» с чужими UA.
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
LOOKBACK_YEARS = 10

limiter("google_patents", 2.0)  # не чаще одного запроса в 2 с
limiter("epo_ops", 2.0)

_TAG = re.compile(r"<[^>]+>")
_TOKEN: dict[str, Any] = {"value": "", "expires": 0.0}


# ------------------------------------------------------------------ общие помощники

def _clean(text: Any) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", str(text or ""))).replace("…", " ").split())


def _ymd(day: date) -> str:
    return day.strftime("%Y%m%d")


def _phrases(terms: list[str]) -> list[str]:
    return short_phrases(terms)


def _is_relevant(text: str, phrases: list[str], terms: list[str]) -> bool:
    return relevant(text, phrases or terms)


def _ops_keys() -> tuple[str, str] | None:
    key, secret = env("EPO_OPS_KEY"), env("EPO_OPS_SECRET")
    return (key, secret) if key and secret else None


def _enabled(_terms: list[str]) -> bool:
    return True  # Google Patents не требует ключа


# ------------------------------------------------------------------ Google Patents xhr

def _google_query(phrases: list[str], *, before: tuple[str, date] | None = None,
                  after: tuple[str, date] | None = None, num: int = 10) -> dict:
    """Ответ xhr/query. before/after — пары ("priority"|"publication", дата)."""
    expression = " OR ".join(f'("{phrase}")' for phrase in phrases)
    inner = [("q", expression)]
    if before:
        inner.append(("before", f"{before[0]}:{_ymd(before[1])}"))
    if after:
        inner.append(("after", f"{after[0]}:{_ymd(after[1])}"))
    inner.append(("num", str(num)))
    # Внутренний запрос уходит одним параметром url, как в браузере.
    return _json_request("google_patents", GOOGLE_XHR,
                         params={"url": urllib.parse.urlencode(inner), "exp": ""},
                         headers={"User-Agent": BROWSER_UA}, timeout=20)


def _google_items(response: Any) -> list[dict]:
    results = (response or {}).get("results") or {}
    return [item.get("patent") or {} for cluster in results.get("cluster") or []
            for item in cluster.get("result") or []]


def _google_total(response: Any) -> int:
    try:
        return int(((response or {}).get("results") or {}).get("total_num_results") or 0)
    except (TypeError, ValueError):
        return 0


def _patent_ref(number: str, title: str, snippet: str, assignee: str, priority: str | None,
                publication: str | None, provider: str) -> dict:
    url = f"https://patents.google.com/patent/{number}/en"
    text = ". ".join(part for part in (title, snippet, f"Assignee: {assignee}" if assignee else "") if part)
    return _document(url, title, text, priority, "en", provider,
                     patent_number=number, assignee=assignee, publication_date=publication,
                     priority_date=priority, source_country=None)


def _google_refs(response: Any, phrases: list[str], terms: list[str], as_of: date) -> list[dict]:
    refs, seen = [], set()
    cutoff = as_of.isoformat()
    for patent in _google_items(response):
        number = str(patent.get("publication_number") or "").strip()
        title, snippet = _clean(patent.get("title")), _clean(patent.get("snippet"))
        priority = _date_prefix(patent.get("priority_date") or patent.get("filing_date"))
        publication = _date_prefix(patent.get("publication_date"))
        if not number or not title or not priority or number in seen:
            continue
        # Сервер уже режет по публикации, но проверка дешёвая и защищает от смены формата.
        if priority > cutoff or (publication and publication > cutoff):
            continue
        if not _is_relevant(f"{title} {snippet}", phrases, terms):
            continue
        seen.add(number)
        refs.append(_patent_ref(number, title, snippet, _clean(patent.get("assignee")), priority,
                                publication, "google_patents_xhr"))
    return refs


def _search_google(terms: list[str], as_of: date, limit: int) -> list[dict]:
    phrases = _phrases(terms)
    if not phrases:
        return []
    response = _google_query(phrases, before=("publication", as_of),
                             after=("priority", years_before(as_of, LOOKBACK_YEARS)),
                             num=min(max(limit * 2, 10), 50))
    return _google_refs(response, phrases, terms, as_of)[:limit]


# ------------------------------------------------------------------ EPO OPS (по документации)

def _ops_token(key: str, secret: str) -> str:
    """OAuth2 client credentials. Запрос формой, поэтому не через _json_request."""
    if _TOKEN["value"] and _TOKEN["expires"] > time.monotonic() + 30:
        return _TOKEN["value"]
    limiter("epo_ops").wait()
    response = httpx.post(OPS_TOKEN_URL, data={"grant_type": "client_credentials"},
                          auth=(key, secret), timeout=20)
    response.raise_for_status()
    payload = response.json()
    _TOKEN["value"] = payload["access_token"]
    _TOKEN["expires"] = time.monotonic() + float(payload.get("expires_in") or 1000)
    return _TOKEN["value"]


def _ops_get(path: str, params: dict | None = None) -> Any:
    keys = _ops_keys()
    if not keys:
        raise RuntimeError("EPO_OPS_KEY/EPO_OPS_SECRET не заданы")
    token = _ops_token(*keys)
    # Пустая выдача поиска у OPS — 404 (SERVER.EntityNotFound): это ноль, не сбой.
    return _json_request("epo_ops", f"{OPS_BASE}/rest-services/{path}", params=params, allow_404=True,
                         headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})


def _ops_cql(phrases: list[str], start: date | None, end: date) -> str:
    # ta all требует всех слов где угодно ("prompt injection" → «fault injection» в
    # реферате): ищем фразу. Дефис OPS читает как оператор — заменяем пробелом.
    text = " or ".join(f'ta="{phrase.replace("-", " ")}"' for phrase in phrases)
    if len(phrases) > 1:
        text = f"({text})"
    return f'{text} and pd within "{_ymd(start or date(1900, 1, 1))} {_ymd(end)}"'


def _list(value: Any) -> list:
    return value if isinstance(value, list) else [] if value is None else [value]


def _dollar(node: Any) -> str:
    return str(node.get("$", "")) if isinstance(node, dict) else str(node or "")


def _ops_total(response: Any) -> int:
    search = ((response or {}).get("ops:world-patent-data") or {}).get("ops:biblio-search") or {}
    try:
        return int(search.get("@total-result-count") or 0)
    except (TypeError, ValueError):
        return 0


def _ops_numbers(response: Any) -> list[str]:
    search = ((response or {}).get("ops:world-patent-data") or {}).get("ops:biblio-search") or {}
    refs = (search.get("ops:search-result") or {}).get("ops:publication-reference")
    numbers = []
    for ref in _list(refs):
        doc = ref.get("document-id") or {}
        parts = [_dollar(doc.get("country")), _dollar(doc.get("doc-number")), _dollar(doc.get("kind"))]
        if all(parts):
            numbers.append(".".join(parts))
    return numbers


def _ops_date(documents: list[dict], *kinds: str) -> str | None:
    for doc in documents:
        if doc.get("@document-id-type") in kinds and _dollar(doc.get("date")):
            return _date_prefix(_dollar(doc["date"]))
    return None


def _ops_records(response: Any) -> list[dict]:
    """Разбор ответа biblio: список записей с названием, рефератом, датами и заявителем."""
    data = ((response or {}).get("ops:world-patent-data") or {}).get("exchange-documents") or {}
    records = []
    for doc in _list(data.get("exchange-document")):
        biblio = doc.get("bibliographic-data") or {}
        titles = _list(biblio.get("invention-title"))
        title = next((_dollar(t) for t in titles if t.get("@lang") == "en"), "") or (
            _dollar(titles[0]) if titles else "")
        abstract = " ".join(_dollar(p) for a in _list(doc.get("abstract"))
                            if a.get("@lang", "en") == "en" for p in _list(a.get("p")))
        pub_docs = _list((biblio.get("publication-reference") or {}).get("document-id"))
        app_docs = _list((biblio.get("application-reference") or {}).get("document-id"))
        claims = _list((biblio.get("priority-claims") or {}).get("priority-claim"))
        claim_docs = [d for claim in claims for d in _list(claim.get("document-id"))]
        applicants = _list(((biblio.get("parties") or {}).get("applicants") or {}).get("applicant"))
        assignee = next((_dollar((a.get("applicant-name") or {}).get("name")) for a in applicants), "")
        number = f"{doc.get('@country', '')}{doc.get('@doc-number', '')}{doc.get('@kind', '')}"
        records.append({
            "number": number, "title": _clean(title), "abstract": _clean(abstract),
            "assignee": _clean(assignee), "publication": _ops_date(pub_docs, "docdb", "epodoc"),
            "priority": _ops_date(claim_docs, "docdb", "epodoc") or _ops_date(app_docs, "docdb", "epodoc")})
    return records


def _search_ops(terms: list[str], as_of: date, limit: int) -> list[dict]:
    phrases = _phrases(terms)
    if not phrases:
        return []
    # Ядра из двух слов ("key distribution") приносят соседние темы: сначала полные
    # фразы, ядра — только на оставшиеся места.
    full = [p for p in phrases if not any(p != other and p in other for other in phrases)]
    cores = [p for p in phrases if p not in full]
    refs: list[dict] = []
    for group in (full, cores):
        if group and len(refs) < limit:
            seen = {ref["patent_number"] for ref in refs}
            refs += [ref for ref in _ops_refs(group, terms, as_of, limit) if ref["patent_number"] not in seen]
    return refs[:limit]


def _ops_refs(phrases: list[str], terms: list[str], as_of: date, limit: int) -> list[dict]:
    start = years_before(as_of, LOOKBACK_YEARS)
    found = _ops_get("published-data/search", {"q": _ops_cql(phrases, start, as_of),
                                               "Range": f"1-{min(max(limit * 2, 10), 50)}"})
    numbers = _ops_numbers(found)
    if not numbers:
        return []
    biblio = _ops_get(f"published-data/publication/docdb/{','.join(numbers)}/biblio")
    refs, cutoff = [], as_of.isoformat()
    for record in _ops_records(biblio):
        day = record["priority"] or record["publication"]
        if not record["number"] or not record["title"] or not day or day > cutoff:
            continue
        if record["publication"] and record["publication"] > cutoff:
            continue
        if not _is_relevant(f"{record['title']} {record['abstract']}", phrases, terms):
            continue
        refs.append(_patent_ref(record["number"], record["title"], record["abstract"],
                                record["assignee"], day, record["publication"], "epo_ops"))
    return refs


# ------------------------------------------------------------------ FPO

def _search_fpo(terms: list[str], as_of: date, limit: int) -> list[dict]:
    from app.ml.features.evidence import search_patents_fpo

    phrases = _phrases(terms)
    if not phrases:
        return []
    refs = search_patents_fpo(phrases, as_of, limit=limit * 2, years_back=LOOKBACK_YEARS)
    out = []
    for ref in refs:
        title = _clean(ref.get("title"))
        if title and _is_relevant(title, phrases, terms):
            out.append({**ref, "title": title, "source_provider": "freepatentsonline_search",
                        "observed_at": None, "source_country": None,
                        "document": {"title": title, "text": title, "date": None,
                                     "language": "en", "extraction": "freepatentsonline_search"}})
    return out[:limit]


def search(terms: list[str], as_of: date, limit: int = 5) -> list[dict]:
    """Первый непустой источник из EPO OPS, Google Patents и FPO. Ошибки прячем до последнего."""
    branches = []
    if _ops_keys():
        branches.append(_search_ops)
    branches += [_search_google, _search_fpo]
    errors: list[Exception] = []
    for branch in branches:
        try:
            refs = branch(terms, as_of, limit)
        except Exception as exc:  # noqa: BLE001 — источник недоступен, пробуем следующий
            errors.append(exc)
            continue
        if refs:
            return refs
    if errors and len(errors) == len(branches):
        raise RuntimeError(f"патентные источники недоступны: {errors[0]}") from errors[0]
    return []


# ------------------------------------------------------------------ patent_counts

def _share(items: list[dict], phrases: list[str], terms: list[str]) -> tuple[float, list[dict]]:
    """Доля релевантных в выборке окна и сами релевантные записи."""
    good = [p for p in items if _is_relevant(f"{_clean(p.get('title'))} {_clean(p.get('snippet'))}",
                                              phrases, terms)]
    return (len(good) / len(items) if items else 0.0), good


def _counts_google(terms: list[str], phrases: list[str], as_of: date) -> Patents:
    out = Patents()
    # Окна те же, что у FPO-версии: заявки публикуются через ~18 месяцев, поэтому окна
    # роста сдвинуты на 1,5 года назад.
    lag = years_before(as_of, 1.5)
    b2, b4, b6 = (years_before(lag, y) for y in (2, 4, 6))
    b3, b10 = years_before(as_of, 3), years_before(as_of, 10)

    def window(kind: str, after: date | None, before: date) -> tuple[int, list[dict]]:
        response = _google_query(phrases, before=(kind, before), after=(kind, after) if after else None,
                                 num=15 if kind == "publication" or after is None else 10)
        items = _google_items(response)
        share, good = _share(items, phrases, terms)
        # Google ищет по словам, а не по фразе: totals сверяем с релевантностью первой страницы.
        return round(_google_total(response) * share), good

    w0, _ = window("priority", b2, lag)
    w1, _ = window("priority", b4, b2 - timedelta(days=1))
    w2, _ = window("priority", b6, b4 - timedelta(days=1))
    three, three_items = window("publication", b3, as_of)  # опубликовано к as_of
    older, older_items = window("priority", None, b6 - timedelta(days=1))
    older_10 = window("priority", None, b10 - timedelta(days=1))[0] if older else 0
    out.windows, out.count_3y, out.older_6y, out.older_10y = [w0, w1, w2], three, older, older_10
    age = next((a for a, n in ((15, older_10), (9, older - older_10), (6.5, w2), (4.5, w1),
                               (2.5, w0), (1, three)) if n > 0), None)
    out.first_priority = years_before(as_of, age).isoformat() if age else None
    out.titles = [_clean(p.get("title")) for p in three_items[:15]]
    out.assignee_counts_3y = dict(Counter(a for a in (_clean(p.get("assignee")) for p in three_items) if a))
    out.assignees_3y = list(out.assignee_counts_3y)
    out.assignees_before = list(dict.fromkeys(
        a for a in (_clean(p.get("assignee")) for p in older_items) if a))
    out.source, out.ok = "google_patents_xhr", True
    return out


def _counts_ops(phrases: list[str], as_of: date) -> Patents:
    out = Patents()
    lag = years_before(as_of, 1.5)
    b2, b4, b6 = (years_before(lag, y) for y in (2, 4, 6))
    b3, b10 = years_before(as_of, 3), years_before(as_of, 10)

    def total(after: date | None, before: date) -> int:
        # Range=1-1: нужен только @total-result-count. OPS индексирует даты публикации (pd).
        return _ops_total(_ops_get("published-data/search", {"q": _ops_cql(phrases, after, before),
                                                              "Range": "1-1"}))

    w0, w1, w2 = total(b2, lag), total(b4, b2 - timedelta(days=1)), total(b6, b4 - timedelta(days=1))
    three, older = total(b3, as_of), total(None, b6 - timedelta(days=1))
    older_10 = total(None, b10 - timedelta(days=1)) if older else 0
    out.windows, out.count_3y, out.older_6y, out.older_10y = [w0, w1, w2], three, older, older_10
    age = next((a for a, n in ((15, older_10), (9, older - older_10), (6.5, w2), (4.5, w1),
                               (2.5, w0), (1, three)) if n > 0), None)
    out.first_priority = years_before(as_of, age).isoformat() if age else None
    out.titles = [r["title"] for r in _search_ops_records(phrases, b3, as_of)]
    out.source, out.ok = "epo_ops", True
    return out


def _search_ops_records(phrases: list[str], start: date, end: date) -> list[dict]:
    numbers = _ops_numbers(_ops_get("published-data/search", {"q": _ops_cql(phrases, start, end),
                                                              "Range": "1-15"}))
    if not numbers:
        return []
    records = _ops_records(_ops_get(f"published-data/publication/docdb/{','.join(numbers)}/biblio"))
    return [r for r in records if r["title"] and _is_relevant(f"{r['title']} {r['abstract']}", phrases, [])]


def patent_counts(terms: list[str], as_of: date) -> Patents:
    """Патентные признаки на дату as_of в формате evidence.patents().

    Источники по порядку: EPO OPS (если есть ключи), Google Patents xhr, FPO; при сбое
    всех — Patents(ok=False, warning=...) без исключения, как у evidence.patents().
    """
    phrases = _phrases(terms)
    if not phrases:
        return Patents(source="unavailable", warning="нет латинских фраз для патентного поиска")
    warnings: list[str] = []
    empty: Patents | None = None
    branches = []
    if _ops_keys():
        branches.append(lambda: _counts_ops(phrases, as_of))
    branches.append(lambda: _counts_google(terms, phrases, as_of))
    for branch in branches:
        try:
            result = branch()
        except Exception as exc:  # noqa: BLE001 — переходим к следующему источнику
            warnings.append(str(exc)[:160])
            continue
        if result.ok and (result.count_3y or result.older_6y or any(result.windows)):
            return result
        if result.ok:  # пустой ответ: честный ноль, но пробуем следующий источник
            warnings.append(f"{result.source}: пусто")
            empty = empty or result
    from app.ml.features.evidence import _patents_from_fpo

    fpo = _patents_from_fpo(phrases, as_of)
    if fpo.ok:
        return fpo
    if empty:
        return empty
    fpo.warning = "; ".join([*warnings, fpo.warning])[:300]
    fpo.source = "unavailable"
    return fpo


CONNECTOR = Connector(
    name="Patents (Google/EPO)", channel="science", date_support="strict", search=search,
    enabled=_enabled, language="en", radar_limit=5, default_on=False,
    # Google Patents блокирует IP после нескольких запросов подряд: до ключа EPO OPS
    # патенты ищутся только по веткам запроса, не по каждому кандидату.
    scopes=("branch",))
