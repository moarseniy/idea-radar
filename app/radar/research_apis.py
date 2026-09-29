"""Structured public research, funding, adoption and deployment sources.

The adapters in this module are shared by live radar search and training-data
collection. They return the same URL/title/date/language/document shape so both
pipelines can ingest and quote-check source material identically.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.getenv("FEATURE_CACHE_DIR", str(ROOT / "ml" / ".cache" / "features"))) / "research_apis"
CACHE_TTL = 6 * 60 * 60
USER_AGENT = "IDEA-Research-Radar/1.0 (open technology-signal research)"


def openalex_enabled() -> bool:
    """Whether OpenAlex queries are enabled for both discovery and feature enrichment."""
    return os.getenv("OPENALEX_ENABLED", "0").strip() != "0"


class _Limiter:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_at)
            self.next_at = slot + self.interval
        time.sleep(max(0.0, slot - now))


LIMITERS = {
    # OpenAlex's anonymous public endpoint is especially sensitive to bursts.
    "openalex": _Limiter(1.1), "nih": _Limiter(1.05), "cordis": _Limiter(1.0),
    "github": _Limiter(6.1), "huggingface": _Limiter(0.5), "ted": _Limiter(0.2),
    "clinicaltrials": _Limiter(0.5), "openfda": _Limiter(0.5),
}
_SLOTS = threading.BoundedSemaphore(8)
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMITED_UNTIL: dict[str, float] = {}


def _cache_key(provider: str, method: str, url: str, params: dict | None,
               body: dict | None) -> str:
    payload = json.dumps([provider, method, url, params or {}, body or {}],
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _json_request(provider: str, url: str, *, params: dict | None = None,
                  body: dict | None = None, headers: dict | None = None,
                  allow_404: bool = False, timeout: float = 18) -> dict:
    key = _cache_key(provider, "POST" if body is not None else "GET", url, params, body)
    cache = CACHE_DIR / provider / f"{key}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL:
        return json.loads(cache.read_text(encoding="utf-8"))
    with _RATE_LIMIT_LOCK:
        retry_at = _RATE_LIMITED_UNTIL.get(provider, 0.0)
    if retry_at > time.monotonic():
        remaining = max(1, round(retry_at - time.monotonic()))
        raise RuntimeError(f"{provider} rate limited; cooling down for {remaining}s")
    if not _SLOTS.acquire(timeout=timeout):
        raise RuntimeError(f"{provider} request queue is full")
    try:
        last: Exception | None = None
        last_status: int | None = None
        for attempt in range(3):
            LIMITERS[provider].wait()
            try:
                request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}
                if body is not None:
                    response = httpx.post(url, json=body, params=params, headers=request_headers,
                                          timeout=timeout, follow_redirects=True)
                else:
                    response = httpx.get(url, params=params, headers=request_headers,
                                         timeout=timeout, follow_redirects=True)
                if allow_404 and response.status_code == 404:
                    return {}
                if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                response.raise_for_status()
                result = response.json()
                cache.parent.mkdir(parents=True, exist_ok=True)
                temp = cache.with_suffix(".tmp")
                temp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                temp.replace(cache)
                return result
            except Exception as exc:  # noqa: BLE001 — external APIs have different exception types
                last = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                last_status = status if isinstance(status, int) else None
                if status == 429:
                    # A public 429 applies to the shared service IP. Retrying every
                    # candidate/window only amplifies the throttle and stalls LR.
                    try:
                        cooldown = int(os.getenv("RESEARCH_API_429_COOLDOWN", "180"))
                    except ValueError:
                        cooldown = 180
                    cooldown = max(30, min(600, cooldown))
                    with _RATE_LIMIT_LOCK:
                        _RATE_LIMITED_UNTIL[provider] = time.monotonic() + cooldown
                    break
                if status not in {408, 425, 429, 500, 502, 503, 504} and status is not None:
                    break
                if attempt < 2:
                    retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After", "")
                    try:
                        delay = min(12.0, max(0.5, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = 0.75 * (2 ** attempt)
                    time.sleep(delay)
        status_note = f", HTTP {last_status}" if last_status else ""
        raise RuntimeError(f"{provider} API request failed ({type(last).__name__}{status_note})") from last
    finally:
        _SLOTS.release()


def _clean_terms(terms: list[str], limit: int = 4) -> list[str]:
    return list(dict.fromkeys(" ".join(str(t).split())[:180] for t in terms if str(t).strip()))[:limit]


def _abstract_from_index(index: dict | None) -> str:
    if not index:
        return ""
    words = [""] * (1 + max((max(pos) for pos in index.values() if pos), default=-1))
    for word, positions in index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = word
    return " ".join(w for w in words if w)


def _binding_value(row: dict, key: str) -> Any:
    return (row.get(key) or {}).get("value")


def _document(url: str, title: str, text: str, published: str | None,
              language: str | None, extraction: str, **meta: Any) -> dict:
    return {"url": url, "title": title or url, "source_language": language,
            "observed_at": published, "source_provider": extraction,
            "document": {"title": title or url, "text": text[:12000], "date": published,
                         "language": language, "extraction": extraction}, **meta}


def _date_prefix(value: Any) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    match = re.match(r"^(\d{4})(?:-?(\d{2}))?(?:-?(\d{2}))?", value)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{month or '01'}-{day or '01'}"


def openalex_works(terms: list[str], start: date, end: date, *, limit: int = 100,
                   include_affiliations: bool = True) -> list[dict]:
    """Normalized OpenAlex records; skip the larger affiliation payload when unused."""
    if not openalex_enabled():
        return []
    terms = _clean_terms(terms)
    if not terms:
        return []
    query = " OR ".join(f'"{term}"' for term in terms)
    fields = "id,doi,title,publication_date,cited_by_count,type,language,abstract_inverted_index,primary_location"
    if include_affiliations:
        fields = fields.replace("abstract_inverted_index", "authorships,abstract_inverted_index")
    params: dict[str, Any] = {
        "search": query,
        "filter": f"from_publication_date:{start.isoformat()},to_publication_date:{end.isoformat()}",
        "per-page": max(1, min(limit, 100)),
        "sort": "publication_date:desc",
        "select": fields,
        "mailto": os.getenv("OPENALEX_MAILTO", "").strip() or "hackathon@example.org",
    }
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    payload = _json_request("openalex", "https://api.openalex.org/works", params=params)
    rows = []
    for item in payload.get("results", []):
        title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
        abstract = _abstract_from_index(item.get("abstract_inverted_index"))
        if not title:
            continue
        doi = str(item.get("doi") or "").replace("https://doi.org/", "")
        authorships = item.get("authorships") or []
        institutions, countries = {}, set()
        industry_authorships = 0
        for authorship in authorships:
            author_institutions = authorship.get("institutions") or []
            if any(str(inst.get("type") or "").casefold() == "company" for inst in author_institutions):
                industry_authorships += 1
            for inst in author_institutions:
                if inst.get("id") or inst.get("display_name"):
                    institutions[str(inst.get("id") or inst.get("display_name"))] = inst.get("type") or ""
                if inst.get("country_code"):
                    countries.add(str(inst["country_code"]).upper())
        language = ((item.get("language") or {}).get("id") if isinstance(item.get("language"), dict)
                    else item.get("language"))
        primary_location = item.get("primary_location") or {}
        primary_source = primary_location.get("source") or {}
        rows.append({
            "id": str(item.get("id") or ""), "doi": doi, "title": title,
            "abstract": abstract, "publicationDate": _date_prefix(item.get("publication_date")),
            "citationCount": int(item.get("cited_by_count") or 0),
            "preprint": str(item.get("type") or "").casefold() == "preprint",
            "language": str(language or "") or None,
            "institutions": institutions, "countries": countries,
            "industry_affiliation_share": industry_authorships / len(authorships) if authorships else "",
            "primary_location_url": primary_location.get("landing_page_url"),
            "primary_source": primary_source.get("display_name"),
        })
    return rows


def search_openalex(terms: list[str], as_of: date, *, limit: int = 8) -> list[dict]:
    if not openalex_enabled():
        return []
    start = as_of - timedelta(days=365 * 6)
    refs = []
    for item in openalex_works(terms, start, as_of, limit=max(12, min(100, limit * 3))):
        abstract = item["abstract"]
        if not abstract:
            continue
        url = (f"https://doi.org/{item['doi']}" if item["doi"] else item["id"])
        if not url.startswith("https://"):
            continue
        refs.append(_document(url, item["title"], f"{item['title']}\n\n{abstract}",
                              item["publicationDate"], item["language"], "openalex_abstract",
                              source_country=min(item["countries"]) if item["countries"] else None))
    return refs[:limit]


def _nih_rows(terms: list[str], as_of: date, limit: int) -> list[dict]:
    terms = _clean_terms(terms, 3)
    if not terms:
        return []
    after = as_of - timedelta(days=365 * 4)
    search_text = " OR ".join(f'"{term}"' for term in terms)
    body = {
        "criteria": {
            "advanced_text_search": {"operator": "or", "search_field": "projecttitle,abstracttext",
                                     "search_text": search_text},
            "award_notice_date": {"from_date": after.isoformat(), "to_date": as_of.isoformat()},
            "fiscal_years": [], "include_active_projects": True,
        },
        "include_fields": ["ProjectTitle", "AbstractText", "FiscalYear", "AwardAmount", "OrgName",
                           "CoreProjectNum", "ProjectNum", "ProjectStartDate", "ProjectEndDate",
                           "AwardNoticeDate", "AgencyCode", "OpportunityNumber"],
        "offset": 0, "limit": max(1, min(limit, 10)), "sort_field": "award_notice_date", "sort_order": "desc",
    }
    response = _json_request("nih", "https://api.reporter.nih.gov/v2/projects/search", body=body, timeout=25)
    return response.get("results", [])


def _field(row: dict, *names: str) -> Any:
    normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in row.items()}
    for name in names:
        if re.sub(r"[^a-z0-9]", "", name.lower()) in normalized:
            return normalized[re.sub(r"[^a-z0-9]", "", name.lower())]
    return None


def _localized_text(value: Any) -> str:
    if isinstance(value, dict):
        localized = {str(key).lower(): val for key, val in value.items()}
        value = localized.get("eng") or localized.get("en") or next(iter(value.values()), "")
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    return str(value or "").strip()


def search_nih_grants(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    refs = []
    for row in _nih_rows(terms, as_of, limit):
        title = str(_field(row, "ProjectTitle", "project_title") or "").strip()
        abstract = str(_field(row, "AbstractText", "abstract_text") or "").strip()
        project_number = str(_field(row, "CoreProjectNum", "ProjectNum", "project_num") or "").strip()
        if not title or not project_number:
            continue
        org_data = row.get("organization") or row.get("org") or {}
        org = str(_field(row, "OrgName", "org_name") or
                  _field(org_data, "OrgName", "org_name", "name") or "").strip()
        awarded = _date_prefix(_field(row, "AwardNoticeDate")) or _date_prefix(_field(row, "ProjectStartDate"))
        amount = _field(row, "AwardAmount")
        try:
            amount_usd_m = round(float(amount) / 1_000_000, 6) if amount is not None else None
        except (TypeError, ValueError):
            amount_usd_m = None
        url = f"https://reporter.nih.gov/project-details/{project_number}"
        text = (f"Federal research grant. Award/project: {title}. Organization: {org}. "
                f"Award notice date: {awarded or 'unknown'}. Award amount (USD): {amount or 'unknown'}.\n\n{abstract}")
        refs.append(_document(url, title, text, awarded, "en", "nih_reporter_project",
                              structured_event={"type": "grant", "date": awarded, "company": org,
                                                "amount_usd_m": amount_usd_m, "round": "grant",
                                                "strategic": False, "source_id": project_number},
                              source_country="US"))
    return refs


def _sparql_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def search_cordis_grants(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    terms = _clean_terms(terms, 3)
    if not terms:
        return []
    start = as_of - timedelta(days=365 * 5)
    tests = " || ".join(f"CONTAINS(LCASE(STR(?title)), LCASE({_sparql_literal(t)})) || "
                        f"CONTAINS(LCASE(STR(?abstract)), LCASE({_sparql_literal(t)}))" for t in terms)
    query = f"""PREFIX eurio: <http://data.europa.eu/s66#>
SELECT DISTINCT ?project_id ?title ?abstract ?start_date ?funding_amount WHERE {{
  ?project a eurio:Project ; eurio:identifier ?project_id ; eurio:title ?title ; eurio:startDate ?start_date .
  OPTIONAL {{ ?project eurio:abstract ?abstract . }}
  OPTIONAL {{
    ?project eurio:hasInvolvedParty ?participant .
    ?participant eurio:isRecipientOf ?grant_amount .
    ?grant_amount eurio:hasPaymentAmount ?monetary_amount .
    ?monetary_amount eurio:value ?funding_amount .
  }}
  FILTER(?start_date >= \"{start.isoformat()}\"^^<http://www.w3.org/2001/XMLSchema#date> &&
         ?start_date <= \"{as_of.isoformat()}\"^^<http://www.w3.org/2001/XMLSchema#date>)
  FILTER({tests})
}} LIMIT {max(1, min(limit, 8))}"""
    payload = _json_request("cordis", "https://cordis.europa.eu/datalab/sparql",
                            params={"query": query}, headers={"Accept": "application/sparql-results+json"},
                            timeout=25)
    bindings = ((payload.get("results") or {}).get("bindings") or [])
    by_id = {}
    for row in bindings:
        project_id = str(_binding_value(row, "project_id") or "").strip()
        title = str(_binding_value(row, "title") or "").strip()
        if not project_id or not title:
            continue
        entry = by_id.setdefault(project_id, {"title": title, "abstract": str(_binding_value(row, "abstract") or ""),
                                              "date": _date_prefix(_binding_value(row, "start_date")), "amount": None})
        try:
            funding_amount = _binding_value(row, "funding_amount")
            if funding_amount is not None:
                entry["amount"] = max(float(entry["amount"] or 0), float(funding_amount))
        except (TypeError, ValueError):
            pass
    refs = []
    for project_id, item in by_id.items():
        title, abstract, published = item["title"], item["abstract"], item["date"]
        amount = item["amount"]
        text = (f"European Commission-funded research project. Project: {title}. "
                f"Start date: {published or 'unknown'}. Maximum observed funding amount (EUR): "
                f"{amount if amount is not None else 'unknown'}.\n\n{abstract}")
        refs.append(_document(f"https://cordis.europa.eu/project/id/{project_id}", title, text,
                              published, "en", "cordis_project",
                              structured_event={"type": "grant", "date": published, "company": "",
                                                "amount_usd_m": None, "amount_native": amount,
                                                "currency": "EUR", "round": "grant", "strategic": False,
                                                "source_id": project_id}, source_country="EU"))
    return refs


def search_funding(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    if os.getenv("FEATURE_STRUCTURED_SOURCES", "1") == "0":
        return []
    refs = []
    for provider in (search_nih_grants, search_cordis_grants):
        try:
            refs.extend(provider(terms, as_of, limit=limit))
        except Exception:  # noqa: BLE001, S112 — isolate provider failures and keep partial evidence
            continue
    unique = {str(item.get("structured_event", {}).get("source_id") or item["url"]): item for item in refs}
    return list(unique.values())[:limit * 2]


def search_github(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    terms = _clean_terms(terms, 2)
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    params = {"q": f"{expression} pushed:>={(as_of - timedelta(days=1095)).isoformat()}",
              "sort": "updated", "order": "desc", "per_page": max(1, min(limit, 8))}
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = _json_request("github", "https://api.github.com/search/repositories", params=params,
                            headers=headers)
    refs = []
    for repo in payload.get("items", []):
        name, url = str(repo.get("full_name") or ""), str(repo.get("html_url") or "")
        if not name or not url:
            continue
        title = str(repo.get("description") or name).strip()
        pushed = _date_prefix(repo.get("pushed_at"))
        text = (f"Public GitHub repository: {name}. Description: {title}. "
                f"Primary language: {repo.get('language') or 'unknown'}. "
                f"Stars: {repo.get('stargazers_count', 0)}; forks: {repo.get('forks_count', 0)}; "
                f"open issues: {repo.get('open_issues_count', 0)}; last push: {pushed or 'unknown'}. "
                f"Topics: {', '.join(repo.get('topics') or [])}.")
        refs.append(_document(url, title, text, pushed, "en", "github_repository",
                              source_country=None,
                              adoption_metrics={"stars": repo.get("stargazers_count", 0),
                                                "forks": repo.get("forks_count", 0),
                                                "pushed_at": pushed}))
    return refs


def search_huggingface(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    terms = _clean_terms(terms, 2)
    refs = []
    for term in terms:
        params = {"search": term, "limit": max(1, min(limit, 8)), "sort": "downloads",
                  "direction": -1, "full": "true"}
        headers = {}
        token = os.getenv("HF_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = _json_request("huggingface", "https://huggingface.co/api/models", params=params,
                                headers=headers)
        for model in payload if isinstance(payload, list) else []:
            model_id = str(model.get("modelId") or model.get("id") or "").strip()
            if not model_id:
                continue
            title = f"Hugging Face model {model_id}"
            updated = _date_prefix(model.get("lastModified"))
            pipeline = str(model.get("pipeline_tag") or "unknown")
            tags = model.get("tags") or []
            text = (f"Public Hugging Face model: {model_id}. Task: {pipeline}. "
                    f"Downloads: {model.get('downloads', 0)}; likes: {model.get('likes', 0)}; "
                    f"last update: {updated or 'unknown'}. Tags: {', '.join(map(str, tags[:12]))}.")
            refs.append(_document(f"https://huggingface.co/{model_id}", title, text,
                                  updated, "en", "huggingface_model",
                                  adoption_metrics={"downloads": model.get("downloads", 0),
                                                    "likes": model.get("likes", 0), "last_update": updated}))
    unique = {item["url"]: item for item in refs}
    return list(unique.values())[:limit]


def search_developer_adoption(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    refs = []
    for provider in (search_github, search_huggingface):
        try:
            refs.extend(provider(terms, as_of, limit=limit))
        except Exception:  # noqa: BLE001, S112 — isolate provider failures and keep partial evidence
            continue
    unique = {item["url"]: item for item in refs}
    return list(unique.values())[:limit * 2]


def search_ted(terms: list[str], as_of: date, *, limit: int = 6) -> list[dict]:
    terms = _clean_terms(terms, 2)
    if not terms:
        return []
    q = " OR ".join(f"FT = {_sparql_literal(term)}" for term in terms)
    q += f" AND PD >= {(as_of - timedelta(days=365 * 5)):%Y%m%d}"
    body = {"query": q, "fields": ["publication-number", "notice-title", "publication-date", "buyer-name",
                                    "description-proc", "description-lot", "notice-purpose",
                                    "contract-nature", "classification-cpv"],
            "limit": max(1, min(limit, 10)), "scope": "ALL", "checkQuerySyntax": False,
            "paginationMode": "PAGE_NUMBER"}
    payload = _json_request("ted", "https://api.ted.europa.eu/v3/notices/search", body=body)
    notices = payload.get("notices") or payload.get("results") or []
    refs = []
    for notice in notices:
        pub = str(_field(notice, "publication-number", "publicationNumber", "publication_number") or "").strip()
        title_value = _field(notice, "notice-title", "noticeTitle", "title")
        title = _localized_text(title_value)
        published = _date_prefix(_field(notice, "publication-date", "publicationDate", "publication_date"))
        buyer = _field(notice, "buyer-name", "buyerName", "buyer_name")
        descriptions = [_localized_text(_field(notice, field)) for field in
                        ("description-proc", "description-lot", "notice-purpose")]
        contract_nature = _localized_text(_field(notice, "contract-nature"))
        cpv = _localized_text(_field(notice, "classification-cpv"))
        buyer_text = _localized_text(buyer)
        description_text = " ".join(part for part in descriptions if part)
        if not pub or not title:
            continue
        url = f"https://ted.europa.eu/en/notice/{pub}/xml"
        text = (f"EU public procurement notice. Title: {title}. Buyer: {buyer_text}. "
                f"Date: {published or 'unknown'}. Contract nature: {contract_nature or 'unknown'}. "
                f"CPV: {cpv or 'unknown'}. {description_text}")
        refs.append(_document(url, title, text, published, "en", "ted_procurement_notice",
                              structured_event={"type": "procurement", "date": published,
                                                "company": "", "customer": buyer_text or None,
                                                "source_id": pub}))
    return refs


HEALTH_HINTS = ("health", "medical", "medicine", "clinical", "biotech", "biomedical", "pharma", "drug",
                "diagnostic", "therapeutic", "therapy", "cancer", "vaccine", "patient", "disease",
                "insulin", "diabetes", "implant", "prosthetic", "hospital", "surgical", "gene therapy",
                "здрав", "медицин", "клиничес", "биотех", "фарма", "лекарств", "диагност",
                "терап", "онколог", "вакцин", "пациент", "болезн", "инсулин", "диабет", "имплант",
                "хирург", "больниц")


def is_health_topic(terms: list[str]) -> bool:
    text = " ".join(terms).casefold()
    return any(hint in text for hint in HEALTH_HINTS)


def search_clinicaltrials(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    if not is_health_topic(terms):
        return []
    query = " OR ".join(_clean_terms(terms, 3))
    payload = _json_request("clinicaltrials", "https://clinicaltrials.gov/api/v2/studies",
                            params={"query.term": query, "pageSize": max(1, min(limit, 10)),
                                    "format": "json", "countTotal": "false",
                                    "sort": "StudyFirstPostDate:desc"})
    refs = []
    for study in payload.get("studies", []):
        protocol = study.get("protocolSection") or {}
        ident = protocol.get("identificationModule") or {}
        status = protocol.get("statusModule") or {}
        design = protocol.get("designModule") or {}
        desc = protocol.get("descriptionModule") or {}
        nct = str(ident.get("nctId") or "").strip()
        title = str(ident.get("briefTitle") or "").strip()
        summary = str(desc.get("briefSummary") or "").strip()
        posted = _date_prefix((status.get("studyFirstPostDateStruct") or {}).get("date"))
        updated = _date_prefix((status.get("studyFirstPostDateStruct") or {}).get("date"))
        current_status = str(status.get("overallStatus") or "unknown")
        if not nct or not title:
            continue
        text = (f"Clinical study {nct}. Title: {title}. Study type: {design.get('studyType') or 'unknown'}. "
                f"Status: {current_status}. First posted: {posted or 'unknown'}. {summary}")
        refs.append(_document(f"https://clinicaltrials.gov/study/{nct}", title, text, updated,
                              "en", "clinicaltrials_api",
                              structured_event={"type": "trial", "date": posted,
                                                "source_id": nct, "status": current_status}))
    return refs


def _fda_rows(endpoint: str, query: str, limit: int, api_key: str) -> list[dict]:
    params: dict[str, Any] = {"search": query, "limit": max(1, min(limit, 5))}
    if api_key:
        params["api_key"] = api_key
    try:
        payload = _json_request("openfda", f"https://api.fda.gov/{endpoint}.json", params=params,
                                allow_404=True)
    except RuntimeError:
        return []
    return payload.get("results", [])


def search_openfda(terms: list[str], as_of: date, *, limit: int = 4) -> list[dict]:
    if not is_health_topic(terms):
        return []
    term = _clean_terms(terms, 1)[0]
    tokens = list(dict.fromkeys(re.findall(r"[A-Za-z0-9-]{3,}", term)))[:4]
    if not tokens:
        return []
    device_query = " OR ".join(f"device_name:{token}" for token in tokens)
    drug_query = " OR ".join(f"openfda.{field}:{token}" for token in tokens
                              for field in ("generic_name", "brand_name"))
    api_key = os.getenv("OPENFDA_API_KEY", "").strip()
    results = []
    for row in _fda_rows("device/510k", device_query, limit, api_key):
        name = str(row.get("device_name") or "").strip()
        k_number = str(row.get("k_number") or "").strip()
        decision = _date_prefix(row.get("decision_date"))
        if not name or not k_number:
            continue
        url = f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={k_number}"
        text = (f"FDA 510(k) medical-device record. Device: {name}. Decision: {row.get('decision_description') or ''}. "
                f"Decision date: {decision or 'unknown'}. Applicant: {row.get('applicant') or 'unknown'}. "
                f"Product code: {row.get('product_code') or 'unknown'}. K-number: {k_number}.")
        results.append(_document(url, name, text, decision, "en", "openfda_510k",
                                 structured_event={"type": "regulation", "date": decision,
                                                   "source_id": k_number}))
    for row in _fda_rows("drug/drugsfda", drug_query, limit, api_key):
        application = str(row.get("application_number") or "").strip()
        products = row.get("products") or []
        name = str((products[0].get("brand_name") or products[0].get("active_ingredients") or "FDA drug record")
                   if products else "FDA drug record")
        decision = _date_prefix(row.get("submissions", [{}])[0].get("submission_status_date")
                                if row.get("submissions") else None)
        if not application:
            continue
        url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={application}"
        text = f"FDA drug approval record. Product: {name}. Application: {application}. Latest submission date: {decision or 'unknown'}."
        results.append(_document(url, name, text, decision, "en", "openfda_drug_approval",
                                 structured_event={"type": "regulation", "date": decision,
                                                   "source_id": application}))
    return results[:limit]


def search_health(terms: list[str], as_of: date, *, limit: int = 5) -> list[dict]:
    refs = []
    for provider in (search_clinicaltrials, search_openfda):
        try:
            refs.extend(provider(terms, as_of, limit=limit))
        except Exception:  # noqa: BLE001, S112 — isolate provider failures and keep partial evidence
            continue
    unique = {item["url"]: item for item in refs}
    return list(unique.values())[:limit * 2]


def search_structured_sources(terms: list[str], as_of: date, *, per_source: int = 5,
                              include_openalex: bool = True) -> list[dict]:
    """All default-on direct API connectors, shared by radar and data mining."""
    if os.getenv("FEATURE_STRUCTURED_SOURCES", "1") == "0":
        return []
    providers = (search_funding, search_developer_adoption, search_ted, search_health)
    if include_openalex and openalex_enabled():
        providers = (search_openalex, *providers)
    refs = []
    for provider in providers:
        try:
            refs.extend(provider(terms, as_of, limit=per_source))
        except Exception:  # noqa: BLE001, S112 — isolate provider failures and keep partial evidence
            continue
    unique = {item["url"]: item for item in refs}
    return list(unique.values())
