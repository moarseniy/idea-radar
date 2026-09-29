"""Keyless, bounded searches for corroborating patent and media records.

These sources supplement a candidate. Search-result snippets do not establish
commercial deployment, market size, or an exhaustive count of patent families.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlsplit

import httpx

from scripts.build_positive_dataset import ROOT, relevance, tokens

PATENT_SEARCH = "https://patents.google.com/xhr/query"
NEWS_SEARCH = "https://www.bing.com/news/search"
CACHE = ROOT / "storage" / "public_evidence_cache"
HEADERS = {"User-Agent": "IDEA-weak-signals/1.0 (research metadata)"}

EXTRA_COLUMNS = [
    "patent_search_status", "patent_source_title_original", "patent_source_url",
    "patent_publication_number", "patent_publication_date", "patent_assignee",
    "patent_evidence_snippet_original", "media_search_status",
    "media_source_title_original", "media_source_url", "media_source_date",
    "media_source_publisher", "media_evidence_snippet_original",
    "official_source_title_original", "official_source_url", "official_source_date",
    "funding_source_title_original", "funding_source_url", "funding_source_date",
    "funding_amount_original", "deployment_source_title_original",
    "deployment_source_url", "deployment_source_date",
    "technology_media_search_status", "technology_media_source_title_original",
    "technology_media_source_url", "technology_media_source_date",
    "technology_media_source_publisher", "technology_official_source_url",
    "technology_funding_source_title_original", "technology_funding_source_url",
    "technology_funding_source_date", "technology_funding_amount_original",
    "technology_context_scope",
]


def _cache_path(channel: str, query: str) -> Path:
    digest = hashlib.sha256(f"{channel}\0{query}".encode()).hexdigest()
    return CACHE / f"{digest}.{channel}"


def _fetch(channel: str, query: str) -> bytes:
    path = _cache_path(channel, query)
    if path.exists():
        return path.read_bytes()
    if channel == "patent":
        endpoint = PATENT_SEARCH
        params = {"url": "q=" + quote_plus(query), "exp": ""}
    else:
        endpoint = NEWS_SEARCH
        params = {"q": query, "format": "rss"}
    with httpx.Client(timeout=15, follow_redirects=True, headers=HEADERS) as client:
        response = client.get(endpoint, params=params)
        response.raise_for_status()
        payload = response.content
    CACHE.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{time.time_ns()}.tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    return payload


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", html.unescape(text or ""))).strip()


def _date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        try:
            return parsedate_to_datetime(value).date()
        except (TypeError, ValueError):
            return None


def _patents(candidate: dict, as_of: date) -> dict:
    response = json.loads(_fetch("patent", candidate["search_query"]))
    results = response.get("results", {}).get("cluster", [])
    choices = []
    for cluster in results:
        for result in cluster.get("result", []):
            patent = result.get("patent") or {}
            pub_date = _date(patent.get("publication_date"))
            number = patent.get("publication_number")
            if not number or not pub_date or pub_date > as_of:
                continue
            title = _clean(patent.get("title") or "")
            snippet = _clean(patent.get("snippet") or "")
            score = relevance(candidate, {"title": title, "abstract_text": snippet})
            if score > 0:
                choices.append((score, pub_date, patent, title, snippet))
    if not choices:
        return {"patent_search_status": "no_relevant_result"}
    _, pub_date, patent, title, snippet = max(choices, key=lambda x: (x[0], x[1]))
    number = patent["publication_number"]
    return {
        "patent_search_status": "found",
        "patent_source_title_original": title,
        "patent_source_url": f"https://patents.google.com/patent/{number}/en",
        "patent_publication_number": number,
        "patent_publication_date": pub_date.isoformat(),
        "patent_assignee": _clean(patent.get("assignee") or ""),
        "patent_evidence_snippet_original": snippet[:600],
    }


def _direct_news_url(link: str) -> str:
    parsed = urlsplit(link)
    if parsed.hostname and parsed.hostname.endswith("bing.com"):
        url = parse_qs(parsed.query).get("url", [""])[0]
        if url.startswith(("https://", "http://")):
            return url
    return link if link.startswith(("https://", "http://")) else ""


def _is_official(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host.endswith((".gov", ".edu", ".ac.uk", ".gov.uk", ".europa.eu"))


MONEY = re.compile(r"(?:[$€£¥]\s?\d+(?:[.,]\d+)?\s?(?:m|b|million|billion|млн|млрд)|\d+(?:[.,]\d+)?\s?(?:million|billion|млн|млрд)\s?(?:dollars|euros|USD|EUR))", re.IGNORECASE)
FUNDING = re.compile(r"\b(rais(?:es|ed|ing)|funding|investment|series [a-f]|seed round|grant|invests?)\b", re.IGNORECASE)
DEPLOYMENT = re.compile(r"\b(pilot|trial|deployed|deployment|rollout|first customer|commercial launch)\b", re.IGNORECASE)


def _news(candidate: dict, as_of: date) -> dict:
    root = ET.fromstring(_fetch("news", candidate["search_query"]))
    choices = []
    for entry in root.findall(".//item"):
        pub_date = _date(entry.findtext("pubDate"))
        if not pub_date or pub_date > as_of:
            continue
        url = _direct_news_url(entry.findtext("link") or "")
        if not url:
            continue
        title = _clean(entry.findtext("title") or "")
        description = _clean(entry.findtext("description") or "")
        if re.search(r"\bmarket (?:outlook|size|research report|forecast)\b", title, re.IGNORECASE):
            continue
        application_terms = tokens(candidate["application_original"])
        present_terms = tokens(f"{title} {description}")
        # A two-word application needs both terms. Matching only "security"
        # in "maritime security", for instance, is not corroboration.
        if len(application_terms) <= 2 and not application_terms.issubset(present_terms):
            continue
        score = relevance(candidate, {"title": title, "abstract_text": description})
        if score <= 0:
            continue
        source = entry.find("source")
        publisher = source.text.strip() if source is not None and source.text else (urlsplit(url).hostname or "")
        choices.append((score, pub_date, title, url, description, publisher))
    if not choices:
        return {"media_search_status": "no_relevant_result"}
    choices.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, pub_date, title, url, description, publisher = choices[0]
    result = {
        "media_search_status": "found",
        "media_source_title_original": title,
        "media_source_url": url,
        "media_source_date": pub_date.isoformat(),
        "media_source_publisher": publisher,
        "media_evidence_snippet_original": description[:600],
    }
    for _, day, item_title, item_url, snippet, _ in choices:
        text = f"{item_title} {snippet}"
        if _is_official(item_url) and "official_source_url" not in result:
            result.update(official_source_title_original=item_title,
                          official_source_url=item_url, official_source_date=day.isoformat())
        amount = MONEY.search(text)
        if amount and FUNDING.search(text) and "funding_source_url" not in result:
            result.update(funding_source_title_original=item_title,
                          funding_source_url=item_url, funding_source_date=day.isoformat(),
                          funding_amount_original=amount.group(0))
        if DEPLOYMENT.search(text) and "deployment_source_url" not in result:
            result.update(deployment_source_title_original=item_title,
                          deployment_source_url=item_url, deployment_source_date=day.isoformat())
    return result


def enrich_one(row: dict, as_of: date, channels: tuple[str, ...] = ("patent", "news")) -> dict:
    candidate = {"search_query": row["search_query"],
                 "technology_original": row["technology_original"],
                 "application_original": row["application_original"]}
    result = {}
    for channel, operation in (("patent", _patents), ("news", _news)):
        if channel not in channels:
            result[f"{channel if channel == 'patent' else 'media'}_search_status"] = "unavailable"
            continue
        try:
            result.update(operation(candidate, as_of))
        except (httpx.HTTPError, ET.ParseError, ValueError, KeyError, json.JSONDecodeError):
            result[f"{channel if channel == 'patent' else 'media'}_search_status"] = "failed"
    row.update(result)
    channel_count = 1 + int(bool(row.get("patent_source_url"))) + int(bool(row.get("media_source_url")))
    row["source_type_diversity"] = channel_count
    # Search snippets cannot establish a complete time series or verified pilots.
    # The corresponding missing_* flags and numeric features stay unchanged.
    return row


def enrich_rows(rows: list[dict], as_of: date, workers: int = 3,
                channels: tuple[str, ...] = ("patent", "news")) -> list[dict]:
    if rows and "patent" in channels:
        try:
            _fetch("patent", rows[0]["search_query"])
        except httpx.HTTPError:
            # Public patent search can temporarily deny bulk access. Stop here
            # instead of repeating hundreds of failing requests.
            channels = tuple(channel for channel in channels if channel != "patent")
    enriched = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 3))) as pool:
        tasks = {pool.submit(enrich_one, dict(row), as_of, channels): i for i, row in enumerate(rows)}
        for future in as_completed(tasks):
            enriched[tasks[future]] = future.result()
            done = sum(item is not None for item in enriched)
            if done % 50 == 0:
                print(f"external_enrichment={done}/{len(rows)}", flush=True)
    supplements = json.loads((ROOT / "scripts/manual_patent_sources.json").read_text(encoding="utf-8"))
    by_name = {item["candidate_name_original"]: item for item in supplements}
    for row in enriched:
        supplement = by_name.get(row["name_original"])
        if not supplement:
            continue
        publication = _date(supplement["patent_publication_date"])
        if publication is None or publication > as_of:
            continue
        row.update({k: supplement[k] for k in (
            "patent_source_title_original", "patent_source_url", "patent_publication_number",
            "patent_publication_date", "patent_assignee", "patent_evidence_snippet_original")})
        row["patent_search_status"] = "manual_page_review"
        row["source_type_diversity"] = 2 + int(bool(row.get("media_source_url")))
    events = json.loads((ROOT / "scripts/manual_event_sources.json").read_text(encoding="utf-8"))
    event_by_name = {item["candidate_name_original"]: item for item in events}
    for row in enriched:
        event = event_by_name.get(row["name_original"])
        if not event:
            continue
        funding_date = _date(event["funding_source_date"])
        if funding_date and funding_date <= as_of:
            for key in ("funding_source_title_original", "funding_source_url",
                        "funding_source_date", "funding_amount_original"):
                row[key] = event[key]
            if not row.get("media_source_url"):
                row.update(media_search_status="manual_page_review",
                           media_source_title_original=event["funding_source_title_original"],
                           media_source_url=event["funding_source_url"],
                           media_source_date=event["funding_source_date"],
                           media_source_publisher="GlobeNewswire (company statement)",
                           media_evidence_snippet_original=event["funding_evidence_snippet_original"])
        official_date = _date(event["official_source_date"])
        if official_date and official_date <= as_of:
            for key in ("official_source_title_original", "official_source_url", "official_source_date"):
                row[key] = event[key]
        row["source_type_diversity"] = (1 + int(bool(row.get("patent_source_url")))
                                        + int(bool(row.get("media_source_url")))
                                        + int(bool(row.get("official_source_url"))))
    return enriched


def _technology_news(technology: str, as_of: date) -> dict:
    try:
        root = ET.fromstring(_fetch("news", technology))
    except (httpx.HTTPError, ET.ParseError, ValueError):
        return {"technology_media_search_status": "failed"}
    topic = tokens(technology)
    matches = []
    for entry in root.findall(".//item"):
        day = _date(entry.findtext("pubDate"))
        url = _direct_news_url(entry.findtext("link") or "")
        if not day or day > as_of or not url:
            continue
        title = _clean(entry.findtext("title") or "")
        description = _clean(entry.findtext("description") or "")
        overlap = len(topic & tokens(f"{title} {description}")) / max(1, len(topic))
        if overlap < 0.6:
            continue
        source = entry.find("source")
        publisher = source.text.strip() if source is not None and source.text else (urlsplit(url).hostname or "")
        matches.append((overlap, day, title, url, description, publisher))
    if not matches:
        return {"technology_media_search_status": "no_relevant_result"}
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, day, title, url, _, publisher = matches[0]
    result = {
        "technology_media_search_status": "found",
        "technology_media_source_title_original": title,
        "technology_media_source_url": url,
        "technology_media_source_date": day.isoformat(),
        "technology_media_source_publisher": publisher,
        "technology_context_scope": "technology_only_not_application",
    }
    for _, event_day, event_title, event_url, description, _ in matches:
        text = f"{event_title} {description}"
        if _is_official(event_url) and not result.get("technology_official_source_url"):
            result["technology_official_source_url"] = event_url
        amount = MONEY.search(text)
        if amount and FUNDING.search(text) and not result.get("technology_funding_source_url"):
            result.update(technology_funding_source_title_original=event_title,
                          technology_funding_source_url=event_url,
                          technology_funding_source_date=event_day.isoformat(),
                          technology_funding_amount_original=amount.group(0))
    return result


def enrich_technology_context(rows: list[dict], as_of: date, workers: int = 3) -> list[dict]:
    technologies = sorted({row["technology_original"] for row in rows})
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 3))) as pool:
        futures = {pool.submit(_technology_news, technology, as_of): technology for technology in technologies}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    for row in rows:
        row.update(results[row["technology_original"]])
    return rows
