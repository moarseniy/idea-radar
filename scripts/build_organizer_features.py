"""Build local-validation 63-feature rows from translated titles and open evidence.

Only organizer ID/title/domain are used from the workbook. Author rationale,
stage, trend, score, companies, and curated link counts never enter features.
Unknown/unmeasured values stay blank; this collector does not invent zeros.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import quote

import httpx

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from scripts.build_positive_dataset import (ROOT, CrossrefClient, normalize_crossref_work,
                                            relevance, row_from_candidate, tokens)
from scripts.enrich_public_sources import EXTRA_COLUMNS, enrich_one
from scripts.prepare_organizer_validation import DEFAULT_OUTPUT, build as build_manifest

SNAPSHOT = date(2026, 9, 22)
DEFAULT_OUTPUT = ROOT / "data/training/organizer_100_features.csv"
TRANSLATION_CACHE = ROOT / "storage/organizer_feature_cache/translations"

CONNECTORS = re.compile(
    r"\s+(?:for use in|for|within|across|inside|in|on|with|using|through|as|для|во|в|на|при|под|от|между|с помощью|как)\s+",
    re.I,
)
WHITESPACE = re.compile(r"\s+")


def _translation_path(text: str) -> Path:
    return TRANSLATION_CACHE / f"{hashlib.sha256(text.encode()).hexdigest()}.json"


def translate_title(text: str) -> tuple[str, str]:
    latin = sum("LATIN" in __import__("unicodedata").name(c, "") for c in text if c.isalpha())
    cyr = sum("CYRILLIC" in __import__("unicodedata").name(c, "") for c in text if c.isalpha())
    if cyr == 0 or latin / max(1, latin + cyr) >= 0.8:
        return text, "original_mostly_latin"
    cache = _translation_path(text)
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data["translated"], data.get("provider", "cached_translation")
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = httpx.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": "ru|en"},
            timeout=25,
            headers={"User-Agent": "IDEA-weak-signals/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        translated = html.unescape((payload.get("responseData") or {}).get("translatedText") or "").strip()
        if int(payload.get("responseStatus") or 0) != 200:
            return text, f"mymemory_status_{payload.get('responseStatus', 'unknown')}_fallback_original"
    except (httpx.HTTPError, ValueError, TypeError):
        return text, "mymemory_failed_fallback_original"
    translated = re.sub(r"\bedge iron\b", "edge hardware", translated, flags=re.I)
    if not translated:
        return text, "translation_empty_fallback_original"
    cache.write_text(json.dumps({"original": text, "translated": translated,
                                 "provider": "mymemory"},
                                ensure_ascii=False), encoding="utf-8")
    time.sleep(0.7)
    return translated, "mymemory"


def split_technology_application(title_en: str, domain_en: str) -> tuple[str, str, str]:
    title = WHITESPACE.sub(" ", title_en).strip(" .,:;–—-")
    for match in CONNECTORS.finditer(title):
        tech = title[:match.start()].strip(" .,:;–—-")
        app = title[match.end():].strip(" .,:;–—-")
        if len(tokens(tech)) >= 2 and len(tokens(app)) >= 2:
            return tech, app, f"title_connector:{match.group(0).strip().casefold()}"
    if ":" in title:
        left, right = (part.strip() for part in title.split(":", 1))
        if len(tokens(left)) >= 2 and len(tokens(right)) >= 2:
            return left, right, "title_colon"
    return title, domain_en, "full_title_plus_translated_domain"


def query_candidate(client: CrossrefClient, candidate: dict,
                    full_title_query: str) -> tuple[list[tuple[float, dict]], dict[int, int], str]:
    """Search the technology/application pair, then fall back to full-title overlap."""
    query = f"{candidate['technology_original']} {candidate['application_original']}"
    data = client.get(query, SNAPSHOT)
    works = [normalize_crossref_work(item) for item in data.get("items", [])]
    ranked = sorted(((relevance(candidate, work), work) for work in works),
                    key=lambda item: (item[0], item[1].get("publication_date") or ""), reverse=True)
    ranked = [(score, work) for score, work in ranked if score >= 0.28]
    scope = "technology_and_application_query"
    if not ranked:
        data = client.get(full_title_query, SNAPSHOT)
        works = [normalize_crossref_work(item) for item in data.get("items", [])]
        query_tokens = tokens(full_title_query)
        ranked = []
        for work in works:
            corpus_tokens = tokens(f"{work.get('title') or ''} {work.get('abstract_text') or ''}")
            score = len(query_tokens & corpus_tokens) / max(1, len(query_tokens))
            if score >= 0.2:
                ranked.append((score, work))
        ranked.sort(key=lambda item: (item[0], item[1].get("publication_date") or ""), reverse=True)
        scope = "full_title_overlap_fallback"
    years = Counter(work["publication_year"] for score, work in ranked
                    if work.get("publication_year"))
    return ranked, dict(years), scope


def build_one(row: dict, client: CrossrefClient) -> dict:
    translated, translation_status = translate_title(row["name_original"])
    domain_en, domain_status = translate_title(row["domain_original"])
    technology_part, application_part, split_method = split_technology_application(translated, domain_en)
    technology, technology_translation_status = translate_title(technology_part)
    application, application_translation_status = translate_title(application_part)
    candidate = {
        "domain_original": row["domain_original"], "domain_ru": row["domain_original"],
        "technology_original": technology, "technology_ru": row["name_original"],
        "application_original": application, "application_ru": row["domain_original"],
        "name_original": translated, "name_ru": row["name_original"],
        "original_language": "mixed_or_ru", "search_query": f"{technology} {application}",
    }
    result = {name: "" for name in FEATURE_NAMES}
    result.update({"missing_science": 1, "missing_patents": 1, "missing_media": 1,
                   "missing_funding": 1, "missing_adoption": 1})
    ranked, years, query_scope = query_candidate(client, candidate, translated)
    base = row_from_candidate(candidate, ranked, years, SNAPSHOT,
                              allow_mature=True, require_recent=False) if ranked else None
    metadata = {
        "organizer_id": row["organizer_id"], "name_original": row["name_original"],
        "domain_original": row["domain_original"], "snapshot_date": SNAPSHOT.isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "translated_query": translated, "technology_query": technology,
        "application_query": application, "query_split_method": split_method,
        "technology_original": technology, "application_original": application,
        "search_query": f"{technology} {application}",
        "translation_status": translation_status,
        "technology_translation_status": technology_translation_status,
        "application_translation_status": application_translation_status,
        "domain_translation_status": domain_status,
        "crossref_status": "matched" if ranked else "no_relevant_work_in_top_200",
        "crossref_match_scope": query_scope,
        "crossref_relevant_works": len(ranked),
    }
    if base:
        result.update({name: base[name] for name in FEATURE_NAMES})
        metadata.update({
            "primary_source_title_original": base["primary_source_title_original"],
            "primary_source_url": base["primary_source_url"],
            "primary_source_date": base["primary_source_date"],
            "primary_source_publisher": base["primary_source_publisher"],
            "source_evidence_excerpt_original": base["source_evidence_excerpt_original"],
            "source_relevance_score": base["source_relevance_score"],
            "science_count_scope": "relevant_works_in_top_200_crossref_query_2018_to_snapshot",
        })
        if query_scope == "full_title_overlap_fallback":
            metadata["science_count_scope"] = "top_200_full_title_query_20pct_token_overlap_preliminary"
    else:
        metadata.update({"primary_source_title_original": "", "primary_source_url": "",
                        "primary_source_date": "", "primary_source_publisher": "",
                        "source_evidence_excerpt_original": "", "source_relevance_score": "",
                        "science_count_scope": "no_relevant_crossref_match"})
    record = {**metadata, **result, **{column: "" for column in EXTRA_COLUMNS}}
    external = enrich_one(record, SNAPSHOT, channels=("news",))
    external["source_type_diversity"] = (
        int(bool(external.get("primary_source_url")))
        + int(bool(external.get("patent_source_url")))
        + int(bool(external.get("media_source_url")))
    )
    return external


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT.parent / "organizer_validation_manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not args.manifest.exists():
        build_manifest(args.manifest)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError(f"Expected 100 organizer validation rows, found {len(rows)}")
    client = CrossrefClient(delay=1.1)
    results: list[dict | None] = [None] * len(rows)
    errors = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 3))) as pool:
            tasks = {pool.submit(build_one, row, client): index for index, row in enumerate(rows)}
            done = 0
            for future in as_completed(tasks):
                index = tasks[future]
                done += 1
                try:
                    results[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    source = rows[index]
                    blank = {name: "" for name in FEATURE_NAMES}
                    blank.update({"missing_science": 1, "missing_patents": 1,
                                  "missing_media": 1, "missing_funding": 1,
                                  "missing_adoption": 1})
                    results[index] = {"organizer_id": source["organizer_id"],
                                      "name_original": source["name_original"],
                                      "domain_original": source["domain_original"],
                                      "snapshot_date": SNAPSHOT.isoformat(),
                                      "feature_schema_version": FEATURE_SCHEMA_VERSION,
                                      "crossref_status": "failed", **blank}
                    errors.append({"organizer_id": source["organizer_id"],
                                   "error": type(exc).__name__,
                                   "detail": str(exc)[:300]})
                if done % 10 == 0:
                    print(f"organizer_feature_rows={done}/{len(rows)}", flush=True)
    finally:
        client.close()
    final_rows = [row for row in results if row is not None]
    fields = ["organizer_id", "name_original", "domain_original", "snapshot_date",
              "feature_schema_version", "translated_query", "technology_query",
              "application_query", "query_split_method", "translation_status",
              "technology_translation_status", "application_translation_status",
              "domain_translation_status", "crossref_status", "crossref_match_scope",
              "crossref_relevant_works",
              "primary_source_title_original", "primary_source_url", "primary_source_date",
              "primary_source_publisher", "source_evidence_excerpt_original",
              "source_relevance_score", "science_count_scope", *EXTRA_COLUMNS, *FEATURE_NAMES]
    # Preserve first occurrence while removing columns already present in EXTRA_COLUMNS.
    fields = list(dict.fromkeys(fields))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)
    summary = {
        "rows": len(final_rows), "schema_version": FEATURE_SCHEMA_VERSION,
        "crossref_status": dict(Counter(row.get("crossref_status", "") for row in final_rows)),
        "media_found": sum(bool(row.get("media_source_url")) for row in final_rows),
        "official_found": sum(bool(row.get("official_source_url")) for row in final_rows),
        "funding_found": sum(bool(row.get("funding_source_url")) for row in final_rows),
        "deployment_found": sum(bool(row.get("deployment_source_url")) for row in final_rows),
        "measured_feature_counts": {feature: sum(bool(row.get(feature, "")) for row in final_rows)
                                    for feature in FEATURE_NAMES},
        "errors": errors,
        "note": "Features are computed from title-query Crossref matches and Bing News snippets; unavailable exhaustive counts remain blank.",
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(report_path),
                      "crossref_status": summary["crossref_status"],
                      "media_found": summary["media_found"], "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
