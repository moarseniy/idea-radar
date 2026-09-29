"""Append 100 distinct, source-backed N3b product-name negatives.

Run after build_negative_dataset.py. GitHub metadata is cached so the same
candidate universe can be inspected without an API key on later runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from scripts.build_negative_dataset import ROOT, SNAPSHOT

DEFAULT_FILE = ROOT / "data/training/negative_weak_signals_global.csv"
EXTENSION_TAG = "github_sector_product_name_v1"
SECTORS = [
    ("personal_finance", "topic:personal-finance", "Personal finance products", "Продукты для личных финансов", 20),
    ("robotics", "topic:robotics", "Robotics products", "Робототехнические продукты", 3),
    ("healthcare", "topic:healthcare", "Healthcare products", "Медицинские программные продукты", 9),
    ("education", "topic:education", "Education products", "Образовательные продукты", 5),
    ("self_hosted", "topic:self-hosted", "Self-hosted products", "Самостоятельно размещаемое ПО", 23),
    ("project_management", "topic:project-management", "Project management products", "Продукты для управления проектами", 19),
    ("photo_management", "topic:photo-management", "Photo management products", "Продукты для управления фотографиями", 10),
    ("crm", "topic:crm", "CRM products", "CRM-продукты", 10),
    ("home_automation", "topic:home-automation", "Home automation products", "Продукты для автоматизации дома", 1),
]

EXCLUDE = re.compile(
    r"awesome|guide|tutorial|skills|library|sdk|framework|template|mcp|plugin|nvim|docs|examples|workflow builder|tools list|curated|dataset|papers|benchmark|course|algorithm|protocol|model|research|extension|configuration|config|frontend|backend|boilerplate|collection|demo",
    re.IGNORECASE,
)
INCLUDE = re.compile(
    r"\b(app|application|assistant|platform|client|workspace|product|studio|workbench|server|system|dashboard|manager|tool|software)\b",
    re.IGNORECASE,
)
GENERIC_NAMES = {
    "platform", "core", "frontend", "backend", "android", "ios", "server",
    "app", "client", "system", "api", "docs", "website", "web", "mobile",
    "project-management", "crm", "supervisor", "homepage",
}
EXCLUDED_NAMES = {
    "clickhouse", "p5.js", "ctakes", "node-fhir-server-core", "ccpm",
    "rt", "egos-2000", "rathena", "vue-crud", "apriltag",
    "home-assistantconfig", "home-assistant-config", "node-red-contrib-home-assistant-websocket",
}


def repository_items(key: str, query: str) -> list[dict]:
    cache = ROOT / "storage/github_negative_cache" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["items"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, headers={"Accept": "application/vnd.github+json",
                                          "User-Agent": "IDEA-weak-signals/1.0"}) as client:
        for attempt in range(4):
            response = client.get("https://api.github.com/search/repositories",
                                  params={"q": query, "sort": "stars", "per_page": 100})
            if response.status_code in (403, 429):
                time.sleep(60 if attempt else 10)
                continue
            response.raise_for_status()
            payload = response.json()
            cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload["items"]
    raise RuntimeError(f"GitHub rate limit persisted for {query}")


def eligible(repo: dict, used_urls: set[str], used_names: set[str]) -> bool:
    name = repo.get("name") or ""
    description = repo.get("description") or ""
    url = repo.get("html_url") or ""
    if not name or not url or not description:
        return False
    if url.casefold() in used_urls or name.casefold() in used_names:
        return False
    if name.casefold() in GENERIC_NAMES | EXCLUDED_NAMES:
        return False
    if not repo.get("homepage") or repo.get("fork") or repo.get("archived"):
        return False
    if repo.get("created_at", "9999")[:10] > SNAPSHOT.isoformat():
        return False
    if EXCLUDE.search(f"{name} {description}") or not INCLUDE.search(description):
        return False
    return True


def choose_repositories(existing: list[dict], target: int = 100) -> list[tuple[dict, tuple]]:
    used_urls = {r["primary_source_url"].casefold() for r in existing}
    used_names = {r["technology_original"].casefold() for r in existing}
    selected: list[tuple[dict, tuple]] = []
    backups: list[tuple[dict, tuple]] = []
    for spec in SECTORS:
        key, query, *_rest, quota = spec
        candidates = sorted(repository_items(key, query),
                            key=lambda repo: repo.get("stargazers_count") or 0,
                            reverse=True)
        sector_count = 0
        for repo in candidates:
            if not eligible(repo, used_urls, used_names):
                continue
            if sector_count < quota:
                selected.append((repo, spec))
                used_urls.add(repo["html_url"].casefold())
                used_names.add(repo["name"].casefold())
                sector_count += 1
            else:
                backups.append((repo, spec))
    if len(selected) < target:
        for repo, spec in backups:
            if not eligible(repo, used_urls, used_names):
                continue
            selected.append((repo, spec))
            used_urls.add(repo["html_url"].casefold())
            used_names.add(repo["name"].casefold())
            if len(selected) == target:
                break
    if len(selected) < target:
        raise RuntimeError(f"Only {len(selected)} unique sourced products; need {target}")
    return selected[:target]


def product_row(repo: dict, spec: tuple) -> dict:
    key, _, domain, domain_ru, _ = spec
    name = repo["name"]
    full_name = repo["full_name"]
    stable = hashlib.sha256(f"{full_name}|{SNAPSHOT}".encode()).hexdigest()[:16]
    group = hashlib.sha256(full_name.casefold().encode()).hexdigest()[:12]
    features = {feature: "" for feature in FEATURE_NAMES}
    features.update({"source_type_diversity": 1, "missing_science": 1,
                     "missing_patents": 1, "missing_media": 1,
                     "missing_funding": 1, "missing_adoption": 1})
    return {
        "id": f"neg_{stable}", "label": 0, "data_tier": "bronze",
        "label_status": "provisional_negative", "review_status": "needs_human_review",
        "negative_type": "N3b", "stage": "unknown", "trend": "unknown",
        "snapshot_date": SNAPSHOT.isoformat(), "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "technology_group_id": f"product_{group}",
        "domain_original": domain, "domain_ru": domain_ru,
        "technology_original": name, "technology_ru": name,
        "application_original": "software product", "application_ru": "программный продукт",
        "name_original": f"{name} (software product)",
        "name_ru": f"{name} (программный продукт)",
        "original_language": "en", "search_query": f"{name} software product",
        "primary_source_title_original": full_name,
        "primary_source_url": repo["html_url"],
        "primary_source_date": repo["created_at"][:10],
        "primary_source_date_precision": "day",
        "primary_source_language": "unknown",
        "primary_source_publisher": "GitHub repository",
        "source_evidence_excerpt_original": (repo.get("description") or "")[:600],
        "source_match_method": "named_repository_product_identity",
        "source_match_strength": "repository_entity",
        "source_relevance_score": 1,
        "science_count_scope": "not_searched",
        "selection_rule": f"{EXTENSION_TAG}:{key}",
        "evidence_note": "Repository describes a named software product; review entity-versus-method classification before training.",
        "patent_search_status": "not_searched", "media_search_status": "not_searched",
        **features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--add", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with args.file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = list(reader)
    already = sum(row.get("selection_rule", "").startswith(EXTENSION_TAG)
                  for row in existing)
    if already == args.add:
        print(f"already_extended={already} rows={len(existing)}")
        return
    if already:
        raise RuntimeError(f"Partial extension ({already} rows); inspect before retrying")
    selected = choose_repositories(existing, args.add)
    added = [product_row(repo, spec) for repo, spec in selected]
    if len({r["name_original"].casefold() for r in existing + added}) != len(existing) + len(added):
        raise RuntimeError("Name collision after selection")
    if len({r["primary_source_url"].casefold() for r in added}) != len(added):
        raise RuntimeError("Source collision among new rows")
    print("new_sectors", dict(Counter(r["domain_original"] for r in added)))
    for row in added:
        print(row["technology_original"], row["primary_source_url"], row["source_evidence_excerpt_original"][:100])
    if args.dry_run:
        return
    temp = args.file.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + added)
    temp.replace(args.file)
    print(f"appended={len(added)} total={len(existing) + len(added)} file={args.file}")


if __name__ == "__main__":
    main()
