from __future__ import annotations

import json
from datetime import date

from scripts import enrich_public_sources as external


def test_keyless_corrob_filters_dates_and_keeps_numeric_features_missing(monkeypatch):
    patent = {"results": {"cluster": [{"result": [
        {"patent": {"publication_number": "US20260234567A1", "publication_date": "2026-06-03",
                     "title": "Photonic neural accelerators for AI data centers",
                     "snippet": "Optical accelerator for AI data center inference.", "assignee": "Example University"}},
        {"patent": {"publication_number": "US20270234567A1", "publication_date": "2027-01-01",
                     "title": "Photonic neural accelerators for AI data centers",
                     "snippet": "Future optical accelerator.", "assignee": "Future Inc."}},
    ]}]}}
    rss = """<rss><channel>
    <item><title>Photonic neural accelerators for AI data centers raise $35M</title>
      <link>http://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Fexample.org%2Ffunding</link>
      <pubDate>Mon, 15 Jun 2026 10:00:00 GMT</pubDate>
      <description>Startup raised $35M in a seed funding round for photonic neural accelerators in AI data centers.</description>
      <source>Industry Example</source></item>
    <item><title>Photonic neural accelerators for AI data centers open a pilot</title>
      <link>https://energy.gov/example/pilot</link>
      <pubDate>Fri, 10 Jul 2026 10:00:00 GMT</pubDate>
      <description>Official pilot of photonic neural accelerators in AI data centers.</description>
      <source>Energy Department</source></item>
    <item><title>Photonic neural accelerators for AI data centers deploy</title>
      <link>https://example.org/future</link>
      <pubDate>Fri, 10 Oct 2026 10:00:00 GMT</pubDate>
      <description>Deployment announced later.</description></item>
    </channel></rss>"""

    def fake_fetch(channel, query):
        return json.dumps(patent).encode() if channel == "patent" else rss.encode()

    monkeypatch.setattr(external, "_fetch", fake_fetch)
    row = {
        "search_query": "photonic neural accelerators AI data centers",
        "technology_original": "photonic neural accelerators",
        "application_original": "AI data centers",
        "missing_patents": 1, "missing_media": 1,
        "patent_families_log_3y": "", "media_volume_normalized_12m": "",
        "verified_pilots_log_24m": "", "source_type_diversity": 1,
    }
    result = external.enrich_one(row, date(2026, 9, 22))
    assert result["patent_publication_number"] == "US20260234567A1"
    assert result["patent_source_url"].endswith("US20260234567A1/en")
    assert result["media_source_url"] == "https://energy.gov/example/pilot"
    assert result["funding_amount_original"] == "$35M"
    assert result["funding_source_url"] == "https://example.org/funding"
    assert result["official_source_url"] == "https://energy.gov/example/pilot"
    assert result["source_type_diversity"] == 3
    assert result["missing_patents"] == 1 and result["patent_families_log_3y"] == ""
    assert result["missing_media"] == 1 and result["media_volume_normalized_12m"] == ""
    assert result["verified_pilots_log_24m"] == ""


def test_reviewed_patent_supplement_respects_snapshot(monkeypatch):
    monkeypatch.setattr(external, "_fetch", lambda channel, query: b"<rss><channel /></rss>")
    candidate = {
        "name_original": "optical circuit switching for low-energy AI data centers",
        "search_query": "optical circuit switching low-energy AI data centers",
        "technology_original": "optical circuit switching",
        "application_original": "low-energy AI data centers",
        "source_type_diversity": 1,
    }
    early = external.enrich_rows([candidate], date(2026, 1, 1), channels=("news",))[0]
    assert not early.get("patent_source_url")
    current = external.enrich_rows([candidate], date(2026, 9, 22), channels=("news",))[0]
    assert current["patent_publication_number"] == "US20260110848A1"
    assert current["patent_search_status"] == "manual_page_review"
    assert current["source_type_diversity"] == 2


def test_reviewed_funding_and_developer_sources_respect_snapshot(monkeypatch):
    monkeypatch.setattr(external, "_fetch", lambda channel, query: b"<rss><channel /></rss>")
    candidate = {
        "name_original": "precision fermentation for alternative dairy production",
        "search_query": "precision fermentation alternative dairy production",
        "technology_original": "precision fermentation",
        "application_original": "alternative dairy production",
        "source_type_diversity": 1,
    }
    early = external.enrich_rows([candidate], date(2026, 1, 1), channels=("news",))[0]
    assert not early.get("funding_source_url")
    assert not early.get("official_source_url")
    current = external.enrich_rows([candidate], date(2026, 9, 22), channels=("news",))[0]
    assert current["funding_amount_original"] == "$4M"
    assert current["media_source_url"] == current["funding_source_url"]
    assert current["official_source_url"].startswith("https://dyadic.com/")
    assert current["source_type_diversity"] == 3


def test_technology_context_is_explicitly_not_application_evidence(monkeypatch):
    rss = """<rss><channel><item>
    <title>Photonic neural accelerators raise $12M</title>
    <link>https://example.org/photonic-round</link>
    <pubDate>Mon, 15 Jun 2026 10:00:00 GMT</pubDate>
    <description>Funding for photonic neural accelerators.</description>
    <source>Example News</source>
    </item></channel></rss>"""
    monkeypatch.setattr(external, "_fetch", lambda channel, query: rss.encode())
    rows = [{"technology_original": "photonic neural accelerators",
             "application_original": "underwater robotics", "source_type_diversity": 1}]
    result = external.enrich_technology_context(rows, date(2026, 9, 22))[0]
    assert result["technology_funding_amount_original"] == "$12M"
    assert result["technology_context_scope"] == "technology_only_not_application"
    assert result["source_type_diversity"] == 1
