import threading
import time
from datetime import date

import pytest

from app.radar.pipeline import (_DeadlineThreadPoolExecutor, _attach_candidate_evidence_counts,
                               branch_coverage_warning)


def test_plan_shortfall_is_reported_without_hiding_original_query_coverage():
    assert branch_coverage_warning(8, 8) is None
    assert branch_coverage_warning(1, 8) == (
        "План сформировал 1 из 8 запрошенных веток. Исходная формулировка всё равно была включена в поиск."
    )


def test_deadline_executor_does_not_wait_for_an_unresponsive_task_after_timeout():
    release_task = threading.Event()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            with _DeadlineThreadPoolExecutor(1, time.monotonic() + 10, threading.Event()) as pool:
                pool.submit(release_task.wait)
                raise TimeoutError("run deadline")
        assert time.monotonic() - started < 0.2
    finally:
        release_task.set()


def test_dated_candidate_sources_fill_unknown_channel_years_and_extend_chart():
    item = {
        "_search_source_ids": ["patent-1", "patent-duplicate", "patent-old", "news-old", "news-1",
                               "science-old", "grant-old"],
        "model": {"visualization": {"evidence_timeline": {
            "years": ["2025", "2026"],
            "channels": {
                "patents": {"label": "Патенты", "counts": {"2025": None, "2026": None}},
                "news": {"label": "Новости и медиа", "counts": {"2025": 0, "2026": 0}},
                "science": {"label": "Научные работы", "counts": {"2025": 4, "2026": 5}},
                "funding": {"label": "Финансирование и гранты", "counts": {"2025": 0, "2026": 0}},
            },
        }}},
    }
    source_map = {
        "patent-1": {"type": "patent", "status": "read", "published_at": "2025-04-01"},
        "patent-duplicate": {"type": "patent", "status": "read", "published_at": "2026-02-01",
                             "duplicate_of": "patent-1"},
        "patent-old": {"type": "patent", "status": "read", "published_at": "2010-06-01"},
        "news-old": {"type": "news_agency", "channel": "market", "status": "read",
                      "published_at": "2016-06-01"},
        "news-1": {"type": "news_agency", "channel": "market", "status": "read",
                    "published_at": "2026-01-01"},
        "science-old": {"type": "scientific_journal", "channel": "science", "status": "read",
                        "published_at": "2008-06-01"},
        "grant-old": {"type": "institution", "channel": "money", "status": "read",
                       "published_at": "2014-05-01",
                       "structured_event": {"type": "grant", "date": "2014-03-01"}},
    }

    _attach_candidate_evidence_counts(item, source_map, date(2026, 12, 31))

    timeline = item["model"]["visualization"]["evidence_timeline"]
    assert timeline["years"] == [str(year) for year in range(2008, 2027)]
    assert timeline["channels"]["patents"]["counts"]["2010"] == 1
    assert timeline["channels"]["patents"]["counts"]["2025"] == 1
    assert timeline["channels"]["patents"]["counts"]["2026"] is None
    assert timeline["channels"]["news"]["counts"]["2016"] == 1
    assert timeline["channels"]["news"]["counts"]["2026"] == 0
    assert timeline["channels"]["science"]["counts"]["2008"] == 1
    assert timeline["channels"]["science"]["counts"]["2025"] == 4
    assert timeline["channels"]["funding"]["counts"]["2014"] == 1
