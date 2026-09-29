import threading
import time

import pytest

from app.radar.pipeline import _DeadlineThreadPoolExecutor, branch_coverage_warning


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
