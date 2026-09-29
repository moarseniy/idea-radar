from app.radar.store import RadarStore


def test_delete_run_removes_only_that_run_and_cascades_sources(tmp_path):
    store = RadarStore("", tmp_path)
    store.initialize()
    for run_id in ("first", "second"):
        store.save({"id": run_id, "created_at": "2026-09-28T00:00:00+00:00",
                    "status": "completed", "signals": []})
        store.put_source(run_id, {"id": f"{run_id}-source", "title": "source"})

    assert store.delete_run("first") is True
    assert store.get("first") is None
    assert store.source("first", "first-source") is None
    assert store.get("second") is not None
    assert store.source("second", "second-source") is not None
    assert store.delete_run("missing") is False
