import csv

from app.ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.ml.scorer import default_scorer
from app.radar import dataset


def test_catalog_attaches_only_the_features_used_by_the_serving_model(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    training_dir = data_dir / "training"
    training_dir.mkdir(parents=True)
    (data_dir / "examples_сигнал.xlsx").write_bytes(b"fixture")
    feature_path = training_dir / "organizer_100_features.csv"
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "organizer_id", "feature_schema_version", *FEATURE_NAMES,
        ])
        writer.writeheader()
        writer.writerow({
            "organizer_id": "organizer_001",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "stage_research": "1",
            "papers_log_3y": "",
        })

    monkeypatch.setattr(dataset, "_read", lambda *_: {
        "filename": "examples_сигнал.xlsx",
        "records": [{"id": 1, "title": "Пример", "sources": []}],
        "count": 1,
        "domains": {},
    })

    result = dataset.read_catalog(data_dir)
    record = result["records"][0]
    active_features = default_scorer().features

    assert result["feature_count"] == len(active_features)
    assert "feature_schema_version" not in result
    assert "feature_schema_version" not in record
    assert [feature["name"] for feature in record["features"]] == active_features
    assert record["measured_feature_count"] == 1
    stage_index = active_features.index("stage_research")
    paper_index = active_features.index("papers_log_3y")
    assert record["features"][stage_index]["display_value"] == "да"
    assert record["features"][paper_index]["display_value"] == "нет данных"
    assert record["features"][paper_index]["description"]
