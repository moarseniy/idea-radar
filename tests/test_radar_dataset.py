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


def test_catalog_falls_back_to_validation_csv_when_workbook_is_missing(tmp_path):
    data_dir = tmp_path / "data"
    training_dir = data_dir / "training"
    training_dir.mkdir(parents=True)
    feature_path = training_dir / "organizer_100_features.csv"
    fields = ["organizer_id", "name_original", "domain_original", "feature_schema_version",
              "technology_query", "application_query", "stage_pilot", "crossref_status",
              "primary_source_title_original", "primary_source_url", *FEATURE_NAMES]
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "organizer_id": "organizer_001",
            "name_original": "Пример технологии",
            "domain_original": "Энергетика",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "technology_query": "твердотельные батареи",
            "application_query": "транспорт",
            "stage_pilot": "1",
            "crossref_status": "matched",
            "primary_source_title_original": "Исследование батарей",
            "primary_source_url": "https://example.org/paper",
            "papers_log_3y": "",
        })
    with (training_dir / "organizer_validation_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["organizer_id", "label"])
        writer.writeheader()
        writer.writerow({"organizer_id": "organizer_001", "label": "1"})

    catalog = dataset.read_catalog(data_dir)
    record = catalog["records"][0]

    assert catalog["count"] == 1
    assert catalog["filename"] == "organizer_100_features.csv"
    assert "отсутствует" in catalog["note"]
    assert record["title"] == "Пример технологии"
    assert record["application"] == "твердотельные батареи · транспорт"
    assert record["stage"] == "Пилот"
    assert record["reference_label"] == "Положительный пример"
    assert record["reference_score"] is None
    assert record["sources"] == [{"title": "Исследование батарей", "url": "https://example.org/paper"}]
    assert len(record["features"]) == len(default_scorer().features)
