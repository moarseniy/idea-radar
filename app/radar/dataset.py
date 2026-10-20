from __future__ import annotations

import csv
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.ml.features.metadata import FEATURE_DESCRIPTIONS, FEATURE_GROUPS, FEATURE_GROUP_BY_NAME
from app.ml.features.schema import FEATURE_SCHEMA_VERSION
from app.ml.scorer import FEATURE_LABELS, default_scorer, display_value

_CATALOG_STAGE_LABELS = (
    ("stage_scaling_or_mature", "Масштабирование / зрелый продукт"),
    ("stage_early_adoption", "Раннее внедрение"),
    ("stage_pilot", "Пилот"),
    ("stage_poc", "Прототип / проверка концепции"),
    ("stage_research", "Исследование"),
)
_CATALOG_SOURCE_URL_FIELDS = (
    ("primary_source_title_original", "primary_source_url", "Научный источник"),
    ("patent_source_title_original", "patent_source_url", "Патент"),
    ("media_source_title_original", "media_source_url", "Медиа"),
    ("official_source_title_original", "official_source_url", "Официальный источник"),
    ("funding_source_title_original", "funding_source_url", "Финансирование"),
    ("deployment_source_title_original", "deployment_source_url", "Внедрение"),
    ("technology_media_source_title_original", "technology_media_source_url", "Медиа о технологии"),
    ("technology_official_source_title_original", "technology_official_source_url", "Официальный источник о технологии"),
    ("technology_funding_source_title_original", "technology_funding_source_url", "Финансирование технологии"),
)


def _catalog_stage(row: dict[str, str]) -> str:
    for feature, label in _CATALOG_STAGE_LABELS:
        try:
            if float(row.get(feature, "") or 0) > 0:
                return label
        except ValueError:
            continue
    return "Стадия не установлена"


def _catalog_sources(row: dict[str, str]) -> list[dict[str, str]]:
    sources = []
    seen = set()
    for title_field, url_field, fallback in _CATALOG_SOURCE_URL_FIELDS:
        url = (row.get(url_field) or "").strip()
        if not url.startswith(("https://", "http://")) or url in seen:
            continue
        seen.add(url)
        sources.append({"title": (row.get(title_field) or "").strip() or fallback, "url": url})
    return sources


def _read_feature_catalog(data_dir: Path) -> dict | None:
    """Recover the organizer examples when the source workbook is absent from a deployment."""
    path = data_dir / "training" / "organizer_100_features.csv"
    if not path.is_file():
        return None

    manifest_path = data_dir / "training" / "organizer_validation_manifest.csv"
    labels: dict[str, str] = {}
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            labels = {row.get("organizer_id", ""): row.get("label", "") for row in csv.DictReader(stream)}

    records = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                continue
            match = re.fullmatch(r"organizer_(\d+)", row.get("organizer_id", ""))
            if not match or not row.get("name_original", "").strip():
                continue
            example_id = int(match.group(1))
            label = labels.get(row.get("organizer_id", ""), "")
            status_fields = []
            for field, title in (("crossref_status", "Научный поиск"),
                                 ("patent_search_status", "Патенты"),
                                 ("media_search_status", "Медиа")):
                status = (row.get(field) or "").strip()
                if status:
                    status_fields.append(f"{title}: {status}")
            excerpt = (row.get("source_evidence_excerpt_original") or "").strip()
            records.append({
                "id": example_id,
                "title": row["name_original"].strip(),
                "domain": (row.get("domain_original") or "Без области").strip(),
                "application": " · ".join(value.strip() for value in
                    (row.get("technology_query", ""), row.get("application_query", "")) if value.strip()),
                "companies": "",
                "stage": _catalog_stage(row),
                "reference_score": None,
                "reference_label": "Положительный пример" if label == "1" else "",
                "reference_explanation": excerpt or "В исходном наборе нет поясняющего фрагмента.",
                "mention_trend": "; ".join(status_fields) or "Статусы каналов не указаны.",
                "sources": _catalog_sources(row),
                "sheet": "organizer_100_features.csv",
                "row": example_id + 1,
                "origin": "organizer_validation_csv",
                "independently_verified": False,
            })

    return {
        "filename": path.name,
        "records": records,
        "count": len(records),
        "domains": dict(Counter(record["domain"] for record in records)),
        "note": ("Excel-таблица отсутствует в развернутом наборе файлов. "
                 "Каталог восстановлен из organizer_100_features.csv; это исходные примеры для валидации, "
                 "а не ответы живого поиска."),
    }


@lru_cache(maxsize=4)
def _read_features(path: str, modified: int, feature_names: tuple[str, ...]) -> dict[str, list[dict]]:
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                continue
            features = []
            for name in feature_names:
                raw = row.get(name, "").strip()
                try:
                    value = float(raw) if raw else None
                    if value is not None and not math.isfinite(value):
                        value = None
                except ValueError:
                    value = None
                features.append({
                    "name": name,
                    "label": FEATURE_LABELS[name],
                    "description": FEATURE_DESCRIPTIONS[name],
                    "group": FEATURE_GROUP_BY_NAME[name],
                    "group_label": FEATURE_GROUPS[FEATURE_GROUP_BY_NAME[name]],
                    "value": value,
                    "display_value": display_value(name, value),
                })
            result[row.get("organizer_id", "")] = features
    return result


def _with_features(records: list[dict], data_dir: Path, feature_names: tuple[str, ...]) -> list[dict]:
    path = data_dir / "training" / "organizer_100_features.csv"
    feature_rows = (_read_features(str(path.resolve()), path.stat().st_mtime_ns, feature_names)
                    if path.is_file() else {})
    for record in records:
        features = feature_rows.get(f"organizer_{record['id']:03d}", [])
        record["features"] = features
        record["measured_feature_count"] = sum(feature["value"] is not None for feature in features)
    return records


@lru_cache(maxsize=4)
def _read(path: str, modified: int) -> dict:
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    records = []
    try:
        for sheet in workbook:
            header = None
            for row_no, row in enumerate(sheet.iter_rows(values_only=True), 1):
                values = [str(v).strip() if v is not None else "" for v in row]
                if "Технология (слабый сигнал)" in values and "Источники" in values:
                    header = {name: i for i, name in enumerate(values) if name}
                    continue
                if not header:
                    continue

                def cell(name):
                    index = header.get(name)
                    return row[index] if index is not None and index < len(row) else None

                if not isinstance(cell("№"), (int, float)):
                    continue
                links = [{"title": title, "url": url} for title, url in re.findall(
                    r"\[([^\]]+)\]\((https?://[^\s)]+)\)", str(cell("Источники") or ""))]
                records.append({"id": int(cell("№")), "title": cell("Технология (слабый сигнал)"),
                    "domain": cell("Область"), "companies": cell("Компании"),
                    "reference_explanation": cell("Почему это слабый сигнал"), "stage": cell("Стадия развития"),
                    "mention_trend": cell("Тренд упоминаний"), "reference_score": cell("Балл (стадия+тренд)"),
                    "sources": links, "sheet": sheet.title, "row": row_no,
                    "origin": "organizer_catalog", "independently_verified": False})
    finally:
        workbook.close()
    return {"filename": Path(path).name, "records": records, "count": len(records),
            "domains": dict(Counter(r["domain"] for r in records)),
            "note": "Примеры организаторов. Объяснения и баллы сохранены из Excel; это не результаты живого поиска и не независимая валидация."}


def read_catalog(data_dir: Path) -> dict:
    path = next(iter(sorted(data_dir.rglob("*сигнал*.xlsx"))), None)
    if path is None:
        path = next(iter(sorted(data_dir.rglob("*.xlsx"))), None)
    if path is not None:
        try:
            catalog = _read(str(path.resolve()), path.stat().st_mtime_ns)
        except Exception:  # noqa: BLE001 — повреждённый/непоставленный Excel не должен скрывать доступный CSV
            catalog = _read_feature_catalog(data_dir)
        if catalog is not None and not catalog["records"]:
            catalog = _read_feature_catalog(data_dir) or catalog
    else:
        catalog = _read_feature_catalog(data_dir)
    if catalog is None:
        return {"filename": "", "records": [], "count": 0, "domains": {},
                "note": "Файл Excel и резервный CSV примеров не найдены."}
    feature_names = tuple(default_scorer().features)
    records = _with_features([dict(record) for record in catalog["records"]], data_dir, feature_names)
    return {**catalog, "records": records,
            "feature_count": len(feature_names),
            "feature_groups": FEATURE_GROUPS}
