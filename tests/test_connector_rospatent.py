"""Роспатент: записи по форме ответа из «Открытые API ИС Поисковая платформа» (ФИПС)."""
from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request

from app.ml.features.evidence import years_before
from app.radar.connectors import rospatent

AS_OF = date(2021, 12, 31)


def _hit(doc_id, title_ru, title_en, published, priority=None, description="", office="RU"):
    common = {"publishing_office": office, "document_number": doc_id[2:9], "kind": "C1",
              "publication_date": published, "application": {"number": "1", "filing_date": "2019.01.10"}}
    if priority:
        common["priority"] = {"number": "2", "country": "RU", "filing_date": priority}
    return {"id": doc_id, "common": common, "dataset": "ru_since_1994",
            "biblio": {"ru": {"title": title_ru, "patentee": [{"name": "ООО Ромашка (RU)"}]},
                       "en": {"title": title_en}},
            "snippet": {"title": title_ru, "description": description, "lang": "ru"}}


ON_TOPIC = _hit("RU2700001C1_20200110", "Способ квантового распределения ключей",
                "METHOD OF QUANTUM KEY DISTRIBUTION", "2020.01.10", "2018.05.04",
                "Система <em>квантового распределения ключей</em> по оптоволокну.")
NEIGHBOUR = _hit("RU2700002C1_20200210", "Квантовый генератор для лазерной резки",
                 "QUANTUM GENERATOR FOR LASER CUTTING", "2020.02.10",
                 description="Лазер для резки металла; ключ на корпусе.")


def test_relevant_patent_kept_with_priority_date_and_strict_window(monkeypatch):
    monkeypatch.setenv("ROSPATENT_API_KEY", "test-key")
    calls = fake_json_request(monkeypatch, rospatent, {"total": 2, "hits": [ON_TOPIC, NEIGHBOUR]})
    refs = rospatent.CONNECTOR.run(["quantum key distribution"], AS_OF, 5)

    assert_connector_contract(rospatent.CONNECTOR, refs, AS_OF)
    assert [r["url"] for r in refs] == ["https://searchplatform.rospatent.gov.ru/docs/RU2700001C1_20200110"]
    assert refs[0]["observed_at"] == "2018-05-04" and refs[0]["publication_date"] == "2020-01-10"
    assert refs[0]["assignee"] == "ООО Ромашка (RU)"
    assert calls[0]["body"]["qn"] == "quantum key distribution"
    assert calls[0]["body"]["datasets"] == rospatent.RU_DATASETS
    start = years_before(AS_OF, rospatent.LOOKBACK_YEARS).strftime("%Y%m%d")
    assert calls[0]["body"]["filter"] == {"date_published": {"range": {"gte": start, "lte": "20211231"}}}
    assert calls[0]["headers"] == {"Authorization": "Bearer test-key"}


def test_russian_topic_is_searched_as_is(monkeypatch):
    monkeypatch.setenv("ROSPATENT_API_KEY", "test-key")
    calls = fake_json_request(monkeypatch, rospatent, {"total": 1, "hits": [ON_TOPIC]})
    refs = rospatent.search(["quantum key distribution", "квантовое распределение ключей"], AS_OF, 5)
    # Сначала обе фразы в массивах РФ и СНГ, затем — на оставшиеся места — в мировых.
    assert [(c["body"]["qn"], c["body"]["datasets"] == rospatent.RU_DATASETS) for c in calls] == [
        ("quantum key distribution", True), ("квантовое распределение ключей", True),
        ("quantum key distribution", False), ("квантовое распределение ключей", False)]
    assert len(refs) == 1  # один и тот же патент из всех ответов


def test_family_members_with_same_title_are_merged(monkeypatch):
    monkeypatch.setenv("ROSPATENT_API_KEY", "test-key")
    wo = _hit("WO2020000001A1_20200110", "", "System and method for quantum key distribution", "2020.01.10",
              office="WO")
    us = _hit("US2020000002A1_20200210", "", "SYSTEM AND METHOD FOR QUANTUM KEY DISTRIBUTION", "2020.02.10",
              office="US")
    fake_json_request(monkeypatch, rospatent, {"total": 2, "hits": [wo, us]})
    refs = rospatent.search(["quantum key distribution"], AS_OF, 5)
    assert [r["source_country"] for r in refs] == ["WO"]


def test_later_publication_and_missing_key_give_nothing(monkeypatch):
    monkeypatch.setenv("ROSPATENT_API_KEY", "test-key")
    late = _hit("RU2700003C1_20220301", "Способ квантового распределения ключей", "QKD", "2022.03.01")
    fake_json_request(monkeypatch, rospatent, {"total": 1, "hits": [late]})
    assert rospatent.CONNECTOR.run(["quantum key distribution"], AS_OF, 5) == []
    monkeypatch.setenv("ROSPATENT_API_KEY", "")
    assert not rospatent.CONNECTOR.enabled(["quantum key distribution"])
    assert rospatent.search(["quantum key distribution"], AS_OF, 5) == []
