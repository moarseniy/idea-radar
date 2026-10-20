"""SEDIA: форма записей сверена с живым ответом (октябрь 2026).

Конкурс — type "1" с metadata.identifier/title/startDate списками и копией на каждом языке
портала; тендер — type "0"; FAQ — "3"; объявления — "ANNOUNCEMENT".
"""
from datetime import date

from connector_testkit import assert_connector_contract, fake_json_request

from app.radar.connectors import sedia

SNAPSHOT = date(2021, 12, 31)


def _item(identifier, title, types, start=None, summary="", sort_ms=None, language="en"):
    metadata = {"type": types, "identifier": [identifier], "title": [title]}
    if start:
        metadata["startDate"] = [start]
    if sort_ms is not None:
        metadata["es_SortDate"] = sort_ms
    return {"reference": f"{identifier}{language}", "summary": summary or title, "language": language,
            "metadata": metadata}


def _payload(*items):
    return {"totalResults": len(items), "results": list(items)}


CALL = _item("HORIZON-TEST-2021-1", "Trustworthy artificial intelligence for industry", ["1"],
             "2021-06-01T00:00:00.000+0000", "Call for AI research and innovation actions.")
CALL_ES = _item("HORIZON-TEST-2021-1", "Trustworthy artificial intelligence for industry", ["1"],
                "2021-06-01T00:00:00.000+0000", language="es")
TENDER = _item("EC-TENDER-2021-7", "Artificial intelligence security study", ["0"],
               "2021-09-15T00:00:00.000+0000", "Study on AI security for public administrations.")
FAQ = _item("FAQ-1", "How to apply for artificial intelligence calls", ["3"], "2021-03-01T00:00:00.000+0000")
LATE = _item("HORIZON-TEST-2022-9", "Artificial intelligence call 2022", ["1"], "2022-02-01T00:00:00.000+0000")


def test_calls_and_tenders_mapped_by_type_and_date_cut(monkeypatch):
    calls = fake_json_request(monkeypatch, sedia, _payload(CALL, CALL_ES, TENDER, FAQ, LATE))
    connector = sedia.CONNECTOR
    refs = connector.run(["artificial intelligence"], SNAPSHOT, 5)
    assert_connector_contract(connector, refs, SNAPSHOT, money=True)
    assert [r["structured_event"]["kind"] for r in refs] == ["grant_call", "tender"]
    assert refs[0]["url"].endswith("/HORIZON-TEST-2021-1")
    assert refs[0]["structured_event"]["date"] == "2021-06-01"
    assert refs[1]["structured_event"]["type"] == "procurement"
    assert calls[0]["params"]["apiKey"] == "SEDIA"
    assert calls[0]["params"]["text"] == '"artificial intelligence"'
    assert '"language": ["en"]' in calls[0]["files"]["query"]


def test_epoch_sort_date_is_used_when_start_date_missing(monkeypatch):
    item = _item("EC-TEST-EPOCH", "Artificial intelligence call", ["1"], sort_ms=1622505600000,
                 summary="AI call.")  # 2021-06-01 UTC
    fake_json_request(monkeypatch, sedia, _payload(item))
    refs = sedia.search_sedia(["artificial intelligence"], SNAPSHOT)
    assert refs[0]["structured_event"]["date"] == "2021-06-01"


def test_irrelevant_and_empty_responses_give_nothing(monkeypatch):
    off_topic = _item("EC-OFF-1", "Motorcycle carrier attachment", ["1"], "2021-06-01T00:00:00.000+0000")
    fake_json_request(monkeypatch, sedia, _payload(off_topic))
    assert sedia.search_sedia(["AI agent security"], SNAPSHOT) == []
    fake_json_request(monkeypatch, sedia, {"totalResults": 0, "results": []})
    assert sedia.search_sedia(["artificial intelligence"], SNAPSHOT) == []
