"""Схема денежного и регуляторного события (``ref["structured_event"]``).

Признаки (app/ml/features/extract.py) понимают типы EVENT_TYPES. Коннектор называет
событие подробно (``kind``), а ``type`` выводится из него, поэтому новые виды денег
сразу попадают в существующие признаки финансирования, закупок и регуляторики.
"""
from __future__ import annotations

from datetime import date
from typing import Any

# kind → type признаков; None — событие сохраняется для аудита, но признаки его не считают.
KINDS: dict[str, str | None] = {
    "grant": "grant",               # выданный грант (NIH, NSF, UKRI, CORDIS)
    "grant_call": None,             # объявленный конкурс: намерение, а не деньги
    "contract": "procurement",      # госконтракт (USAspending, SAM.gov)
    "tender": "procurement",        # закупка или тендер (TED)
    "funding_round": "funding",     # венчурный раунд
    "accelerator": "funding",       # вход в акселератор (YC)
    "filing_mention": None,         # упоминание в отчётности (SEC 10-K, 8-K)
    "trial": "trial",
    "regulation": "regulation",
}

# Среднегодовые курсы, единиц валюты за 1 USD (округлённо, по данным ЦБ и ФРС).
# Точность до процентов достаточна: признаки берут log1p суммы.
USD_RATES: dict[str, dict[int, float]] = {
    "EUR": {2015: 0.90, 2016: 0.90, 2017: 0.89, 2018: 0.85, 2019: 0.89, 2020: 0.88,
            2021: 0.85, 2022: 0.95, 2023: 0.92, 2024: 0.92, 2025: 0.89, 2026: 0.87},
    "GBP": {2015: 0.65, 2016: 0.74, 2017: 0.78, 2018: 0.75, 2019: 0.78, 2020: 0.78,
            2021: 0.73, 2022: 0.81, 2023: 0.80, 2024: 0.78, 2025: 0.76, 2026: 0.75},
    "RUB": {2015: 61.0, 2016: 67.0, 2017: 58.3, 2018: 62.7, 2019: 64.7, 2020: 72.1,
            2021: 73.7, 2022: 68.5, 2023: 85.2, 2024: 92.6, 2025: 84.0, 2026: 80.0},
}


def to_usd_m(amount: Any, currency: str, year: int) -> float | None:
    """Сумма в миллионах долларов по курсу года события; None, если пересчитать нельзя."""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    currency = (currency or "USD").upper()
    if currency == "USD":
        return round(value / 1_000_000, 6)
    rates = USD_RATES.get(currency)
    if not rates:
        return None
    rate = rates.get(year) or rates[min(rates, key=lambda known: abs(known - year))]
    return round(value / rate / 1_000_000, 6)


def make_event(kind: str, when: date | str | None, source_id: str, *, amount: Any = None,
               currency: str = "USD", company: str = "", customer: str | None = None,
               funder: str | None = None, round_name: str | None = None,
               strategic: bool = False, **extra: Any) -> dict:
    """Нормализованное событие. Дата обязательна: без неё событие не попадёт в окна признаков."""
    if kind not in KINDS:
        raise ValueError(f"неизвестный вид события {kind!r}")
    if isinstance(when, date):
        day = when.isoformat()
    else:
        day = str(when or "")[:10]
        date.fromisoformat(day)  # ValueError для пустой или кривой даты
    event = {"type": KINDS[kind] or kind, "kind": kind, "date": day, "source_id": str(source_id),
             "company": company or "", "customer": customer, "funder": funder,
             "amount_native": amount, "currency": currency.upper(),
             "amount_usd_m": to_usd_m(amount, currency, int(day[:4])) if amount is not None else None,
             "round": round_name or ("grant" if kind == "grant" else None), "strategic": strategic}
    event.update(extra)
    return event
