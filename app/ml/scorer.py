"""Оценка кандидата обученной логистической регрессией с объяснением.

    from app.ml.scorer import score_candidate
    result = score_candidate("Нейроморфные чипы для edge-устройств", "Edge")
    result["probability"], result["is_signal"], result["top_for"], result["exclusion_reasons"]

Модель читается из JSON (app/ml/model/logreg_v1.json, см. ml/scripts/export_model.py),
сам расчёт оценки по входам модели — чистый Python и практически мгновенный: импутация медианой,
стандартизация, линейная часть, калибровка Платта. Задержку создаёт featurize():
сбор данных из источников и два вызова LLM на кандидата. Вклад признака — его слагаемое в откалиброванном логите:
    вклад_i = a · w_i · (x_i − μ_i) / σ_i,
сумма вкладов плюс свободный член равна логиту, поэтому объяснение точное, а не приближение.
"""
from __future__ import annotations

import json
import math
import threading
from datetime import date
from pathlib import Path

from app.ml.env import env, feature_llm, model_slug

MODEL_DIR = Path(__file__).resolve().parent / "model"


def model_path(feature_model: str | None = None) -> Path:
    """Модель обучена на признаках конкретной LLM, поэтому артефакт выбирается по LLM разметки.

    SIGNAL_MODEL_PATH — явный путь; иначе logreg_v1_<LLM разметки>.json, если такой есть
    (например logreg_v1_qwen3.6-35b-a3b.json для qwen3.6:35b-a3b); иначе основная logreg_v1.json
    (обучена на разметке openai/gpt-5.6-luna).
    """
    explicit = env("SIGNAL_MODEL_PATH")
    if explicit:
        return Path(explicit)
    matched = MODEL_DIR / f"logreg_v1_{model_slug(feature_model or feature_llm()['model'])}.json"
    return matched if matched.exists() else MODEL_DIR / "logreg_v1.json"


MODEL_PATH = model_path()

FEATURE_LABELS = {
    "stage_research": "Стадия: исследования",
    "stage_poc": "Стадия: прототип / PoC",
    "stage_pilot": "Стадия: пилоты у заказчиков",
    "stage_early_adoption": "Стадия: раннее внедрение",
    "stage_scaling_or_mature": "Стадия: масштабирование / зрелость",
    "first_evidence_age_years": "Возраст первых свидетельств, лет",
    "first_commercial_event_age_years": "Возраст первого коммерческого события, лет",
    "verified_pilots_log_24m": "Пилоты за 24 месяца",
    "production_deployments_log": "Промышленные внедрения",
    "commercial_vendor_count_log": "Число поставщиков",
    "named_customer_count_log": "Число названных заказчиков",
    "final_standard_flag": "Есть принятый стандарт",
    "regulatory_precursor_count": "Регуляторные инициативы",
    "procurement_presence": "Есть госзакупки",
    "mass_market_flag": "Массово доступна",
    "papers_log_3y": "Научные работы за 3 года",
    "paper_growth_2y": "Рост научных работ за 2 года",
    "paper_acceleration": "Ускорение роста научных работ",
    "paper_volume_percentile_domain": "Объём публикаций относительно области",
    "citation_velocity_median": "Скорость цитирования",
    "preprint_share_2y": "Доля препринтов",
    "institution_count_log_3y": "Число научных организаций",
    "industry_affiliation_share_3y": "Доля авторов из индустрии",
    "science_country_count_log_3y": "Число стран в науке",
    "patent_families_log_3y": "Патенты за 3 года",
    "patent_growth_2y": "Рост патентов",
    "patent_acceleration": "Ускорение патентования",
    "patent_first_priority_age_years": "Возраст первого патента, лет",
    "patent_new_assignee_share_3y": "Доля новых патентообладателей",
    "patent_assignee_count_log_3y": "Число патентообладателей",
    "patent_assignee_hhi_3y": "Концентрация патентообладателей",
    "patent_active_share": "Доля действующих патентов",
    "media_volume_normalized_12m": "Объём упоминаний в СМИ за 12 месяцев",
    "media_growth_6m": "Рост упоминаний в СМИ за полгода",
    "media_burstiness_12m": "Всплескообразность упоминаний",
    "independent_media_domains_log_12m": "Число независимых изданий",
    "press_release_share_12m": "Доля пресс-релизов",
    "source_type_diversity": "Число типов подтверждающих источников",
    "independent_high_trust_source_count_log": "Число изданий высокого доверия",
    "unverifiable_source_share": "Доля непроверяемых источников",
    "source_organization_hhi": "Концентрация источников",
    "funding_log_24m": "Объём финансирования за 24 месяца",
    "funding_growth_24m": "Рост финансирования",
    "funding_round_count_24m": "Раунды финансирования за 24 месяца",
    "early_stage_funding_share": "Доля ранних раундов",
    "strategic_investor_count": "Стратегические инвесторы",
    "funding_company_hhi": "Концентрация финансирования",
    "public_grants_log_36m": "Государственные гранты",
    "marketing_claim_density": "Плотность маркетинговых формулировок",
    "technical_term_density": "Плотность технических терминов",
    "claim_specificity": "Конкретность заявлений",
    "is_technology_flag": "Технология, а не продукт или бренд",
    "rebranding_similarity_to_mature": "Похожесть на ребрендинг зрелой технологии",
    "patent_to_paper_ratio": "Патенты относительно публикаций",
    "patent_minus_paper_growth": "Рост патентов минус рост публикаций",
    "media_to_technical_evidence_ratio": "Медиа относительно науки и патентов",
    "cross_channel_confirmation_count": "Каналы с одновременным ростом",
    "first_time_convergence_flag": "Впервые сошлись наука, патенты и медиа",
    "missing_science": "Нет научных публикаций",
    "missing_patents": "Нет патентов",
    "missing_media": "Нет упоминаний в СМИ",
    "missing_funding": "Нет данных о финансировании",
    "missing_adoption": "Нет пилотов и внедрений",
}


def _num(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def display_value(name: str, raw: float | None) -> str:
    """Значение в единицах, понятных человеку: счётчики — без логарифма, флаги — да/нет."""
    if raw is None:
        return "нет данных"
    if name.startswith(("stage_", "missing_")) or name.endswith("_flag") or name == "procurement_presence":
        return "да" if raw >= 0.5 else "нет"
    if "_log" in name:
        return f"{math.expm1(raw):.0f}"
    if name.endswith(("_share", "_hhi")) or "_share_" in name or "_hhi_" in name:
        return f"{raw:.0%}"
    return f"{raw:.2f}".rstrip("0").rstrip(".")


def exclusion_reasons(f: dict) -> list[str]:
    """Причины, по которым кандидат похож на зрелую технологию, шум или не технологию.

    Это объяснение для пользователя, решение принимает модель. Правила опираются только
    на признаки, поэтому одинаковы для обучения и сервиса.
    """
    reasons = []
    if _num(f.get("stage_scaling_or_mature")) == 1:
        reasons.append("По публикациям в СМИ технология на стадии масштабирования или зрелости")
    if _num(f.get("mass_market_flag")) == 1:
        reasons.append("Технология уже массово доступна потребителям или бизнесу")
    if _num(f.get("is_technology_flag")) == 0:
        reasons.append("Это продукт, бренд или бизнес-модель, а не технология")
    if _num(f.get("final_standard_flag")) == 1:
        reasons.append("По технологии уже принят стандарт")
    age, growth = _num(f.get("first_evidence_age_years")), _num(f.get("paper_growth_2y"))
    if age is not None and age >= 15 and growth is not None and growth <= 0:
        reasons.append(f"Область существует {age:.0f} лет, а число публикаций не растёт")
    if _num(f.get("missing_science")) == 1 and _num(f.get("missing_media")) == 1:
        reasons.append("Нет подтверждений ни в научных публикациях, ни в СМИ")
    unverifiable = _num(f.get("unverifiable_source_share"))
    if unverifiable is not None and unverifiable >= 0.7:
        reasons.append("Большинство упоминаний — в непроверяемых источниках (блоги, соцсети)")
    return reasons


class Scorer:
    def __init__(self, artifact: dict) -> None:
        self.artifact = artifact
        self.features: list[str] = artifact["features"]
        self.threshold: float = artifact["threshold"]
        self._a = artifact["calibration"]["a"]
        self._b = artifact["calibration"]["b"]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> Scorer:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, features: dict) -> dict:
        art, a = self.artifact, self._a
        contributions = []
        logit = a * art["intercept"] + self._b
        for i, name in enumerate(self.features):
            raw = _num(features.get(name))
            value = art["median"][i] if raw is None else raw
            part = a * art["coef"][i] * (value - art["mean"][i]) / art["scale"][i]
            logit += part
            contributions.append({"feature": name, "label": FEATURE_LABELS.get(name, name),
                                  "value": raw, "display_value": display_value(name, raw),
                                  "imputed": raw is None,
                                  "relative": "нет данных" if raw is None else
                                  ("выше типичного" if value > art["mean"][i] else "ниже типичного"),
                                  "contribution": round(part, 4)})
        probability = 1 / (1 + math.exp(-logit))
        contributions.sort(key=lambda c: -abs(c["contribution"]))
        is_signal = probability >= self.threshold
        return {
            "probability": probability, "logit": logit, "is_signal": is_signal,
            "threshold": self.threshold, "confident": probability > 0.75,
            "top_for": [c for c in contributions if c["contribution"] > 0][:5],
            "top_against": [c for c in contributions if c["contribution"] < 0][:5],
            "contributions": contributions,
            "exclusion_reasons": [] if is_signal else exclusion_reasons(features),
            "warnings": exclusion_reasons(features) if is_signal else [],
            "model": art["name"], "feature_schema_version": art["feature_schema_version"],
        }


_lock = threading.Lock()
_default: Scorer | None = None


def default_scorer() -> Scorer:
    global _default
    with _lock:
        if _default is None:
            _default = Scorer.load()
        return _default


def score_candidate(name: str, domain: str, snapshot: date | None = None, scorer: Scorer | None = None) -> dict:
    """Разметка признаков (сеть + LLM, кэшируется) и оценка одного кандидата."""
    from app.ml.features.extract import featurize

    scorer = scorer or default_scorer()
    row = featurize(name, domain, snapshot or date.today(), requested_features=scorer.features)
    result = scorer.score(row)
    return {"name": name, "domain": domain, "search_term": row["search_term"],
            "snapshot_date": row["snapshot_date"], "extraction_status": row["extraction_status"],
            "patent_source": row.get("patent_source", "unavailable"),
            "patent_warning": row.get("patent_warning", ""),
            "features": {k: row[k] for k in row if k in FEATURE_LABELS}, **result}
