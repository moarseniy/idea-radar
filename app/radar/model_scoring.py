"""Оценка кандидатов радара обученной моделью (app/ml/scorer.py).

Разметка одного кандидата — сеть и два вызова LLM, 20–60 секунд, поэтому кандидаты
размечаются параллельно и начинают размечаться сразу после выделения, одновременно
с проверочным поиском и LLM-проверкой цитат.

Решение по кандидату (apply_model):
  - LLM-проверка сочла кандидата не относящимся к запросу → «не соответствует запросу»
    (модель релевантность запросу не оценивает);
  - вероятность модели ниже порога → «зрелая» или «шум», причины — от модели;
  - вероятность выше порога, но у описания нет дословно подтверждённых цитат →
    «недостаточно доказательств» (выдача не должна держаться на знаниях LLM);
  - иначе — слабый сигнал. Ранжирование — по вероятности модели.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date

from app.radar.config import RadarSettings
from app.registry import supports_signal

logger = logging.getLogger("idea.radar.model")

MODEL_FIELDS = ("probability", "is_signal", "threshold", "confident", "top_for", "top_against",
                "exclusion_reasons", "warnings", "search_term", "extraction_status", "snapshot_date",
                "patent_source", "patent_warning", "model", "fallback", "fallback_reason")
MATURE_HINTS = ("зрелост", "массово", "стандарт", "существует")


def candidate_name(candidate: dict) -> str:
    title, application = candidate.get("title", "").strip(), candidate.get("application", "").strip()
    return f"{title}: {application}" if application and application.lower() not in title.lower() else title


def public_result(result: dict) -> dict:
    out = {k: result.get(k) for k in MODEL_FIELDS}
    for key in ("top_for", "top_against"):
        out[key] = [{k: c[k] for k in ("feature", "label", "display_value", "relative", "contribution")}
                    for c in out[key] or []]
    out["probability"] = round(float(out["probability"]), 4)
    return out


class ModelScorer:
    """Пул разметки кандидатов одного запуска радара."""

    def __init__(self, settings: RadarSettings, as_of: date, workers: int = 6) -> None:
        # featurize() читает ключ из окружения; сервис хранит его в настройках.
        if settings.openrouter_api_key:
            os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
        from app.ml.scorer import default_scorer
        self.scorer = default_scorer()
        self.as_of = as_of
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="radar-model")
        self.futures: dict[str, Future] = {}

    def submit(self, candidates: list[dict]) -> None:
        from app.ml.scorer import score_candidate
        for c in candidates:
            if c["id"] not in self.futures:
                self.futures[c["id"]] = self.pool.submit(
                    score_candidate, candidate_name(c), c.get("domain", ""), self.as_of, self.scorer)

    def result(self, candidate_id: str, timeout: float) -> dict | None:
        future = self.futures.get(candidate_id)
        if future is None:
            return None
        try:
            return public_result(future.result(timeout=max(0.1, timeout)))
        except FutureTimeoutError:
            # Feature extraction is still running; the caller can keep polling
            # without treating a pending candidate as a model failure.
            return None
        except Exception as exc:  # noqa: BLE001 — кандидат остаётся с оценкой LLM и пометкой
            logger.warning("Model scoring failed for %s: %s", candidate_id, type(exc).__name__)
            return None

    def fallback_result(self, candidate_id: str, reason: str) -> dict:
        """Always produce a real LR result when network feature extraction misses its budget.

        Scorer.score applies the model's documented median imputation. Keeping this
        result explicit lets the UI distinguish it from a complete feature extraction.
        """
        result = self.scorer.score({})
        result.update({
            "search_term": "",
            "snapshot_date": self.as_of.isoformat(),
            "extraction_status": "fallback_median_imputation",
            "patent_source": "unavailable",
            "patent_warning": "",
            "feature_schema_version": self.scorer.artifact.get("feature_schema_version", ""),
            "fallback": True,
            "fallback_reason": reason,
            "confident": False,
            "warnings": list(dict.fromkeys([*(result.get("warnings") or []), reason])),
        })
        return public_result(result)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def supporting_sources(item: dict, source_map: dict) -> list[dict]:
    ids = {e["source_id"] for e in item.get("evidence", []) if e.get("relation") == "supports"}
    return [source_map[i] for i in sorted(ids) if i in source_map]


def apply_model(item: dict, model: dict | None, sources: list[dict] | None = None) -> dict:
    """Итоговое решение по кандидату: вероятность модели + проверки LLM + реестр источников.

    sources — источники подтверждающих цитат; если переданы, сигнал должен опираться хотя бы
    на один независимый источник (не соцсеть, блог, агрегатор или пресс-релиз, ТЗ).
    """
    if model is None:
        limitations = [*item.get("limitations", []),
                       "Модель не успела оценить кандидата; решение принято по критериям проверки источников."]
        verdict = item.get("verdict")
        reason = item.get("reason", "")
        if verdict == "weak_signal":
            verdict = "insufficient_evidence"
            reason = "Логистическая оценка не завершена; статус слабого сигнала не подтверждён."
            limitations.append("Статус слабого сигнала требует завершённой оценки логистической регрессии.")
        return {**item, "verdict": verdict, "reason": reason,
                "llm_verdict": item.get("llm_verdict", item.get("verdict")),
                "model": None, "limitations": list(dict.fromkeys(limitations))}

    if model.get("fallback"):
        # A median-imputed baseline is a valid model computation but is not
        # candidate-specific evidence. Never let it promote an item to a signal.
        verdict = item.get("verdict", "insufficient_evidence")
        reason = item.get("reason", "")
        if verdict == "weak_signal":
            verdict = "insufficient_evidence"
            reason = ("Логистическая регрессия рассчитала предварительную оценку, но сбор признаков "
                      "не завершился; слабый сигнал не подтверждён.")
        limitations = list(item.get("limitations", [])) + [
            "Оценка логистической регрессии рассчитана с медианной подстановкой всех признаков, "
            "поэтому не отражает индивидуальные данные кандидата и не подтверждает слабый сигнал.",
        ]
        if model.get("fallback_reason"):
            limitations.append(model["fallback_reason"])
        return {**item, "verdict": verdict, "reason": reason,
                "llm_verdict": item.get("llm_verdict", item.get("verdict")),
                "rubric_score": item.get("score"), "score": round(100 * model["probability"]),
                "score_kind": "model_partial", "score_label": "Предварительная оценка модели, %",
                "model": model, "limitations": list(dict.fromkeys(limitations))}

    llm_verdict, verdict, reason = item["verdict"], item["verdict"], item.get("reason", "")
    supported, support_notes = supports_signal(sources) if sources is not None else (True, [])
    grounded = bool(item.get("evidence")) and item.get("description", {}).get("grounded", False)
    if llm_verdict == "irrelevant":
        pass  # модель не оценивает соответствие запросу
    elif not model["is_signal"]:
        reasons = model["exclusion_reasons"] or [
            f"{c['label']}: {c['display_value']}" for c in model["top_against"][:2]]
        verdict = "mature" if any(h in r for r in reasons for h in MATURE_HINTS) else "hype_or_noise"
        reason = (f"Модель: вероятность слабого сигнала {model['probability']:.0%} ниже порога "
                  f"{model['threshold']:.0%}. " + "; ".join(reasons))
    elif not grounded:
        verdict = "insufficient_evidence"
        reason = ("Модель считает кандидата слабым сигналом, но описание не подтверждено дословной "
                  "цитатой из прочитанных источников.")
    elif not supported:
        verdict = "insufficient_evidence"
        reason = f"Модель: вероятность слабого сигнала {model['probability']:.0%}. {support_notes[0]}"
    else:
        verdict = "weak_signal"
        if llm_verdict != "weak_signal":
            reason = (f"Модель: вероятность слабого сигнала {model['probability']:.0%}. "
                      f"Проверка источников отмечала: {item.get('reason', '')}")
    limitations = list(item.get("limitations", [])) + model["warnings"]
    if verdict == "weak_signal":
        limitations += support_notes
    if verdict == "weak_signal" and llm_verdict == "mature":
        # LLM часто называет «зрелым» уже единичное внедрение; по рубрике заказчика это
        # стадия «ранние внедрения», то есть ещё сигнал. Расхождение показываем явно.
        maturity = next((p for p in item.get("predictors", []) if p.get("name") == "maturity"), {})
        limitations.append("Проверка источников нашла признаки внедрения: "
                           f"{maturity.get('explanation') or item.get('reason', '')} "
                           "Модель относит это к ранней стадии; нужна экспертная проверка масштаба.")
    if model["extraction_status"] != "ok":
        limitations.append(f"Часть каналов признаков недоступна: {model['extraction_status']}.")
    return {**item, "verdict": verdict, "reason": reason, "llm_verdict": llm_verdict,
            "rubric_score": item.get("score"), "score": round(100 * model["probability"]),
            "score_kind": "model", "score_label": "Уверенность модели, %",
            "model": model, "limitations": list(dict.fromkeys(limitations))}
