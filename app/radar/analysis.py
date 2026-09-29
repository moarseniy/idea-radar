from __future__ import annotations

import hashlib
import re
from datetime import date
from difflib import SequenceMatcher
from itertools import pairwise

from app.radar.models import Assessment
from app.radar.sources import quote_present

RUBRIC_VERSION = "rubric-1.0"
WEIGHTS = {"early_stage": 30, "novelty": 25, "momentum": 20, "evidence": 25}
VALUES = {"strong": 1, "partial": .5, "unknown": 0, "contradictory": 0}


def candidate_key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def merge_candidates(candidates: list[dict], limit: int) -> list[dict]:
    merged: list[dict] = []
    for item in candidates:
        key = candidate_key(item["title"] + " " + item["application"])
        duplicate = next((c for c in merged if SequenceMatcher(None, key, c["_key"]).ratio() > .86), None)
        if duplicate:
            duplicate["source_ids"] = sorted(set(duplicate["source_ids"] + item["source_ids"]))
        elif len(merged) < limit:
            merged.append({**item, "id": "cand_" + hashlib.sha256(key.encode()).hexdigest()[:12], "_key": key})
    return [{k: v for k, v in c.items() if k != "_key"} for c in merged]


def checked_assessment(candidate: dict, assessment: Assessment, source_map: dict, as_of: date) -> dict:
    item = assessment.model_dump()
    limitations = list(item["limitations"])
    evidence = []
    for ev in item["evidence"]:
        source = source_map.get(ev["source_id"])
        if not source or source["status"] != "read" or source.get("duplicate_of"):
            continue
        if ev["event_date"]:
            try:
                if date.fromisoformat(ev["event_date"]) > as_of:
                    limitations.append("Цитата о будущем событии исключена из основания текущего решения.")
                    continue
            except ValueError:
                ev["event_date"] = None
                limitations.append("Точная дата события не подтверждена.")
        if quote_present(ev["quote"], source["text"]):
            evidence.append({**ev, "id": f"ev_{candidate['id']}_{len(evidence)+1}", "quote_verified": True})
        else:
            limitations.append("Один фрагмент модели не найден дословно в документе и исключён.")
    grounded = {ev["source_id"] for ev in evidence}
    supporting = {ev["source_id"] for ev in evidence if ev["relation"] == "supports"}
    for name in ["description", "why_now", "why_early", "advantage", "case"]:
        finding = item[name]
        finding["source_ids"] = [sid for sid in finding["source_ids"] if sid in grounded]
        finding["grounded"] = bool(finding["source_ids"])
    if item["case_kind"] == "observed" and not item["case"]["grounded"]:
        item["case_kind"] = "unknown"
        item["case"]["text"] = "Реальный кейс не подтверждён проверяемым фрагментом."
    predictors = {}
    for p in item["predictors"]:
        p["source_ids"] = [sid for sid in p["source_ids"] if sid in grounded]
        if not p["source_ids"]:
            p["value"] = "unknown"
        predictors[p["name"]] = p
    for name in [*WEIGHTS, "maturity"]:
        predictors.setdefault(name, {"name": name, "value": "unknown", "explanation": "Данных недостаточно.", "source_ids": []})
    score = round(sum(WEIGHTS[k] * VALUES[predictors[k]["value"]] for k in WEIGHTS))
    decision = item["verdict"]
    if decision == "weak_signal":
        if predictors["maturity"]["value"] == "strong":
            decision = "mature"
            item["reason"] = "Есть подтверждённые признаки зрелости рассматриваемого применения."
        elif (not supporting or not item["why_early"]["grounded"] or not item["description"]["grounded"]
              or predictors["early_stage"]["value"] not in {"strong", "partial"}
              or predictors["novelty"]["value"] not in {"strong", "partial"}):
            decision = "insufficient_evidence"
            item["reason"] = "Не хватает проверяемых оснований для ранней стадии и конкретной новизны."
    if not grounded and decision == "mature":
        decision = "insufficient_evidence"
        item["reason"] = "Вывод о зрелости не подтверждён дословным фрагментом источника."
    if any(not source_map[sid].get("published_at") for sid in grounded):
        limitations.append("Дата публикации части источников не установлена.")
    return {**candidate, **item, "id": candidate["id"], "verdict": decision,
            "predictors": list(predictors.values()), "evidence": evidence,
            "source_ids": sorted(grounded), "score": score, "score_kind": "rubric",
            "score_label": "Оценка по критериям, не вероятность", "rubric_version": RUBRIC_VERSION,
            "limitations": list(dict.fromkeys(limitations)), "summary_generated": True,
            "rank": None}


def unassessed(candidate: dict, reason: str) -> dict:
    return {**candidate, "verdict": "insufficient_evidence", "reason": reason, "score": 0,
            "score_kind": "rubric", "evidence": [], "predictors": [], "limitations": [reason], "rank": None}


TERM_STOP = {"ai", "for", "and", "the", "of", "in", "based", "system", "systems", "technology", "technologies",
             "platform", "solution", "solutions", "application", "applications"}


def technology_words(item: dict) -> list[str]:
    """Значимые слова технологии по порядку: термин модели (канонический английский), иначе название."""
    term = (item.get("model") or {}).get("search_term") or item.get("title", "")
    words = re.findall(r"[\w-]+", term.casefold())
    return [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if w not in TERM_STOP and len(w) > 1]


def same_technology(a: list[str], b: list[str]) -> bool:
    """Доля общих слов ≥ 0,5, вложение терминов или общая пара слов подряд («synthetic data»)."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return False
    if len(sa & sb) / len(sa | sb) >= 0.5 or sa <= sb or sb <= sa:
        return True
    return bool(set(pairwise(a)) & set(pairwise(b)))


def rank_results(assessments: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    """ТОП по убыванию оценки; одна технология — одно место.

    Сигналы одной технологии (разные применения) склеиваются в кластер по связности
    same_technology(); место в ТОП занимает лучший по оценке, остальные — в его поле related.
    """
    pool = sorted((a for a in assessments if a["verdict"] == "weak_signal"),
                  key=lambda a: (-a["score"], -len(a["evidence"]), a["id"]))
    rejected = [a for a in assessments if a["verdict"] != "weak_signal"]
    words = [technology_words(a) for a in pool]
    parent = list(range(len(pool)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if same_technology(words[i], words[j]):
                parent[max(root(i), root(j))] = min(root(i), root(j))  # корень — лучший по оценке
    clusters: dict[int, list[dict]] = {}
    for i, item in enumerate(pool):
        clusters.setdefault(root(i), []).append(item)
    selected = []
    for members in list(clusters.values())[:limit]:  # pool отсортирован, порядок кластеров — по лидеру
        leader, *others = members
        selected.append({**leader, "rank": len(selected) + 1,
                         "related": [{"id": o["id"], "title": o["title"], "score": o["score"],
                                      "application": o.get("application", "")} for o in others]})
    return selected, rejected


def rank_candidates(assessments: list[dict], limit: int) -> list[dict]:
    """Prioritize weak signals, then fill any remaining slots by confidence.

    The final shortlist may contain mature, irrelevant, or insufficient-evidence
    items when fewer than ``limit`` candidates pass the weak-signal gates. Within
    both groups, full feature scores rank before median-imputed fallback scores;
    scores then sort descending and the verdict is preserved.
    """
    ordered = sorted(assessments,
                     key=lambda item: (item.get("verdict") != "weak_signal",
                                       bool((item.get("model") or {}).get("fallback")),
                                       -item.get("score", 0), -len(item.get("evidence", [])), item["id"]))
    return [{**item, "rank": index} for index, item in enumerate(ordered[:limit], 1)]
