from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from typing import Any


DOMAIN_TERMS: list[tuple[str, str, tuple[str, ...]]] = [
    ("material", "хвосты", ("tailings", "хвост")),
    ("material", "руда", ("ore", "сырье")),
    ("material", "концентрат", ("concentrate",)),
    ("material", "медь", ("copper", "cu")),
    ("material", "никель", ("nickel", "ni")),
    ("material", "золото", ("gold", "au")),
    ("material", "серебро", ("silver", "ag")),
    ("material", "платина", ("platinum", "pt", "платиноид")),
    ("process", "флотация", ("flotation", "флот")),
    ("process", "измельчение", ("grinding", "доизмельчение", "помол")),
    ("process", "классификация", ("гидроциклон", "classification")),
    ("process", "выщелачивание", ("leaching", "цианирование")),
    ("process", "окислительный обжиг", ("roasting", "обжиг")),
    ("process", "магнитная сепарация", ("magnetic separation",)),
    ("reagent", "собиратель", ("collector", "ксантогенат", "xanthate")),
    ("reagent", "пенообразователь", ("frother", "пена")),
    ("reagent", "депрессор", ("depressant", "подавитель")),
    ("reagent", "активатор", ("activator", "медный купорос", "cuso4")),
    ("reagent", "известь", ("lime", "cao")),
    ("property", "извлечение", ("recovery", "извлеч")),
    ("property", "содержание", ("grade", "массовая доля")),
    ("property", "крупность", ("particle size", "мкм", "класс крупности")),
    ("property", "pH", ("кислотность",)),
    ("property", "расход реагента", ("дозировка", "кг/т", "g/t")),
    ("metric", "экономический эффект", ("себестоимость", "opex", "capex", "стоимость")),
    ("risk", "шламование", ("slime", "тонкие шламы")),
    ("risk", "переизмельчение", ("overgrinding",)),
    ("equipment", "флотомашина", ("машина флотации", "камера флотации")),
    ("equipment", "мельница", ("mill", "мельницы")),
]

UNIT_RE = r"%|г/т|кг/т|g/t|kg/t|мкм|µm|μm|um|мм|mm|т/ч|t/h|tph|ppm|°c|°с|℃"
PH_RE = r"pH|рН"
NUMBER_RE = r"\d+(?:[,.]\d+)?(?:\s*[-–—]\s*\d+(?:[,.]\d+)?)?"
NUMERIC_RE = re.compile(
    rf"(?<![\w/])(?:{PH_RE}\s*)?{NUMBER_RE}\s?(?:{UNIT_RE})(?![\w/])|(?<![\w/])(?:{PH_RE})\s?{NUMBER_RE}(?![\w/])",
    re.IGNORECASE,
)
RAW_NUMERIC_LABEL_RE = re.compile(
    rf"^\s*(?:[~≈<>≤≥±]\s*)?(?:{PH_RE}\s*)?{NUMBER_RE}(?:\s?(?:{UNIT_RE}))?\s*$",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"(?<=[.!?。;])\s+|\n+")


def stable_id(kind: str, label: str) -> str:
    digest = hashlib.sha1(f"{kind}:{label.lower()}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{digest}"


def edge_id(source: str, target: str, relation: str) -> str:
    digest = hashlib.sha1(f"{source}|{target}|{relation}".encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"


def score_hypothesis(novelty: float, feasibility: float, impact: float, risk: float, weights: dict[str, float] | None = None) -> float:
    weights = weights or {}
    novelty_w = float(weights.get("novelty", 0.25))
    feasibility_w = float(weights.get("feasibility", 0.25))
    impact_w = float(weights.get("impact", 0.35))
    risk_w = float(weights.get("risk", 0.15))
    total = novelty_w + feasibility_w + impact_w + risk_w or 1
    raw = novelty * novelty_w + feasibility * feasibility_w + impact * impact_w + (100 - risk) * risk_w
    return round(raw / total, 1)


def heuristic_graph_from_text(text: str, source_id: str, source_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    source_node_id = f"source:{source_id}"
    nodes[source_node_id] = {
        "id": source_node_id,
        "label": source_name[:80],
        "type": "source",
        "summary": "Загруженный источник",
        "weight": 1.0,
        "source_ids": [source_id],
    }

    lower_text = text.lower()
    for kind, label, aliases in DOMAIN_TERMS:
        aliases_full = (label, *aliases)
        count = sum(lower_text.count(alias.lower()) for alias in aliases_full)
        if count:
            nid = stable_id(kind, label)
            nodes[nid] = {
                "id": nid,
                "label": label,
                "type": kind,
                "summary": f"Упомянуто в источнике {source_name}",
                "weight": min(8.0, 1.0 + math.log(count + 1)),
                "source_ids": [source_id],
            }
            eid = edge_id(source_node_id, nid, "contains")
            edges[eid] = {
                "id": eid,
                "source": source_node_id,
                "target": nid,
                "relation": "contains",
                "evidence": f"{source_name}: {count} упомин.",
                "weight": min(6.0, 1.0 + math.log(count + 1)),
                "source_ids": [source_id],
            }

    for label, values, evidence, count in _extract_parameter_groups(text):
        nid = stable_id("parameter", label)
        examples = ", ".join(value for value, _ in values.most_common(8))
        evidence_text = " ".join(evidence[:2])
        summary = f"Группа числовых параметров. Примеры значений: {examples}."
        if evidence_text:
            summary = f"{summary} Контекст: {evidence_text[:260]}"
        nodes[nid] = {
            "id": nid,
            "label": label,
            "type": "parameter",
            "summary": summary,
            "weight": min(5.8, 1.2 + math.log(count + len(values) + 1)),
            "source_ids": [source_id],
        }
        eid = edge_id(source_node_id, nid, "reports_parameter")
        edges[eid] = {
            "id": eid,
            "source": source_node_id,
            "target": nid,
            "relation": "reports_parameter",
            "evidence": examples,
            "weight": min(5.0, 1.0 + math.log(count + 1)),
            "source_ids": [source_id],
        }

    for sentence in SENTENCE_RE.split(text[:60_000]):
        sentence_l = sentence.lower()
        present = []
        for kind, label, aliases in DOMAIN_TERMS:
            if any(alias.lower() in sentence_l for alias in (label, *aliases)):
                present.append((stable_id(kind, label), kind, label))
        for (left_id, left_kind, left_label), (right_id, right_kind, right_label) in itertools.combinations(present[:6], 2):
            relation = _relation_for(left_kind, right_kind)
            eid = edge_id(left_id, right_id, relation)
            if eid not in edges:
                evidence = sentence.strip()
                edges[eid] = {
                    "id": eid,
                    "source": left_id,
                    "target": right_id,
                    "relation": relation,
                    "evidence": evidence[:220],
                    "weight": 1.0,
                    "source_ids": [source_id],
                }
            else:
                edges[eid]["weight"] = float(edges[eid].get("weight", 1)) + 0.3

    return list(nodes.values()), list(edges.values())


def graph_from_hypothesis(hypothesis: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hid = f"hypothesis:{hypothesis['id']}"
    title = hypothesis.get("title") or hypothesis.get("statement", "Гипотеза")
    text = " ".join(
        str(hypothesis.get(key, ""))
        for key in ("title", "statement", "rationale", "mechanism", "uncertainty")
    )
    nodes = [
        {
            "id": hid,
            "label": title[:90],
            "type": "hypothesis",
            "summary": hypothesis.get("statement", ""),
            "weight": 3.0,
            "source_ids": [hypothesis["id"]],
        }
    ]
    edges = []
    lower = text.lower()
    for kind, label, aliases in DOMAIN_TERMS:
        if any(alias.lower() in lower for alias in (label, *aliases)):
            nid = stable_id(kind, label)
            nodes.append(
                {
                    "id": nid,
                    "label": label,
                    "type": kind,
                    "summary": "Связано с гипотезой",
                    "weight": 2.0,
                    "source_ids": [hypothesis["id"]],
                }
            )
            edges.append(
                {
                    "id": edge_id(nid, hid, "supports"),
                    "source": nid,
                    "target": hid,
                    "relation": "supports",
                    "evidence": title[:160],
                    "weight": 2.0,
                    "source_ids": [hypothesis["id"]],
                }
            )
    return nodes, edges


def _relation_for(left_kind: str, right_kind: str) -> str:
    pair = {left_kind, right_kind}
    if "process" in pair and "property" in pair:
        return "influences"
    if "reagent" in pair and "process" in pair:
        return "used_in"
    if "material" in pair and "process" in pair:
        return "processed_by"
    if "risk" in pair:
        return "constrained_by"
    if "equipment" in pair and "process" in pair:
        return "implemented_with"
    return "associated_with"


def is_low_signal_numeric_label(label: str) -> bool:
    text = _clean_parameter_value(label)
    return bool(RAW_NUMERIC_LABEL_RE.fullmatch(text))


def is_low_signal_numeric_node(node: dict[str, Any]) -> bool:
    kind = str(node.get("type") or "").strip().lower()
    if kind not in {"metric", "parameter", "property", "concept", "observation"}:
        return False
    return is_low_signal_numeric_label(str(node.get("label") or ""))


def _extract_parameter_groups(text: str) -> list[tuple[str, Counter[str], list[str], int]]:
    counters: dict[str, Counter[str]] = {}
    evidence: dict[str, list[str]] = {}
    for sentence in SENTENCE_RE.split(text[:100_000]):
        cleaned_sentence = " ".join(sentence.split())
        if not cleaned_sentence:
            continue
        for match in NUMERIC_RE.finditer(cleaned_sentence):
            value = _clean_parameter_value(match.group(0))
            label = _parameter_label(value, cleaned_sentence)
            if not label:
                continue
            counters.setdefault(label, Counter())[value] += 1
            examples = evidence.setdefault(label, [])
            if len(examples) < 3 and cleaned_sentence not in examples:
                examples.append(cleaned_sentence[:220])

    groups: list[tuple[str, Counter[str], list[str], int]] = []
    for label, values in counters.items():
        groups.append((label, values, evidence.get(label, []), sum(values.values())))
    groups.sort(key=lambda item: (-item[3], item[0]))
    return groups[:12]


def _parameter_label(value: str, context: str) -> str | None:
    value_l = value.lower().replace(" ", "")
    context_l = context.lower()

    if "ph" in value_l or "рн" in value_l:
        return "pH / кислотность"
    if any(unit in value_l for unit in ("г/т", "g/t", "кг/т", "kg/t")):
        return "Расход реагента / дозировка"
    if any(unit in value_l for unit in ("мкм", "µm", "μm", "um")):
        return "Крупность / размер частиц"
    if any(unit in value_l for unit in ("мм", "mm")):
        if _contains_any(context_l, ("крупн", "частиц", "зерн", "сито", "mesh", "particle", "size", "class")):
            return "Крупность / размер частиц"
        return "Размер / геометрия"
    if any(unit in value_l for unit in ("т/ч", "t/h", "tph")):
        return "Производительность / поток"
    if "ppm" in value_l:
        return "Концентрация / ppm"
    if any(unit in value_l for unit in ("°c", "°с", "℃")):
        return "Температура"
    if "%" in value_l:
        if _contains_any(context_l, ("извлеч", "recovery", "yield", "выход")):
            return "Извлечение / выход"
        if _contains_any(context_l, ("содерж", "grade", "массов", "доля", "концентрат")):
            return "Содержание / массовая доля"
        if _contains_any(context_l, ("влажн", "moisture")):
            return "Влажность"
        return "Процентный показатель"
    return None


def _clean_parameter_value(value: str) -> str:
    text = " ".join(str(value or "").replace(",", ".").split()).strip()
    replacements = (
        (r"(?i)^рН", "pH"),
        (r"(?i)\bum\b", "мкм"),
        (r"µm|μm", "мкм"),
        (r"(?i)\bmm\b", "мм"),
        (r"(?i)\bkg/t\b", "кг/т"),
        (r"(?i)\bg/t\b", "г/т"),
        (r"(?i)\bt/h\b|\btph\b", "т/ч"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
