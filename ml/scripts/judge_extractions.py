"""LLM-судья: слепая оценка того, как модели прочитали одни и те же заголовки.

Вход — результат compare_feature_models.py: headlines.jsonl (одинаковые для всех моделей
заголовки и статьи) и extractions.jsonl (ответ каждой модели). Судья видит строку один раз
со всеми ответами под буквами в случайном порядке (зерно — id строки) и не знает, какая
модель какая. Для каждого ответа он проверяет:
  события — есть ли оно в указанном заголовке, про эту ли технологию, верен ли тип;
  пропуски — сколько явных событий из заголовков не выписано;
  стадию — сначала ставит свою по заголовкам, потом оценивает обоснованность каждой;
  «технология или нет» и «массово доступна» — верно ли.

Судья из другого семейства моделей, чем кандидаты, чтобы не оценивать «своих». Проверка
самого судьи: его стадия на примерах заказчика сравнивается со стадией заказчика.

Выход: ml/reports/judge_extractions.{json,md}, ответы судьи — в кэше LLM.
Запуск: python ml/scripts/judge_extractions.py [--per-part 40] [--judge anthropic/claude-opus-5.5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_training_set as bts  # грузит .env

import app.ml.features.evidence as ev

SRC = bts.DATASETS / "model_compare"
REPORTS = bts.ROOT / "ml" / "reports"
LETTERS = "ABCDEFGH"

PROMPT = """Ты проверяешь, как разные системы извлекли факты о технологии из новостных заголовков.
Технология: «{name}» (область: {domain}; поисковый термин: {term}). Срез: {snapshot}.

Заголовки (номер. дата | издание | заголовок):
{headlines}

Недавние научные работы: {papers}

Шкала стадии: 1 — исследования; 2 — прототип/PoC; 3 — пилоты у заказчиков; 4 — раннее
коммерческое внедрение (единицы-десятки внедрений); 5 — масштабирование/зрелая массовая
технология с лидерами рынка и стандартами.
Типы событий: pilot, deployment, product, funding, grant, standard_final, regulation, procurement.

Сначала сам, до чтения ответов, оцени по заголовкам стадию ("judge_stage", 1-5), массовую
доступность ("judge_mass_market") и является ли объект технологией или техническим подходом,
а не продуктом/брендом/бизнес-моделью ("judge_is_technology").

Затем оцени каждый ответ:
{variants}

Для каждого варианта верни:
  "events": список вердиктов по его событиям в том же порядке: "ok" — событие прямо описано
    в указанном заголовке, про эту технологию, тип верный; "wrong_type" — событие есть, тип
    неверный; "unsupported" — в заголовке такого события нет; "other_tech" — событие про
    другую технологию или компанию без связи с ней;
  "missed": сколько явных событий из заголовков вариант не выписал (целое);
  "stage_justified": 1-5 — насколько стадия варианта обоснована заголовками (5 — точно);
  "is_technology_ok": true/false; "mass_market_ok": true/false;
  "overall": 1-10 — общая точность и полнота извлечения.
Оценивай только по заголовкам и работам выше, не по своим знаниям о рынке.

Ответ строго JSON:
{{"judge_stage":3,"judge_mass_market":false,"judge_is_technology":true,
"variants":{{"A":{{"events":["ok"],"missed":0,"stage_justified":4,"is_technology_ok":true,"mass_market_ok":true,"overall":8}}}}}}"""


def judge_call(model: str, prompt: str) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": 16000,
            "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}

    def parse(text: str) -> dict:
        content = (json.loads(text).get("choices") or [{}])[0].get("message", {}).get("content") or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("пустой ответ судьи")
        return json.loads(match.group(0))

    return ev.cached_fetch("llm", "https://openrouter.ai/api/v1/chat/completions", parse=parse, body=body,
                           headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"},
                           cache_key=f"judge\n{model}\n{prompt}", tries=4)


def variant_text(letter: str, x: dict) -> str:
    events = [{k: v for k, v in e.items() if k in ("h", "type", "company", "customer", "round", "amount_usd_m") and v not in (None, "")}
              for e in x["events"] or []]
    lines = [f"Вариант {letter}: технология: {x['own_query']['is_technology']}; стадия: {x['stage']}; "
             f"массово доступна: {x['mass_market']}"]
    lines += [f"  событие {i}: {json.dumps(e, ensure_ascii=False)}" for i, e in enumerate(events, 1)] or ["  событий нет"]
    return "\n".join(lines)


def sample(heads: dict, per_part: int) -> list[str]:
    """Детерминированная выборка: золото, валидационные негативы, обучение (поровну по меткам)."""
    def order(ids):
        return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    usable = [h for h in heads.values() if len(h["headlines"]) >= 3]
    gold = [h["id"] for h in usable if h["source"] == "gold"]
    val = [h["id"] for h in usable if h["source"] == "validation"]
    pos = [h["id"] for h in usable if h["source"] not in ("gold", "validation") and h["label"] == "1"]
    neg = [h["id"] for h in usable if h["source"] not in ("gold", "validation") and h["label"] == "0"]
    return order(gold)[:per_part] + order(val)[:per_part] + order(pos)[: per_part // 2] + order(neg)[: per_part // 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="anthropic/claude-opus-5.5")
    parser.add_argument("--per-part", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", default="", help="через запятую; по умолчанию все из extractions.jsonl")
    args = parser.parse_args()

    heads = {}
    for line in (SRC / "headlines.jsonl").open(encoding="utf-8"):
        h = json.loads(line)
        heads[h["id"]] = h
    ext = defaultdict(dict)
    for line in (SRC / "extractions.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        ext[e["id"]][e["model"]] = e
    models = sorted({m for per in ext.values() for m in per})
    if args.models:
        models = [m for m in models if m in args.models.split(",")]
    ids = [i for i in sample(heads, args.per_part) if all(m in ext[i] for m in models)]
    print(f"судья {args.judge}: {len(ids)} строк × {len(models)} моделей")

    def one(row_id: str) -> dict:
        h = heads[row_id]
        order = models[:]
        random.Random(row_id).shuffle(order)
        letters = dict(zip(LETTERS, order))
        listing = "\n".join(f"{i}. {x['date']} | {x['publisher']} | {x['title']}" for i, x in enumerate(h["headlines"], 1))
        prompt = PROMPT.format(name=h["name"], domain=h["domain"], term=h["term"], snapshot=bts.SNAPSHOT.isoformat(),
                               headlines=listing, papers="; ".join(h["papers"]) or "нет",
                               variants="\n\n".join(variant_text(k, ext[row_id][m]) for k, m in letters.items()))
        verdict = judge_call(args.judge, prompt)
        return {"id": row_id, "letters": letters, "verdict": verdict}

    results = [r for r in bts.parallel(one, ids, args.workers, "судья") if not isinstance(r, Exception)]
    print(f"оценено строк: {len(results)} из {len(ids)}")

    agg = {m: defaultdict(float) for m in models}
    position = defaultdict(list)
    judge_stage = []
    for r in results:
        v, h = r["verdict"], heads[r["id"]]
        if h["source"] == "gold" and str(h["stage_ref"]).isdigit() and isinstance(v.get("judge_stage"), int):
            judge_stage.append((int(h["stage_ref"]), v["judge_stage"]))
        scores = {}
        for letter, model in r["letters"].items():
            x, e, a = v.get("variants", {}).get(letter), ext[r["id"]][model], agg[model]
            if not isinstance(x, dict):
                continue
            verdicts = x.get("events") or []
            a["rows"] += 1
            a["events"] += len(verdicts)
            a["events_ok"] += sum(s == "ok" for s in verdicts)
            a["missed"] += int(x.get("missed") or 0)
            a["stage_justified"] += float(x.get("stage_justified") or 0)
            a["is_technology_ok"] += bool(x.get("is_technology_ok"))
            a["mass_market_ok"] += bool(x.get("mass_market_ok"))
            a["overall"] += float(x.get("overall") or 0)
            if isinstance(v.get("judge_stage"), int) and e["stage"]:
                a["stage_abs_err"] += abs(v["judge_stage"] - e["stage"])
                a["stage_n"] += 1
            scores[model] = float(x.get("overall") or 0)
            position[letter].append(float(x.get("overall") or 0))
        if scores:
            best = max(scores.values())
            for model, s in scores.items():
                agg[model]["wins"] += (s == best) / sum(1 for t in scores.values() if t == best)

    table = {}
    for m, a in agg.items():
        n = max(1, a["rows"])
        found = a["events_ok"]
        table[m] = {
            "rows": int(a["rows"]),
            "event_precision": round(found / max(1, a["events"]), 3),
            "event_recall": round(found / max(1, found + a["missed"]), 3),
            "events_per_row": round(a["events"] / n, 2),
            "stage_justified": round(a["stage_justified"] / n, 2),
            "stage_abs_err_vs_judge": round(a["stage_abs_err"] / max(1, a["stage_n"]), 2),
            "is_technology_acc": round(a["is_technology_ok"] / n, 3),
            "mass_market_acc": round(a["mass_market_ok"] / n, 3),
            "overall": round(a["overall"] / n, 2),
            "win_share": round(a["wins"] / max(1, len(results)), 3),
        }
    check = {"n": len(judge_stage),
             "exact": round(sum(a == b for a, b in judge_stage) / max(1, len(judge_stage)), 3),
             "within_1": round(sum(abs(a - b) <= 1 for a, b in judge_stage) / max(1, len(judge_stage)), 3)}
    report = {"judge": args.judge, "rows": len(results), "models": table, "judge_stage_vs_customer": check,
              "overall_by_position": {k: round(sum(v) / len(v), 2) for k, v in sorted(position.items())}}
    (REPORTS / "judge_extractions.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# LLM-судья: качество извлечения ({args.judge})", "",
             f"{len(results)} строк (золото, валидационные негативы, обучение), ответы моделей вслепую "
             "и в случайном порядке. Точность событий — доля событий, которые есть в указанном заголовке "
             "с верным типом; полнота — найдено / (найдено + пропущено по мнению судьи).", "",
             "| Модель | Точность событий | Полнота событий | Событий на строку | Стадия обоснована (1-5) "
             "| Ошибка стадии vs судья | «Технология» верно | Массовость верно | Общая (1-10) | Доля побед |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, t in sorted(table.items(), key=lambda kv: -kv[1]["overall"]):
        lines.append(f"| `{m}` | {t['event_precision']} | {t['event_recall']} | {t['events_per_row']} | {t['stage_justified']} "
                     f"| {t['stage_abs_err_vs_judge']} | {t['is_technology_acc']} | {t['mass_market_acc']} | {t['overall']} | {t['win_share']} |")
    lines += ["", f"Проверка судьи: его стадия на {check['n']} примерах заказчика совпадает со стадией заказчика "
              f"в {check['exact']:.0%}, с ошибкой не больше единицы — в {check['within_1']:.0%}.",
              f"Средняя общая оценка по позиции варианта (смещение порядка): {report['overall_by_position']}."]
    (REPORTS / "judge_extractions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
