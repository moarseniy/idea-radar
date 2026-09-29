"""Трудные негативы: зрелый «родитель» каждого позитива.

Для позитива «делегирование полномочий ИИ-агентам» зрелая пара — «ролевая модель доступа
(RBAC)»: та же область и лексика, другая стадия. Такие пары учат модель именно границе
«ранний сигнал / зрелая технология», а не различию областей или словаря.

Лёгкие негативы (учебные понятия прошлых десятилетий из категорий Википедии) этого не
дают: модель выучивает «старое понятие — негатив».

Отбор:
  1. Модель предлагает для позитива 1-2 зрелые технологии-предшественника той же задачи
     и точное название статьи Википедии. Это генерация кандидатов, не разметка.
  2. stage_judge определяет стадию по тексту статьи. Принимается только стадия 5
     (проверено: 13/14 зрелых узнаёт, 0/49 слабых сигналов зрелыми не называет).
  3. Пересечения с золотом, датасетом Арсения и собранными строками исключаются.

Запуск: python ml/scripts/contrastive_negatives.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402
import stage_judge  # noqa: E402

PROMPT = """Технология на ранней стадии: «{name}» (область: {domain}).

Назови {k} ЗРЕЛЫЕ технологии, которые решают ту же или смежную задачу давно и массово:
предшественники или устоявшиеся альтернативы, у которых есть лидеры рынка и стандарты.
Они должны быть из той же области и близки по смыслу, а не просто «что-то старое».

Для каждой: точное название статьи английской Википедии и русское название
4-10 слов в стиле «что делается для какой задачи».

Ответ строго JSON: {{"items":[{{"wiki_title":"...","name_ru":"..."}}]}}"""


def parents(row: dict, k: int = 2) -> list[dict]:
    response = collect._post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": collect.MODEL, "temperature": 0, "response_format": {"type": "json_object"},
         "messages": [{"role": "user", "content": PROMPT.format(name=row["name"], domain=row["domain"], k=k)}]},
        {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    content = response["choices"][0]["message"]["content"] or "{}"
    match = re.search(r"\{.*\}", content, re.DOTALL)
    try:
        items = json.loads(match.group(0)).get("items", []) if match else []
    except json.JSONDecodeError:
        return []
    return [i for i in items if i.get("wiki_title") and i.get("name_ru")][:k]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="*", help="добор только для этих областей")
    parser.add_argument("--k", type=int, default=2, help="сколько родителей просить на позитив")
    parser.add_argument("--from-gold", action="store_true",
                        help="пары к золоту → валидационные негативы в --out (не в обучение)")
    parser.add_argument("--out", type=Path, default=collect.LABELED)
    args = parser.parse_args()
    rows = list(csv.DictReader(collect.LABELED.open(encoding="utf-8")))
    # Пара строится один раз: повторный запуск добирает только новые позитивы.
    paired = {r["rationale"].split("зрелая пара к «")[1].split("»")[0]
              for r in rows if "зрелая пара к «" in r["rationale"]}
    if args.from_gold:
        positives = [dict(r, label="1") for r in csv.DictReader(collect.GOLD.open(encoding="utf-8"))]
    elif args.domains:  # добор: у всех позитивов области просим больше родителей
        positives = [r for r in rows if r["label"] == "1" and r["domain"] in args.domains]
    else:
        positives = [r for r in rows if r["label"] == "1" and r["name"][:80] not in paired]
    print(f"позитивов без пары: {len(positives)}")
    names = []
    for path in (collect.GOLD, collect.LABELED,
                 *sorted(p for p in collect.EXCLUSIONS.glob("*.csv") if not p.name.endswith("_terms.csv"))):
        if path.exists() and path.stat().st_size > 0:
            names += [r["name"] for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("name")]
    names = [n for n in names]  # названия из золота, собранного и Арсения
    seen = collect.Seen(names)
    # Статья Википедии = одна строка, в том числе среди резерва и удалённых дублей.
    titles_taken: set[str] = set()
    for path in (collect.LABELED, collect.LABELED.with_name("reserve.csv"),
                 collect.LABELED.with_name("dedupe_removed.csv"), args.out):
        if path.exists():
            for r in csv.DictReader(path.open(encoding="utf-8")):
                for url in (r.get("evidence_urls") or "").split():
                    if "wikipedia.org/wiki/" in url:
                        titles_taken.add(urllib.parse.unquote(url.split("/wiki/")[1]).replace("_", " "))

    def one(pos):
        out = []
        try:
            for cand in parents(pos, args.k):
                result = stage_judge.wiki_stage(cand["wiki_title"])
                out.append((pos, cand, result))
        except Exception:  # noqa: BLE001
            pass
        return out

    with ThreadPoolExecutor(6) as pool:
        results = [x for batch in pool.map(one, positives) for x in batch]

    accepted, stats = [], {"кандидатов": len(results), "нет статьи": 0, "не стадия 5": 0, "дубли": 0}
    for pos, cand, result in results:
        if not result:
            stats["нет статьи"] += 1
            continue
        title, verdict = result
        if verdict.get("stage") != 5:
            stats["не стадия 5"] += 1
            continue
        if title in titles_taken or title.replace("_", " ") in titles_taken or not seen.claim(cand["name_ru"]):
            stats["дубли"] += 1
            continue
        titles_taken.add(title)
        accepted.append({
            "id": f"neg-c{abs(hash(title)) % 10**7:07d}", "name": cand["name_ru"].strip(),
            "domain": pos["domain"], "label": 0, "stage": 5, "trend": verdict.get("trend", ""),
            "negative_type": "N1",
            "rationale": f"зрелая пара к «{pos['name'][:80]}» | Википедия «{title}»: "
                         f"{verdict.get('stage_quote', '')[:180]}",
            "annotator": f"contrastive:{collect.MODEL}",
            "evidence_urls": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            "added_at": date.today().isoformat(),
        })
    if args.from_gold:
        for a in accepted:
            a["annotator"] = f"contrastive-gold:{collect.MODEL}"
    collect.Writer().append(args.out, accepted, collect.LABELED_FIELDS)
    stats["принято"] = len(accepted)
    print(f"итог: {stats}")
    for a in accepted[:10]:
        print(f"  [{a['domain'][:10]}] {a['name'][:60]}\n      {a['rationale'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
