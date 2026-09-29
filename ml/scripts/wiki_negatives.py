"""Зрелые технологии (N1) из Википедии — бесплатно и проверяемо.

Кандидаты: статьи из технологических категорий Википедии по шести областям датасета.
Отбор:
  1. Модель отсеивает то, что не является конкретной технологией (области знания,
     компании, персоны) и даёт русское название в стиле датасета.
  2. Стадию определяет stage_judge по тексту статьи. Негатив — только стадия 5
     (массовое внедрение). Проверено: 13 из 14 заведомо зрелых технологий модель узнаёт,
     и ни одного из 49 настоящих слабых сигналов зрелым не называет.
  3. Исключаются пересечения с золотом, датасетом Арсения и уже собранными строками.

ВАЖНО: источник обнаружения (Википедия у негативов, новости у позитивов) НЕ должен
попадать в признаки. При расчёте признаков доказательная база собирается заново,
одинаково для каждой строки — иначе классы разделятся по отпечатку пайплайна.

Запуск: python ml/scripts/wiki_negatives.py --per-domain 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402
import stage_judge  # noqa: E402

ROOT = collect.ROOT
CACHE = ROOT / "ml" / ".cache" / "wiki_categories"
UA = {"User-Agent": f"weak-signal-radar/0.1 (hackathon research; {os.getenv('OPENALEX_MAILTO', '')})"}

CATEGORIES = {
    "Индустриальный ИИ": ["Industrial automation", "Industrial computing", "Manufacturing",
                          "Industrial processes", "Quality control"],
    "Роботы": ["Industrial robots", "Robot kinematics", "Robotics engineering", "Mobile robots",
               "Robot control"],
    "Инфраструктура ИИ": ["Data centers", "Parallel computing", "Computer memory", "Supercomputers",
                          "Computer hardware cooling"],
    "Финтех": ["Payment systems", "Electronic funds transfer", "Banking technology", "Stock exchanges",
               "Payment cards"],
    "Защита ИИ": ["Computer security", "Authentication methods", "Computer network security",
                  "Cryptographic protocols", "Malware"],
    "Edge": ["Embedded systems", "Internet of things", "Mobile computers", "Wireless networking",
             "Microcontrollers"],
}

FILTER_PROMPT = """Ниже названия статей Википедии из области «{domain}».
Для каждой определи, является ли она КОНКРЕТНОЙ технологией (метод, устройство, протокол,
класс систем). Не являются: области знания («Robotics»), компании, персоны, продукты
одной компании, списки, организации, стандарты-документы, общие понятия.
Для технологий дай русское название 4-10 слов в стиле «что делается для какой задачи».

Статьи:
{titles}

Ответ строго JSON: {{"items":[{{"title":"...","technology":true,"name_ru":"..."}}]}}"""


def category_members(category: str) -> list[str]:
    cached = CACHE / (re.sub(r"[^a-z0-9]+", "_", category.lower()) + ".json")
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
        "cmnamespace": 0, "cmlimit": 200, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as response:
        titles = [m["title"] for m in json.loads(response.read())["query"]["categorymembers"]]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    return titles


def filter_technologies(domain: str, titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 25):
        chunk = titles[i:i + 25]
        response = collect._post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"model": collect.MODEL, "temperature": 0, "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": FILTER_PROMPT.format(
                 domain=domain, titles="\n".join(f"- {t}" for t in chunk))}]},
            {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        content = response["choices"][0]["message"]["content"] or "{}"
        match = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            items = json.loads(match.group(0)).get("items", []) if match else []
        except json.JSONDecodeError:
            continue
        out += [x for x in items if x.get("technology") and x.get("name_ru") and x.get("title") in chunk]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-domain", type=int, default=40, help="сколько кандидатов проверять на область")
    args = parser.parse_args()

    names = []
    for path in (collect.GOLD, collect.LABELED, *sorted(collect.EXCLUSIONS.glob("*.csv"))):
        if path.exists() and path.stat().st_size > 0:
            names += [r["name"] for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("name")]
    seen = collect.Seen(names)
    writer = collect.Writer()
    rng = random.Random(7)
    total = {"кандидатов": 0, "технологий": 0, "стадия 5": 0, "принято": 0, "дубли": 0}

    for domain, categories in CATEGORIES.items():
        titles = sorted({t for c in categories for t in category_members(c)})
        rng.shuffle(titles)
        techs = filter_technologies(domain, titles[: args.per_domain * 2])[: args.per_domain]
        total["кандидатов"] += min(len(titles), args.per_domain * 2)
        total["технологий"] += len(techs)

        def judge(item):
            try:
                return item, stage_judge.wiki_stage(item["title"])
            except Exception:  # noqa: BLE001
                return item, None

        with ThreadPoolExecutor(6) as pool:
            judged = list(pool.map(judge, techs))

        rows = []
        for item, result in judged:
            if not result or result[1].get("stage") != 5:
                continue
            total["стадия 5"] += 1
            if not seen.claim(item["name_ru"]):
                total["дубли"] += 1
                continue
            title, verdict = result
            rows.append({
                "id": f"neg-w{abs(hash(title)) % 10**7:07d}", "name": item["name_ru"].strip(), "domain": domain,
                "label": 0, "stage": 5, "trend": verdict.get("trend", ""), "negative_type": "N1",
                "rationale": f"Википедия «{title}»: {verdict.get('stage_quote', '')[:200]}",
                "annotator": f"wiki_negatives:{collect.MODEL}",
                "evidence_urls": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "added_at": date.today().isoformat(),
            })
        writer.append(collect.LABELED, rows, collect.LABELED_FIELDS)
        total["принято"] += len(rows)
        print(f"  [{domain}] технологий {len(techs)}, стадия 5: {sum(1 for _, r in judged if r and r[1].get('stage') == 5)}, "
              f"принято {len(rows)}", flush=True)

    print(f"\nитог: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
