"""Дедупликация по общим источникам с проверкой моделью.

Словесное совпадение ловит только точные повторы. Одна и та же технология, извлечённая
из одной статьи в разных прогонах, получает разные формулировки: «Тактильное восприятие
для точного манипулирования» и «Мультимодальное тактильное очувствление роботов» — это
Xense Robotics дважды.

Правило: строки, у которых есть общая ссылка-источник, образуют группу (связность по
общим ссылкам). Группу из двух и более строк модель разбивает на действительно разные
технологии — из одного обзорного материала их законно бывает несколько. Из каждой
технологии остаётся одна строка: с научным подтверждением раньше, чем «только медиа»,
иначе самая ранняя.

Негативы из Википедии: одна статья — одна технология, модель не нужна.

Понадобится и при слиянии с датасетом Арсения.

Запуск: python ml/scripts/dedupe.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402

REMOVED = collect.ROOT / "ml" / "datasets" / "dedupe_removed.csv"

PROMPT = """Ниже названия технологий, извлечённые из одних и тех же источников.
Разбей их на группы: в одну группу — названия, описывающие ОДНУ И ТУ ЖЕ технологию
(разными словами), в разные — действительно разные технологии.

{items}

Ответ строго JSON: {{"groups": [[1, 3], [2]]}} — номера из списка, каждый ровно в одной группе."""


def components(rows: list[dict]) -> list[list[dict]]:
    """Связные группы строк по общим ссылкам (объединение-поиск)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_url = defaultdict(list)
    for i, row in enumerate(rows):
        for url in row["evidence_urls"].split(";"):
            url = url.strip().rstrip("/")
            # Главная страница сайта — не статья, по ней строки не связываем.
            if url.startswith("http") and url.count("/") > 3:
                by_url[url].append(i)
    for idx in by_url.values():
        for j in idx[1:]:
            parent[find(j)] = find(idx[0])
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(rows[i])
    return [g for g in groups.values() if len(g) > 1]


def split_same(group: list[dict]) -> list[list[dict]]:
    items = "\n".join(f"{i}. {r['name']}" for i, r in enumerate(group, start=1))
    response = collect._post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": collect.MODEL, "temperature": 0, "response_format": {"type": "json_object"},
         "messages": [{"role": "user", "content": PROMPT.format(items=items)}]},
        {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    content = response["choices"][0]["message"]["content"] or "{}"
    try:
        raw = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0)).get("groups", [])
    except (AttributeError, json.JSONDecodeError):
        return [[r] for r in group]  # не разобрали — ничего не удаляем
    used, clusters = set(), []
    for cluster in raw:
        members = [group[i - 1] for i in cluster if isinstance(i, int) and 1 <= i <= len(group) and i not in used]
        used.update(i for i in cluster if isinstance(i, int))
        if members:
            clusters.append(members)
    clusters += [[group[i - 1]] for i in range(1, len(group) + 1) if i not in used]
    return clusters


def keeper(cluster: list[dict]) -> dict:
    return sorted(cluster, key=lambda r: ("media_only" in r["annotator"], r["added_at"]))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(collect.LABELED.open(encoding="utf-8")))
    fields = list(rows[0])
    drop: dict[str, str] = {}

    # Негативы: одна статья Википедии — одна технология
    seen_wiki = {}
    for row in rows:
        url = row["evidence_urls"]
        if row["label"] == "0" and "wikipedia.org/wiki/" in url:
            if url in seen_wiki:
                drop[row["id"]] = f"та же статья Википедии, что у {seen_wiki[url]}"
            else:
                seen_wiki[url] = row["id"]

    positives = [r for r in rows if r["label"] == "1"]
    groups = components(positives)
    with ThreadPoolExecutor(6) as pool:
        splits = list(pool.map(split_same, groups))
    for clusters in splits:
        for cluster in clusters:
            keep = keeper(cluster)
            for row in cluster:
                if row is not keep:
                    drop[row["id"]] = f"та же технология, что «{keep['name'][:70]}»"

    print(f"групп позитивов с общими источниками: {len(groups)}")
    print(f"к удалению: {len(drop)} (позитивов {sum(1 for r in rows if r['id'] in drop and r['label'] == '1')}, "
          f"негативов {sum(1 for r in rows if r['id'] in drop and r['label'] == '0')})")
    for row in [r for r in rows if r["id"] in drop][:12]:
        print(f"  - {row['name'][:70]}\n      {drop[row['id']][:90]}")
    if args.dry_run:
        return 0

    with collect.LABELED.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in rows if r["id"] not in drop)
    collect.Writer().append(REMOVED, [{"id": r["id"], "name": r["name"], "reason": drop[r["id"]]}
                                      for r in rows if r["id"] in drop], ["id", "name", "reason"])
    print(f"\nосталось строк: {len(rows) - len(drop)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
