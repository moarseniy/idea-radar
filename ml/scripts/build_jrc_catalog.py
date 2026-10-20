"""Собирает ml/datasets/external/jrc_weak_signals.csv из PDF отчётов JRC
«Weak signals in Science and Technologies» (выпуски 2021 и 2024).

Запуск из корня репозитория:

    .venv/bin/python ml/scripts/build_jrc_catalog.py \
        --pdf-2021 /private/tmp/claude-501/jrc/jrc2021.pdf \
        --pdf-2024 /private/tmp/claude-501/jrc/jrc2024.pdf

Сеть не используется, PDF передаются аргументами. Скрипт завершается с ошибкой,
если число сигналов не совпадает с числом, заявленным в самом отчёте (93 и 221).
Текст извлекается pymupdf, а если его нет, то pypdf.

Поля page и cluster берутся из отчёта: page — печатный номер страницы, где стоит
подпись к сигналу (2021) или запись приложения (2024; если записи нет, то страница
списка «Figure 1»). cluster — раздел отчёта (2021: по страницам оглавления).
short_description — текст из отчёта (2021: абзац под заголовком сигнала; 2024: пункт a)
приложения), пустой, если в отчёте такого текста не нашлось.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "ml" / "datasets" / "external" / "jrc_weak_signals.csv"

REPORTS = {
    2021: {
        "report": "Weak signals in Science and Technologies in 2021",
        "published_date": "2022-08-22",  # dc.date.available в записи репозитория JRC129501
        "source_url": "https://publications.jrc.ec.europa.eu/repository/handle/JRC129501",
        "expected": 93,
    },
    2024: {
        "report": "Weak signals in Science and Technologies - 2024",
        "published_date": "2025-02-17",  # dc.date.available в записи репозитория JRC140959
        "source_url": "https://publications.jrc.ec.europa.eu/repository/handle/JRC140959",
        "expected": 221,
    },
}

COLUMNS = [
    "signal_id", "report", "report_year", "published_date", "title_en",
    "cluster", "short_description", "page", "source_url",
]

# Кластеры 2024 в порядке «Figure 1»: (название для CSV, строка-заголовок в тексте PDF).
CLUSTERS_2024 = [
    ("Advanced Materials and Advanced Manufacturing", "advanced materials and advanced manufacturing"),
    ("Aerospace", "aerospace"),
    ("Mobility and Transport", "mobility and transport"),
    ("Digital Twin", "digital twins"),
    ("Artificial Intelligence and Machine Learning", "artificial intelligence & machine learning"),
    ("Information and Communication Technologies", "information and communication technologies"),
    ("Medical Imaging", "medical imaging"),
    ("Therapeutics and Biotechnologies", "therapeutics and biotechnologies"),
    ("e-Health", "e-health"),
    ("Environment and Agriculture", "environment and agriculture"),
    ("Energy", "energy"),
    ("Quantum and Cryptography", "quantum and cryptography"),
]

CAPTION_2021 = re.compile(r"^Figure (\d+): Visualisations\s*(?:for\s+)?(.+?)\s*\.?$")
ANY_FIGURE = re.compile(r"^Figure \d+[:.]")
TOC_TOPIC_2021 = re.compile(r"^2\.\d (.+?)\s*\.{4,}\s*(\d+)\s*$")
TOPIC_HEADING = re.compile(r"^2\.\d [A-Za-z&]")
DOT_LEADER = re.compile(r"\.{4,}")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def read_pages(pdf: Path) -> list[str]:
    try:
        import pymupdf  # type: ignore

        with pymupdf.open(pdf) as doc:
            return [page.get_text() for page in doc]
    except ImportError:
        from pypdf import PdfReader  # type: ignore

        return [page.extract_text() or "" for page in PdfReader(str(pdf)).pages]


def printed_numbers(pages: list[str]) -> list[int]:
    """Печатные номера страниц по порядку PDF.

    Номер ищется в первых и последних трёх строках страницы (строка из одних цифр).
    Там, где его нет (в приложении многих страниц), берётся номер PDF минус наиболее
    частое смещение, найденное по страницам с номером. В обоих отчётах смещение 3.
    """
    detected: dict[int, int] = {}
    for idx, text in enumerate(pages):
        lines = text.splitlines()
        for raw in lines[:3] + lines[-3:]:
            m = re.fullmatch(r"\s*(\d{1,3})\s*", raw)
            if m:
                detected[idx] = int(m.group(1))
                break
    offsets = [idx + 1 - num for idx, num in detected.items()]
    offset = max(set(offsets), key=offsets.count) if offsets else 0
    return [detected.get(idx, idx + 1 - offset) for idx in range(len(pages))]


def flat_lines(pages: list[str]) -> list[tuple[int, str]]:
    out = []
    for pno, text in enumerate(pages, 1):
        for raw in text.splitlines():
            out.append((pno, " ".join(raw.split())))
    return out


def is_noise(line: str) -> bool:
    return (
        not line
        or re.fullmatch(r"\d{1,3}", line) is not None
        or line.startswith(("Dashboard", "http"))
        or TOPIC_HEADING.match(line) is not None
    )


def parse_2021(pages: list[str]) -> list[dict]:
    lines = flat_lines(pages)
    printed = printed_numbers(pages)

    # Оглавление раздела 2: семь тем с печатными номерами страниц.
    topics: list[tuple[int, str]] = []
    for _, line in lines:
        m = TOC_TOPIC_2021.match(line)
        if m:
            topics.append((int(m.group(2)), m.group(1)))
        if len(topics) == 7:
            break
    topics.sort()

    # Подписи «Figure N: Visualisations for <сигнал>»: номера с 6 (1, 2, 4 — обзорные графики).
    captions = []
    for i, (pno, line) in enumerate(lines):
        if DOT_LEADER.search(line):
            continue  # строки оглавления
        m = CAPTION_2021.match(line)
        if m and int(m.group(1)) >= 6:
            captions.append((i, pno, m.group(2).strip()))

    boundaries = [i for i, (_, line) in enumerate(lines) if ANY_FIGURE.match(line) and not DOT_LEADER.search(line)]

    records = []
    for i, pno, name in captions:
        prev = max((b for b in boundaries if b < i), default=-1)
        segment = [line for _, line in lines[prev + 1 : i]]

        # Заголовок сигнала — короткая строка, наиболее похожая на название из подписи.
        # Берём последнее лучшее совпадение: сам заголовок стоит ближе к подписи, чем список на стр. 9.
        best_j, best_ratio = None, 0.0
        for j, line in enumerate(segment):
            if not line or len(line) > 80 or is_noise(line):
                continue
            ratio = difflib.SequenceMatcher(None, norm(line), norm(name)).ratio()
            if ratio >= best_ratio:
                best_j, best_ratio = j, ratio
        if best_j is not None and best_ratio >= 0.75:
            desc = " ".join(l for l in segment[best_j + 1 :] if not is_noise(l))
        else:
            desc = ""

        page_no = printed[pno - 1]
        cluster = next((t for start, t in reversed(topics) if page_no >= start), "")
        records.append({
            "title_en": name,
            "cluster": cluster,
            "short_description": desc,
            "page": page_no,
        })
    return records


def parse_2024(pages: list[str]) -> list[dict]:
    lines = flat_lines(pages)
    printed = printed_numbers(pages)

    # Список сигналов по кластерам: «Figure 1. Weak signals organized by clusters».
    start = next(i for i, (_, l) in enumerate(lines) if l == "Figure 1. Weak signals organized by clusters")
    end = next(i for i in range(start + 1, len(lines)) if lines[i][1].startswith("Dashboard:"))

    entries: list[dict] = []
    cur = 0
    for pno, line in lines[start + 1 : end]:
        if not line or line == "Source: JRC" or re.fullmatch(r"\d{1,3}", line):
            continue
        low = line.lower()
        if cur + 1 < len(CLUSTERS_2024) and low == CLUSTERS_2024[cur + 1][1]:
            cur += 1
            continue
        if cur > 0 and low == CLUSTERS_2024[0][1]:
            continue  # заголовок AM, попавший в текст колонки AI (артефакт извлечения PDF)
        entries.append({"title_en": line, "cluster": CLUSTERS_2024[cur][0], "list_page": pno,
                        "short_description": "", "page": None})

    # Приложение с описаниями: пункты «a) ... b) ...», имя сигнала стоит перед «a)».
    annex_first = next(
        i for i, p in enumerate(pages)
        if any(line.strip() == "Annexes" for line in p.splitlines())
    )
    annex = "\n".join(pages[annex_first:])
    page_of_offset = []
    running = 0
    for idx in range(annex_first, len(pages)):
        page_of_offset.append((running, printed[idx]))
        running += len(pages[idx]) + 1

    def page_at(offset: int) -> int:
        return [p for o, p in page_of_offset if o <= offset][-1]

    norm_names = [norm(e["title_en"]) for e in entries]
    for m in re.finditer(r"(?<!\w)a\)\s", annex):
        window = norm(annex[max(0, m.start() - 150) : m.start()])
        best, best_end = None, -1
        for idx, nn in enumerate(norm_names):
            if len(nn) < 6 or entries[idx]["short_description"]:
                continue
            pos = window.rfind(nn)
            if pos != -1 and pos + len(nn) > best_end:
                best, best_end = idx, pos + len(nn)
        if best is None:
            continue
        a_start = m.end()
        b = re.search(r"(?<!\w)b\)\s", annex[a_start:])
        a_text = annex[a_start : a_start + (b.start() if b else 600)]
        entries[best]["short_description"] = " ".join(a_text.split())
        entries[best]["page"] = page_at(m.start())

    for e in entries:
        if e["page"] is None:
            e["page"] = printed[e["list_page"] - 1]
    return entries


def build(pdf_2021: Path, pdf_2024: Path, out: Path) -> None:
    rows = []
    for year, pdf, parser in ((2021, pdf_2021, parse_2021), (2024, pdf_2024, parse_2024)):
        recs = parser(read_pages(pdf))
        expected = REPORTS[year]["expected"]
        if len(recs) != expected:
            sys.exit(f"{year}: извлечено {len(recs)} сигналов, в отчёте заявлено {expected}")
        meta = REPORTS[year]
        for n, rec in enumerate(recs, 1):
            rows.append({
                "signal_id": f"JRC{year}-{n:03d}",
                "report": meta["report"],
                "report_year": year,
                "published_date": meta["published_date"],
                "title_en": rec["title_en"],
                "cluster": rec["cluster"],
                "short_description": rec["short_description"],
                "page": rec["page"],
                "source_url": meta["source_url"],
            })
        missing = sum(1 for r in recs if not r["short_description"])
        print(f"{year}: {len(recs)} сигналов, без описания в отчёте: {missing}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"записано {len(rows)} строк: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-2021", type=Path, required=True)
    ap.add_argument("--pdf-2024", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build(args.pdf_2021, args.pdf_2024, args.out)


if __name__ == "__main__":
    main()
