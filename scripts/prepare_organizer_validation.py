"""Freeze organizer validation identities without exporting label hints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from app.radar.dataset import read_catalog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/training/organizer_validation_manifest.csv"
SNAPSHOT_DATE = "2026-09-22"


def build(output: Path = DEFAULT_OUTPUT) -> list[dict[str, str]]:
    catalog = read_catalog(ROOT / "data")
    records = catalog["records"]
    if len(records) != 100:
        raise ValueError(f"Expected exactly 100 organizer examples; got {len(records)}")
    if {record["id"] for record in records} != set(range(1, 101)):
        raise ValueError("Organizer IDs must be unique integers 1–100")
    rows = []
    for record in records:
        title = str(record["title"] or "").strip()
        rows.append({
            "organizer_id": f"organizer_{record['id']:03d}",
            "name_original": title,
            "domain_original": str(record["domain"] or "").strip(),
            "snapshot_date": SNAPSHOT_DATE,
            "label": "1",
        })
    if any(not row["name_original"] for row in rows):
        raise ValueError("Organizer title is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build(args.output)
    print(f"organizer_validation={len(rows)} manifest={args.output}")


if __name__ == "__main__":
    main()
