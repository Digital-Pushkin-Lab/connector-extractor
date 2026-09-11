#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build data/linkers.csv from Коннекторы_правка.xlsx.

Rerun this script whenever the spreadsheet changes:

    python3 build_linkers_csv.py --xlsx /path/to/Коннекторы_правка.xlsx

Source sheets:
  * "Исходные" - original connector list with revised characteristics
  * "Доп"      - new connectors added during the revision

Columns read from the spreadsheet: Союз | POS | other_POS | semfield1 |
semfield2 | ПУ | Atoms.
  * "Союз" becomes the `linker` column (the single column
    `patterns.build_patterns_from_csv` reads to build match patterns).
  * POS / other_POS are dropped entirely (unreliable, not used downstream).
  * semfield1: ';'-separated alternatives (a connector's primary meaning
    may be ambiguous between two or more readings) -> written back
    ';'-joined.
  * semfield2 / ПУ (pragmatics) / Atoms: ','-separated sets -> written back
    ','-joined.
  * A row is dropped if its "Союз" cell is prefixed with "DEL" or filled
    orange (FFFFC000) - these were marked for removal by the reviewer.
  * "…" (ellipsis, U+2026) in a connector is normalized to the three-dot
    "..." marker used to detect discontinuous connectors (e.g.
    "если ... то").
  * Duplicate connectors (same spelling appearing more than once, across
    or within sheets) are merged: the first-seen row sets the base entry,
    and later duplicates' semfield1/semfield2/pragmatics/atoms are unioned
    into it (order preserved, deduplicated).
"""

import argparse
import csv
import re
from pathlib import Path

import openpyxl

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "linkers.csv"

SHEETS = ["Исходные", "Доп"]
DEL_FILL_RGB = "FFFFC000"
FIELDNAMES = ["linker", "semfield1", "semfield2", "pragmatics", "atoms"]


def is_deleted(cell) -> bool:
    value = cell.value
    if value and str(value).strip().upper().startswith("DEL"):
        return True
    fill = cell.fill
    if fill and fill.fgColor and fill.fgColor.rgb == DEL_FILL_RGB:
        return True
    return False


def normalize_key(raw: str) -> str:
    """Normalize a spreadsheet connector key to match the surface string
    `matching.pattern_surface` reconstructs for discontinuous connectors:
    an ellipsis (however spaced/typed in the sheet) becomes a bare "..."
    with no surrounding spaces, e.g. "если ... то" -> "если...то"."""
    key = str(raw).strip().replace("…", "...")
    return re.sub(r"\s*\.\.\.\s*", "...", key)


def split_alternatives(raw) -> list:
    if not raw or not str(raw).strip():
        return []
    return [p.strip() for p in str(raw).split(";") if p.strip()]


def split_set(raw) -> list:
    if not raw or not str(raw).strip():
        return []
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def merge_unique(existing: list, new: list) -> list:
    for item in new:
        if item not in existing:
            existing.append(item)
    return existing


def collect_linkers(xlsx_path: str) -> list:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    linkers = []
    by_key = {}

    for sheet_name in SHEETS:
        ws = wb[sheet_name]
        for row in ws.iter_rows(min_row=2):
            key_cell = row[0]
            if key_cell.value is None or not str(key_cell.value).strip():
                continue
            if is_deleted(key_cell):
                continue

            key = normalize_key(key_cell.value)
            semfield1 = split_alternatives(row[3].value)
            semfield2 = split_set(row[4].value)
            pragmatics = split_set(row[5].value)
            atoms = split_set(row[6].value)

            if key not in by_key:
                entry = {
                    "linker": key,
                    "semfield1": list(semfield1),
                    "semfield2": list(semfield2),
                    "pragmatics": list(pragmatics),
                    "atoms": list(atoms),
                }
                by_key[key] = entry
                linkers.append(entry)
            else:
                entry = by_key[key]
                merge_unique(entry["semfield1"], semfield1)
                merge_unique(entry["semfield2"], semfield2)
                merge_unique(entry["pragmatics"], pragmatics)
                merge_unique(entry["atoms"], atoms)

    return linkers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", required=True, help="Path to Коннекторы_правка.xlsx.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Output CSV path (default: {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def main():
    args = parse_args()
    linkers = collect_linkers(args.xlsx)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for entry in linkers:
            writer.writerow({
                "linker": entry["linker"],
                "semfield1": ";".join(entry["semfield1"]),
                "semfield2": ",".join(entry["semfield2"]),
                "pragmatics": ",".join(entry["pragmatics"]),
                "atoms": ",".join(entry["atoms"]),
            })

    print(f"Wrote {len(linkers)} linkers to {args.output}")


if __name__ == "__main__":
    main()
