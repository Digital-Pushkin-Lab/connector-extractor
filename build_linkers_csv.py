#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build data/linkers.csv from Коннекторы_правка.xlsx.

Rerun this script whenever the spreadsheet changes:

    python3 build_linkers_csv.py --xlsx /path/to/Коннекторы_правка.xlsx

Source sheets:
  * "Исходные" - original connector list with revised characteristics
  * "Доп"      - new connectors added during the revision

Only the "Союз" column is used (the single "linker" column
`patterns.build_patterns_from_csv` reads). A row is dropped if its "Союз"
cell is prefixed with "DEL" or filled orange (FFFFC000) - these were marked
for removal by the reviewer. "…" (ellipsis, U+2026) in a connector is
normalized to the three-dot "..." marker used to detect discontinuous
connectors (e.g. "если ... то"). Duplicate connectors (same spelling
appearing more than once, across or within sheets) are deduplicated while
preserving first-seen order.
"""

import argparse
import csv
from pathlib import Path

import openpyxl

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "linkers.csv"

SHEETS = ["Исходные", "Доп"]
DEL_FILL_RGB = "FFFFC000"


def is_deleted(cell) -> bool:
    value = cell.value
    if value and str(value).strip().upper().startswith("DEL"):
        return True
    fill = cell.fill
    if fill and fill.fgColor and fill.fgColor.rgb == DEL_FILL_RGB:
        return True
    return False


def normalize_key(raw: str) -> str:
    return str(raw).strip().replace("…", "...")


def collect_linkers(xlsx_path: str) -> list:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    linkers = []
    seen = set()

    for sheet_name in SHEETS:
        ws = wb[sheet_name]
        for row in ws.iter_rows(min_row=2):
            key_cell = row[0]
            if key_cell.value is None or not str(key_cell.value).strip():
                continue
            if is_deleted(key_cell):
                continue

            key = normalize_key(key_cell.value)
            if key not in seen:
                seen.add(key)
                linkers.append(key)

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
        writer = csv.writer(f)
        writer.writerow(["linker"])
        for linker in linkers:
            writer.writerow([linker])

    print(f"Wrote {len(linkers)} linkers to {args.output}")


if __name__ == "__main__":
    main()
