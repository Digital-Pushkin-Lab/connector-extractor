#!/usr/bin/env python3
"""Extract Russian linkers from text.

Uses stanza dependency parsing plus a rule-based scorer to find and score
candidate spans matching a dictionary of known linker expressions
(`data/linkers.csv`; ported from `linkers (1).ipynb` and
`connectors within linkers.ipynb`).

Examples:
    # Analyze one sentence, print JSON to stdout
    python extract.py --text "Более того, к Швеции отошли города Ивангород и Копорье."

    # Analyze a text file
    python extract.py --input-file article.txt

    # Batch-analyze a CSV column
    python extract.py --input-csv texts.csv --text-column text --output results.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline import (
    DEFAULT_LINKERS_CSV,
    DEFAULT_THRESHOLD,
    analyze_parsed,
    build_engine,
    load_patterns,  # noqa: F401  (re-exported for callers that still `from extract import load_patterns`)
    parse_sentences,
    summarize_linkers,
)
from tables import pick_text_column, read_table, write_table

# Табличные расширения: такой --output для одиночного текста пишется
# таблицей (одна строка), а не JSON; такой --input-file читается как
# таблица (пакетный режим, как --input-csv), а не как один текст.
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx"}


def compute_stats(parsed_sentences, patterns_by_type: dict, checker, threshold: float, word_count: int) -> dict:
    linker_results = analyze_parsed(parsed_sentences, checker, patterns_by_type["linker"])
    return summarize_linkers(linker_results, threshold, word_count, parsed_sentences)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--text", help="Raw text to analyze.", default=None)
    input_group.add_argument(
        "--input-file", default=None,
        help="Path to a plain text file to analyze as one text. A .csv/.tsv/.xlsx "
             "file is read as a table instead (same as --input-csv).",
    )
    input_group.add_argument("--input-csv", help="Path to a table (.csv/.tsv/.xlsx) with a text column to analyze in batch.", default=None)

    parser.add_argument(
        "--text-column", default=None,
        help="Column holding the text in --input-csv. Only this column is "
             "analyzed. Default: the column with the longest average cell length.",
    )
    parser.add_argument(
        "--output",
        help="Output path. Required with --input-csv (writes a CSV). "
             "With --text/--input-file: a .csv/.tsv/.xlsx path writes a one-row "
             "table, any other path writes JSON; JSON is printed to stdout if omitted.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum scored probability to keep a match (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument("--linkers-csv", default=str(DEFAULT_LINKERS_CSV), help="Linkers word list CSV.")

    return parser.parse_args()


def get_ext(file_name):
    """Return a file's extension with the leading dot, e.g. '.csv'."""
    return Path(file_name).suffix


def run_single(text, patterns_by_type, nlp, checker, threshold, output_path):
    parsed_sentences, word_count = parse_sentences(text, nlp)
    stats = compute_stats(parsed_sentences, patterns_by_type, checker, threshold, word_count)

    if output_path and get_ext(output_path).lower() in TABLE_EXTENSIONS:
        # те же колонки, что и у строки пакетного режима: текст + статистика
        write_table(pd.DataFrame([{"text": text, **stats}]), output_path)
        print(f"Wrote results to {output_path}")
        return

    output_json = json.dumps(stats, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(output_json, encoding="utf-8")
        print(f"Wrote results to {output_path}")
    else:
        print(output_json)


def run_batch(input_csv, text_column, output_path, patterns_by_type, nlp, checker, threshold):
    # Reads .csv (comma) or .tsv/.txt/.xlsx (tab / sheet); see tables.read_table.
    df = read_table(input_csv)

    rows = []

    try:
        text_column = pick_text_column(df, text_column)
    except ValueError as e:
        sys.exit(f"{input_csv}: {e}")
    print(f"Reading texts from column '{text_column}'", file=sys.stderr)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        parsed_sentences, word_count = parse_sentences(text, nlp)
        rows.append(compute_stats(parsed_sentences, patterns_by_type, checker, threshold, word_count))

    stats_df = pd.DataFrame(rows)
    result_df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)

    # Comma only for a real .csv; everything else (e.g. .tsv) gets tabs.
    sep = "," if get_ext(output_path) == ".csv" else "\t"
    result_df.to_csv(output_path, index=False, sep=sep)

    print(f"Wrote {len(result_df)} rows to {output_path}")


def table_input(args):
    """Path of the table to analyze in batch mode, or None for single-text
    mode: --input-csv, or an --input-file with a table extension."""
    if args.input_csv:
        return args.input_csv
    if args.input_file and get_ext(args.input_file).lower() in TABLE_EXTENSIONS:
        return args.input_file
    return None


def run(args, patterns_by_type, nlp, checker):
    """Dispatch to batch or single-text mode based on `args`."""
    table = table_input(args)
    if table:
        run_batch(table, args.text_column, args.output, patterns_by_type, nlp, checker, args.threshold)
    else:
        text = args.text if args.text is not None else Path(args.input_file).read_text(encoding="utf-8")
        run_single(text, patterns_by_type, nlp, checker, args.threshold, args.output)


def main():
    args = parse_args()

    table = table_input(args)
    if table and not args.output:
        sys.exit(f"--output is required when analyzing a table ({table})")
    if not table and args.text_column:
        print("Warning: --text-column is ignored: the input is a single text, not a table.", file=sys.stderr)

    print("Loading stanza pipeline (tokenize,pos,lemma,depparse)...", file=sys.stderr)
    engine = build_engine(args.linkers_csv)

    run(args, engine.patterns_by_type, engine.nlp, engine.checker)


if __name__ == "__main__":
    main()
