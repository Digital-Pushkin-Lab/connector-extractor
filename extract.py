#!/usr/bin/env python3
"""Extract Russian linkers and/or introductory words from text.

Uses stanza dependency parsing plus a rule-based scorer to find and score
candidate spans matching a dictionary of known linker/introductory-word
expressions (ported from `linkers (1).ipynb` and
`connectors within linkers.ipynb`).

Linkers and introductory words are always matched in independent passes,
each with its own greedy non-overlapping match state -- exactly as in the
two source notebooks. This matters because the two word lists overlap and
otherwise diverge: a longer match in one list (e.g. the introductory phrase
"конечно же") would otherwise be able to consume tokens that a shorter,
unrelated entry in the other list (e.g. the linker "же") should be free to
match on its own.

Examples:
    # Analyze one sentence for linkers only, print JSON to stdout
    python extract.py --mode linkers --text "Более того, к Швеции отошли города Ивангород и Копорье."

    # Analyze a text file for introductory words only
    python extract.py --mode intro --input-file article.txt

    # Batch-analyze a CSV column for both linkers and introductory words
    python extract.py --mode both --input-csv texts.csv --text-column text --output results.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import stanza
from tqdm import tqdm

from patterns import build_patterns_from_csv
from pipeline import (
    DEFAULT_THRESHOLD,
    analyze_parsed,
    combine_stats,
    parse_sentences,
    summarize_intro,
    summarize_linkers,
)
from rules import build_default_checker

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_LINKERS_CSV = DATA_DIR / "linkers.csv"
DEFAULT_INTRO_CSV = DATA_DIR / "intro_words.csv"


def load_patterns(mode: str, linkers_csv: str, intro_csv: str) -> dict:
    """Return {"linker": [...]} and/or {"intro": [...]} pattern lists,
    loaded independently -- they are never merged into a shared list."""
    patterns_by_type = {}
    if mode in ("linkers", "both"):
        patterns_by_type["linker"] = build_patterns_from_csv(linkers_csv)
    if mode in ("intro", "both"):
        patterns_by_type["intro"] = build_patterns_from_csv(intro_csv)
    return patterns_by_type


def compute_stats(mode: str, parsed_sentences, patterns_by_type: dict, checker, threshold: float, word_count: int) -> dict:
    linker_stats = None
    intro_stats = None

    if "linker" in patterns_by_type:
        linker_results = analyze_parsed(parsed_sentences, checker, patterns_by_type["linker"])
        linker_stats = summarize_linkers(linker_results, threshold, word_count)

    if "intro" in patterns_by_type:
        intro_results = analyze_parsed(parsed_sentences, checker, patterns_by_type["intro"])
        intro_stats = summarize_intro(intro_results, threshold, word_count)

    if mode == "linkers":
        return linker_stats
    if mode == "intro":
        return intro_stats
    return combine_stats(linker_stats, intro_stats, word_count)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["linkers", "intro", "both"], required=True,
        help="Which kind of entity to extract from the text.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Raw text to analyze.")
    input_group.add_argument("--input-file", help="Path to a plain text file to analyze.")
    input_group.add_argument("--input-csv", help="Path to a CSV file with a text column to analyze in batch.")

    parser.add_argument(
        "--text-column", default="text",
        help="Column holding the text in --input-csv (default: 'text').",
    )
    parser.add_argument(
        "--output",
        help="Output path. Required with --input-csv (writes a CSV). "
             "With --text/--input-file, JSON is printed to stdout if omitted.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum scored probability to keep a match (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument("--linkers-csv", default=str(DEFAULT_LINKERS_CSV), help="Linkers word list CSV.")
    parser.add_argument("--intro-csv", default=str(DEFAULT_INTRO_CSV), help="Introductory words word list CSV.")

    return parser.parse_args()


def run_single(text, mode, patterns_by_type, nlp, checker, threshold, output_path):
    parsed_sentences, word_count = parse_sentences(text, nlp)
    stats = compute_stats(mode, parsed_sentences, patterns_by_type, checker, threshold, word_count)
    output_json = json.dumps(stats, ensure_ascii=False, indent=2)

    if output_path:
        Path(output_path).write_text(output_json, encoding="utf-8")
        print(f"Wrote results to {output_path}")
    else:
        print(output_json)


def run_batch(input_csv, text_column, output_path, mode, patterns_by_type, nlp, checker, threshold):
    df = pd.read_csv(input_csv)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        parsed_sentences, word_count = parse_sentences(text, nlp)
        rows.append(compute_stats(mode, parsed_sentences, patterns_by_type, checker, threshold, word_count))

    stats_df = pd.DataFrame(rows)
    result_df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)
    result_df.to_csv(output_path, index=False)
    print(f"Wrote {len(result_df)} rows to {output_path}")


def main():
    args = parse_args()

    if args.input_csv and not args.output:
        sys.exit("--output is required when using --input-csv")

    patterns_by_type = load_patterns(args.mode, args.linkers_csv, args.intro_csv)

    print("Loading stanza pipeline (tokenize,pos,lemma,depparse)...", file=sys.stderr)
    nlp = stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse")
    checker = build_default_checker()

    if args.input_csv:
        run_batch(args.input_csv, args.text_column, args.output, args.mode, patterns_by_type, nlp, checker, args.threshold)
    else:
        text = args.text if args.text is not None else Path(args.input_file).read_text(encoding="utf-8")
        run_single(text, args.mode, patterns_by_type, nlp, checker, args.threshold, args.output)


if __name__ == "__main__":
    main()
