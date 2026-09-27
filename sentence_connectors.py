#!/usr/bin/env python3
"""Split each text of a table into sentences and list the connectors found
in every sentence.

Input: a table (.csv / .tsv / .xlsx) with one text per row in the `text`
column. Output: a table with one row per sentence:

    text_id     -- row number of the text in the input table (1 = first text)
    sentence    -- the sentence (razdel, within a paragraph; see
                   `pipeline.parse_sentences`)
    connectors  -- connectors found in the sentence, in order of appearance,
                   separated by "; " (a discontinuous connector is listed
                   once, e.g. "если...то")

Sentences and connectors are exactly those behind the per-sentence
statistics of extract.py (same threshold, same sentence splitting).

Example:
    python sentence_connectors.py --input texts.xlsx --output sentences.csv
"""

import argparse
import sys

import pandas as pd
from tqdm import tqdm

from pipeline import (
    DEFAULT_LINKERS_CSV,
    DEFAULT_THRESHOLD,
    analyze_parsed,
    build_engine,
    parse_sentences,
    summarize_linkers,
)
from tables import read_table, write_table

CONNECTOR_SEP = "; "


def sentence_rows(text_id: int, text: str, engine, threshold: float) -> list:
    """One {"text_id", "sentence", "connectors"} dict per sentence of `text`."""
    parsed_sentences, word_count = parse_sentences(text, engine.nlp)
    results = analyze_parsed(parsed_sentences, engine.checker, engine.patterns_by_type["linker"])
    by_sentence = summarize_linkers(results, threshold, word_count)["linkers_by_appearance"]

    return [
        {
            "text_id": text_id,
            "sentence": sentence_text,
            "connectors": CONNECTOR_SEP.join(connectors),
        }
        for (sentence_text, *_), connectors in zip(parsed_sentences, by_sentence)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input table (.csv / .tsv / .xlsx).")
    parser.add_argument("--output", required=True, help="Output table (.csv / .tsv / .xlsx).")
    parser.add_argument("--text-column", default="text", help="Column holding the text (default: 'text').")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum scored probability to keep a match (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument("--linkers-csv", default=str(DEFAULT_LINKERS_CSV), help="Linkers word list CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = read_table(args.input)
    if args.text_column not in df.columns:
        sys.exit(f"Column '{args.text_column}' not found in {args.input}. Columns: {list(df.columns)}")

    print("Loading stanza pipeline (tokenize,pos,lemma,depparse)...", file=sys.stderr)
    engine = build_engine(args.linkers_csv)

    rows = []
    texts = df[args.text_column].tolist()
    for text_id, text in enumerate(tqdm(texts, desc="Analyzing"), start=1):
        text = str(text) if pd.notna(text) else ""
        rows.extend(sentence_rows(text_id, text, engine, args.threshold))

    out = pd.DataFrame(rows, columns=["text_id", "sentence", "connectors"])
    write_table(out, args.output)
    print(f"Wrote {len(out)} sentences from {len(texts)} texts to {args.output}")


if __name__ == "__main__":
    main()
