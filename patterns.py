"""Build matching patterns from the linker/introductory-word CSV word lists.

Ported from the `linker_list_to_patterns` helper shared by both source
notebooks.
"""

from typing import List, Tuple

import pandas as pd

Part = Tuple[str, ...]
Pattern = Tuple[Part, ...]


def normalize_ellipsis(expr: str) -> str:
    return expr.replace("…", "...")


def linker_list_to_patterns(raw_expressions: List[str]) -> List[Pattern]:
    """Turn raw expressions like 'если ... то' into part tuples, e.g.
    (('если',), ('то',)). Deduplicates while preserving first-seen order."""
    patterns: List[Pattern] = []

    for expr in raw_expressions:
        expr = normalize_ellipsis(str(expr).strip().lower())
        if not expr:
            continue

        parts = []
        for part in expr.split("..."):
            words = tuple(part.strip().split())
            if words:
                parts.append(words)

        if parts:
            patterns.append(tuple(parts))

    unique: List[Pattern] = []
    seen = set()
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique.append(pattern)

    return unique


def build_patterns_from_csv(csv_path: str, column: str = "linker") -> List[Pattern]:
    df = pd.read_csv(csv_path)
    return linker_list_to_patterns(df[column].dropna().tolist())
