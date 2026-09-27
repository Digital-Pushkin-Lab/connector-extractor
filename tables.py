"""Tabular I/O helpers shared by the CLI (`extract.py`) and the evaluation
harness (`evaluate.py`).

`read_table` reads a `.csv` (comma-separated), a `.tsv`/`.txt`
(tab-separated) or an `.xlsx` spreadsheet into a DataFrame, and
`write_table` writes one back by the same extension rules, so callers don't
each reinvent extension sniffing.
"""

from pathlib import Path

import pandas as pd


def read_table(path: str, *, as_str: bool = False) -> pd.DataFrame:
    """Load `path` into a DataFrame.

    - `.csv`               -> comma-separated
    - `.tsv` / `.txt` / *  -> tab-separated
    - `.xlsx`              -> first sheet

    With `as_str=True` every cell is read as a string and blank cells become
    "" instead of NaN -- the shape the benchmark evaluation expects.
    """
    ext = Path(path).suffix.lower()
    read_kwargs = {"dtype": str, "keep_default_na": False} if as_str else {}

    if ext == ".xlsx":
        return pd.read_excel(path, **read_kwargs)

    sep = "," if ext == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, **read_kwargs)


def write_table(df: pd.DataFrame, path: str) -> None:
    """Write `df` to `path` without the index, choosing the format by
    extension exactly as `read_table` does."""
    ext = Path(path).suffix.lower()

    if ext == ".xlsx":
        df.to_excel(path, index=False)
        return

    sep = "," if ext == ".csv" else "\t"
    df.to_csv(path, sep=sep, index=False)


def pick_text_column(df: pd.DataFrame, column: str = None) -> str:
    """Name of the column holding the texts.

    With `column` given, that column is used (ValueError if it is missing).
    Otherwise the column with the longest average cell length wins; empty
    cells count as length 0, so a mostly-empty column doesn't win on a few
    long values."""
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Columns: {list(df.columns)}")
        return column
    if len(df.columns) == 0:
        raise ValueError("The table has no columns.")

    def mean_length(col) -> float:
        return df[col].fillna("").astype(str).str.len().mean() if len(df) else 0.0

    return max(df.columns, key=mean_length)
