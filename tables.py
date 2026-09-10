"""Tabular I/O helpers shared by the CLI (`extract.py`) and the evaluation
harness (`evaluate.py`).

One entry point, `read_table`, reads a `.csv` (comma-separated), a
`.tsv`/`.txt` (tab-separated) or an `.xlsx` spreadsheet into a DataFrame,
so callers don't each reinvent extension sniffing.
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
