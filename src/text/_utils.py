# src/text/_utils.py

from __future__ import annotations
import re
import pandas as pd

def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cand_low = {c.lower(): c for c in candidates}
    for c in df.columns:
        if c.lower() in cand_low:
            return c
    # fallback: fuzzy contains "brief" and "summary"
    for c in df.columns:
        s = c.strip().lower()
        if "brief" in s and "summary" in s:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)[:20]}…")
