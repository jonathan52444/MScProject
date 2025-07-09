from __future__ import annotations

"""
src/features/helpers.py

Utility functions used in feature engineering pipeline for ClinicalTrials.gov flat tables.
"""


import ast,json
import re
from typing import Any, List, Dict 

import numpy as np
import pandas as pd
from dateutil import parser


def list_len(x: Any) -> int | float:
    """Length of list or pipe-delimited string; else NaN."""
    if isinstance(x, list):
        return len(x)
    if isinstance(x, str):
        return len([v for v in x.split("|") if v])
    return np.nan


def parse_date_any(x: Any) -> pd.Timestamp | pd.NaT:
    """Parse CT.gov date strings robustly."""
    try:
        return pd.to_datetime(
            parser.parse(str(x), fuzzy=True, default=pd.Timestamp("1900-01-01"))
        )
    except Exception:  # noqa: BLE001
        return pd.NaT


def count_any(x: Any) -> int | float:
    """Generic “how many?” that works for lists, arrays, strings, numerics."""
    if isinstance(x, (list, tuple, set)):
        return len(x)
    if isinstance(x, np.ndarray):
        return len(x)
    if isinstance(x, str):
        parts = [t for t in x.split("|") if t.strip()]
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0])
        return len(parts)
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return np.nan


def complexity_bucket(n: float | int | None) -> str | float:
    if pd.isna(n):
        return np.nan
    return "Low" if n <= 3 else "Medium" if n <= 6 else "High"


def n_elig_criteria(text: str | float | int | None) -> float:
    """
    Return the number of discrete eligibility-criteria items.

    • Detects lines beginning with *, •, or - (common bullets on CT.gov)  
    • If no bullet markers are found, falls back to counting non-empty
      paragraphs separated by blank lines.
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan

    bullet_lines = re.findall(r"^[ \t]*[•\*\-][ \t]+", text, flags=re.MULTILINE)
    if bullet_lines:
        return float(len(bullet_lines))

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return float(len(paragraphs))


def count_arm_groups(x: Any) -> float:
    """
    Return len(x) when x is a list / ndarray, or when x is a stringified list.
    Otherwise np.nan.
    """
    if isinstance(x, (list, np.ndarray)):
        return float(len(x))
    if isinstance(x, str) and x.lstrip().startswith("["):
        try:
            return float(len(ast.literal_eval(x)))
        except Exception:
            return np.nan
    return np.nan


# Therapeutic-area helpers
TA_MAP: dict[str, str] = {
    "Neoplasms": "Oncology",
    "Cardiovascular Diseases": "Cardio-Metabolic",
    "Nervous System Diseases": "CNS / Neurology",
    "Immune System Diseases": "Immunology",
    "Respiratory Tract Diseases": "Respiratory",
    "Musculoskeletal Diseases": "Musculoskeletal",
    "Infectious Diseases": "Infectious",
}

def browse_to_ta(branches: Any) -> str:
    """
    branches = list OR '|'-joined string like
      ['Neoplasms', 'Digestive System Diseases']
    Returns broad TA label for the *first* branch; else 'Other'.
    """
    if isinstance(branches, list) and branches:
        first = str(branches[0])
    elif isinstance(branches, str) and branches:
        first = branches.split("|", 1)[0]
    else:
        return "Other"
    return TA_MAP.get(first, "Other")

_age_re = re.compile(r"(?P<num>\d+(?:\.\d+)?)[ ]*(?P<unit>year|yr|month|week|day)s?",
                     flags=re.I)


def age_to_years(x: Any) -> float:
    """
    Convert CT.gov “minimum/maximum age” strings to *years* (float).

    Examples
    --------
    >>> age_to_years("18 Years")      -> 18
    >>> age_to_years("6 Months")      -> 0.5
    >>> age_to_years("N/A")           -> np.nan
    """
    if not isinstance(x, str):
        return np.nan
    x = x.strip()
    if not x or x.upper().startswith("N/A"):
        return np.nan
    m = _age_re.search(x)
    if not m:
        return np.nan
    num = float(m["num"])
    unit = m["unit"].lower()
    if unit.startswith("month"):
        return num / 12
    if unit.startswith("week"):
        return num / 52
    if unit.startswith("day"):
        return num / 365
    return num  # years by default

def count_sites(locations):
    """
    Count the number of sites from the locations list or array.
    """
    if locations is None or (isinstance(locations, (list, np.ndarray)) and len(locations) == 0):
        return 0
    return len(locations)

def count_unique_countries(locations):
    """
    Count the number of unique countries from the locations list or array.
    """
    if locations is None or (isinstance(locations, (list, np.ndarray)) and len(locations) == 0):
        return 0
    countries = {loc.get('country', '') for loc in locations if loc.get('country')}
    return len(countries)
