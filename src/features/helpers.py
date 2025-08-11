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
from functools import lru_cache
from typing import Optional


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


# # Therapeutic-area helpers
# TA_MAP: dict[str, str] = {
#     "Neoplasms": "Oncology",
#     "Cardiovascular Diseases": "Cardio-Metabolic",
#     "Nervous System Diseases": "CNS / Neurology",
#     "Immune System Diseases": "Immunology",
#     "Respiratory Tract Diseases": "Respiratory",
#     "Musculoskeletal Diseases": "Musculoskeletal",
#     "Infectious Diseases": "Infectious",
# }

# ---------------- master pattern table (ordered; first match wins) -------
_PATTERNS = [
    # Oncology
    (r"\b(cancer|tumou?r|carcinom|sarcom|lymphom|leukemi|myelom|neoplasm|oncolog|melanom|glioblastom)\b",
     "Oncology"),

    # Neurology / CNS
    (r"\b(alzheimer|parkinson|dementia|epileps|seizure|stroke|tbi|migraine|multiple sclerosis|amyotrophic lateral sclerosis|als|huntington|neuropath|meningit|neuro)\b",
     "Neurology"),

    # Cardiovascular
    (r"\b(cardio|cardiac|coronar|ischemi|myocard|angina|atrial|ventricular|heart|hypertens|aort|vascular)\b",
     "Cardiovascular"),

    # Respiratory
    (r"\b(asthma|copd|bronchi|pulmon|lung|respirat|cystic fibrosis|cf|influenza|covid|pneumon)\b",
     "Respiratory"),

    # Metabolic & endocrine
    (r"\b(diabet|obes|metabolic|lipid|cholesterol|dyslipid|hyperlipid|thyroid|hashimoto|graves|parathyroid|pituitar|endocrin)\b",
     "Metabolic & Endocrine"),

    # Gastro-Hepatology
    (r"\b(hepatitis|liver|hepat|cirrhos|steatohepat|nafld|nash|crohn|colitis|ibs|ibd|gastroenter)\b",
     "Gastro-Hepatology"),

    # Renal / Urology
    (r"\b(kidney|renal|nephro|ckd|esrd|glomerulo|urinar|bladder|prostat|urolog)\b",
     "Renal & Urology"),

    # Rheumatology
    (r"\b(arthritis|rheumat|lupus|sj[oö]gren|scleroderma|osteopor|ankylosing spondyl|fibromyalg|musculoskelet)\b",
     "Rheumatology"),

    # Dermatology
    (r"\b(psoriasis|eczema|dermatit|acne|rosacea|vitiligo|skin)\b",
     "Dermatology"),

    # Infectious disease
    (r"\b(hiv|aids|malaria|tubercul|tb\b|ebola|dengue|zika|chikung|cmv|hsv|herpes|hpv|hpylori|infectious)\b",
     "Infectious Disease"),

    # Obstetrics / Gynaecology
    (r"\b(pregnan|obstet|gynaec|gynec|endometrio|preeclamps|ivf|fertilit|uterin|ovarian)\b",
     "Ob-Gyn"),

    # Mental health / Psychiatry
    (r"\b(depress|anxi|schizophren|bipolar|psych|adhd|autism|asd|ptsd)\b",
     "Psychiatry"),

    # Ophthalmology
    (r"\b(retina|macular|glaucom|ocular|ophthal|uveitis|vision|eye)\b",
     "Ophthalmology"),

    # Haematology (non-oncology)
    (r"\b(haemoph|sickle|thalassem|anemi|coagulat|bleeding|platelet|haematolog)\b",
     "Haematology (non-onc)"),

    # Paediatrics catch-all (keep last)
    (r"\b(pediatr|paediatr|childhood)\b", "Paediatrics"),
]

_PAT_COMPILED = [(re.compile(p, re.I), ta) for p, ta in _PATTERNS]

@lru_cache(maxsize=8192)
def _match_ta(text: str) -> Optional[str]:
    for rgx, area in _PAT_COMPILED:
        if rgx.search(text):
            return area
    return None

def browse_to_ta(raw: str | None) -> str:
    """
    Map a free-text condition string to a therapeutic area.
    Returns "Other" if no pattern matches.
    """
    if not raw or not isinstance(raw, str):
        return "Other"

    txt = re.sub(r"[^a-z0-9\s]", " ", raw.lower())
    txt = re.sub(r"\s+", " ", txt).strip()
    return _match_ta(txt) or "Other"


_age_re  = re.compile(r"(\d+\.?\d*)")               # captures the number
_unit_re = re.compile(r"(year|month|week|day)s?", re.I)  # captures the unit


def age_to_years(x: Any) -> float:
    """
    Convert CT.gov age strings to years (float).

    Examples
    --------
    >>> age_to_years("18 Years")   -> 18
    >>> age_to_years("6 Months")   -> 0.5
    >>> age_to_years("21 Days")    -> 0.06
    >>> age_to_years("N/A")        -> np.nan
    """
    if not isinstance(x, str):
        return np.nan

    s = x.strip().lower()
    if not s or s.startswith("n/a"):
        return np.nan

    # ---------------- number -------------------------------------------------
    m_val = _age_re.search(s)
    if not m_val:
        return np.nan
    value = float(m_val.group(1))                # ▲ <── changed

    # ---------------- unit  --------------------------------------------------
    m_unit = _unit_re.search(s)
    unit = m_unit.group(1) if m_unit else "year"   # ▲ <── changed

    if unit.startswith("month"):
        return value / 12
    if unit.startswith("week"):
        return value / 52
    if unit.startswith("day"):
        return value / 365
    return value            # default = years


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
