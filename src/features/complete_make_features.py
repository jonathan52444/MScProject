#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined, self‑contained script that performs feature engineering for
ClinicalTrials.gov flattened data and saves a modelling‑ready feature table.

This file merges the original modular implementation:
    • src/features/helpers.py
    • src/features/scoring.py
    • src/features/engineering.py
    • src/features/make_features.py
…into a *single* stand‑alone module (`complete_make_features.py`).

No functional changes have been introduced—only the intra‑package
imports were removed because the relevant code now lives in this file.
Run from the project root with:

    python complete_make_features.py

All relative paths behave exactly as before (they resolve relative to the
project root).  The script will read:
    data/interim/ctgov_flat.parquet
…build features, and write:
    data/processed/features_v3.parquet

"""

from __future__ import annotations

# ── Standard library ────────────────────────────────────────────────────
import ast
import pathlib
import pickle
import re
from typing import Any

# ── Third‑party ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pyarrow as pa  # noqa: F401 (kept for parity with original code)
import pyarrow.parquet as pq
from dateutil import parser
from sklearn.preprocessing import StandardScaler  # noqa: F401 (kept for parity)

# ────────────────────────────────────────────────────────────────────────
# Helpers (originally src/features/helpers.py)
# ────────────────────────────────────────────────────────────────────────

TA_MAP: dict[str, str] = {
    "Neoplasms": "Oncology",
    "Cardiovascular Diseases": "Cardio-Metabolic",
    "Nervous System Diseases": "CNS / Neurology",
    "Immune System Diseases": "Immunology",
    "Respiratory Tract Diseases": "Respiratory",
    "Musculoskeletal Diseases": "Musculoskeletal",
    "Infectious Diseases": "Infectious",
}


def list_len(x: Any) -> int | float:
    """Length of list or pipe‑delimited string; else NaN."""
    if isinstance(x, list):
        return len(x)
    if isinstance(x, str):
        return len([v for v in x.split("|") if v])
    return np.nan


def parse_date_any(x: Any) -> pd.Timestamp | pd.NaT:
    """Parse CT.gov date strings robustly."""
    try:
        return pd.to_datetime(parser.parse(str(x), fuzzy=True, default=pd.Timestamp("1900-01-01")))
    except Exception:  # noqa: BLE001
        return pd.NaT


def count_any(x: Any) -> int | float:
    """Generic *how many?* that works for lists, arrays, strings, numerics."""
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
    """Return the number of discrete eligibility‑criteria items."""
    if not isinstance(text, str) or not text.strip():
        return np.nan

    bullet_lines = re.findall(r"^[ \t]*[•\*\-][ \t]+", text, flags=re.MULTILINE)
    if bullet_lines:
        return float(len(bullet_lines))

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return float(len(paragraphs))


def count_arm_groups(x: Any) -> float:
    """Return len(x) for list/array or stringified JSON list; else NaN."""
    if isinstance(x, (list, np.ndarray)):
        return float(len(x))
    if isinstance(x, str) and x.lstrip().startswith("["):
        try:
            return float(len(ast.literal_eval(x)))
        except Exception:  # noqa: BLE001
            return np.nan
    return np.nan


def browse_to_ta(branches: Any) -> str:
    """Map CT.gov *browse* branches → broad Therapeutic Area label."""
    if isinstance(branches, list) and branches:
        first = str(branches[0])
    elif isinstance(branches, str) and branches:
        first = branches.split("|", 1)[0]
    else:
        return "Other"
    return TA_MAP.get(first, "Other")

# ────────────────────────────────────────────────────────────────────────
# Scoring utilities (originally src/features/scoring.py)
# ────────────────────────────────────────────────────────────────────────

def _load_pickle(path: pathlib.Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def add_complexity_score(df: pd.DataFrame, model_path: pathlib.Path) -> pd.DataFrame:
    """Append a 0–100 *complexity_score_100* column using a pre‑fit Ridge pipeline."""
    obj = _load_pickle(model_path)

    X = obj["pipeline"].named_steps["pre"].transform(df)
    raw = (X * obj["beta"]).sum(axis=1)
    df["complexity_score_100"] = 100 * (raw - obj["Smin"]) / (obj["Smax"] - obj["Smin"])
    return df


def add_attractiveness_score(
    df: pd.DataFrame,
    model_path: pathlib.Path,
    out_col: str = "attractiveness_score_100",
) -> pd.DataFrame:
    """Append a 0–100 attractiveness score column."""
    obj = _load_pickle(model_path)

    if "design" in obj:  # our notebook saved this
        needed_cols = obj["design"]
    else:  # scikit‑learn ≥1.2
        needed_cols = list(obj["pipeline"].named_steps["pre"].feature_names_in_)

    X = obj["pipeline"].transform(df[needed_cols]).ravel()
    df[out_col] = 100 * (X - obj["Smin"]) / (obj["Smax"] - obj["Smin"])
    return df

# ────────────────────────────────────────────────────────────────────────
# Feature engineering (originally src/features/engineering.py)
# ────────────────────────────────────────────────────────────────────────

def build_features(flat_path: pathlib.Path) -> pd.DataFrame:
    """Return a modelling‑ready feature table from a flattened CT.gov dataset."""
    df = pq.read_table(flat_path).to_pandas()
    df.columns = df.columns.str.strip()

    # Basic Cleanup
    df["# patients"] = pd.to_numeric(df["# patients"], errors="coerce")
    df["Primary Completion Type"] = df["Primary Completion Type"].astype(str).str.strip().str.upper()
    df["Overall status"] = df["Overall status"].astype(str).str.strip().str.upper()

    # dates
    df["start_date"] = df["Study Start Date"].apply(parse_date_any)
    df["complete_date"] = df["Primary Completion Date"].apply(parse_date_any)
    df["duration_days"] = (df["complete_date"] - df["start_date"]).dt.days
    df["start_year"] = df["start_date"].dt.year

    # Initial filters (modern, finished studies)
    today = pd.Timestamp.today().normalize()
    df = df[df["start_date"] >= pd.Timestamp("2000-01-01")]
    df = df[df["complete_date"].notna() & (df["complete_date"] <= today)]
    df = df[df["Primary Completion Type"] == "ACTUAL"]
    df = df[df["Overall status"].isin(["COMPLETED", "TERMINATED"])]

    # phase column guard
    if "phase" not in df.columns and "Phase" in df.columns:
        df.rename(columns={"Phase": "phase"}, inplace=True)

    phase_map = {
        "PHASE1": "Phase 1",
        "PHASE2": "Phase 2",
        "PHASE3": "Phase 3",
        "PHASE4": "Phase 4",
        "PHASE1/PHASE2": "Phase 1_2",
        "PHASE2/PHASE3": "Phase 2_3",
        "EARLY PHASE 1": "Phase 1",
    }
    phase_raw = df["phase"].fillna("").str.upper().map(phase_map)
    phase_cat = [
        "Phase 1",
        "Phase 1_2",
        "Phase 2",
        "Phase 2_3",
        "Phase 3",
        "Phase 4",
        "Unknown",
    ]
    df["phase"] = pd.Categorical(phase_raw.fillna("Unknown"), categories=phase_cat, ordered=True)

    # simple engineered cols
    df["sponsor_class"] = np.where(
        df["sponsorship type"].str.contains("univer|hospital|nih|institute", case=False, na=False),
        "Academic",
        "Industry",
    )
    df["condition_top"] = df["indication/disease area"].str.split("|").str[0]

    browse_col = "rare, non-rare (established disease area and clear diagnosis criteria)"
    df["therapeutic_area"] = df[browse_col].apply(browse_to_ta)

    df["intervention_type"] = df["mode of administration (ex. NBE, NCE, iv vs pill)"].str.split("|").str[0]

    site_col = "# sites"
    df["site_n"] = df[site_col].apply(count_any)

    # Fallback for removed “geographies” column
    geo_col = (
        "# geographies - global, regions involved, single country" " (consider start-up timings by country)"
    )
    if geo_col in df.columns:
        df["country_n"] = df[geo_col].apply(count_any)
    else:
        df["country_n"] = df["site_n"]

    # outcomes / endpoints
    df["primary_out_n"] = df["primary_outcomes"].apply(count_any).fillna(0)
    df["secondary_out_n"] = df["secondary_outcomes"].apply(count_any).fillna(0)
    df["other_out_n"] = df["other_outcomes"].apply(count_any).fillna(0)

    df["assessments_n"] = df["primary_out_n"] + df["secondary_out_n"] + df["other_out_n"]

    df["assessments_complexity"] = df["assessments_n"].apply(complexity_bucket)
    df.drop(columns=["primary_out_n", "secondary_out_n", "other_out_n"], inplace=True)

    # placeholders (may be absent in XML flatten)
    for col in ["screen fail rate", "evaluability / drop out rate", "safety events"]:
        if col not in df.columns:
            df[col] = np.nan

    # trial arms
    arm_col = "Number of arms"
    df["num_arms"] = df[arm_col].apply(count_arm_groups).fillna(1.0)

    # masking / placebo flags
    df["masking_flag"] = np.nan
    if "mask level" in df.columns:
        df["masking_flag"] = df["mask level"].astype(str).str.contains("mask", case=False, na=False).astype(int)

    df["placebo_flag"] = np.nan
    if "placebo included" in df.columns:
        df["placebo_flag"] = df["placebo included"].astype(str).str.contains("yes|true|1", case=False, na=False).astype(int)

    # Patient‑facing “active treatment probability”
    df["active_prob"] = 1 - df["placebo_flag"] / df["num_arms"].replace(0, np.nan)
    df["active_prob"] = df["active_prob"].fillna(1.0)

    # eligibility length
    elig_col = (
        "Eligibility Criteria: The stringency and number of eligibility criteria for" " participants"
    )
    df["elig_crit_n"] = df.get(elig_col, "").apply(n_elig_criteria)

    # novelty: leakage‑free rolling window
    df = df.sort_values(["condition_top", "start_date"])
    df["freq_in_window"] = (
        df.groupby("condition_top")["start_date"].transform(lambda s: pd.Series(1, index=s).rolling("1825D").sum().to_numpy())
    )
    df["novelty_score"] = 1 / df["freq_in_window"]

    # patients / site
    df["patients_per_site"] = df["# patients"] / df["site_n"].replace({0: np.nan})

    # global vs single‑country
    df["global_trial"] = np.where(df["country_n"] > 1, "Global", "Single")
    df["global_trial"] = pd.Categorical(df["global_trial"], categories=["Single", "Global"])

    # post‑feature filter
    df = df[df["duration_days"].between(30, 3650)]
    df = df[df["# patients"].notna() & (df["# patients"] >= 10)]
    df["phase"] = df["phase"].fillna("Unknown")

    # ---------- Add Complexity Score ----------
    MODEL_PKL = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "complexity_score_artifacts"
        / "complexity_score_model.pkl"
    )
    df = add_complexity_score(df, MODEL_PKL)

    # ---------- Add Attractiveness Score ----------
    ATTR_PKL = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "attractiveness_score_artifacts"
        / "attractiveness_score_model.pkl"
    )
    df = add_attractiveness_score(df, ATTR_PKL)

    return df.reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────
# Command‑line entry point (originally src/features/make_features.py)
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    FLAT = ROOT / "data" / "interim" / "ctgov_flat.parquet"
    OUT = ROOT / "data" / "processed" / "features_v3.parquet"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Reading", FLAT)
    features = build_features(FLAT)

    # ── DEBUG ───────────────────────────────────────────────────────────
    print("\n─── DEBUG ──────────────────────────────────")
    print("Rows after feature engineering :", len(features))
    print(" • non-NaN duration_days       :", features["duration_days"].notna().sum())
    print(
        " • duration range (days)       :",
        int(features["duration_days"].min()),
        "/",
        int(features["duration_days"].max()),
    )
    print(" • phase unique values         :", features["phase"].unique()[:10])
    print(
        " • novelty min / max           :",
        features["novelty_score"].min(),
        "/",
        features["novelty_score"].max(),
    )
    print(
        " • complexity_score_100 min / max :",
        round(features["complexity_score_100"].min(), 2),
        "/",
        round(features["complexity_score_100"].max(), 2),
    )
    print(
        " • attractiveness_score_100 min / max :",
        round(features["attractiveness_score_100"].min(), 2),
        "/",
        round(features["attractiveness_score_100"].max(), 2),
    )

    used_num = [
        "# patients",
        "country_n",
        "site_n",
        "assessments_n",
        "start_year",
        "novelty_score",
        "complexity_score_100",
        "attractiveness_score_100",
        "patients_per_site",
        "num_arms",
        "masking_flag",
        "placebo_flag",
        "elig_crit_n",
    ]
    used_cat = [
        "phase",
        "sponsor_class",
        "condition_top",
        "therapeutic_area",
        "intervention_type",
        "assessments_complexity",
        "global_trial",
    ]

    print("Numeric features  ({}): {}".format(len(used_num), ", ".join(used_num)))
    print("Categoric features({}): {}".format(len(used_cat), ", ".join(used_cat)))

    print("──────────────────────────────────────────────\n")

    pq.write_table(pa.Table.from_pandas(features), OUT)
    print("Features saved →", OUT, "| shape:", features.shape)


if __name__ == "__main__":
    main()
