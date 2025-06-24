"""
Generate modelling-ready features from ctgov_flat.parquet

Input  : data/interim/ctgov_flat.parquet
Output : data/processed/features_v3.parquet
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler  # noqa: F401  (kept for parity)

from .helpers import (
    list_len,
    parse_date_any,
    count_any,
    complexity_bucket,
    n_elig_criteria,
    count_arm_groups,
    browse_to_ta,
)
from .scoring import add_complexity_score, add_attractiveness_score



# ---------- Pipeline --------------------------------------------------


def build_features(flat_path: pathlib.Path) -> pd.DataFrame:
    """Return a modelling-ready feature table."""
    df = pq.read_table(flat_path).to_pandas()
    df.columns = df.columns.str.strip()

    # Basic Cleanup
    df["# patients"] = pd.to_numeric(df["# patients"], errors="coerce")
    df["Primary Completion Type"] = (
        df["Primary Completion Type"].astype(str).str.strip().str.upper()
    )
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
    df["phase"] = pd.Categorical(
        phase_raw.fillna("Unknown"), categories=phase_cat, ordered=True
    )

    # simple engineered cols
    df["sponsor_class"] = np.where(
        df["sponsorship type"].str.contains(
            "univer|hospital|nih|institute", case=False, na=False
        ),
        "Academic",
        "Industry",
    )
    df["condition_top"] = df["indication/disease area"].str.split("|").str[0]

    browse_col = (
        "rare, non-rare (established disease area and clear diagnosis criteria)"
    )
    df["therapeutic_area"] = df[browse_col].apply(browse_to_ta)

    df["intervention_type"] = (
        df["mode of administration (ex. NBE, NCE, iv vs pill)"]
        .str.split("|")
        .str[0]
    )

    site_col = "# sites"
    df["site_n"] = df[site_col].apply(count_any)

    # Fallback for removed “geographies” column
    geo_col = (
        "# geographies - global, regions involved, single country"
        " (consider start-up timings by country)"
    )
    if geo_col in df.columns:
        df["country_n"] = df[geo_col].apply(count_any)
    else:
        df["country_n"] = df["site_n"]

    # outcomes / endpoints
    df["primary_out_n"] = df["primary_outcomes"].apply(list_len).fillna(0)
    df["secondary_out_n"] = df["secondary_outcomes"].apply(list_len).fillna(0)
    df["other_out_n"] = df["other_outcomes"].apply(list_len).fillna(0)

    df["assessments_n"] = (
        df["primary_out_n"] + df["secondary_out_n"] + df["other_out_n"]
    )
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
        df["masking_flag"] = (
            df["mask level"]
            .astype(str)
            .str.contains("mask", case=False, na=False)
            .astype(int)
        )

    df["placebo_flag"] = np.nan
    if "placebo included" in df.columns:
        df["placebo_flag"] = (
            df["placebo included"]
            .astype(str)
            .str.contains("yes|true|1", case=False, na=False)
            .astype(int)
        )

    # Patient-facing “active treatment probability”
    df["active_prob"] = 1 - df["placebo_flag"] / df["num_arms"].replace(0, np.nan)
    df["active_prob"] = df["active_prob"].fillna(1.0)

    # eligibility length
    elig_col = (
        "Eligibility Criteria: The stringency and number of eligibility criteria for"
        " participants"
    )
    elig_col = (
        "Eligibility Criteria: The stringency and number of eligibility criteria for"
        " participants"
    )
    df["elig_crit_n"] = df.get(elig_col, "").apply(n_elig_criteria)

    # novelty: leakage-free rolling window
    df = df.sort_values(["condition_top", "start_date"])

    df["freq_in_window"] = (
        df.groupby("condition_top")["start_date"]
        .transform(
            lambda s: pd.Series(1, index=s).rolling("1825D").sum().to_numpy()
        )
    )

    df["novelty_score"] = 1 / df["freq_in_window"]

    # patients / site
    df["patients_per_site"] = df["# patients"] / df["site_n"].replace({0: np.nan})

    # global vs single-country
    df["global_trial"] = np.where(df["country_n"] > 1, "Global", "Single")
    df["global_trial"] = pd.Categorical(
        df["global_trial"], categories=["Single", "Global"]
    )

    # post-feature filter
    df = df[df["duration_days"].between(30, 3650)]
    df = df[df["# patients"].notna() & (df["# patients"] >= 10)]
    df["phase"] = df["phase"].fillna("Unknown")

    #  ---------- Add Complexity Score (From BCG paper) ----------
    MODEL_PKL = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "complexity_score_artifacts"
        / "complexity_score_model.pkl"
    )
    df = add_complexity_score(df, MODEL_PKL)

    #  ---------- Add Attractiveness Score ----------
    ATTR_PKL = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "attractiveness_score_artifacts"
        / "attractiveness_score_model.pkl"
    )
    df = add_attractiveness_score(df, ATTR_PKL)

    return df.reset_index(drop=True)
