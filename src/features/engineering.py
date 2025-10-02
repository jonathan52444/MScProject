"""
src/features/engineering.py
Generate modelling-ready features from ctgov_flat.parquet

Input  : data/interim/ctgov_flat.parquet
Output : data/processed/features_v5.parquet
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler 
import pathlib
from unidecode import unidecode 
import re             
from typing import Any
from pandas import Int8Dtype
from .helpers import (
    list_len,
    parse_date_any,
    count_any,
    complexity_bucket,
    n_elig_criteria,
    count_arm_groups,
    browse_to_ta,
    age_to_years,
    count_sites,
    count_unique_countries,
)
from .scoring import add_complexity_score, add_attractiveness_score


# Pipeline --------------------------------------------------
def build_features(flat_path: pathlib.Path) -> pd.DataFrame:
    """Return a modelling-ready feature table."""
    df = pq.read_table(flat_path).to_pandas()
    df.columns = df.columns.str.strip()

    # Ensure NCT identifier is present as `nct_id`
    id_variants = {
        "nct_id", "NCT ID", "NCT_ID", "nctId", "NCTId", "NCT Number", "NCTNumber"
    }
    found = next(
        (
            c for c in df.columns
            if (c in id_variants)
            or (c.lower().replace(" ", "").replace("_", "") in {"nctid", "nctnumber"})
        ),
        None,
    )

    if found:
        df.rename(columns={found: "nct_id"}, inplace=True)
        df["nct_id"] = df["nct_id"].astype(str).str.strip().str.upper()
    else:
        # best-effort extraction from any URL-like column if present
        url_like = next((c for c in df.columns if "url" in c.lower()), None)
        if url_like:
            df["nct_id"] = (
                df[url_like]
                .astype(str)
                .str.extract(r"(NCT\d{8})", expand=False)
                .str.upper()
            )
        else:
            df["nct_id"] = pd.NA


    # Basic Cleanup
    df["# patients"] = pd.to_numeric(df["# patients"], errors="coerce")
    df["Primary Completion Type"] = (
        df["Primary Completion Type"].astype(str).str.strip().str.upper()
    )
    df["Overall status"] = df["Overall status"].astype(str).str.strip().str.upper()

    # ------------------------------------------------------------------
    # Location-based counts  →  site_n   |   country_n
    # Extract number of sites
    df["site_n"] = df["# sites"].apply(count_sites)

    # Extract number of unique countries
    df["country_n"] = df["# sites"].apply(count_unique_countries)
    # dates
    df["start_date"] = df["Study Start Date"].apply(parse_date_any)
    df["complete_date"] = df["Study Full Completion Date"].apply(parse_date_any) # Fixed: Changed from Primary Completion to Full Completion
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

    df["therapeutic_area"] = df["condition_top"].apply(browse_to_ta)    

    df["intervention_type"] = (
        df["mode of administration (ex. NBE, NCE, iv vs pill)"]
        .str.split("|")
        .str[0]
    )

    # outcomes / endpoints  (use count_any not list_len)
    df["primary_out_n"]   = df["primary_outcomes"].apply(count_any).fillna(0)
    df["secondary_out_n"] = df["secondary_outcomes"].apply(count_any).fillna(0)
    df["other_out_n"]     = df["other_outcomes"].apply(count_any).fillna(0)

    df["assessments_n"] = (
        df["primary_out_n"] + df["secondary_out_n"] + df["other_out_n"]
    )

    df["assessments_complexity"] = df["assessments_n"].apply(complexity_bucket)
    # df.drop(columns=["primary_out_n", "secondary_out_n", "other_out_n"], inplace=True)

    # placeholders (may be absent in XML flatten)
    for col in ["screen fail rate", "evaluability / drop out rate", "safety events"]:
        if col not in df.columns:
            df[col] = np.nan

    # trial arms
    arm_col = "Number of arms"
    df["num_arms"] = df[arm_col].apply(count_arm_groups).fillna(1.0)

    # ── Masking / placebo ────────────────────────────────────────────────
    if "mask level" in df.columns:
        level_map = {
            "none":        "None",
            "open label":  "None",
            "single":      "Single",
            "double":      "Double",
            "triple":      "Triple",
            "quadruple":   "Quadruple",
        }

        # extract first match of any keyword above
        df["masking_level"] = (
            df["mask level"]
            .astype(str)
            .str.extract(
                r"(none|open label|single|double|triple|quadruple)",
                flags=re.I,
                expand=False,
            )
            .str.lower()
            .map(level_map)
            .fillna("Other")
        )
    else:
        df["masking_level"] = "Unknown"

    df["masking_level"] = pd.Categorical(
        df["masking_level"],
        categories=["None", "Single", "Double", "Triple", "Quadruple", "Other", "Unknown"],
        ordered=True,
    )
    mask_levels_with_blinding = ["Single", "Double", "Triple", "Quadruple"]

    df["masking_flag"] = (
        df["masking_level"].isin(mask_levels_with_blinding)
        & df["masking_level"].notna()
    ).astype(int)

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


    # Cohort design
    df["cohort_design"] = (
        df["cohorts (sequential or parallel)"]
        .astype(str)
        .str.lower()
        .map(lambda s: "Sequential" if "seq" in s
             else "Parallel" if "par" in s
             else "Other")
    )
    df["cohort_design"] = pd.Categorical(
        df["cohort_design"], categories=["Parallel", "Sequential", "Other"]
    )

    # Safety cuts / DMCs
    yes_pat = r"true"
    df["safety_cuts_flag"] = (
    df["safety cuts, DMCs"]
    .astype(str)
    .str.contains(yes_pat, case=False, na=False)
    .astype(int)
)

    # Study type
    df["study_type"] = (
        df["study type"].astype(str).str.strip().str.title().replace("", np.nan)
    )
    df["study_type"] = pd.Categorical(df["study_type"])

    # Age windows
    from .helpers import age_to_years                # NEW IMPORT
    df["age_min"] = df["minimum age"].apply(age_to_years)
    df["age_max"] = df["maximum age"].apply(age_to_years)

    # 1) treat 0 as “not specified”
    df.loc[df["age_max"] == 0, "age_max"] = np.nan
    df.loc[df["age_min"] == 0, "age_min"] = np.nan

    # 2) swap obvious entry mistakes (min > max)
    swap_mask = df["age_min"].notna() & df["age_max"].notna() & (df["age_min"] > df["age_max"])
    df.loc[swap_mask, ["age_min", "age_max"]] = (
        df.loc[swap_mask, ["age_max", "age_min"]].values
    )

    # 3) finally compute the range
    df["age_range"] = df["age_max"] - df["age_min"]


    # ── Adult / paediatric population ─────────────────────────────────────────
    # 1) Text-based rules
    pop_col = df["population - adults vs peds"].astype(str).str.lower()

    # keyword flags (vectorised, fast)
    is_adult_kw  = pop_col.str.contains(r"\badult\b",                na=False)
    is_child_kw  = pop_col.str.contains(r"\b(?:child|p(?:a|e)diat|peds?)\b", na=False)
    is_mixed_kw  = (
        pop_col.str.contains(r"\b(?:both|mixed)\b", na=False) |
        (is_adult_kw & is_child_kw)                 # mentions both groups
    )

    df["population_class"] = np.select(
        [is_mixed_kw, is_adult_kw, is_child_kw],
        ["Mixed",     "Adult",     "Pediatric"],
        default="Unknown",
    )

    # 2) Age-based fallback (only for remaining “Unknown” rows) ---------------
    need_fallback = df["population_class"] == "Unknown"

    age_min = df["age_min"]
    age_max = df["age_max"]

    df.loc[
        need_fallback & age_min.notna() & age_max.notna()
        & (age_min < 18) & (age_max >= 18),
        "population_class"
    ] = "Mixed"

    df.loc[
        need_fallback & age_min.notna() & (age_min >= 18),
        "population_class"
    ] = "Adult"

    df.loc[
        need_fallback & age_max.notna() & (age_max < 18),
        "population_class"
    ] = "Pediatric"

    # 3) Categorical dtype with explicit order --------------------------------
    df["population_class"] = pd.Categorical(
        df["population_class"],
        categories=["Adult", "Pediatric", "Mixed", "Unknown"]
    )


    # Randomisation
    alloc_map = {
        "RANDOMISED":       "Randomized",
        "RANDOMIZED":       "Randomized",
        "NON-RANDOMISED":   "Non-Randomized",
        "NON-RANDOMIZED":   "Non-Randomized",
    }
    df["allocation"] = (
        df["Allocation (Randomised / Non-randomised)"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(alloc_map)
        .fillna("Unknown")
    )
    df["randomized_flag"] = (df["allocation"] == "Randomized").astype(int)

    # FDA oversight flags
    yes_pat = r"yes|true|1"
    df["fda_drug_flag"] = (
        df["FDA-regulated drug"]
        .astype(str)
        .str.contains(yes_pat, case=False, na=False)
        .astype(int)
    )
    df["fda_device_flag"] = (
        df["FDA-regulated device"]
        .astype(str)
        .str.contains(yes_pat, case=False, na=False)
        .astype(int)
    )

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

    # Complexity Score (BCG Paper)
    MODEL_PKL = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "complexity_score_artifacts"
        / "complexity_score_model.pkl"
    )
    df = add_complexity_score(df, MODEL_PKL)

    #  Attractivness SCore 
    ATTR_PKL = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "attractiveness_score_artifacts"
        / "attractiveness_score_model.pkl"
    )
    df = add_attractiveness_score(df, ATTR_PKL)

    binary_cols = [
    "fda_drug_flag", "fda_device_flag",
    "placebo_flag", "randomized_flag",
    "safety_cuts_flag",
]
    for col in binary_cols:
        # ensure 0/1 then cast to an ordered categorical
        df[col] = pd.Categorical(
            df[col].fillna(0).astype(int), categories=[0, 1], ordered=True
        )
    df[["age_min", "age_max", "age_range"]] = df[["age_min", "age_max", "age_range"]].apply(
    pd.to_numeric, errors="coerce"
)
    return df.reset_index(drop=True)