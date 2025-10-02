# src/models/ablation_rf_global.py
"""
Ablation (global) — RandomForest leave-one-out over ALL features.

Purpose
-------
Run a single ablation experiment across the combined feature set
(RAW + DERIVED + ENGINEERED). Also computes three baselines:
  - All features
  - Raw features only
  - Raw + Derived
    
"""

from __future__ import annotations
import argparse, pathlib, time, warnings, re, sys
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ------------------------ Config (edit if needed or pass via CLI) ------------------------

ROOT   = pathlib.Path(__file__).resolve().parents[2]  # project root two levels up
DATA   = ROOT / "data" / "processed" / "features_v6.parquet"
SPLITS = ROOT / "data" / "splits"
TRAIN_IDS = SPLITS / "oot_filtered_train_nctids.txt"
TEST_IDS  = SPLITS / "oot_filtered_test_nctids.txt"

FIGDIR = ROOT / "reports" / "figures_209" / "filtered_data";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";                                 MODELD.mkdir(exist_ok=True)

TARGET = "duration_days"

# Seeds 
SEEDS = [42, 43, 44, 45, 46]

# Random‑Forest parameters 
RF_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 20,
    min_samples_leaf = 10,
    max_features     = 1.0,
    bootstrap        = True,
    max_samples      = 0.5,
    criterion        = "squared_error",
    n_jobs           = -1,
)

# Desired canonical feature groups (as in your ablation spec)
RAW = [
    "# patients", "start_date_ts", "age_min", "age_max",
    "phase", "study_type", "allocation",
]
DERIVED = [
    "site_n", "country_n", "primary_out_n", "secondary_out_n", "other_out_n",
    "num_arms", "masking_flag", "placebo_flag", "randomized_flag",
    "fda_drug_flag", "fda_device_flag", "safety_cuts_flag",
    "patients_per_site", "age_range",
    "sponsor_class", "intervention_type", "cohort_design",
    "masking_level", "assessments_n", "elig_crit_n", "freq_in_window",
    "global_trial", "therapeutic_area", "population_class",
]
ENGINEERED = [
    "active_prob", "novelty_score",
    "complexity_score_100", "attractiveness_score_100",
    "assessments_complexity",
]

ALL_CANONICAL = RAW + DERIVED + ENGINEERED

# Your EDA display labels (plus start_date_ts -> Start Date)
DISPLAY_LABELS = {
    "duration_days": "Trial Duration (days)",
    "duration_years": "Trial Duration (years)",
    "start_year": "Start Year",
    "num_arms": "Number of Treatment Arms",
    "assessments_n": "Number of Assessments",
    "primary_out_n": "Number of Primary Outcomes",
    "secondary_out_n": "Number of Secondary Outcomes",
    "other_out_n": "Number of Other Outcomes",
    "elig_crit_n": "Number of Eligibility Criteria",
    "active_prob": "Probability of Receiving Active Treatment",
    "masking_flag": "Blinded Masking",
    "placebo_flag": "Placebo Included",
    "randomized_flag": "Randomised Allocation",
    "fda_drug_flag": "FDA-regulated Drug",
    "fda_device_flag": "FDA-regulated Device",
    "safety_cuts_flag": "Formal Safety Oversight / Interim Rules",
    "country_n": "Number of Countries",
    "site_n": "Number of Sites",
    "# patients": "Number of Patients",
    "patients_per_site": "Patients per Site",
    "age_min": "Minimum Participant Age (years)",
    "age_max": "Maximum Participant Age (years)",
    "age_range": "Participant Age Range (years)",
    "phase": "Study Phase",
    "sponsor_class": "Sponsor Class",
    "intervention_type": "Intervention Type",
    "masking_level": "Masking Level",
    "condition_top": "Top Condition",
    "study_type": "Study Type",
    "allocation": "Allocation",
    "cohort_design": "Cohort Design",
    "therapeutic_area": "Therapeutic Area",
    "assessments_complexity": "Assessments Complexity",
    "global_trial": "Single Country / Global Trial",
    "population_class": "Population Class",
    "start_date": "Start Date",
    "freq_in_window": "Frequency in Window",
    "novelty_score": "Novelty Score",
    "complexity_score_100": "Complexity Score",
    "attractiveness_score_100": "Attractiveness Score",
    # Extension for the engineered timestamp:
    "start_date_ts": "Start Date",
}

# ------------------------ Helpers: matching, labels, splits ------------------------

def norm(s: str) -> str:
    """Loosened normalization for fuzzy matching."""
    s = s.lower().strip()
    s = s.replace("#", "num ").replace("number", "num")
    s = re.sub(r"[_\-/]+", " ", s)
    s = s.replace("count", "n")
    s = s.replace("randomised", "randomized")
    s = s.replace("trial", "")
    s = re.sub(r"\s+", " ", s)
    return s

SYNONYMS: Dict[str, List[str]] = {
    "# patients": ["num patients", "n patients", "patients n", "enrollment", "enrolment", "enrollment n", "enrolled", "enrollmentcount"],
    "site_n": ["n sites", "num sites", "site count", "sites n"],
    "country_n": ["n countries", "num countries", "country count", "countries n"],
    "primary_out_n": ["n primary outcomes", "num primary outcomes", "primary outcomes n"],
    "secondary_out_n": ["n secondary outcomes", "num secondary outcomes", "secondary outcomes n"],
    "other_out_n": ["n other outcomes", "num other outcomes", "other outcomes n"],
    "num_arms": ["n arms", "num arms", "arms n"],
    "masking_flag": ["masking", "blinded", "blinding", "blinded_flag"],
    "placebo_flag": ["placebo", "placebo included"],
    "randomized_flag": ["randomized", "randomised", "randomisation", "is randomized"],
    "fda_drug_flag": ["fda drug", "fda_regulated_drug"],
    "fda_device_flag": ["fda device", "fda_regulated_device"],
    "safety_cuts_flag": ["safety", "safety oversight", "data monitoring", "interim rules"],
    "patients_per_site": ["patients per site", "patients/site", "enrollment per site"],
    "age_min": ["min age", "minimum age"],
    "age_max": ["max age", "maximum age"],
    "age_range": ["age range", "age span"],
    "sponsor_class": ["sponsor class", "sponsor_type", "sponsor category"],
    "intervention_type": ["intervention type"],
    "cohort_design": ["cohort design"],
    "masking_level": ["masking level", "blinding level"],
    "assessments_n": ["n assessments", "num assessments"],
    "elig_crit_n": ["n eligibility criteria", "num eligibility criteria", "eligibility criteria n"],
    "freq_in_window": ["frequency in window", "freq window", "freq_in_time_window"],
    "global_trial": ["global", "multi country", "single country", "global trial"],
    "therapeutic_area": ["therapeutic area"],
    "population_class": ["population class", "population category"],
    "assessments_complexity": ["assessments complexity", "assessment complexity"],
    "start_date_ts": ["start date ts", "start date time", "start date epoch", "start ts"],
}

def best_match(canonical: str, df_cols: List[str]) -> Tuple[Optional[str], float, str]:
    """Return (best_column, confidence, reason) for a canonical name."""
    n_can = norm(canonical)
    candidates = [(c, norm(c)) for c in df_cols]

    # Exact or normalized exact
    for c, n_c in candidates:
        if c.lower() == canonical.lower() or n_c == n_can:
            return c, 1.0, "exact"

    # Synonym contains / contained-in checks
    for syn in SYNONYMS.get(canonical, []):
        n_syn = norm(syn)
        for c, n_c in candidates:
            if n_syn == n_c or n_syn in n_c or n_c in n_syn:
                return c, 0.95, f"synonym:{syn}"

    # Fuzzy similarity
    best_c, best_score = None, -1.0
    for c, n_c in candidates:
        score = SequenceMatcher(None, n_can, n_c).ratio()
        if score > best_score:
            best_score, best_c = score, c

    reason = "fuzzy"
    return (best_c if best_score >= 0.70 else None), float(best_score), reason

def detect_nctid_column(df: pd.DataFrame) -> str:
    """Find the NCT id column robustly."""
    candidates = [c for c in df.columns if "nct" in c.lower() and "id" in c.lower()]
    if candidates:
        return candidates[0]
    # fallbacks
    for c in df.columns:
        n = norm(c)
        if n in {"nctid", "nct id", "nct_number", "nctnumber"}:
            return c
    # try fuzzy
    best_c, best_score = None, -1
    for c in df.columns:
        s = SequenceMatcher(None, norm("nctid"), norm(c)).ratio()
        if s > best_score:
            best_score, best_c = s, c
    if best_score >= 0.65:
        return best_c
    raise ValueError("Could not detect NCT ID column in dataframe.")

def pretty_label(canonical_or_col: str) -> str:
    """Prefer DISPLAY_LABELS; otherwise title‑case fallback."""
    if canonical_or_col in DISPLAY_LABELS:
        return DISPLAY_LABELS[canonical_or_col]
    if canonical_or_col.lower() in DISPLAY_LABELS:
        return DISPLAY_LABELS[canonical_or_col.lower()]
    # Best-effort tidy label
    s = re.sub(r"[_\s]+", " ", canonical_or_col).strip()
    s = s.replace(" ts", "")
    return s[:1].upper() + s[1:]

def load_id_list(path: pathlib.Path) -> List[str]:
    with open(path, "r") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    # standardize case
    return [i.upper() for i in ids]

# ------------------------ Data preparation ------------------------

def add_start_date_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Add start_date_ts from start_date if available."""
    if "start_date_ts" in df.columns:
        return df
    if "start_date" in df.columns:
        dt = pd.to_datetime(df["start_date"], errors="coerce")
        ts = dt.view("int64")  # ns
        ts[dt.isna()] = np.nan
        df["start_date_ts"] = ts / 1e9  # seconds
    return df

def coerce_flags_to_cats(df: pd.DataFrame, cols: List[str]) -> None:
    """Turn 0/1-like flags into 'No'/'Yes' categoricals for readability."""
    for col in cols:
        if col not in df.columns:
            continue
        ser = df[col]
        # Only if dtype numeric or boolean and has <=2 unique non-nans
        vals = pd.Series(ser.dropna().unique())
        if len(vals) <= 2 and (pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_bool_dtype(ser)):
            mapping = {0: "No", 1: "Yes", False: "No", True: "Yes"}
            df[col] = ser.map(mapping).astype("category")

def split_num_cat(df: pd.DataFrame, columns: List[str]) -> Tuple[List[str], List[str]]:
    nums, cats = [], []
    for c in columns:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_categorical_dtype(df[c]):
            nums.append(c)
        else:
            cats.append(c)
    return nums, cats

# ------------------------ Modeling ------------------------

def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    # handle sparse_output vs sparse for different sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    ohe),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], verbose_feature_names_out=False)
    return pre

def evaluate_feature_set(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                         y_tr: pd.Series, y_te: pd.Series,
                         features: List[str], seed: int, name: str) -> dict:
    kept = list(dict.fromkeys([c for c in features if c in df_tr.columns]))
    if not kept:
        raise ValueError(f"No valid columns for feature set '{name}' – check names")

    num_cols, cat_cols = split_num_cat(df_tr, kept)
    pre = make_preprocessor(num_cols, cat_cols)
    rf = RandomForestRegressor(random_state=seed, **RF_PARAMS)

    pipe = Pipeline([("pre", pre), ("rf", rf)])

    t0 = time.perf_counter()
    pipe.fit(df_tr[kept], y_tr)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(df_te[kept])

    mae_days  = mean_absolute_error(y_te, y_pred)
    rmse_days = mean_squared_error(y_te, y_pred) ** 0.5
    r2        = r2_score(y_te, y_pred)

    return {
        "left_out": name,   # for baselines this is the feature-set label
        "seed":     seed,
        "MAE":      mae_days / 365.25,
        "RMSE":     rmse_days / 365.25,
        "R2":       r2,
        "seconds":  fit_sec,
    }

# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=DATA, help="Path to features parquet")
    parser.add_argument("--train_ids", type=pathlib.Path, default=TRAIN_IDS, help="Train NCT IDs txt")
    parser.add_argument("--test_ids", type=pathlib.Path, default=TEST_IDS, help="Test  NCT IDs txt")
    parser.add_argument("--target", type=str, default=TARGET, help="Target column name")
    parser.add_argument("--figdir", type=pathlib.Path, default=FIGDIR, help="Output figure dir")
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in dataframe columns.")

    # Ensure engineered timestamp exists
    df = add_start_date_ts(df)

    # Detect NCT ID column and prepare OOT split
    nct_col = detect_nctid_column(df)
    train_ids = set(load_id_list(args.train_ids))
    test_ids  = set(load_id_list(args.test_ids))

    # Standardize df IDs for matching
    df[nct_col] = df[nct_col].astype(str).str.upper()

    is_train = df[nct_col].isin(train_ids)
    is_test  = df[nct_col].isin(test_ids)

    overlap = (train_ids & test_ids)
    if overlap:
        print(f"[WARN] {len(overlap)} IDs are present in BOTH train & test lists; they will be treated as TEST.")

    # Ensure disjoint
    is_train = is_train & ~is_test

    X_train_all, X_test_all = df.loc[is_train], df.loc[is_test]
    y_train, y_test = X_train_all[args.target], X_test_all[args.target]

    print(f"Train rows: {len(X_train_all):,} | Test rows: {len(X_test_all):,} | Train IDs: {len(train_ids):,} | Test IDs: {len(test_ids):,}")

    # Identify likely flag columns & coerce to categorical for readability
    likely_flags = [c for c in df.columns if c.endswith("_flag")] + [
        "placebo_flag", "masking_flag", "randomized_flag",
        "fda_drug_flag", "fda_device_flag", "safety_cuts_flag"
    ]
    coerce_flags_to_cats(df, likely_flags)

    # Column matching: map canonical -> actual df column
    matches = []
    canon_to_dfcol: Dict[str, str] = {}
    for canon in ALL_CANONICAL:
        if canon in df.columns:
            canon_to_dfcol[canon] = canon
            matches.append((canon, canon, 1.0, "exact"))
            continue
        found, score, reason = best_match(canon, list(df.columns))
        if found is not None:
            canon_to_dfcol[canon] = found
            matches.append((canon, found, score, reason))
        else:
            matches.append((canon, "", score, "unmatched"))

    # Save mapping for auditability
    mapping_df = pd.DataFrame(matches, columns=["canonical", "df_column", "confidence", "reason"])
    mapping_csv = args.figdir / "column_matching.csv"
    mapping_df.to_csv(mapping_csv, index=False)
    print(f"Saved column matching → {mapping_csv}")

    # Assemble feature groups using the matched columns that actually exist
    RAW_cols        = [canon_to_dfcol[c] for c in RAW if c in canon_to_dfcol and canon_to_dfcol[c] in df.columns]
    DERIVED_cols    = [canon_to_dfcol[c] for c in DERIVED if c in canon_to_dfcol and canon_to_dfcol[c] in df.columns]
    ENGINEERED_cols = [canon_to_dfcol[c] for c in ENGINEERED if c in canon_to_dfcol and canon_to_dfcol[c] in df.columns]

    ALL_COLS = list(dict.fromkeys(RAW_cols + DERIVED_cols + ENGINEERED_cols))
    if not ALL_COLS:
        raise ValueError("No usable features after matching. Check 'column_matching.csv'.")

    print(f"Using {len(ALL_COLS)} total features ({len(RAW_cols)} RAW, {len(DERIVED_cols)} DERIVED, {len(ENGINEERED_cols)} ENGINEERED).")

    # Build baseline sets
    BASELINE_SETS = {
        "All Features (Raw + Derived + Engineered)": ALL_COLS,
        "Raw Features Only":     RAW_cols,
        "Raw + Derived Features":  list(dict.fromkeys(RAW_cols + DERIVED_cols)),
    }

    # Subset frames to identical rows (avoid re-index pitfalls)
    X_train = X_train_all[ALL_COLS]
    X_test  = X_test_all[ALL_COLS]

    # Run experiments
    print(f"Running {len(SEEDS)} seeds × {len(BASELINE_SETS) + len(ALL_COLS)} feature sets")
    metrics: List[dict] = []

    # Baselines
    for seed in SEEDS:
        for name, cols in BASELINE_SETS.items():
            if not cols:
                continue
            metrics.append(
                evaluate_feature_set(X_train, X_test, y_train, y_test, cols, seed, name)
            )

    # Leave‑one‑out ablation
    for seed in SEEDS:
        for left_out_col in ALL_COLS:
            kept = [c for c in ALL_COLS if c != left_out_col]
            if not kept:
                continue
            # Use the *canonical* pretty label if possible; else fall back to df col
            # Find canonical corresponding to this df column
            canonical_for_col = None
            for k, v in canon_to_dfcol.items():
                if v == left_out_col:
                    canonical_for_col = k
                    break
            abl_name = pretty_label(canonical_for_col or left_out_col)
            res = evaluate_feature_set(X_train, X_test, y_train, y_test, kept, seed, abl_name)
            metrics.append(res)

    # Save raw results
    results = pd.DataFrame(metrics)
    raw_csv = args.figdir / "ablation_rf_metrics.csv"
    results.to_csv(raw_csv, index=False)
    print("Saved raw metrics    →", raw_csv)

    # Aggregate & visualize (mean ± sd)
    summary = (
        results.groupby("left_out", as_index=False)
               .agg(MAE_mean=("MAE", "mean"),
                    MAE_std =("MAE", "std"),
                    RMSE_mean=("RMSE", "mean"),
                    R2_mean=("R2", "mean"))
               .sort_values("MAE_mean", ascending=False)
    )
    sum_csv = args.figdir / "ablation_rf_metrics_summary.csv"
    summary.to_csv(sum_csv, index=False)
    print("Saved summary        →", sum_csv)

    # Plot MAE with error bars (±1 SD). Labels already pretty for ablations; baselines keep their names.
    sns.set_style("whitegrid")
    plt.figure(figsize=(8.5, max(6, 0.33 * len(summary))))
    y = summary["left_out"]
    x = summary["MAE_mean"]
    err = summary["MAE_std"].fillna(0.0)

    plt.barh(y, x, xerr=err, capsize=3)
    plt.xlabel("MAE (years)")
    plt.ylabel("Feature Left Out / Model")
    plt.title(f"Random Forest ablation – mean MAE ±1 SD across {len(SEEDS)} seeds\n(lower is better)")
    plt.xlim(max(0, x.min() * 0.95), x.max() * 1.05)
    plt.tight_layout()

    png_path = args.figdir / "ablation_rf_mae.png"
    pdf_path = args.figdir / "ablation_rf_mae.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    print("Saved chart          →", png_path)
    print("Saved vector (PDF)   →", pdf_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
