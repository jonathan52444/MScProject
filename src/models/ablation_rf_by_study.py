# src/models/ablation_rf_by_study.py
"""
Ablation (by study) — RandomForest leave-one-out per feature group (v2).

Purpose
-------
Run two focused ablation studies and save separate results & plots:
  1. Raw Features Ablation (leave each raw feature out once)
  2. Raw+Derived Ablation (leave each derived feature out once)

"""

from __future__ import annotations
import argparse, pathlib, time, warnings, re, sys
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Config (update here or via CLI)
ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v6.parquet"
SPLITS = ROOT / "data" / "splits"
TRAIN_IDS = SPLITS / "oot_filtered_train_nctids.txt"
TEST_IDS  = SPLITS / "oot_filtered_test_nctids.txt"

FIGDIR = ROOT / "reports" / "figures209v2" / "filtered_data";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";                                 MODELD.mkdir(exist_ok=True)

TARGET = "duration_days"
PLOT_ERRORBARS = True

# Seeds → **5** runs
SEEDS = [42, 43, 44, 45, 46]

# Random‑Forest parameters (as requested)
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

# Feature groups (canonical names)
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

# DISPLAY_LABELS (from your EDA; with a label for start_date_ts)
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
    "start_date_ts": "Start Date",
}

# Matching helpers (canonical → df columns) + labels

def norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("#", "num ").replace("number", "num")
    s = re.sub(r"[_\-/]+", " ", s)
    s = s.replace("count", "n").replace("randomised", "randomized")
    s = re.sub(r"\s+", " ", s)
    return s

SYNONYMS: Dict[str, List[str]] = {
    "# patients": ["num patients", "n patients", "patients n", "enrollment", "enrolment", "enrollmentcount", "enrolled"],
    "site_n": ["n sites", "num sites", "site count", "sites n"],
    "country_n": ["n countries", "num countries", "country count", "countries n"],
    "primary_out_n": ["n primary outcomes", "num primary outcomes", "primary outcomes n"],
    "secondary_out_n": ["n secondary outcomes", "num secondary outcomes", "secondary outcomes n"],
    "other_out_n": ["n other outcomes", "num other outcomes", "other outcomes n"],
    "num_arms": ["n arms", "num arms", "arms n"],
    "masking_flag": ["masking", "blinded", "blinding", "blinded_flag"],
    "placebo_flag": ["placebo", "placebo included"],
    "randomized_flag": ["randomized", "randomised", "is randomized"],
    "fda_drug_flag": ["fda drug", "fda_regulated_drug"],
    "fda_device_flag": ["fda device", "fda_regulated_device"],
    "safety_cuts_flag": ["safety", "oversight", "data monitoring", "interim rules", "dmc_flag", "dmsc_flag"],
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
    "start_date_ts": ["start date ts", "start date epoch", "start timestamp", "start ts"],
}

def best_match(canonical: str, df_cols: List[str]) -> Tuple[Optional[str], float, str]:
    n_can = norm(canonical)
    candidates = [(c, norm(c)) for c in df_cols]

    # exact or normalized exact
    for c, n_c in candidates:
        if c.lower() == canonical.lower() or n_c == n_can:
            return c, 1.0, "exact"

    # synonyms
    for syn in SYNONYMS.get(canonical, []):
        n_syn = norm(syn)
        for c, n_c in candidates:
            if n_syn == n_c or n_syn in n_c or n_c in n_syn:
                return c, 0.95, f"synonym:{syn}"

    # fuzzy
    best_c, best_score = None, -1.0
    for c, n_c in candidates:
        score = SequenceMatcher(None, n_can, n_c).ratio()
        if score > best_score:
            best_score, best_c = score, c
    return (best_c if best_score >= 0.70 else None), float(best_score), "fuzzy"

def pretty_label(name: str) -> str:
    if name in DISPLAY_LABELS:
        return DISPLAY_LABELS[name]
    s = re.sub(r"[_\s]+", " ", name).strip()
    s = s.replace(" ts", "")
    return s[:1].upper() + s[1:]


# Splits + light preprocessing
def detect_nctid_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if "nct" in c.lower() and "id" in c.lower()]
    if cands:
        return cands[0]
    # fallback: fuzzy to "nctid"
    best_c, best_score = None, -1
    for c in df.columns:
        s = SequenceMatcher(None, "nctid", c.lower()).ratio()
        if s > best_score:
            best_score, best_c = s, c
    if best_score >= 0.65:
        return best_c
    raise ValueError("Could not detect NCT ID column in dataframe.")

def load_id_list(path: pathlib.Path) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip().upper() for ln in f if ln.strip()]

def add_start_date_ts(df: pd.DataFrame) -> pd.DataFrame:
    if "start_date_ts" in df.columns:
        return df
    if "start_date" in df.columns:
        dt = pd.to_datetime(df["start_date"], errors="coerce")
        ts = dt.view("int64")
        ts[dt.isna()] = np.nan
        df["start_date_ts"] = ts / 1e9
    return df

def coerce_flags_to_cats(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col not in df.columns: 
            continue
        ser = df[col]
        vals = pd.Series(ser.dropna().unique())
        if len(vals) <= 2 and (pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_bool_dtype(ser)):
            df[col] = ser.map({0: "No", 1: "Yes", False: "No", True: "Yes"}).astype("category")

def split_num_cat(df: pd.DataFrame, columns: List[str]) -> Tuple[List[str], List[str]]:
    nums, cats = [], []
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_categorical_dtype(df[c]):
            nums.append(c)
        else:
            cats.append(c)
    return nums, cats

def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    ohe),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], verbose_feature_names_out=False)



# Core evaluation
def evaluate_feature_set(
    X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series,
    features: List[str], seed: int, model_label: str, study: str, is_baseline: bool
) -> dict:
    kept = [c for c in features if c in X_tr.columns]
    if not kept:
        raise ValueError(f"No valid columns for model '{model_label}'")

    num_cols, cat_cols = split_num_cat(X_tr, kept)
    pre = make_preprocessor(num_cols, cat_cols)
    rf  = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    pipe = Pipeline([("pre", pre), ("rf", rf)])

    t0 = time.perf_counter()
    pipe.fit(X_tr[kept], y_tr)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(X_te[kept])

    mae_days  = mean_absolute_error(y_te, y_pred)
    rmse_days = mean_squared_error(y_te, y_pred) ** 0.5
    r2        = r2_score(y_te, y_pred)

    return {
        "study":      study,
        "model":      model_label,  # used for grouping/rows
        "seed":       seed,
        "is_baseline": int(is_baseline),
        "MAE":        mae_days  / 365.25,
        "RMSE":       rmse_days / 365.25,
        "R2":         r2,
        "seconds":    fit_sec,
    }

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=DATA)
    parser.add_argument("--train_ids", type=pathlib.Path, default=TRAIN_IDS)
    parser.add_argument("--test_ids",  type=pathlib.Path, default=TEST_IDS)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--figdir", type=pathlib.Path, default=FIGDIR)
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in dataframe.")

    df = add_start_date_ts(df)

    # OOT split by NCT IDs
    nct_col = detect_nctid_column(df)
    df[nct_col] = df[nct_col].astype(str).str.upper()
    train_ids = set(load_id_list(args.train_ids))
    test_ids  = set(load_id_list(args.test_ids))
    overlap   = train_ids & test_ids
    if overlap:
        print(f"[WARN] {len(overlap)} IDs appear in BOTH files → treated as TEST.")

    is_test  = df[nct_col].isin(test_ids)
    is_train = df[nct_col].isin(train_ids) & ~is_test

    X_train_all, X_test_all = df.loc[is_train].copy(), df.loc[is_test].copy()
    y_train, y_test = X_train_all[args.target], X_test_all[args.target]
    print(f"Train rows: {len(X_train_all):,} | Test rows: {len(X_test_all):,}")

    # Coerce likely flags for readability
    likely_flags = [c for c in df.columns if c.endswith("_flag")] + [
        "placebo_flag","masking_flag","randomized_flag","fda_drug_flag","fda_device_flag","safety_cuts_flag"
    ]
    coerce_flags_to_cats(df, likely_flags)

    # Canonical→df matching
    matches, canon_to_dfcol = [], {}
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

    mapping_df = pd.DataFrame(matches, columns=["canonical","df_column","confidence","reason"])
    mapping_csv = args.figdir / "ablation_rf_v2_column_matching.csv"
    mapping_df.to_csv(mapping_csv, index=False)
    print("Saved column mapping →", mapping_csv)

    # Build final feature groups that exist
    RAW_cols        = [canon_to_dfcol[c] for c in RAW if c in canon_to_dfcol and canon_to_dfcol[c] in df.columns]
    DERIVED_cols    = [canon_to_dfcol[c] for c in DERIVED if c in canon_to_dfcol and canon_to_dfcol[c] in df.columns]
    ENGINEERED_cols = [canon_to_dfcol[c] for c in ENGINEERED if c in canon_to_dfcol and canon_to_dfcol[c] in df.columns]

    ALL_COLS = list(dict.fromkeys(RAW_cols + DERIVED_cols + ENGINEERED_cols))
    if not ALL_COLS:
        raise ValueError("No usable features after matching. Check the mapping CSV.")

    X_train = X_train_all[ALL_COLS]
    X_test  = X_test_all[ALL_COLS]

    # Define studies
    STUDIES = {
        "Raw Features Ablation Study": {
            "baseline":   "All Raw Features",
            "feature_set": RAW_cols,
            "ablate":      RAW_cols,          # leave each raw feature once
        },
        "Raw+Derived Features Ablation Study": {
            "baseline":   "All Raw+Derived Features",
            "feature_set": list(dict.fromkeys(RAW_cols + DERIVED_cols)),
            "ablate":      DERIVED_cols,      # leave each derived feature once
        },
    }

    # Run experiments
    print(f"Running {len(SEEDS)} seeds × (baselines + ablations per study)")
    sns.set_style("whitegrid")
    metrics: List[dict] = []

    for seed in tqdm(SEEDS, desc="seeds", unit="seed"):
        for study_name, cfg in STUDIES.items():
            base_name = cfg["baseline"]
            base_cols = cfg["feature_set"]

            # Baseline
            if base_cols:
                metrics.append(
                    evaluate_feature_set(X_train, X_test, y_train, y_test,
                                         base_cols, seed,
                                         model_label=f"{study_name} | {base_name}",
                                         study=study_name, is_baseline=True)
                )

            # Leave‑one‑out
            for df_col in cfg["ablate"]:
                kept = [c for c in base_cols if c != df_col]
                if not kept:
                    continue
                # Find the canonical name for this df column (for pretty label)
                canonical = None
                for k, v in canon_to_dfcol.items():
                    if v == df_col:
                        canonical = k
                        break
                nice = pretty_label(canonical or df_col)
                model_label = f"{study_name} | {base_name} - {nice}"
                metrics.append(
                    evaluate_feature_set(X_train, X_test, y_train, y_test,
                                         kept, seed,
                                         model_label=model_label,
                                         study=study_name, is_baseline=False)
                )

    # Save raw metrics
    results = pd.DataFrame(metrics)
    raw_csv = args.figdir / "ablation_rf_v2_metrics.csv"
    results.to_csv(raw_csv, index=False)
    print("Saved raw metrics →", raw_csv)

    # Summary (mean ± sd), keep baselines at top within each study
    summary = (
        results.groupby(["study","model"], as_index=False)
               .agg(MAE_mean=("MAE","mean"),
                    MAE_std =("MAE","std"),
                    RMSE_mean=("RMSE","mean"),
                    R2_mean=("R2","mean"),
                    is_baseline=("is_baseline","max"))
    )

    # Derive a clean label for plotting: after " | "
    summary["model_clean"] = summary["model"].str.split(" \| ", n=1, expand=True)[1]

    # Order rows per study: baselines first, then descending MAE (or choose ascending)
    ordered = []
    for study, grp in summary.groupby("study", sort=False):
        # Baseline first (is_baseline=1), then by MAE_mean descending to highlight worst‑to‑best
        grp = grp.sort_values(by=["is_baseline","MAE_mean"], ascending=[False, False])
        ordered.append(grp)
    summary_ordered = pd.concat(ordered, ignore_index=True)

    sum_csv = args.figdir / "ablation_rf_v2_summary.csv"
    summary_ordered.to_csv(sum_csv, index=False)
    print("Saved summary →", sum_csv)

    # Visualisation (per study) – ±1 SD error bars; also save PDF for LaTeX
    if PLOT_ERRORBARS:
        for study, grp in summary_ordered.groupby("study", sort=False):
            plt.figure(figsize=(10.5, max(6, 0.33 * len(grp))))
            plt.barh(grp["model_clean"], grp["MAE_mean"], xerr=grp["MAE_std"].fillna(0.0), capsize=3)
            plt.xlabel("MAE (years)")
            plt.ylabel(None)
            plt.title(f"{study} – mean MAE ±1 SD across {len(SEEDS)} seeds\n(lower is better)")
            plt.tight_layout()
            fname_png = "ablation_rf_v2_raw_mae.png" if "Raw Features" in study else "ablation_rf_v2_rawderived_mae.png"
            png_path = args.figdir / fname_png
            pdf_path = png_path.with_suffix(".pdf")
            plt.savefig(png_path, dpi=200)
            plt.savefig(pdf_path)
            plt.close()
            print("Saved chart →", png_path, "| PDF →", pdf_path)
    else:
        print("Skipping plots (PLOT_ERRORBARS = False)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
