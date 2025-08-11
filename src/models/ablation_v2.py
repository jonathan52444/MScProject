"""
Random‑Forest ablation study (v2)
================================
• Runs **20×** repetitions per feature‑set (configurable via ``N_REPEATS``)
• Separate ablation tables for:
    1. **Raw‐features only** models (every raw feature left out once)
    2. **Raw + Derived** models (every derived feature left out once)
• Baselines are included at the top of each table
• Summary tables capture **mean ± sd** across seeds and are saved to CSV
• MAE + error‑bars plotted *per study* (optional)

Outputs
-------
reports/figures/filtered_data/
    ├── ablation_rf_v2_metrics.csv          (all repetitions)
    ├── ablation_rf_v2_summary.csv          (mean ± sd per row)
    ├── ablation_rf_v2_raw_mae.png          (plot for Raw‑only study)
    └── ablation_rf_v2_rawderived_mae.png   (plot for Raw+Derived study)
"""

from __future__ import annotations
import pathlib, time, warnings, itertools, joblib

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

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
N_REPEATS   = 20                 # number of different random seeds per feature‑set
SEEDS       = list(range(42, 42 + N_REPEATS))
PLOT_ERRORBARS = True            # generate barplot with ±1 SD error bars

RF_PARAMS = dict(
    n_estimators      = 500,
    n_jobs            = -1,
    max_features      = "sqrt",
    min_samples_leaf  = 2,
    bootstrap         = True,
)

ROOT   = pathlib.Path(__file__).resolve().parents[2]  # adapt if needed
DATA   = ROOT / "data" / "processed" / "features_v4.parquet"
FIGDIR = ROOT / "reports" / "figures" / "filtered_data";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";                                 MODELD.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Load data + light preprocessing
# -----------------------------------------------------------------------------

df = pd.read_parquet(DATA)
TARGET = "duration_days"

# Convert date columns to numeric (seconds since UNIX epoch)
for col in ["start_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    ts = df[col].view("int64")
    ts[df[col].isna()] = np.nan
    df[f"{col}_ts"] = ts / 1e9  # seconds since epoch

# -----------------------------------------------------------------------------
# Feature groups (update as the dataframe evolves)
# -----------------------------------------------------------------------------

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
    "masking_level",
]

ENGINEERED = [
    "assessments_n", "active_prob", "elig_crit_n",
    "freq_in_window", "novelty_score",
    "complexity_score_100", "attractiveness_score_100",
    "assessments_complexity", "global_trial", "therapeutic_area",
    "population_class",
]

# All columns present in the dataframe
ALL_COLS = [c for c in RAW + DERIVED + ENGINEERED if c in df.columns]
missing = sorted(set(RAW + DERIVED + ENGINEERED) - set(ALL_COLS))
if missing:
    print("⚠️  Skipping missing columns:", ", ".join(missing))

# -----------------------------------------------------------------------------
# Train/test split (temporal)
# -----------------------------------------------------------------------------

cutoff = pd.Timestamp("2019-01-01")
is_test = df["start_date"] >= cutoff

X_train, X_test = df.loc[~is_test, ALL_COLS], df.loc[is_test, ALL_COLS]
y_train, y_test = df.loc[~is_test, TARGET],   df.loc[is_test, TARGET]

print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

# -----------------------------------------------------------------------------
# Pre‑processing helpers
# -----------------------------------------------------------------------------

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])

def split_num_cat(columns: list[str]) -> tuple[list[str], list[str]]:
    """Return ([numeric_cols], [categorical_cols]) for *columns* based on df dtypes."""
    num, cat = [], []
    for c in columns:
        (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return num, cat

# -----------------------------------------------------------------------------
# Core evaluation routine
# -----------------------------------------------------------------------------

def evaluate_feature_set(name: str, features: list[str], seed: int) -> dict:
    """Train/test a RF on *features* using *seed* and return metrics dict."""
    kept = [c for c in features if c in df.columns]
    if not kept:
        raise ValueError(f"No valid columns for feature set '{name}' – check names")

    num_cols, cat_cols = split_num_cat(kept)

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], verbose_feature_names_out=False)

    rf = RandomForestRegressor(random_state=seed, **RF_PARAMS)

    pipe = Pipeline([
        ("pre", pre),
        ("rf",  rf),
    ])

    t0 = time.perf_counter()
    pipe.fit(X_train[kept], y_train)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(X_test[kept])

    mae_days  = mean_absolute_error(y_test, y_pred)
    rmse_days = mean_squared_error(y_test, y_pred) ** 0.5

    return {
        "model":      name,
        "seed":       seed,
        "MAE":        mae_days  / 365,
        "RMSE":       rmse_days / 365,
        "R2":         r2_score(y_test, y_pred),
        "seconds":    fit_sec,
    }

# -----------------------------------------------------------------------------
# Ablation experiments: two separate studies
# -----------------------------------------------------------------------------

metrics: list[dict] = []

STUDIES = {
    "Raw Features Ablation Study": {
        "baseline_name": "All Raw Features",
        "feature_space": RAW,
        "ablate_these":  RAW,          # drop each raw feature once
    },
    "Raw+Derived Features Ablation Study": {
        "baseline_name": "All Raw+Derived Features",
        "feature_space": RAW + DERIVED,
        "ablate_these":  DERIVED,      # drop each derived feature once
    },
}

print("Running ablations …")
for seed in tqdm(SEEDS, desc="seeds", unit="seed"):
    for study_name, cfg in STUDIES.items():
        base_name = cfg["baseline_name"]
        base_cols = cfg["feature_space"]

        # 1) baseline
        metrics.append(
            evaluate_feature_set(f"{study_name} | {base_name}", base_cols, seed)
        )

        # 2) leave‑one‑out within this study
        for feat in cfg["ablate_these"]:
            kept = [c for c in base_cols if c != feat]
            model_name = f"{study_name} | {base_name} - {feat}"
            metrics.append(evaluate_feature_set(model_name, kept, seed))

# -----------------------------------------------------------------------------
# Aggregate & save
# -----------------------------------------------------------------------------

results = pd.DataFrame(metrics)
raw_csv = FIGDIR / "ablation_rf_v2_metrics.csv"
results.to_csv(raw_csv, index=False)
print("Saved raw metrics →", raw_csv)

summary = (
    results.groupby("model", as_index=False)
           .agg(MAE_mean=("MAE", "mean"),
                MAE_std =("MAE", "std"),
                RMSE_mean=("RMSE", "mean"),
                R2_mean=("R2", "mean"))
           .sort_values(["model"])
)

# Split study column back out for readability
summary[["study", "model_clean"]] = summary["model"].str.split(" \| ", expand=True)
summary = summary[["study", "model_clean", "MAE_mean", "MAE_std", "RMSE_mean", "R2_mean"]]

sum_csv = FIGDIR / "ablation_rf_v2_summary.csv"
summary.to_csv(sum_csv, index=False)
print("Saved summary →", sum_csv)

# -----------------------------------------------------------------------------
# Visualisation (per study)
# -----------------------------------------------------------------------------

if PLOT_ERRORBARS:
    for study, grp in summary.groupby("study"):
        plt.figure(figsize=(8, max(6, 0.25 * len(grp))))
        plt.barh(grp["model_clean"], grp["MAE_mean"], xerr=grp["MAE_std"], capsize=3)
        plt.xlabel("MAE (years)")
        plt.title(f"{study} – mean MAE ±1 SD across seeds\n(lower is better)")
        plt.tight_layout()
        fname = ("ablation_rf_v2_raw_mae.png" if "Raw Features" in study else "ablation_rf_v2_rawderived_mae.png")
        plt.savefig(FIGDIR / fname, dpi=150)
        plt.close()
        print("Saved chart →", FIGDIR / fname)
else:
    print("Skipping plots (PLOT_ERRORBARS = False)")
