"""
Engineered‑Features Benchmark Study
===================================
Evaluates how far we can get by mimicking a *“naïve human heuristic”* that only looks at
hand‑crafted (engineered) signals—no raw or derived inputs.

Experiments ------------------------------------------------------------
1. **Engineered‑only baseline** – model trained on the full ENGINEERED set.
2. **Single‑feature models** – one model per engineered column (e.g. just
   ``novelty_score``).

All models are 500‑tree *RandomForestRegressor*s and are repeated ``N_REPEATS``
(times/seeds) to measure variance.  The temporal train/test split and metric
suite (*MAE, RMSE, R²*) match the original ablation scripts so results are
comparable.

Outputs ---------------------------------------------------------------
    reports/figures/filtered_data/engineered_rf_metrics.csv
    reports/figures/filtered_data/engineered_rf_metrics_summary.csv
    reports/figures/filtered_data/engineered_rf_mae.png

Usage -----------------------------------------------------------------
$ python engineered_features_benchmark.py                # default 20× repeats
$ N_REPEATS=50 python engineered_features_benchmark.py   # override via env var
"""

from __future__ import annotations

import itertools
import os
import pathlib
import time
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Config ──────────────────────────────────────────────────────────────
N_REPEATS = int(os.getenv("N_REPEATS", 20))
SEEDS = list(range(42, 42 + N_REPEATS))

RF_PARAMS = dict(
    n_estimators=500,
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=2,
    bootstrap=True,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "features_v4.parquet"
FIGDIR = ROOT / "reports" / "figures" / "filtered_data"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── Feature sets ───────────────────────────────────────────────────────
ENGINEERED: List[str] = [
    "assessments_n",
    "active_prob",
    "elig_crit_n",
    "freq_in_window",
    "novelty_score",
    "complexity_score_100",
    "attractiveness_score_100",
    "assessments_complexity",
    "global_trial",
    "therapeutic_area",
    "population_class",
]

TARGET = "duration_days"

# ── Data loading & basic prep ──────────────────────────────────────────
print("Loading", DATA)

df = pd.read_parquet(DATA)

# Ensure engineered columns exist
missing = sorted(set(ENGINEERED) - set(df.columns))
if missing:
    raise ValueError("Missing engineered columns in dataframe: " + ", ".join(missing))

# Temporal split (same rule as other scripts)
cutoff = pd.Timestamp("2019-01-01")
is_test = df["start_date"] >= cutoff

X_train, X_test = df.loc[~is_test, ENGINEERED], df.loc[is_test, ENGINEERED]
y_train, y_test = df.loc[~is_test, TARGET], df.loc[is_test, TARGET]

print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

# ── Helpers ────────────────────────────────────────────────────────────
num_pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)

cat_pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)


def split_num_cat(cols: List[str]) -> tuple[List[str], List[str]]:
    num, cat = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def evaluate_feature_set(name: str, cols: List[str], seed: int) -> Dict[str, float]:
    num_cols, cat_cols = split_num_cat(cols)

    pre = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        verbose_feature_names_out=False,
    )

    model = RandomForestRegressor(random_state=seed, **RF_PARAMS)

    pipe = Pipeline([("pre", pre), ("rf", model)])

    t0 = time.perf_counter()
    pipe.fit(X_train[cols], y_train)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(X_test[cols])

    mae = mean_absolute_error(y_test, y_pred) / 365  # → years
    rmse = (mean_squared_error(y_test, y_pred) ** 0.5) / 365

    return {
        "feature_set": name,
        "seed": seed,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2_score(y_test, y_pred),
        "seconds": fit_sec,
    }


# ── Run experiments ────────────────────────────────────────────────────
results: List[Dict] = []

print(f"Running {len(SEEDS)} seeds for baseline + {len(ENGINEERED)} single‑feature models")

# 1. Engineered‑only baseline
for seed in tqdm(SEEDS, desc="Engineered‑only baseline"):
    results.append(evaluate_feature_set("Engineered‑only", ENGINEERED, seed))

# 2. Single‑feature models
for feature in tqdm(ENGINEERED, desc="Single‑feature loop"):
    for seed in SEEDS:
        results.append(evaluate_feature_set(feature, [feature], seed))

# ── Save raw metrics ───────────────────────────────────────────────────
raw_df = pd.DataFrame(results)
raw_csv = FIGDIR / "engineered_rf_metrics.csv"
raw_df.to_csv(raw_csv, index=False)
print("Saved raw metrics →", raw_csv)

# ── Aggregate & summarise ──────────────────────────────────────────────
summary = (
    raw_df.groupby("feature_set", as_index=False)
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        R2_mean=("R2", "mean"),
    )
    .sort_values("MAE_mean")  # ascending (lower MAE is better)
)

sum_csv = FIGDIR / "engineered_rf_metrics_summary.csv"
summary.to_csv(sum_csv, index=False)
print("Saved summary     →", sum_csv)

# ── Plot MAE with error bars ───────────────────────────────────────────
plt.figure(figsize=(8, max(6, 0.3 * len(summary))))
plt.barh(summary["feature_set"], summary["MAE_mean"], xerr=summary["MAE_std"], capsize=3)
plt.xlabel("MAE (years)")
plt.title("Engineered‑features benchmark – mean MAE ±1 SD\n(lower is better)")
plt.tight_layout()
fig_path = FIGDIR / "engineered_rf_mae.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print("Saved chart       →", fig_path)
