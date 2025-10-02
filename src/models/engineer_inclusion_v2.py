#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engineered-Features Benchmark Study — OOT split + label-aware plotting
- 5 seeds, ±1 SD error bars (MAE in years)
- Baseline: Engineered features only  → "Engineered Features are Used" (Used in bold)

"""

from __future__ import annotations

import pathlib, time, warnings, re, sys
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.container import BarContainer

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Config ──────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).resolve().parents[2]   # .../MSc Project
DATA = ROOT / "data" / "processed" / "features_v6.parquet"
SPLITS = ROOT / "data" / "splits"
TRAIN_IDS = SPLITS / "oot_filtered_train_nctids.txt"
TEST_IDS  = SPLITS / "oot_filtered_test_nctids.txt"

print("Loading", DATA)
if not DATA.exists():
    raise FileNotFoundError(
        "features_v6.parquet not found at expected path:\n"
        f"  {DATA}\n"
        "Tip: ensure ROOT is the project folder and not 'src/'."
    )

FIGDIR = ROOT / "reports" / "figures_209" / "filtered_data"
FIGDIR.mkdir(parents=True, exist_ok=True)

TARGET = "duration_days"

# 5 seeds (as used elsewhere)
SEEDS = [42, 43, 44, 45, 46]

# Random-Forest parameters (matching previous scripts)
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

# Axis bounds (use exactly when provided)
X_FLOOR = 0.80
X_CEIL  = 1.80
DECIMALS = 3  # for "μ ± σ" labels

# Feature set 
ENGINEERED = [
    "active_prob", "novelty_score",
    "complexity_score_100", "attractiveness_score_100",
    "assessments_complexity"]



# Display labels
DISPLAY_LABELS = {
    "assessments_n": "Number of Assessments",
    "active_prob": "Probability of Receiving Active Treatment",
    "elig_crit_n": "Number of Eligibility Criteria",
    "freq_in_window": "Frequency in Window",
    "novelty_score": "Novelty Score",
    "complexity_score_100": "Complexity Score",
    "attractiveness_score_100": "Attractiveness Score",
    "assessments_complexity": "Assessments Complexity",
    "global_trial": "Single Country / Global Trial",
    "therapeutic_area": "Therapeutic Area",
    "population_class": "Population Class",
}

def pretty(col: str) -> str:
    if col in DISPLAY_LABELS:
        return DISPLAY_LABELS[col]
    s = re.sub(r"[_\s]+", " ", col).strip()
    return s[:1].upper() + s[1:]

# Split helpers
def norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("#", "num ").replace("number", "num")
    s = re.sub(r"[_\-/]+", " ", s)
    return re.sub(r"\s+", " ", s)

def detect_nctid_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if "nct" in c.lower() and "id" in c.lower()]
    if cands: return cands[0]
    best, score = None, -1
    for c in df.columns:
        sc = SequenceMatcher(None, "nctid", c.lower()).ratio()
        if sc > score: best, score = c, sc
    if score >= 0.65: return best
    raise ValueError("Could not detect NCT ID column.")

def load_id_list(path: pathlib.Path) -> list[str]:
    return [ln.strip().upper() for ln in open(path) if ln.strip()]

# ── Preprocess pipelines ───────────────────────────────────────────────
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    ohe),
])

def split_num_cat(df: pd.DataFrame, cols: List[str]) -> tuple[List[str], List[str]]:
    num, cat = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_categorical_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat

def make_pre(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], verbose_feature_names_out=False)

# ── Core evaluation ────────────────────────────────────────────────────
def evaluate_feature_set(X_tr, X_te, y_tr, y_te, cols: List[str], seed: int, label: str) -> Dict[str, float]:
    num_cols, cat_cols = split_num_cat(X_tr, cols)
    pre = make_pre(num_cols, cat_cols)
    rf  = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    pipe = Pipeline([("pre", pre), ("rf", rf)])

    t0 = time.perf_counter()
    pipe.fit(X_tr[cols], y_tr)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(X_te[cols])
    mae = mean_absolute_error(y_te, y_pred) / 365.0
    rmse = (mean_squared_error(y_te, y_pred) ** 0.5) / 365.0

    return {
        "feature_set": label,
        "seed": seed,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2_score(y_te, y_pred),
        "seconds": fit_sec,
    }

# ── Load data + OOT split ─────────────────────────────────────────────
print("Loading", DATA)
df = pd.read_parquet(DATA)

nct_col = detect_nctid_column(df)
df[nct_col] = (
    df[nct_col]
    .astype("string")
    .str.strip()
    .str.upper()
)

train_ids = set(load_id_list(TRAIN_IDS))
test_ids  = set(load_id_list(TEST_IDS))

is_test  = df[nct_col].isin(test_ids)
is_train = df[nct_col].isin(train_ids) & ~is_test

X_train_all, X_test_all = df.loc[is_train, ENGINEERED].copy(), df.loc[is_test, ENGINEERED].copy()
y_train, y_test = df.loc[is_train, TARGET], df.loc[is_test, TARGET]

print(f"Train rows: {len(X_train_all):,} | Test rows: {len(X_test_all):,}")

# ── Run experiments ────────────────────────────────────────────────────
results: List[Dict] = []

print(f"Running {len(SEEDS)} seeds for baseline + {len(ENGINEERED)} single-feature models")

# 1) Engineered-only baseline (bold 'Used' in final plot label)
for seed in SEEDS:
    results.append(evaluate_feature_set(X_train_all, X_test_all, y_train, y_test,
                                        ENGINEERED, seed, "Engineered Features (all)"))

# 2) Single-feature models
for feat in ENGINEERED:
    for seed in SEEDS:
        results.append(evaluate_feature_set(X_train_all, X_test_all, y_train, y_test,
                                            [feat], seed, pretty(feat)))

# ── Save raw metrics ───────────────────────────────────────────────────
raw_df = pd.DataFrame(results)
raw_csv = FIGDIR / "engineered_rf_metrics.csv"
raw_df.to_csv(raw_csv, index=False)
print("Saved raw metrics →", raw_csv)

# ── Aggregate & summarise ──────────────────────────────────────────────
summary = (
    raw_df.groupby("feature_set", as_index=False)
          .agg(MAE_mean=("MAE","mean"),
               MAE_std =("MAE","std"),
               RMSE_mean=("RMSE","mean"),
               R2_mean=("R2","mean"))
)

# Mark/rename baseline and order with baseline first, then worst→best MAE
summary["is_baseline"] = (summary["feature_set"] == "Engineered Features (all)").astype(int)
summary.loc[summary["is_baseline"] == 1, "plot_label"] = r"All Engineered Features are used"
summary.loc[summary["is_baseline"] == 0, "plot_label"] = summary.loc[summary["is_baseline"] == 0, "feature_set"]

base = summary[summary["is_baseline"] == 1]
rest = summary[summary["is_baseline"] == 0].sort_values("MAE_mean", ascending=False)
summary_plot = pd.concat([base, rest], ignore_index=True)

sum_csv = FIGDIR / "engineered_rf_metrics_summary.csv"
summary_plot.drop(columns=["plot_label"], errors="ignore").to_csv(sum_csv, index=False)
print("Saved summary     →", sum_csv)

# ── Plot MAE with error bars (μ ± σ annotations, exact axis bounds) ────
sns.set_style("whitegrid")
plt.rcParams["mathtext.default"] = "regular"

fig, ax = plt.subplots(figsize=(10.5, max(6, 0.35 * len(summary_plot))))

ax.barh(summary_plot["plot_label"], summary_plot["MAE_mean"],
        xerr=summary_plot["MAE_std"].fillna(0.0), capsize=3)

# Get bars (skip errorbar containers)
bars = None
for c in ax.containers:
    if isinstance(c, BarContainer):
        bars = list(c)
if bars is None:
    bars = ax.patches

# Axis limits: use floor/ceil exactly
xmin, xmax = (X_FLOOR, X_CEIL) if (X_FLOOR is not None and X_CEIL is not None) else (None, None)
if xmin is not None and xmax is not None and xmax <= xmin:
    xmax = xmin + 0.2
if xmin is not None: ax.set_xlim(left=xmin)
if xmax is not None: ax.set_xlim(right=xmax)

# Labels + title
ax.set_xlabel("MAE (years)")
ax.set_ylabel("Feature / Model")
ax.set_title(f"Engineered-features benchmark – mean MAE ±1 SD across {len(SEEDS)} seeds\n(lower is better)")

# Annotate μ ± σ
fmt = f"{{:.{DECIMALS}f}} ± {{:.{DECIMALS}f}}"
x0, x1 = ax.get_xlim()
offset = (x1 - x0) * 0.01
for rect, mu, sd in zip(bars, summary_plot["MAE_mean"].values, summary_plot["MAE_std"].fillna(0.0).values):
    x = rect.get_width()
    y = rect.get_y() + rect.get_height() / 2
    ax.text(x + offset, y, fmt.format(mu, sd), va="center", ha="left", fontsize=10)

fig.tight_layout(rect=(0, 0, 1, 0.93))
png_path = FIGDIR / "engineered_rf_mae.png"
pdf_path = png_path.with_suffix(".pdf")
fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("Saved chart       →", png_path)
print("Saved PDF         →", pdf_path)
