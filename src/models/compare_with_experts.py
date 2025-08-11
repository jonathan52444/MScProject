#!/usr/bin/env python
"""
src/models/expert_compare.py
────────────────────────────────────────────────────────────────────────────
Train a Random‑Forest 20× (seeds 42‑61), each time holding out the 15 trials
that already have expert duration estimates.  For every held‑out trial we
accumulate the absolute error so we can print:

    NCT01234567: 0.92±0.21   <-- mean ± 1 SD (years)

At the end we also show the MAE distribution across runs and its
overall mean ± 1 SD (days & years), and save all outputs to files.
"""

from __future__ import annotations
import pathlib, time, joblib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
N_REPEATS = 20
SEEDS     = list(range(42, 42 + N_REPEATS))       # 42 … 61

RF_PARAMS = dict(
    n_estimators     = 500,
    n_jobs           = -1,
    max_features     = "sqrt",
    min_samples_leaf = 2,
    bootstrap        = True,
)

EXPERT_NCTS = [
    "NCT06011733", "NCT05292131", "NCT03928704", "NCT03928743",
    "NCT04242446", "NCT04242498", "NCT03536884", "NCT05020249",
    "NCT06312566", "NCT01550003", 
    "NCT04740814", "NCT04163016",
    "NCT04294667", "NCT06315335", "NCT04643457","NCT04867642", "NCT03340064", "NCT06533475", "NCT04658186",
    "NCT05845645", "NCT03728933", "NCT04875975", "NCT05643794",
    "NCT05681715", "NCT04651153", "NCT05514873",
]

ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v4.parquet"
MODELD = ROOT / "models"; MODELD.mkdir(exist_ok=True)

# paths for outputs
OUT_DIR = MODELD
SUMMARY_TXT = OUT_DIR / "expert_compare_summary.txt"
PER_TRIAL_CSV = OUT_DIR / "per_trial_error_stats.csv"
MAE_RUNS_CSV = OUT_DIR / "mae_days_runs.csv"

# target & columns
target   = "duration_days"
nct_col  = "NCT ID"

num_cols = [
    "# patients", "country_n", "site_n", "assessments_n", "start_year",
    "patients_per_site", "num_arms", "masking_flag", "placebo_flag",
    "active_prob", "elig_crit_n", "age_min", "age_max", "age_range",
    "randomized_flag", "fda_drug_flag", "fda_device_flag", "freq_in_window",
    "novelty_score", "complexity_score_100", "attractiveness_score_100",
]
cat_cols = [
    "phase", "sponsor_class", "condition_top", "therapeutic_area",
    "intervention_type", "assessments_complexity", "global_trial",
    "masking_level", "population_class", "cohort_design",
    "study_type", "allocation",
]

# ──────────────────────────────────────────────────────────────────────────
# PREPROCESSOR
# ──────────────────────────────────────────────────────────────────────────
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], verbose_feature_names_out=False)

# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA & SPLIT
# ──────────────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA)
is_test  = df[nct_col].isin(EXPERT_NCTS)
train_df = df.loc[~is_test]
test_df  = df.loc[ is_test]
X_train, y_train = train_df[num_cols + cat_cols], train_df[target]
X_test , y_test  =  test_df[num_cols + cat_cols],  test_df[target]

print(f"Train on {len(train_df):,} trials  |  Test on {len(test_df):,} expert‑predicted trials")

# ──────────────────────────────────────────────────────────────────────────
# CONTAINERS FOR RESULTS
# ──────────────────────────────────────────────────────────────────────────
mae_days_runs   = []                                  # one MAE per run
per_trial_err   = {nct: [] for nct in EXPERT_NCTS}    # accumulate per‑trial errors

# ──────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────
for run_idx, seed in enumerate(SEEDS, start=1):
    print(f"\n─── Run {run_idx}/{N_REPEATS}  (seed={seed}) ───")

    rf   = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    pipe = Pipeline([("pre", pre), ("rf", rf)])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    print(f"Model fitted in {time.perf_counter() - t0:,.1f}s")

    joblib.dump(pipe, MODELD / f"rf_expert_compare_seed{seed}.joblib")

    # predictions & errors
    y_pred      = pipe.predict(X_test)
    err_days    = np.abs(y_test.values - y_pred)      # absolute error
    mae_days    = err_days.mean()

    mae_days_runs.append(mae_days)
    for nct, e in zip(test_df[nct_col].values, err_days):
        per_trial_err[nct].append(e)

    print(f"MAE this run: {mae_days:,.1f} days  ({mae_days/365:.3f} years)")

# ──────────────────────────────────────────────────────────────────────────
# PER‑TRIAL STATS
# ──────────────────────────────────────────────────────────────────────────
err_stats = (
    pd.DataFrame({
        "NCT"          : list(per_trial_err.keys()),
        "mean_days"    : [np.mean(v)         for v in per_trial_err.values()],
        "std_days"     : [np.std(v, ddof=1)  for v in per_trial_err.values()],
        "n_runs"       : [len(v)             for v in per_trial_err.values()],
    })
    .assign(std_years=lambda d: d["std_days"] / 365,
            mean_years=lambda d: d["mean_days"] / 365)
    .set_index("NCT")
    .reindex(EXPERT_NCTS)
    .reset_index()
)

print("\nAbsolute error per held‑out trial  (mean ± 1 SD, years)")
for row in err_stats.itertuples(index=False):
    print(f"{row.NCT}: {row.mean_years:.3f}±{row.std_years:.3f}")
print("(* ± values are standard deviations across the 20 seeds)")

# ──────────────────────────────────────────────────────────────────────────
# RUN‑LEVEL MAE DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────
mae_days_arr   = np.array(mae_days_runs)
mean_mae_days  = mae_days_arr.mean()
std_mae_days   = mae_days_arr.std(ddof=1)

print("\nRun‑level MAE (days):", np.round(mae_days_arr, 1))
print(f"\nMean MAE = {mean_mae_days:,.1f} days ± {std_mae_days:,.1f}  (±1 SD)")
print(f"           = {mean_mae_days/365:.3f} years ± {std_mae_days/365:.3f}")

# ──────────────────────────────────────────────────────────────────────────
# SAVE OUTPUT TO FILES
# ──────────────────────────────────────────────────────────────────────────
# 1) CSV of per-trial error statistics
err_stats.to_csv(PER_TRIAL_CSV, index=False)

# 2) CSV of run-level MAE values
pd.DataFrame({"mae_days": mae_days_arr}).to_csv(MAE_RUNS_CSV, index=False)

# 3) Plain-text summary
lines = []
lines.append("Absolute error per held-out trial  (mean ± 1 SD, years)")
for row in err_stats.itertuples(index=False):
    lines.append(f"{row.NCT}: {row.mean_years:.3f}±{row.std_years:.3f}")
lines.append("")
lines.append("Run-level MAE (days): " + ", ".join([f"{d:.1f}" for d in mae_days_arr]))
lines.append("")
lines.append(f"Mean MAE = {mean_mae_days:.1f} days ± {std_mae_days:.1f} (1 SD)")
lines.append(f"         = {mean_mae_days/365:.3f} years ± {std_mae_days/365:.3f}")

SUMMARY_TXT.write_text("\n".join(lines))

print(f"\nSaved per-trial stats to: {PER_TRIAL_CSV}")
print(f"Saved MAE runs to: {MAE_RUNS_CSV}")
print(f"Saved summary text to: {SUMMARY_TXT}")
