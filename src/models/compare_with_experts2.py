#!/usr/bin/env python
"""
src/models/compare_with_experts2.py
────────────────────────────────────────────────────────────────────────────
Train a Random-Forest 5x, each time evaluating on a fixed
test set of trials (by NCT IDs). Train/test splits are driven by two text
files of NCT IDs and matched on the Parquet column 'nct_id'.

Also prints a feature report:
  • raw numeric/categorical feature lists (preview)
  • transformed feature count and a sample (full list saved to file)
  • Saves per-seed predictions (true, pred, |err|) per test NCT.
  • Aggregates per-trial true, pred mean±SD, |err| mean±SD across seeds.
"""

from __future__ import annotations
import pathlib, time, json, joblib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Configuration
N_REPEATS = 5
SEEDS     = list(range(42, 42 + N_REPEATS))  # 42 … 51

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=10,
    max_features=1.0,   # same for meta-only and meta+HAN
    bootstrap=True,
    max_samples=0.5,
    criterion="squared_error",
    n_jobs=-1,
)

ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v6_han.parquet"
MODELD = ROOT / "models"; MODELD.mkdir(exist_ok=True)

# paths for outputs
OUT_DIR = MODELD
SUMMARY_TXT   = OUT_DIR / "expert_compare_summary_179_han_extratest.txt"
PER_TRIAL_CSV = OUT_DIR / "per_trial_error_stats_179_han_extratest.csv"        # aggregated
MAE_RUNS_CSV  = OUT_DIR / "mae_days_runs_179_han_extratest.csv"
TRANSFORMED_FEATURES_TXT = OUT_DIR / "feature_names_179_han_extratest.txt"
PRED_BY_SEED_CSV = OUT_DIR / "per_trial_predictions_by_seed_179_han_extratest.csv"

# target & columns
target   = "duration_days"
nct_col  = "NCT_ID"  

# Load dataframe
df = pd.read_parquet(DATA)

# Normalize NCT column now (string, trimmed, uppercase) for stable matching
if nct_col not in df.columns:
    raise KeyError(f"Column '{nct_col}' not found in the Parquet file: {DATA}")
df[nct_col] = df[nct_col].astype(str).str.strip().str.upper()

# Build feature lists AFTER loading df (so we can inspect columns)
num_cols = [
    #"# patients",
    #"country_n",
    #"site_n",
    #"assessments_n",
    "primary_out_n",
    "secondary_out_n",
    "other_out_n",
    #"start_date",
    #"start_year",
    #"patients_per_site",
    "num_arms",
    "masking_flag",
    "placebo_flag",
    #"active_prob",
    "elig_crit_n",
    "safety_cuts_flag",
    "age_min",
    "age_max",
    "age_range",
    "randomized_flag",
    "fda_drug_flag",
    "fda_device_flag",
    "freq_in_window",
    #"novelty_score",
    #"complexity_score_100",
    #"attractiveness_score_100",
    ]
    
num_cols += [c for c in df.columns if c.startswith("brief_han_")]

cat_cols = [
    "phase",
    "sponsor_class",
    "condition_top",
    "therapeutic_area",
    "intervention_type",
    #"assessments_complexity",
    #"global_trial",
    "masking_level",
    #"population_class",
    "cohort_design",
    "study_type",
    "allocation",
    ]


# Guard: drop any columns that don't exist (prevents KeyError downstream)
present = set(df.columns)
missing_num = [c for c in num_cols if c not in present]
missing_cat = [c for c in cat_cols if c not in present]
if missing_num or missing_cat:
    print("Dropping missing columns:", {"num": missing_num, "cat": missing_cat})
num_cols = [c for c in num_cols if c in present]
cat_cols = [c for c in cat_cols if c in present]

# Preprocessor
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

# Feature report 
def _preview_list(title: str, items: list[str], max_n: int = 20) -> None:
    print(f"{title} ({len(items)}):")
    if len(items) <= max_n:
        print("  " + ", ".join(items))
    else:
        head = ", ".join(items[:max_n])
        print(f"  {head}, ... (+{len(items) - max_n} more)")

print("\n── Feature selection (raw columns) ──")
_preview_list("Numeric columns", num_cols)
_preview_list("Categorical columns", cat_cols)

# Load train/test IDs from text files and split by nct_id
def load_ids_txt(path: pathlib.Path) -> list[str]:
    """Read IDs (one per line), normalize to uppercase, keep order & uniqueness."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    seen, ordered = set(), []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            su = s.upper()
            if su not in seen:
                seen.add(su)
                ordered.append(su)
    return ordered

# Default to repo paths
TRAIN_IDS_TXT = ROOT / "data" / "splits" / "expert_filtered_train.txt"
TEST_IDS_TXT  = ROOT / "data" / "splits" / "expert_filtered_test.txt"

TRAIN_NCTS_LIST = load_ids_txt(TRAIN_IDS_TXT)
TEST_NCTS_LIST  = load_ids_txt(TEST_IDS_TXT)
TRAIN_NCTS, TEST_NCTS = set(TRAIN_NCTS_LIST), set(TEST_NCTS_LIST)

# Handle overlaps: keep in TRAIN, drop from TEST (and its ordered list)
overlap = TRAIN_NCTS & TEST_NCTS
if overlap:
    print(f"WARNING: {len(overlap)} IDs are in BOTH train and test lists. "
          f"Assigning to TRAIN and removing from TEST.")
    TEST_NCTS_LIST = [x for x in TEST_NCTS_LIST if x not in overlap]
    TEST_NCTS -= overlap

# Build splits
is_train = df[nct_col].isin(TRAIN_NCTS)
is_test  = df[nct_col].isin(TEST_NCTS)

train_df = df.loc[ is_train].copy()
test_df  = df.loc[ is_test ].copy()

# Diagnostics for missing IDs
present_train_ids = set(train_df[nct_col].unique())
present_test_ids  = set(test_df[nct_col].unique())
missing_train = TRAIN_NCTS - present_train_ids
missing_test  = TEST_NCTS  - present_test_ids
if missing_train:
    print(f"NOTE: {len(missing_train)} train IDs not found in Parquet (e.g., {sorted(list(missing_train))[:5]})")
if missing_test:
    print(f"NOTE: {len(missing_test)} test IDs not found in Parquet (e.g., {sorted(list(missing_test))[:5]})")

# Keep output order following the TEST IDs file, but only those present
EXPERT_NCTS = [n for n in TEST_NCTS_LIST if n in present_test_ids]

X_train, y_train = train_df[num_cols + cat_cols], train_df[target]
X_test , y_test  =  test_df[num_cols + cat_cols],  test_df[target]

print(f"\nTrain on {len(train_df):,} trials  |  Test on {len(test_df):,} trials "
      f"(unique test NCTs present: {len(present_test_ids)})")

# HAN coverage diagnostics (optional but helpful)
han_cols = [c for c in df.columns if c.startswith("brief_han_")]
if han_cols:
    miss_tr = train_df[han_cols].isna().any(axis=1).sum()
    miss_te = test_df[han_cols].isna().any(axis=1).sum()
    print(f"HAN dims detected: {len(han_cols)} | missing in train: {miss_tr} | missing in test: {miss_te}")

# ── Print RF hyper-parameters so they appear in console logs
print("\nRandom-Forest parameters:")
for k, v in RF_PARAMS.items():
    print(f"  {k}: {v}")

# Container FOR RESULTS
mae_days_runs   = []                                  # one MAE per run
per_trial_err   = {nct: [] for nct in EXPERT_NCTS}    # accumulate |error| per trial
per_trial_pred  = {nct: [] for nct in EXPERT_NCTS}    # accumulate predictions per trial
per_trial_true  = {
    nct: float(test_df.loc[test_df[nct_col] == nct, target].iloc[0])
    for nct in EXPERT_NCTS
}
pred_by_seed_rows = []  # long-form audit rows: one per (seed, NCT)


# Main loop
for run_idx, seed in enumerate(SEEDS, start=1):
    print(f"\n─── Run {run_idx}/{N_REPEATS}  (seed={seed}) ───")

    rf   = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    pipe = Pipeline([("pre", pre), ("rf", rf)])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    print(f"Model fitted in {time.perf_counter() - t0:,.1f}s")

    # ── FEATURE REPORT (TRANSFORMED) — print once and save full list
    if run_idx == 1:
        try:
            feat_names = pipe.named_steps["pre"].get_feature_names_out()
            print("\n── Transformed feature space ──")
            print(f"Total features after preprocessing: {len(feat_names)}")
            with open(TRANSFORMED_FEATURES_TXT, "w", encoding="utf-8") as f:
                for n in feat_names:
                    f.write(str(n) + "\n")
            sample = ", ".join(map(str, feat_names[:30]))
            print(f"Saved full list to: {TRANSFORMED_FEATURES_TXT}")
            print(f"Sample (first 30): {sample}")
        except Exception as e:
            print("NOTE: Could not extract transformed feature names from preprocessor:", repr(e))

    joblib.dump(pipe, MODELD / f"rf_expert_compare_seed{seed}.joblib")

    # predictions & errors
    y_pred   = pipe.predict(X_test)
    abs_err  = np.abs(y_test.values - y_pred)  # absolute error
    mae_days = abs_err.mean()
    mae_days_runs.append(mae_days)

    for nct, yt, yp, ae in zip(test_df[nct_col].values, y_test.values, y_pred, abs_err):
        if nct in per_trial_err:   # only record ordered, present NCTs
            per_trial_err[nct].append(float(ae))
            per_trial_pred[nct].append(float(yp))
            pred_by_seed_rows.append({
                "seed": seed,
                "NCT": nct,
                "true_days": float(yt),
                "pred_days": float(yp),
                "abs_err_days": float(ae),
            })

    print(f"MAE this run: {mae_days:,.1f} days  ({mae_days/365.25:.3f} years)")


# PER-TRIAL STATS (truth, prediction, and error aggregated over seeds)
def _std_or_zero(vals: list[float]) -> float:
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

agg = pd.DataFrame({
    "NCT": list(per_trial_pred.keys()),
    "true_days": [per_trial_true[n] for n in per_trial_pred.keys()],
    "pred_days_mean": [float(np.mean(per_trial_pred[n])) for n in per_trial_pred.keys()],
    "pred_days_std":  [_std_or_zero(per_trial_pred[n]) for n in per_trial_pred.keys()],
    "abs_err_days_mean": [float(np.mean(per_trial_err[n])) for n in per_trial_pred.keys()],
    "abs_err_days_std":  [_std_or_zero(per_trial_err[n]) for n in per_trial_pred.keys()],
}).assign(
    true_years=lambda d: d["true_days"] / 365.25,
    pred_years_mean=lambda d: d["pred_days_mean"] / 365.25,
    pred_years_std=lambda d: d["pred_days_std"] / 365.25,
    abs_err_years_mean=lambda d: d["abs_err_days_mean"] / 365.25,
    abs_err_years_std=lambda d: d["abs_err_days_std"] / 365.25,
).set_index("NCT").reindex(EXPERT_NCTS).reset_index()

print("\nPer-trial results (true vs predicted, and absolute error; mean ± 1 SD over seeds)")
for row in agg.itertuples(index=False):
    print(
        f"{row.NCT}: true={row.true_days:.0f}d ({row.true_years:.2f}y) | "
        f"pred={row.pred_days_mean:.0f}±{row.pred_days_std:.0f}d "
        f"({row.pred_years_mean:.2f}±{row.pred_years_std:.2f}y) | "
        f"|err|={row.abs_err_days_mean:.1f}±{row.abs_err_days_std:.1f}d "
        f"({row.abs_err_years_mean:.3f}±{row.abs_err_years_std:.3f}y)"
    )

# RUN-LEVEL MAE DISTRIBUTION
mae_days_arr   = np.array(mae_days_runs)
mean_mae_days  = mae_days_arr.mean()
std_mae_days   = mae_days_arr.std(ddof=1)

print("\nRun-level MAE (days):", np.round(mae_days_arr, 1))
print(f"\nMean MAE = {mean_mae_days:,.1f} days ± {std_mae_days:,.1f}  (±1 SD)")
print(f"           = {mean_mae_days/365.25:.3f} years ± {std_mae_days/365.25:.3f}")

# SAVE OUTPUT TO FILES
# 1) Per-seed long-form predictions (audit-friendly)
pd.DataFrame(pred_by_seed_rows).to_csv(PRED_BY_SEED_CSV, index=False)

# 2) Aggregated per-trial table (truth, pred, error; mean ± SD)
agg.to_csv(PER_TRIAL_CSV, index=False)

# 3) Run-level MAE values
pd.DataFrame({"mae_days": mae_days_arr}).to_csv(MAE_RUNS_CSV, index=False)

# Append RF params to CSV footers (optional)
for path in [PER_TRIAL_CSV, MAE_RUNS_CSV]:
    with open(path, "a", encoding="utf-8") as f:
        f.write("# RF_PARAMS: " + json.dumps(RF_PARAMS) + "\n")

# 4) Plain-text summary
lines = []
lines.append("Absolute error per held-out trial  (mean ± 1 SD, years)")
for row in agg.itertuples(index=False):
    lines.append(f"{row.NCT}: {row.abs_err_years_mean:.3f}±{row.abs_err_years_std:.3f}")

lines.append("")
lines.append("Sample per-trial predictions (first 5):")
for row in agg.head(5).itertuples(index=False):
    lines.append(
        f"{row.NCT}: true={row.true_days:.0f}d, "
        f"pred={row.pred_days_mean:.0f}±{row.pred_days_std:.0f}d, "
        f"|err|={row.abs_err_days_mean:.1f}d"
    )

lines.append("")
lines.append("Run-level MAE (days): " + ", ".join([f"{d:.1f}" for d in mae_days_arr]))
lines.append("")
lines.append(f"Mean MAE = {mean_mae_days:.1f} days ± {std_mae_days:.1f} (1 SD)")
lines.append(f"         = {mean_mae_days/365.25:.3f} years ± {std_mae_days/365.25:.3f}")
lines.append("")
lines.append("Random-Forest parameters:")
lines.extend([f"  {k}: {v}" for k, v in RF_PARAMS.items()])

SUMMARY_TXT.write_text("\n".join(lines))

print(f"\nSaved per-trial predictions (by seed) to: {PRED_BY_SEED_CSV}")
print(f"Saved per-trial aggregates to:            {PER_TRIAL_CSV}")
print(f"Saved MAE runs to:                        {MAE_RUNS_CSV}")
print(f"Saved transformed feature names to:       {TRANSFORMED_FEATURES_TXT}")
print(f"Saved summary text to:                    {SUMMARY_TXT}")
