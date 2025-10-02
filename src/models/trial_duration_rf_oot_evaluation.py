# src/models/trial_duration_rf_oot_evaluation.py
"""
Evaluate:
  - Mean baseline (DummyRegressor(strategy="mean"))
  - RandomForestRegressor (repeated runs over several seeds)

- Loads features from: data/processed/features_v6.parquet
- Splits by explicit NCT ID lists (newline text files):
    data/splits/expert_filtered_train.txt
    data/splits/expert_filtered_test.txt
- Uses the configured numeric and categorical columns (HAN brief features
  prefixed "brief_han_" are included automatically if present).
- Trains and evaluates:
    * a mean baseline (single fit/predict)
    * RandomForest repeated N_REPEATS times (different seeds)
- Records per-run MAE/RMSE/R² and per-trial error statistics; saves models,
  plots and CSV/text summaries.
"""

import pathlib
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v6.parquet"
FIGDIR = ROOT / "reports" / "figures" / "final_v2" / "filtered_data" / "han";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";               MODELD.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)
target = "duration_days"

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

# include HAN brief features if present
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

# ID-based split (uses explicit NCTID lists)
# Reads train/test NCTIDs from newline-delimited .txt files.
# Falls back to the old date-based split if the files aren't found.

TRAIN_IDS_TXT = pathlib.Path("data/splits/expert_filtered_train.txt")
TEST_IDS_TXT  = pathlib.Path("data/splits/expert_filtered_test.txt")

def _read_ids(p: pathlib.Path) -> set[str]:
    s = (pd.read_csv(p, header=None, dtype=str, names=["nctid"], sep=r"\s+", engine="python")
           .loc[:, "nctid"].str.strip().str.upper())
    return set(s[s.ne("") & s.notna()].unique())

# Try to use the ID files if they exist
if TRAIN_IDS_TXT.exists() and TEST_IDS_TXT.exists():
    train_ids = _read_ids(TRAIN_IDS_TXT)
    test_ids  = _read_ids(TEST_IDS_TXT)

    # Find the NCTID column in df (robust to common name variants)
    nct_col = next((c for c in df.columns if c.lower() in {"nct_id", "nctid", "nctnumber", "nct_number", "nct id", "nct id"}), None)
    if nct_col is None:
        raise KeyError("Could not find an NCTID column in the features dataframe.")

    # Upper-case to match the lists
    nct_series = df[nct_col].astype(str).str.upper()

    # Handle any overlap by assigning overlapping IDs to the TEST set
    overlap = train_ids & test_ids
    if overlap:
        print(f"WARNING: {len(overlap)} NCTIDs appear in both train and test lists. "
              f"Assigning these to TEST to avoid leakage.")
        train_ids = train_ids - overlap  # test wins

    is_train = nct_series.isin(train_ids)
    is_test  = nct_series.isin(test_ids)

    # Keep only rows that are in either list
    keep_mask = is_train | is_test
    if not keep_mask.any():
        raise ValueError("No rows matched the provided NCTID lists.")

    X_all = df[num_cols + cat_cols]
    y_all = df[target]

    X_train, X_test = X_all[is_train], X_all[is_test]
    y_train, y_test = y_all[is_train], y_all[is_test]

    print(f"Training set size : {len(X_train):>7} rows "
          f"(from {len(train_ids)} NCTIDs)")
    print(f"Test set size     : {len(X_test):>7} rows "
          f"(from {len(test_ids)} NCTIDs)")

# Mean baseline
print("\n=== MeanBaseline ===")
dummy = DummyRegressor(strategy="mean")

t0 = time.perf_counter()
dummy.fit(X_train, y_train)                 # ignores X; uses y mean
fit_sec = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred_dummy = dummy.predict(X_test)
pred_sec = time.perf_counter() - t0

mae_dummy  = mean_absolute_error(y_test, y_pred_dummy)
rmse_dummy = np.sqrt(mean_squared_error(y_test, y_pred_dummy))
r2_dummy   = r2_score(y_test, y_pred_dummy)
secs_dummy = fit_sec + pred_sec

print(f"MAE {mae_dummy:7.1f} | RMSE {rmse_dummy:7.1f} | R² {r2_dummy:5.3f} | {secs_dummy:6.1f}s")

# save it too, for completeness
joblib.dump(dummy, MODELD / "mean_baseline.joblib")


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

# RF config & seeds (from expert_compare.py) 
N_REPEATS = 5
SEEDS     = list(range(42, 42 + N_REPEATS))  # 42..46

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

print("Random-Forest parameters:")
for k, v in RF_PARAMS.items():
    print(f"  {k}: {v}")

# Repeat-runs evaluation 
test_ncts = df.loc[is_test, nct_col].astype(str).values  # order matters
per_trial_err = {nct: [] for nct in test_ncts}
mae_days_runs = []

for run_idx, seed in enumerate(SEEDS, start=1):
    print(f"\n=== RandomForest run {run_idx}/{N_REPEATS} (seed={seed}) ===")
    rf   = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    pipe = Pipeline([("pre", pre), ("model", rf)])

    # Fit
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0

    # Predict & errors
    y_pred   = pipe.predict(X_test)
    err_days = np.abs(y_test.values - y_pred)
    mae_days = err_days.mean()
    mae_days_runs.append(mae_days)

    # Accumulate per-trial errors
    for nct, e in zip(test_ncts, err_days):
        per_trial_err[nct].append(e)

    print(f"MAE this run: {mae_days:,.1f} days ({mae_days/365.25:.3f} years) | fit {fit_sec:,.1f}s")

    # Save model per seed
    joblib.dump(pipe, MODELD / f"random_forest_seed{seed}.joblib")

    # Keep existing artifacts for the first run only
    if seed == SEEDS[0]:
        # Parity plot
        plt.figure(figsize=(4, 4))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.2, s=10, edgecolor=None)
        lims = [0, max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, "--k", lw=1)
        plt.xlabel("Actual duration (days)")
        plt.ylabel("Predicted duration (days)")
        plt.title("Random-Forest parity plot")
        plt.tight_layout()
        plt.savefig(FIGDIR / "parity_rf.png", dpi=150)
        plt.close()

        # Impurity-based feature importance
        rf_model   = pipe.named_steps["model"]
        feat_names = pipe.named_steps["pre"].get_feature_names_out()
        imp = (pd.DataFrame({
                "feature": feat_names,
                "importance_gini": rf_model.feature_importances_
              })
              .sort_values("importance_gini", ascending=False))
        imp.to_csv(FIGDIR / "imp_gini_rf.csv", index=False)

        plt.figure(figsize=(6, 5))
        sns.barplot(data=imp.head(20),
                    x="importance_gini", y="feature", palette="crest")
        plt.xlabel("Gini importance")
        plt.title("Random-Forest feature importance – top 20")
        plt.tight_layout()
        plt.savefig(FIGDIR / "imp_gini_rf.png", dpi=150)
        plt.close()


# Summary and files
mae_arr   = np.array(mae_days_runs)
mean_mae  = mae_arr.mean()
std_mae   = mae_arr.std(ddof=1)

print("\nRun-level MAE (days):", np.round(mae_arr, 1))
print(f"Mean MAE = {mean_mae:,.1f} days ± {std_mae:,.1f} (±1 SD)")
print(f"         = {mean_mae/365.25:.3f} years ± {std_mae/365.25:.3f}")

err_stats = (
    pd.DataFrame({
        "NCT": list(per_trial_err.keys()),
        "mean_days": [np.mean(v)        for v in per_trial_err.values()],
        "std_days":  [np.std(v, ddof=1) for v in per_trial_err.values()],
        "n_runs":    [len(v)            for v in per_trial_err.values()],
    })
    .assign(mean_years=lambda d: d["mean_days"]/365.25,
            std_years=lambda d:  d["std_days"]/365.25)
    .set_index("NCT").reindex(test_ncts).reset_index()
)

OUT_DIR = MODELD
SUMMARY_TXT   = OUT_DIR / "rf_repeat_summary.txt"
PER_TRIAL_CSV = OUT_DIR / "rf_per_trial_error_stats.csv"
MAE_RUNS_CSV  = OUT_DIR / "rf_mae_days_runs.csv"

# Save CSVs
err_stats.to_csv(PER_TRIAL_CSV, index=False)
pd.DataFrame({"mae_days": mae_arr}).to_csv(MAE_RUNS_CSV, index=False)

# Save plain-text summary 
lines = []
lines.append("Absolute error per held-out trial  (mean ± 1 SD, years)")
for row in err_stats.itertuples(index=False):
    lines.append(f"{row.NCT}: {row.mean_years:.3f}±{row.std_years:.3f}")
lines.append("")
lines.append("Run-level MAE (days): " + ", ".join([f"{d:.1f}" for d in mae_arr]))
lines.append("")
lines.append(f"Mean MAE = {mean_mae:.1f} days ± {std_mae:.1f} (1 SD)")
lines.append(f"         = {mean_mae/365.25:.3f} years ± {std_mae/365.25:.3f}")
lines.append("")
lines.append("Random-Forest parameters:")
lines.extend([f"  {k}: {v}" for k, v in RF_PARAMS.items()])
SUMMARY_TXT.write_text("\n".join(lines))

print(f"\nSaved per-trial stats to: {PER_TRIAL_CSV}")
print(f"Saved MAE runs to: {MAE_RUNS_CSV}")
print(f"Saved summary text to: {SUMMARY_TXT}")
