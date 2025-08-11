# src/models/ablation_rf.py
"""
Random‑Forest ablation study 
---------------------------------------------------------------------
* 150‑tree **RandomForestRegressor** experiments
* Baseline models:
    • All features  (RAW + DERIVED + ENGINEERED)
    • Raw only      (just RAW)
    • Raw + Derived (RAW ∪ DERIVED)
* Leave‑one‑out ablation for every feature in **ALL_COLS**
* Numeric vs. categorical columns are now **inferred from the dataframe dtypes** – the hard‑coded
  `num_cols` / `cat_cols` lists have been retired to avoid future drift.
* Outputs:
    reports/figures/filtered_data/ablation_rf_metrics.csv
    reports/figures/filtered_data/ablation_rf_mae.png
"""

from __future__ import annotations
import pathlib, time, warnings, joblib
import numpy as np, pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v4.parquet"
FIGDIR = ROOT / "reports" / "figures" / "filtered_data";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";                                 MODELD.mkdir(exist_ok=True)


df = pd.read_parquet(DATA)
TARGET = "duration_days"

# Convert date columns to numeric (seconds since UNIX epoch) 
for col in ["start_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    ts = df[col].view("int64") # nanoseconds → int64
    ts[df[col].isna()] = np.nan # NaT → NaN
    df[f"{col}_ts"] = ts / 1e9 # seconds since epoch
# Keep original start_date for the temporal split only

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

# Consolidate & validate features

ALL_FEATURES = RAW + DERIVED + ENGINEERED
ALL_COLS = [c for c in ALL_FEATURES if c in df.columns]
missing = sorted(set(ALL_FEATURES) - set(ALL_COLS))
if missing:
    print("Skipping missing columns:", ", ".join(missing))

# Temporal train / test split 

cutoff = pd.Timestamp("2019-01-01")
is_test = df["start_date"] >= cutoff

X_train, X_test = df.loc[~is_test, ALL_COLS], df.loc[is_test, ALL_COLS]
y_train, y_test = df.loc[~is_test, TARGET],   df.loc[is_test, TARGET]

print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

# Preprocessing 

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])

RF_PARAMS = dict(
    n_estimators=150,
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
)

# Helper: split numeric vs categorical on‑the‑fly 

def split_num_cat(columns: list[str]) -> tuple[list[str], list[str]]:
    """Return ([numeric_cols], [categorical_cols]) for *columns* based on df dtypes."""
    num, cat = [], []
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat

# Evaluation function

def evaluate_feature_set(name: str, features: list[str]) -> dict:
    """Train/test a RF on *features* and return a metrics dict."""
    kept = list(dict.fromkeys([c for c in features if c in df.columns]))  # dedupe & ensure exists
    if not kept:
        raise ValueError(f"No valid columns for feature set '{name}' – check names")

    num_cols, cat_cols = split_num_cat(kept)

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], verbose_feature_names_out=False)

    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(**RF_PARAMS)),
    ])

    t0 = time.perf_counter()
    pipe.fit(X_train[kept], y_train)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(X_test[kept])

    mae_days  = mean_absolute_error(y_test, y_pred)
    rmse_days = mean_squared_error(y_test, y_pred) ** 0.5


    return {
        "left_out": name,               
        "MAE":  mae_days  / 365,
        "RMSE": rmse_days / 365,
        "R2":   r2_score(y_test, y_pred),
        "seconds": fit_sec,
    }

# Run experiments 
metrics: list[dict] = []

# Baseline models 
BASELINE_SETS: dict[str, list[str]] = {
    "All features": ALL_COLS,
    "Raw only": RAW,
    "Raw+Derived": RAW + DERIVED,
}

for name, cols in BASELINE_SETS.items():
    metrics.append(evaluate_feature_set(name, cols))

# Leave‑one‑out ablation 

for left_out in tqdm(ALL_COLS, desc="Ablation", unit="feature"):
    kept = [c for c in ALL_COLS if c != left_out]
    metrics.append(evaluate_feature_set(left_out, kept))

#  Save results & plot 

results = pd.DataFrame(metrics).sort_values("MAE", ascending=False)
out_csv = FIGDIR / "ablation_rf_metrics.csv"
results.to_csv(out_csv, index=False)
print("Saved metrics →", out_csv)

plt.figure(figsize=(8, max(6, 0.25 * len(results))))
sns.barplot(data=results, x="MAE", y="left_out", palette="crest")
plt.xlabel("MAE (years)")
plt.xlim(results["MAE"].min() * 0.95, results["MAE"].max() * 1.05)
plt.ylabel("Feature left out / Model")
plt.title("Random‑Forest ablation & baseline comparison – lower MAE is better")
plt.tight_layout()
plt.savefig(FIGDIR / "ablation_rf_mae.png", dpi=150)
plt.close()

# 95 % confidence-interval plot 
if results["left_out"].duplicated().any():          # only meaningful if ran repeats
    # 1. aggregate mean, SD, n
    grp = (results
           .groupby("left_out", as_index=False)
           .agg(MAE_mean=("MAE", "mean"),
                SD=("MAE", "std"),
                n=("MAE", "size")))
    # 2. 95 % CI half-width = 1.96 × SEM
    grp["CI95"] = 1.96 * grp["SD"] / np.sqrt(grp["n"])
    grp.sort_values("MAE_mean", ascending=False, inplace=True)

    # 3. plot bar + error bars
    plt.figure(figsize=(8, max(6, 0.25 * len(grp))))
    plt.barh(grp["left_out"],
             grp["MAE_mean"],
             xerr=grp["CI95"],
             capsize=3)
    plt.xlabel("MAE (years)")
    plt.xlim(grp["MAE_mean"].min() * 0.95,
             grp["MAE_mean"].max() * 1.05)
    plt.ylabel("Feature left out / Model")
    plt.title("Random-Forest ablation – mean MAE ± 95 % CI")
    plt.tight_layout()

    ci_png = FIGDIR / "ablation_rf_mae_ci95.png"
    plt.savefig(ci_png, dpi=150)
    plt.close()
    print("Saved 95 % CI chart →", ci_png)
else:
    print("Skipped CI plot – only one run per feature (no variance available).")

print("Saved chart→", FIGDIR / "ablation_rf_mae.png")
