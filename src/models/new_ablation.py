"""
Random‑Forest ablation study – repeated runs with error bars
------------------------------------------------------------
* **10×** repeated experiments per feature set (configurable via `N_REPEATS`)
* **500‑tree** *RandomForestRegressor* (higher capacity than previous 150‑tree model)
* Baseline models:
    • All features  (RAW + DERIVED + ENGINEERED)
    • Raw only      (just RAW)
    • Raw + Derived (RAW ∪ DERIVED)
* Leave‑one‑out ablation for every feature in **ALL_COLS**
* Numeric vs. categorical columns are **inferred from the dataframe dtypes** – no hard‑coded lists
* **Error bars** (±1 SD) visualised on MAE for every experiment
* Outputs (new/changed):
    reports/figures/filtered_data/ablation_rf_metrics.csv        (all repetitions)
    reports/figures/filtered_data/ablation_rf_metrics_summary.csv (mean ± sd per feature set)
    reports/figures/filtered_data/ablation_rf_mae.png            (bar + error bars)
"""

from __future__ import annotations
import pathlib, time, warnings, joblib, itertools
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

# Config

N_REPEATS   = 20                 # number of different random seeds per feature‑set
SEEDS       = list(range(42, 42 + N_REPEATS))
PLOT_ERRORBARS = True            # generate barplot with ±1 SD error bars

RF_PARAMS = dict(                # higher‑capacity model than v1
    n_estimators      = 500,
    n_jobs            = -1,
    max_features      = "sqrt",
    min_samples_leaf  = 2,
    bootstrap         = True,
)

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
    df[f"{col}_ts"] = ts / 1e9  # seconds since epoch
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
ALL_COLS     = [c for c in ALL_FEATURES if c in df.columns]
missing      = sorted(set(ALL_FEATURES) - set(ALL_COLS))
if missing:
    print("Skipping missing columns:", ", ".join(missing))

# Temporal train / test split

cutoff = pd.Timestamp("2019-01-01")
is_test = df["start_date"] >= cutoff

X_train, X_test = df.loc[~is_test, ALL_COLS], df.loc[is_test, ALL_COLS]
y_train, y_test = df.loc[~is_test, TARGET],   df.loc[is_test, TARGET]

print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

# Pre‑processing pipelines 

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])

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

def evaluate_feature_set(name: str, features: list[str], seed: int) -> dict:
    """Train/test a RF on *features* using *seed* and return a metrics dict."""
    kept = list(dict.fromkeys([c for c in features if c in df.columns]))
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
        "left_out": name,             
        "seed":      seed,
        "MAE":       mae_days  / 365,
        "RMSE":      rmse_days / 365,
        "R2":        r2_score(y_test, y_pred),
        "seconds":   fit_sec,
    }

# Run experiments 

metrics: list[dict] = []

BASELINE_SETS: dict[str, list[str]] = {
    "All features": ALL_COLS,
    "Raw only":     RAW,
    "Raw+Derived":  RAW + DERIVED,
}

# Iterate over seeds and feature sets using tqdm for transparency 

iterables = itertools.product(SEEDS, BASELINE_SETS.items())
print(f"Running {len(SEEDS)} seeds × {len(BASELINE_SETS) + len(ALL_COLS)} feature sets")

for seed, (name, cols) in tqdm(list(iterables), desc="Baselines", unit="run"):
    metrics.append(evaluate_feature_set(name, cols, seed))

for seed in tqdm(SEEDS, desc="Ablation (outer loop)"):
    for left_out in tqdm(ALL_COLS, leave=False, desc="features", unit="feat"):
        kept = [c for c in ALL_COLS if c != left_out]
        metrics.append(evaluate_feature_set(left_out, kept, seed))

# Save raw results

results = pd.DataFrame(metrics)
raw_csv = FIGDIR / "ablation_rf_metrics.csv"
results.to_csv(raw_csv, index=False)
print("Saved raw metrics →", raw_csv)

# Aggregate & visualize (mean ± sd)

summary = (
    results.groupby("left_out", as_index=False)
           .agg(MAE_mean=("MAE", "mean"),
                MAE_std =("MAE", "std"),
                RMSE_mean=("RMSE", "mean"),
                R2_mean=("R2", "mean"))
           .sort_values("MAE_mean", ascending=False)
)

sum_csv = FIGDIR / "ablation_rf_metrics_summary.csv"
summary.to_csv(sum_csv, index=False)
print("Saved summary     →", sum_csv)

# Plot MAE with error bars

if PLOT_ERRORBARS:
    plt.figure(figsize=(8, max(6, 0.25 * len(summary))))
    plt.barh(summary["left_out"], summary["MAE_mean"], xerr=summary["MAE_std"], capsize=3)
    plt.xlabel("MAE (years)")
    plt.xlim(summary["MAE_mean"].min() * 0.95, summary["MAE_mean"].max() * 1.05)
    plt.ylabel("Feature left out / Model")
    plt.title("Random‑Forest ablation – mean MAE ±1 SD across seeds\n(lower is better)")
    plt.tight_layout()
    fig_path = FIGDIR / "ablation_rf_mae.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved chart       →", fig_path)
else:
    print("Skipping plot as PLOT_ERRORBARS = False")
