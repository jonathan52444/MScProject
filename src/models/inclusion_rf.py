"""
Random‑Forest **inclusion study** – engineered features
------------------------------------------------------
Baseline = RAW + DERIVED.  We then **add each ENGINEERED feature individually**
(on top of RAW+DERIVED) and measure the impact on MAE.

* Repeated `N_REPEATS` times (different random seeds) for robust error bars.
* 500‑tree `RandomForestRegressor` (same capacity as new ablation script).
* Outputs
    reports/figures/filtered_data/inclusion_rf_metrics.csv           (raw per‑run)
    reports/figures/filtered_data/inclusion_rf_metrics_summary.csv   (mean ± sd)
    reports/figures/filtered_data/inclusion_rf_mae.png               (bar ± 95 % CI)
"""

from __future__ import annotations
import pathlib, time, warnings, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Config 
N_REPEATS = 20
SEEDS     = list(range(42, 42 + N_REPEATS)) 

RF_PARAMS = dict(
    n_estimators     = 500,
    n_jobs           = -1,
    max_features     = "sqrt",
    min_samples_leaf = 2,
    bootstrap        = True,
)

# ─── Paths ────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v4.parquet"
FIGDIR = ROOT / "reports" / "figures" / "filtered_data";  FIGDIR.mkdir(parents=True, exist_ok=True)

# Data 

df = pd.read_parquet(DATA)
TARGET = "duration_days"

for col in ["start_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    ts = df[col].view("int64")
    ts[df[col].isna()] = np.nan
    df[f"{col}_ts"] = ts / 1e9

# Feature groups
RAW = [
    "# patients", "start_date_ts", "age_min", "age_max",
    "phase", "study_type", "allocation",
]

DERIVED = [
    "site_n", "country_n",
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

ALL_FEATURES = RAW + DERIVED + ENGINEERED
ALL_COLS     = [c for c in ALL_FEATURES if c in df.columns]
missing      = sorted(set(ALL_FEATURES) - set(ALL_COLS))
if missing:
    print("Skipping missing columns:", ", ".join(missing))

#Train / test split 
cutoff = pd.Timestamp("2019-01-01")
is_test = df["start_date"] >= cutoff

X_train, X_test = df.loc[~is_test, ALL_COLS], df.loc[is_test, ALL_COLS]
y_train, y_test = df.loc[~is_test, TARGET],   df.loc[is_test, TARGET]

print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

# Pre-processing
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])

# Helpers 

def split_num_cat(columns: list[str]) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat

def evaluate(name: str, cols: list[str], seed: int) -> dict:
    num_cols, cat_cols = split_num_cat(cols)

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], verbose_feature_names_out=False)

    rf = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    pipe = Pipeline([("pre", pre), ("rf", rf)])

    t0 = time.perf_counter()
    pipe.fit(X_train[cols], y_train)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(X_test[cols])

    mae_days  = mean_absolute_error(y_test, y_pred)
    rmse_days = mean_squared_error(y_test, y_pred) ** 0.5

    return {
        "feature_set": name,
        "seed":        seed,
        "MAE":         mae_days / 365,
        "RMSE":        rmse_days / 365,
        "R2":          r2_score(y_test, y_pred),
        "seconds":     fit_sec,
    }

# Run inclusion study 
metrics: list[dict] = []
BASE_PLUS = RAW + DERIVED

iters = itertools.product(SEEDS, [None] + ENGINEERED)  # None → baseline
print(f"Running {len(SEEDS)} seeds × {1 + len(ENGINEERED)} feature sets")

for seed, feat in tqdm(list(iters), desc="Inclusion", unit="run"):
    if feat is None:
        name = "Base (Raw+Derived)"
        cols = BASE_PLUS
    else:
        name = f"+ {feat}"
        cols = BASE_PLUS + [feat]
    metrics.append(evaluate(name, cols, seed))

# Save raw metrics 
results = pd.DataFrame(metrics)
raw_csv = FIGDIR / "inclusion_rf_metrics.csv"
results.to_csv(raw_csv, index=False)
print("Saved metrics →", raw_csv)

#  Summary + 95 % CI plot 
summary = (results.groupby("feature_set", as_index=False)
                    .agg(MAE_mean=("MAE", "mean"),
                         SD=("MAE", "std"),
                         n=("MAE", "size")))
summary["CI95"] = 1.96 * summary["SD"] / np.sqrt(summary["n"])
summary.sort_values("MAE_mean", ascending=False, inplace=True)

sum_csv = FIGDIR / "inclusion_rf_metrics_summary.csv"
summary.to_csv(sum_csv, index=False)
print("Saved summary →", sum_csv)

plt.figure(figsize=(8, max(6, 0.25 * len(summary))))
plt.barh(summary["feature_set"], summary["MAE_mean"], xerr=summary["CI95"], capsize=3)
plt.xlabel("MAE (years)")
plt.xlim(summary["MAE_mean"].min() * 0.95,
         summary["MAE_mean"].max() * 1.05)
plt.ylabel("Feature added / Baseline = Raw+Derived)")
plt.title("Random-Forest Inclusion study – mean MAE ± 95 % CI (N=20 seeds)")
plt.tight_layout()

png = FIGDIR / "inclusion_rf_mae.png"
plt.savefig(png, dpi=150)
plt.close()
print("Saved chart →", png)
