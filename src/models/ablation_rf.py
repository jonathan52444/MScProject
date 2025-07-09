# src/models/ablation_rf.py
"""
Random-Forest ablation study – “all-but-one” features
----------------------------------------------------
* 50-tree RandomForestRegressor per feature-left-out set
* tqdm progress bar
* outputs:
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
from tqdm.auto import tqdm      # progress bar (pip install tqdm)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Paths
ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v4.parquet"
FIGDIR = ROOT / "reports" / "figures" / "filtered_data";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";                                 MODELD.mkdir(exist_ok=True)

# Data
df = pd.read_parquet(DATA)
target = "duration_days"

# convert date columns to numeric (seconds since UNIX epoch) 
for col in ["start_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")                # ensure datetime
    ts = df[col].view("int64")                                        # ns → int64
    ts[df[col].isna()] = np.nan                                       # NaT → NaN
    df[f"{col}_ts"] = ts / 1e9                                        # seconds
# keep original start_date for the temporal split only

# Feature lists (original + fixes) --------------------------------------------
num_cols = [
    "# patients", "country_n", "site_n", "assessments_n",
    "start_date_ts",
    "patients_per_site", "num_arms", "masking_flag", "placebo_flag",
    "active_prob", "elig_crit_n", "safety_cuts_flag",   
    "age_min", "age_max", "age_range", "disease_modifying_flag",
    "randomized_flag", "fda_drug_flag", "fda_device_flag",
    "freq_in_window", "novelty_score", "complexity_score_100",
    "attractiveness_score_100",
]

cat_cols = [
    "phase", "sponsor_class", "condition_top", "therapeutic_area",
    "intervention_type", "assessments_complexity", "global_trial",
    "masking_level", "population_class", "cohort_design",
    "study_type", "allocation",
]

# drop anything missing in the dataframe (protects against future changes)
num_cols_exist = [c for c in num_cols if c in df.columns]
cat_cols_exist = [c for c in cat_cols if c in df.columns]
missing = sorted(set(num_cols + cat_cols) - set(num_cols_exist + cat_cols_exist))
if missing:
    print("Skipping missing columns:", ", ".join(missing))

ALL_COLS = num_cols_exist + cat_cols_exist

# Temporal train / test split
cutoff = pd.Timestamp("2019-01-01")
is_test = df["start_date"] >= cutoff

X_train, X_test = df.loc[~is_test, ALL_COLS], df.loc[is_test, ALL_COLS]
y_train, y_test = df.loc[~is_test, target],   df.loc[is_test, target]

print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

# Pipelines
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

#Ablation loop -------------------------------------------------------
metrics: list[dict] = []

for left_out in tqdm(ALL_COLS, desc="Ablation", unit="feature"):

    kept = [c for c in ALL_COLS if c != left_out]

    pre = ColumnTransformer(
        [("num", num_pipe, [c for c in kept if c in num_cols_exist]),
         ("cat", cat_pipe, [c for c in kept if c in cat_cols_exist])],
        verbose_feature_names_out=False,
    )

    model = RandomForestRegressor(**RF_PARAMS)
    pipe = Pipeline([("pre", pre), ("rf", model)])

    t0 = time.perf_counter()
    pipe.fit(X_train[kept], y_train)
    fit_sec = time.perf_counter() - t0

    y_pred = pipe.predict(X_test[kept])
    mae_days  = mean_absolute_error(y_test, y_pred)
    rmse_days = mean_squared_error(y_test, y_pred) ** 0.5

    mae = mae_days / 365
    rmse = rmse_days / 365 
    r2   = r2_score(y_test, y_pred)

    metrics.append({"left_out": left_out, "MAE": mae,
                    "RMSE": rmse, "R2": r2, "seconds": fit_sec})

    # optional: save each fitted model
    # joblib.dump(pipe, MODELD / f"rf_no_{left_out}.joblib")

#  Save results & plot ------------------------------------------------------
tbl = pd.DataFrame(metrics).sort_values("MAE", ascending=False)
out_csv = FIGDIR / "ablation_rf_metrics.csv"
tbl.to_csv(out_csv, index=False)
print("Saved metrics →", out_csv)

plt.figure(figsize=(7, 5))
sns.barplot(data=tbl, x="MAE", y="left_out", palette="crest")
plt.xlabel("MAE (years) when feature removed")
plt.xlim(0.8, 1.4)    
plt.ylabel("Feature left out")
plt.title("Random-Forest ablation study – lower MAE is better")
plt.tight_layout()
plt.savefig(FIGDIR / "ablation_rf_mae.png", dpi=150)
plt.close()
print("Saved chart   →", FIGDIR / "ablation_rf_mae.png")
