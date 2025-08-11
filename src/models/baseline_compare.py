"""
src/models/baseline_compare.py
----------------------------------------------------------------
Compare 3 regressors on the trial-duration feature set:
- LinearRegression        
- RandomForestRegressor
- GradientBoostingRegressor


Outputs
- MAE, RMSE, R^2, seconds per model
- reports/figures/filtered_data/model_metrics.csv
- reports/figures/filtered_data/model_mae.png
- reports/figures/filtered_data/parity_rf.png
- reports/figures/filtered_data/imp_gini_rf.csv
- reports/figures/filtered_data/imp_gini_rf.png
- models/<model>.joblib
"""

import pathlib, time, joblib, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
import numpy as np   

ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v4.parquet"
FIGDIR = ROOT / "reports" / "figures" / "filtered_data";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";               MODELD.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)
target = "duration_days"

num_cols = [
    # raw numeric or engineered numeric columns
    "# patients",
    "country_n",
    "site_n",
    "assessments_n",
    "start_year",
    "patients_per_site",
    "num_arms",
    "masking_flag",
    "placebo_flag",
    "active_prob",
    "elig_crit_n",
    "age_min",
    "age_max",
    "age_range",
    "randomized_flag",
    "fda_drug_flag",
    "fda_device_flag",
    "freq_in_window",
    "novelty_score",
    "complexity_score_100",
    "attractiveness_score_100",
    ]

cat_cols= [
# raw or engineered categoricals
    "phase",
    "sponsor_class",
    "condition_top",
    "therapeutic_area",
    "intervention_type",
    "assessments_complexity",
    "global_trial",
    "masking_level",
    "population_class",
    "cohort_design",
    "study_type",
    "allocation",
    ]

# ---------- Temporal Split (Same as TrialDura) ------------------------------
# Training set: trials that START *before* 2019-01-01
# Test set   : trials that START on/after 2019-01-01

if "start_date" in df.columns:
    df["start_date"] = pd.to_datetime(df["start_date"])    # ensure datetime
    cutoff  = pd.Timestamp("2019-01-01")
    is_test = df["start_date"] >= cutoff
else:                       # fall back to the helper column already in the data
    cutoff_year = 2019
    is_test = df["start_year"] >= cutoff_year

X = df[num_cols + cat_cols]
y = df[target]

X_train, X_test = X[~is_test], X[is_test]
y_train, y_test = y[~is_test], y[is_test]

# Show split sizes 
print(f"Training set size : {len(X_train):>7} rows")

# ---------- Pre-processor --------------------------------------------------
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())          # ← new step
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore"))   # sparse CSR
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], verbose_feature_names_out=False)

# 5-fold shuffled CV (ordinary K-fold)
tscv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "MAE" : "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",   # needs scikit-learn ≥1.4
    "R2"  : "r2",
}


# ---------- Model zoo --------------------------------------------------
models = {
    # new baseline
    "MeanBaseline": DummyRegressor(strategy="mean"),

    # existing models
    #"LinearRegression": LinearRegression(n_jobs=-1),
    "RandomForest": RandomForestRegressor(
        n_estimators=150, max_depth=None, n_jobs=-1, random_state=42),
    #"GradientBoosting": GradientBoostingRegressor(
        #n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),

    # MLP model
    # "MLPRegressor": MLPRegressor(
    #     hidden_layer_sizes=(64, 32),
    #     activation="relu",
    #     solver="adam",
    #     max_iter=500,
    #     random_state=42,
    #     early_stopping=True,
    # ),
}

metrics = []

dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
print("Dummy test R²:", r2_score(y_test, dummy.predict(X_test)))


# ---------- Training and evaluation ----------------------------------------
for name, est in models.items():
    print(f"\n=== {name} ===")
    pipe = Pipeline([("pre", pre), ("model", est)])
    
    # cross-validate
    cv_res = cross_validate(
        pipe, X_train, y_train,
        cv=tscv, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    cv_mae  = -cv_res["test_MAE"].mean()
    cv_rmse = -cv_res["test_RMSE"].mean()
    cv_r2   =  cv_res["test_R2"].mean()
    print(f"CV   MAE {cv_mae:7.1f} | RMSE {cv_rmse:7.1f} | R² {cv_r2:5.3f}")

    # Fit
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0

    # Predict
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    pred_sec = time.perf_counter() - t0

    # Compute metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    secs = fit_sec + pred_sec
    print(f"MAE {mae:7.1f} | RMSE {rmse:7.1f} | R² {r2:5.3f} | {secs:6.1f}s")

    metrics.append({
        "model"   : name,
        # cross-validation averages
        "cv_MAE"  : cv_mae,  "cv_RMSE": cv_rmse,  "cv_R2": cv_r2,
        # hold-out (“future slice”) metrics
        "MAE"     : mae,     "RMSE"   : rmse,     "R2"   : r2,
        "seconds" : secs
    })
    joblib.dump(pipe, MODELD / f"{name.lower()}.joblib")

    # Additional plots & outputs for RandomForest
    if name == "RandomForest":
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

        # Impurity-based feature importance (top-20)
        rf_model   = pipe.named_steps["model"]
        feat_names = pipe.named_steps["pre"].get_feature_names_out()

        imp = (pd.DataFrame({
                  "feature"        : feat_names,
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

tbl = pd.DataFrame(metrics).sort_values("MAE")
tbl.to_csv(FIGDIR / "model_metrics.csv", index=False)
print("\nSaved metrics →", FIGDIR / "model_metrics.csv")

plt.figure(figsize=(6, 4))
sns.barplot(data=tbl, x="MAE", y="model", palette="crest")
plt.xlabel("MAE (days)")
plt.ylabel("")
plt.title("Model comparison – lower MAE is better")
plt.tight_layout()
plt.savefig(FIGDIR / "model_mae.png", dpi=150)
plt.close()
print("Saved bar chart →", FIGDIR / "model_mae.png")