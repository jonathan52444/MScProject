"""
src/models/baseline_compare.py
----------------------------------------------------------------
Compare 3 regressors on the trial-duration feature set:
-RandomForestRegressor
-GradientBoostingRegressor
-LinearRegression        

Outputs
-MAE,RMSE,R^2,seconds per model
-reports/figures/model_metrics.csv
-reports/figures/model_mae.png
-models/<model>.joblib
"""

import pathlib, time, joblib, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "features_v0.parquet"
FIGDIR = ROOT / "reports" / "figures";  FIGDIR.mkdir(parents=True, exist_ok=True)
MODELD = ROOT / "models";               MODELD.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)
target = "duration_days"

num_cols = ["# patients", "country_n", "site_n",
            "assessments_n", "start_year"]
cat_cols = ["phase", "sponsor_class", "condition_top",
            "intervention_type", "assessments_complexity"]

X = df[num_cols + cat_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df["phase"]
)

# Pre-processor
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore"))   # sparse CSR
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], verbose_feature_names_out=False)

# Model zoo
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=150, max_depth=None, n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
    "LinearRegression": LinearRegression(n_jobs=-1)
}

metrics = []

for name, est in models.items():
    print(f"\n=== {name} ===")
    pipe = Pipeline([("pre", pre), ("model", est)])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    pred_sec = time.perf_counter() - t0

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    secs = fit_sec + pred_sec
    print(f"MAE {mae:7.1f} | RMSE {rmse:7.1f} | R² {r2:5.3f} | {secs:6.1f}s")

    metrics.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
                    "seconds": secs})
    joblib.dump(pipe, MODELD / f"{name.lower()}.joblib")

tbl = pd.DataFrame(metrics).sort_values("MAE")
tbl.to_csv(FIGDIR / "model_metrics.csv", index=False)
print("\nSaved metrics →", FIGDIR / "model_metrics.csv")

plt.figure(figsize=(6,4))
sns.barplot(data=tbl, x="MAE", y="model", palette="crest")
plt.xlabel("MAE (days)"); plt.ylabel("")
plt.title("Model comparison – lower MAE is better")
plt.tight_layout()
plt.savefig(FIGDIR / "model_mae.png", dpi=150)
plt.close()
print("Saved bar chart →", FIGDIR / "model_mae.png")

