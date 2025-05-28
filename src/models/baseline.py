"""
src/models/baseline.py
---------------------------------------------------------------
Baseline regression for trial duration
-Loads features_v0.parquet
-Train/test split 80/20
-Pre-processing: median/mode impute + one-hot
-Model: RandomForestRegressor wit 100 trees
"""

import pathlib, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    X, y, test_size=0.20, random_state=42, stratify=df["phase"])


num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

rf = RandomForestRegressor(
        n_estimators=100,    
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        verbose=1            
)

pipe = Pipeline([("pre", pre), ("model", rf)])
pipe.fit(X_train, y_train)


y_pred = pipe.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5    
r2   = r2_score(y_test, y_pred)

print(f"Test MAE : {mae:,.1f} days")
print(f"Test RMSE: {rmse:,.1f} days")
print(f"Test  R² : {r2:,.3f}")
 
# Plots 
plt.figure(figsize=(4,4))
sns.scatterplot(x=y_test, y=y_pred, alpha=.2, s=10, edgecolor=None)
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

plt.figure(figsize=(6,5))
sns.barplot(data=imp.head(20),
            x="importance_gini", y="feature", palette="crest")
plt.xlabel("Gini importance")
plt.title("Random-Forest feature importance – top 20")
plt.tight_layout()
plt.savefig(FIGDIR / "imp_gini_rf.png", dpi=150)
plt.close()

joblib.dump(pipe, MODELD / "rf_baseline.joblib")
print("Artifacts saved to reports/figures & models/")
