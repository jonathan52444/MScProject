"""
Redraw inclusion‑study chart with new axis‐label & title
────────────────────────────────────────────────────────
 ‣ y‑axis  :  "Feature added / Baseline = Raw + Derived"
 ‣ title   :  "Random‑Forest Inclusion study – mean MAE ± 95 % CI (N = 20 seeds)"
Nothing is re‑trained – we just read the CSV that the original pipeline
already wrote and save a fresh PNG alongside it.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]          # adjust if needed
FIGDIR  = ROOT / "reports" / "figures" / "filtered_data"
RAW_CSV = FIGDIR / "inclusion_rf_metrics.csv"
SUM_CSV = FIGDIR / "inclusion_rf_metrics_summary.csv"

# ── Load (or build) summary table ──────────────────────────────────────
if SUM_CSV.exists():
    summary = pd.read_csv(SUM_CSV)
else:                                  # fall back to raw per‑run metrics
    df = pd.read_csv(RAW_CSV)
    summary = (df.groupby("feature_set", as_index=False)
                 .agg(MAE_mean=("MAE", "mean"),
                      SD=("MAE",  "std"),
                      n=("MAE",  "size")))
    summary["CI95"] = 1.96 * summary["SD"] / np.sqrt(summary["n"])

summary.sort_values("MAE_mean", ascending=False, inplace=True)

# ── Plot ────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, max(6, 0.25 * len(summary))))
plt.barh(
    summary["feature_set"],
    summary["MAE_mean"],
    xerr=summary["CI95"],
    capsize=3
)

plt.xlabel("MAE (years)")
plt.ylabel("Feature added / Baseline = Raw + Derived")   # ← new label
plt.title("Random‑Forest Inclusion study – mean MAE ± 95 % CI (N = 20 seeds)")  # ← new title

# make bars fit nicely
plt.xlim(summary["MAE_mean"].min() * 0.99,
         summary["MAE_mean"].max() * 1.01)

plt.tight_layout()
out_png = FIGDIR / "inclusion_rf_mae_v2.png"
plt.savefig(out_png, dpi=150)
plt.close()
print("Saved updated chart →", out_png)
