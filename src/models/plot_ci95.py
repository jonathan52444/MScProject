#!/usr/bin/env python
"""Plot Random-Forest ablation results with 95 % confidence intervals.

Assumes `src/models/new_ablation.py` is ran, which wrote
`ablation_rf_metrics.csv` (one row per seed × feature set).
"""

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locate the CSV written by the ablation script

HERE   = pathlib.Path(__file__).resolve()          # this file
ROOT   = HERE.parents[2]                           # repo root  
FIGDIR = ROOT / "reports" / "figures" / "filtered_data"
CSV    = FIGDIR / "ablation_rf_metrics.csv"

if not CSV.exists():
    raise FileNotFoundError(f"Can't find {CSV}; run the ablation script first.")

df = pd.read_csv(CSV)

# Aggregate: mean ± 95 % CI on MAE

summary = (
    df.groupby("left_out", as_index=False)
      .agg(MAE_mean=("MAE", "mean"),
           SD       =("MAE", "std"),
           n        =("MAE", "size"))
)
summary["SEM"]   = summary["SD"] / np.sqrt(summary["n"])
summary["CI95"]  = 1.96 * summary["SEM"]              # half-width
summary.sort_values("MAE_mean", ascending=False, inplace=True)

# Plot
plt.figure(figsize=(8, max(6, 0.25 * len(summary))))
plt.barh(summary["left_out"],
         summary["MAE_mean"],
         xerr=summary["CI95"],
         capsize=3)
plt.xlabel("MAE (years)")
plt.xlim(summary["MAE_mean"].min() * 0.95,
         summary["MAE_mean"].max() * 1.05)
plt.ylabel("Feature left out / Model")
plt.title("Random-Forest ablation – mean MAE ± 95 % CI (N ≈ 10 seeds)")
plt.tight_layout()

out_png = FIGDIR / "ablation_rf_mae_ci95.png"
plt.savefig(out_png, dpi=150)
plt.close()

print(f"Saved plot → {out_png}")
