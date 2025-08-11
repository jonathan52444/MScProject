import pandas as pd
import matplotlib.pyplot as plt

ENG_CSV = "/Users/jonathanfung/Library/Mobile Documents/com~apple~CloudDocs/UCL MSc DSML/MSc Project/reports/figures/filtered_data/engineered_rf_metrics_summary.csv"
ABL_CSV = "/Users/jonathanfung/Library/Mobile Documents/com~apple~CloudDocs/UCL MSc DSML/MSc Project/reports/figures/filtered_data/ablation_rf_v2_summary.csv"
BLUE    = "#1f77b4"                        # bar colour

import pandas as pd

eng = pd.read_csv(ENG_CSV)
print(eng.columns)       # <‑‑ see the REAL names

def make_plot(df, y, title, path, figsize=(12,6), xlim=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        df[y], df["MAE_mean"], xerr=df["MAE_std"],
        color=BLUE,
        error_kw=dict(ecolor="black", lw=1.4, capsize=4, capthick=1.4),
        zorder=3,
    )
    ax.invert_yaxis()
    ax.set_xlabel("MAE (years)")
    ax.set_title(f"{title}\n(lower is better)", fontsize=12, pad=10)
    if xlim:
        ax.set_xlim(xlim)
    ax.grid(axis="x", ls="--", lw=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

# ── 1) Engineered features benchmark ───────────────────────
eng = pd.read_csv(ENG_CSV).sort_values("MAE_mean", ascending=False)
make_plot(
    eng, "feature_set",
    "Engineered‑features benchmark – mean MAE ±1 SD across 20 seeds",
    "Users/jonathanfung/Library/Mobile Documents/com~apple~CloudDocs/UCL MSc DSML/MSc Project/reports/figures/filtered_data/engineered_rf_mae_v4.png",
)

# ── 2) Raw features ablation study ─────────────────────────
abl = pd.read_csv(ABL_CSV)
raw = abl[abl["study"] == "Raw Features Ablation Study"] \
          .sort_values("MAE_mean", ascending=False)
make_plot(
    raw, "model_clean",
    "Raw Features Ablation Study – mean MAE ±1 SD across 20 seeds",
    "Users/jonathanfung/Library/Mobile Documents/com~apple~CloudDocs/UCL MSc DSML/MSc Project/reports/figures/filtered_data/ablation_rf_v2_raw_mae_v4.png",
)

# ── 3) Raw + Derived features ablation study ───────────────
rd = abl[abl["study"] == "Raw+Derived Features Ablation Study"] \
         .sort_values("MAE_mean", ascending=False)
make_plot(
    rd, "model_clean",
    "Raw + Derived Features Ablation Study – mean MAE ±1 SD across 20 seeds",
    "Users/jonathanfung/Library/Mobile Documents/com~apple~CloudDocs/UCL MSc DSML/MSc Project/reports/figures/filtered_data/ablation_rf_v2_rawderived_mae_v4.png",
    figsize=(14,6), xlim=(0.95, 1.1),        # <-- new axis limits
)
