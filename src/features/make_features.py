"""
Command-line entry point

Run with:
    python -m src.features.make_features
"""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from .engineering import build_features


def main() -> None:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    FLAT = ROOT / "data" / "interim" / "ctgov_flat.parquet"
    OUT = ROOT / "data" / "processed" / "features_v3.parquet"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Reading", FLAT)
    features = build_features(FLAT)

    print("\n─── DEBUG ──────────────────────────────────")
    print("Rows after feature engineering :", len(features))
    print(
        " • non-NaN duration_days       :",
        features["duration_days"].notna().sum(),
    )
    print(
        " • duration range (days)       :",
        int(features["duration_days"].min()),
        "/",
        int(features["duration_days"].max()),
    )
    print(" • phase unique values         :", features["phase"].unique()[:10])
    print(
        " • novelty min / max           :",
        features["novelty_score"].min(),
        "/",
        features["novelty_score"].max(),
    )
    print(
        " • complexity_score_100 min / max :",
        round(features["complexity_score_100"].min(), 2),
        "/",
        round(features["complexity_score_100"].max(), 2),
    )
    print(
        " • attractiveness_score_100 min / max :",
        round(features["attractiveness_score_100"].min(), 2),
        "/",
        round(features["attractiveness_score_100"].max(), 2),
    )

    used_num = [
        "# patients",
        "country_n",
        "site_n",
        "assessments_n",
        "start_year",
        "novelty_score",
        "complexity_score_100",
        "attractiveness_score_100",
        "patients_per_site",
        "num_arms",
        "masking_flag",
        "placebo_flag",
        "elig_crit_n",
    ]
    used_cat = [
        "phase",
        "sponsor_class",
        "condition_top",
        "therapeutic_area",
        "intervention_type",
        "assessments_complexity",
        "global_trial",
    ]

    print("Numeric features  ({}): {}".format(len(used_num), ", ".join(used_num)))
    print("Categoric features({}): {}".format(len(used_cat), ", ".join(used_cat)))

    print("──────────────────────────────────────────────\n")

    pq.write_table(pa.Table.from_pandas(features), OUT)
    print("Features saved →", OUT, "| shape:", features.shape)


if __name__ == "__main__":
    main()
