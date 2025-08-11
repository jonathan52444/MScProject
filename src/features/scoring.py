from __future__ import annotations

"""
src/features/scoring.py

Utility functions that append pre‑computed *Complexity* and *Attractiveness*
model scores to a feature dataframe.

Each score is produced by a pickled scikit‑learn pipeline saved during model
training.  The helpers keep the calling code lightweight:

* **`add_complexity_score`** – loads a Ridge‑regression artefact whose
  coefficients (`beta`) and min/max scaling constants (`Smin`, `Smax`) were
  stored alongside the preprocessing pipeline; returns the dataframe with a
  new column ``complexity_score_100`` scaled to 0–100.
* **`add_attractiveness_score`** – very similar but simpler: the pipeline has a
  single numeric output that is linearly rescaled to 0–100 and stored in a
  configurable column (default ``attractiveness_score_100``).

"""

import pathlib
import pickle

import pandas as pd
import numpy as np


def _load_pickle(path: pathlib.Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def add_complexity_score(
    df: pd.DataFrame, model_path: pathlib.Path
) -> pd.DataFrame:
    """
    Append a 0–100 'complexity_score_100' column using the pre-fit
    Ridge pipeline stored at `model_path`.
    """
    obj = _load_pickle(model_path)

    X = obj["pipeline"].named_steps["pre"].transform(df)
    raw = (X * obj["beta"]).sum(axis=1)
    df["complexity_score_100"] = 100 * (raw - obj["Smin"]) / (
        obj["Smax"] - obj["Smin"]
    )
    return df


def add_attractiveness_score(
    df: pd.DataFrame,
    model_path: pathlib.Path,
    out_col: str = "attractiveness_score_100",
) -> pd.DataFrame:
    """
    Append a 0–100 attractiveness score column.
    """
    obj = _load_pickle(model_path)

    if "design" in obj:  # our notebook saved this
        needed_cols = obj["design"]
    else:  # scikit-learn ≥1.2
        needed_cols = list(
            obj["pipeline"].named_steps["pre"].feature_names_in_
        )

    X = obj["pipeline"].transform(df[needed_cols]).ravel()
    df[out_col] = 100 * (X - obj["Smin"]) / (obj["Smax"] - obj["Smin"])
    return df
