"""
src/etl/flatten_all.py

Flatten gzipped ClinicalTrials.gov JSON batches into a single Parquet file.

"""

from __future__ import annotations

import gzip
import glob
import json
import pathlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.nested import value_from_paths


# Paths
ROOT    = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim" / "ctgov_flat.parquet"

# Load Mapping Sheet
MAP_XLSX = Path(
    "/Users/jonathanfung/Library/Mobile Documents/com~apple~CloudDocs/"
    "UCL MSc DSML/MSc Project/ctgov_mapping_v2.xlsx"
)

map_df = (
    pd.read_excel(MAP_XLSX)
      .dropna(subset=["ctgov_field"])
)
PATHS = dict(map_df[["variable_wishlist", "ctgov_field"]].values)


def flat_row(study: dict) -> dict:
    """Extract all desired fields from one study record."""
    return {var: value_from_paths(study, path) for var, path in PATHS.items()}

def main(max_files: int | None = None) -> None:
    """Flatten every .json.gz in RAW_DIR (or only `max_files` of them)."""
    rows: list[dict] = []

    gz_files = sorted(glob.glob(str(RAW_DIR / "*.json.gz")))
    if max_files is not None:
        gz_files = gz_files[:max_files]

    for idx, gz in enumerate(gz_files, 1):
        print(f"→ [{idx}/{len(gz_files)}] {pathlib.Path(gz).name}", flush=True)

        with gzip.open(gz, "rt") as fp:
            js = json.load(fp) # may take ~10-30 s for big files
            for st in tqdm(js["studies"], desc="studies", leave=False):
                rows.append(flat_row(st))

    # Write parquet
    df = pd.DataFrame(rows)
    INTERIM.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), INTERIM)
    print("Flatten saved →", INTERIM, "| shape:", df.shape)


if __name__ == "__main__":
    # Set max_files to a small integer (e.g. 3) for a quick smoke-test.
    main(max_files=None)        # ← None processes *all* batches