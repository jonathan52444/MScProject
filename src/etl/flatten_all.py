import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))  

import gzip, json, glob, pathlib, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from src.utils.nested import value_from_paths

RAW_DIR   = ROOT / "data" / "raw"
INTERIM   = ROOT / "data" / "interim" / "ctgov_flat.parquet"

# load mapping sheet
map_df = pd.read_excel(Path(
    "/Users/jonathanfung/Library/Mobile Documents/com~apple~CloudDocs/"
    "UCL MSc DSML/MSc Project/ctgov_mapping_v1.xlsx"
)).dropna(subset=["ctgov_field"])
PATHS  = dict(map_df[["variable_wishlist", "ctgov_field"]].values)

def flat_row(study):
    return {var: value_from_paths(study, path) for var, path in PATHS.items()}

rows = []
for gz in glob.glob(str(RAW_DIR / "*.json.gz")):
    with gzip.open(gz, "rt") as fp:
        for st in json.load(fp)["studies"]:
            rows.append(flat_row(st))

df = pd.DataFrame(rows)
INTERIM.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pandas(df), INTERIM)
print("Flatten saved →", INTERIM, "| shape:", df.shape)

