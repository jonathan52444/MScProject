# src/features/merge_text_features.py
import argparse, pathlib, pandas as pd
from src.text._utils import find_col

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=pathlib.Path, required=True)
ap.add_argument("--han",  type=pathlib.Path, required=True)
ap.add_argument("--out",  type=pathlib.Path, required=True)
args = ap.parse_args()

base = pd.read_parquet(args.base)
han  = pd.read_parquet(args.han)

# detect NCT column in each frame independently
nb = "NCT ID" if "NCT ID" in base.columns else find_col(base, ["nct id","nct_id","nctid"])
nh = "NCT ID" if "NCT ID" in han.columns  else find_col(han,  ["nct id","nct_id","nctid"])

# normalise key: same name, type, case
base = base.rename(columns={nb: "NCT_ID"})
han  = han.rename(columns={nh: "NCT_ID"})
base["NCT_ID"] = base["NCT_ID"].astype(str).str.upper().str.strip()
han["NCT_ID"]  = han["NCT_ID"].astype(str).str.upper().str.strip()

merged = base.merge(han, on="NCT_ID", how="left")
args.out.parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(args.out, index=False)

# quick report
han_cols = [c for c in merged.columns if c.startswith("brief_han_")]
missing  = merged[han_cols].isna().any(axis=1).sum() if han_cols else len(merged)
print(f"Merged → {args.out} | rows {len(merged)} | HAN dims {len(han_cols)} | "
      f"rows missing HAN {missing}")
