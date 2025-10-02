# src/text/train_han.py

"""
Encode ClinicalTrials.gov Brief Summary text into HAN vectors (Parquet).

This script loads a flattened CT.gov table, cleans the Brief Summary text,
computes fixed-length embeddings using `HANBrief` (BioBERT → BiGRU + attention),
and saves them as `brief_han_###` columns alongside the trial NCT ID.
"""

import argparse, pathlib, pandas as pd, torch, tqdm, re
from .clean import clean_ctgov_text
from .han_encoder import HANBrief
from src.text._utils import find_col

def _load_nct_txt(path: pathlib.Path) -> set[str]:   # NEW
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # accept comma/space/newline separated; normalise to upper
    toks = re.split(r"[,\s]+", txt)
    ids = {t.strip().upper() for t in toks if t.strip()}
    # keep only NCT* patterns; tolerate NCT + 8–10 digits
    return {x for x in ids if re.fullmatch(r"NCT\d{8,10}", x)}

ap = argparse.ArgumentParser()
ap.add_argument("--flat", type=pathlib.Path, required=True)
ap.add_argument("--out",  type=pathlib.Path, required=True)
ap.add_argument("--batch",type=int, default=64)
ap.add_argument("--ids_txt", type=pathlib.Path, help="TXT with NCT IDs (one per line or separated by commas/spaces)")  # NEW
args = ap.parse_args()

df_flat = pd.read_parquet(args.flat)
nct_col = find_col(df_flat, ["nct id"])
bs_col  = find_col(df_flat, ["brief summary","briefSummary"])

# ---- optional restriction by TXT list ----
if args.ids_txt is not None:
    keep = _load_nct_txt(args.ids_txt)
    df_flat[nct_col] = df_flat[nct_col].astype(str).str.upper()
    before = len(df_flat)
    df_flat = df_flat[df_flat[nct_col].isin(keep)].copy()
    print(f"Restricting to TXT list: kept {len(df_flat):,} / {before:,} rows")
    # simple report of any requested IDs missing in flat
    missing = sorted(list(keep - set(df_flat[nct_col])))
    if missing:
        print(f"WARNING: {len(missing)} IDs from TXT were not found in flat. "
              f"Example: {missing[:5]}")

texts = df_flat[bs_col].fillna("").map(clean_ctgov_text).tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HANBrief().to(device)
model.eval()

vecs = []
for i in tqdm.trange(0, len(texts), args.batch):
    batch = texts[i:i+args.batch]
    with torch.no_grad():
        v = model(batch).to("cpu")
    vecs.append(v)
import torch as _torch
vecs = _torch.cat(vecs, dim=0).numpy().astype("float16")

cols = [f"brief_han_{i:03d}" for i in range(vecs.shape[1])]
out_df = pd.DataFrame(vecs, columns=cols)
out_df.insert(0, nct_col, df_flat[nct_col].values)
args.out.parent.mkdir(parents=True, exist_ok=True)
out_df.to_parquet(args.out, index=False)
print("HAN vectors saved →", args.out, "| shape:", out_df.shape)
