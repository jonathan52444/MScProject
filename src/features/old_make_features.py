"""
src/features/make_features.py
Generate modelling-ready features from ctgov_flat.parquet
Input : data/interim/ctgov_flat.parquet   (538 875 × 24)
Output : data/processed/features_v0.parquet
"""

import pathlib, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
from dateutil import parser


def list_len(x):
    """Return length of list or pipe-delimited string; else 0/NaN."""
    if isinstance(x, list):
        return len(x)
    if isinstance(x, str):
        return len([v for v in x.split("|") if v])
    return 0

def parse_date_any(x):
    """
    Parse CT.gov date strings:
      • ‘2024-05-01’ → 2024-05-01
      • ‘January 2024’ → 2024-01-01   (day default=1)
      • bad / missing  → NaT
    Returns pandas.Timestamp / NaT.
    """
    try:
        return pd.to_datetime(
            parser.parse(str(x), fuzzy=True, default=pd.Timestamp("1900-01-01"))
        )
    except Exception:
        return pd.NaT

# paths
ROOT = pathlib.Path(__file__).resolve().parents[2]
FLAT = ROOT / "data" / "interim"   / "ctgov_flat.parquet"
OUT  = ROOT / "data" / "processed" / "features_v1.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Reading", FLAT)
df = pq.read_table(FLAT).to_pandas()
df["# patients"] = pd.to_numeric(df["# patients"], errors="coerce") # make enrolment numeric

# Normalisation
df.columns = df.columns.str.strip()                                 # trim column names
df["Primary Completion Type"] = df["Primary Completion Type"].astype(str).str.strip().str.upper()
df["Overall status"]          = df["Overall status"].astype(str).str.strip().str.upper()

# dates and duration
df["start_date"]    = df["Study Start Date"].apply(parse_date_any)
df["complete_date"] = df["Primary Completion Date"].apply(parse_date_any)
df["duration_days"] = (df["complete_date"] - df["start_date"]).dt.days

TODAY = pd.Timestamp.today().normalize()

# drop rows with missing or future completion dates
df = df[df["complete_date"].notna() & (df["complete_date"] <= TODAY)]

# keep only Actual primary completion dates
df = df[df["Primary Completion Type"] == "ACTUAL"]

# keep only truly finished studies
df = df[df["Overall status"].isin(["COMPLETED", "TERMINATED"])]


# phase cleaning
phase_map = {
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "PHASE1/PHASE2": "Phase 1_2",
    "PHASE2/PHASE3": "Phase 2_3",
    "EARLY PHASE 1": "Phase 1",
}
phase_raw = df["phase"].fillna("").str.upper().map(phase_map)

phase_cat = ["Phase 1","Phase 1_2","Phase 2","Phase 2_3","Phase 3","Phase 4"]
df["phase"] = pd.Categorical(phase_raw.fillna("Unknown"),
                             categories=phase_cat + ["Unknown"],
                             ordered=True)

# sponsor and metric
df["sponsor_class"] = np.where(
    df["sponsorship type"].str.contains("univer|hospital|nih|institute",
                                        case=False, na=False),
    "Academic", "Industry"
)
df["condition_top"]     = df["indication/disease area"].str.split("|").str[0]
df["intervention_type"] = df["mode of administration (ex. NBE, NCE, iv vs pill)"]\
                            .str.split("|").str[0]
df["start_year"]        = df["start_date"].dt.year

# country and site visits
geo_col  = "# geographies - global, regions involved, single country (consider start-up timings by country)"
site_col = "# sites"

df["country_n"] = df[geo_col].apply(list_len)
df["site_n"]    = df[site_col].apply(list_len)


# outcomes 
df["primary_out_n"]   = df["primary_outcomes"].apply(list_len)
df["secondary_out_n"] = df["secondary_outcomes"].apply(list_len)
df["other_out_n"]     = df["other_outcomes"].apply(list_len)
df["assessments_n"]   = df["primary_out_n"] + df["secondary_out_n"] + df["other_out_n"]

def complexity_bucket(n):
    if pd.isna(n): return np.nan
    return "Low" if n<=3 else "Medium" if n<=6 else "High"

df["assessments_complexity"] = df["assessments_n"].apply(complexity_bucket)
df.drop(columns=["primary_out_n","secondary_out_n","other_out_n"], inplace=True)


# placeholders
for col in ["screen fail rate", "evaluability / drop out rate", "safety events"]:
    if col not in df.columns:
        df[col] = np.nan

# debug
print("\n─── DEBUG ──────────────────────────────────")
print("Raw rows after feature engineering :", len(df))
print("  • non-NaN duration_days          :", df['duration_days'].notna().sum())
print("  • min / max duration_days        :", df['duration_days'].min(), "/", df['duration_days'].max())
print("  • non-NaN phase                  :", df['phase'].notna().sum())
print("  • phase unique values            :", df['phase'].unique()[:10])
print("────────────────────────────────────────────\n")


# Filters (duration / start year / enrolment rules)

df = df[
    (df["duration_days"].between(1, 3650))  # ≤ 10 years
    & (df["start_date"] >= pd.Timestamp("2000-01-01"))  # starts in 2000+
    & (df["# patients"] >= 10)  # at least 10 participants
]

df["phase"] = df["phase"].fillna("Unknown")


# save
pq.write_table(pa.Table.from_pandas(df), OUT)
print("Features saved →", OUT, "| shape:", df.shape)


# ---------------------------------------------------------
print("Unique values BEFORE filters")
print("Primary Completion Type →",
      df["Primary Completion Type"].dropna().unique()[:10])
print("Overall status         →",
      df["Overall status"].dropna().unique()[:10])
print("# rows with complete_date <= today →",
      ((df["complete_date"].notna()) & (df["complete_date"] <= TODAY)).sum())
print("# rows with ACTUAL PCT            →",
      (df["Primary Completion Type"] == "ACTUAL").sum())
print("# rows with COMPLETED|TERMINATED  →",
      df["Overall status"].isin(["COMPLETED", "TERMINATED"]).sum())
# ---------------------------------------------------------