import pyarrow.parquet as pq
import pandas as pd

PATH = "data/processed/features_v4.parquet"        # adjust if you moved it
df = pq.read_table(PATH).to_pandas()

cts = (df["therapeutic_area"]
         .value_counts(dropna=False)
         .rename_axis("Therapeutic area")
         .to_frame("Trials"))

print(cts)
print("\nShare classified:", 1 - cts.loc["Other", "Trials"] / len(df))
