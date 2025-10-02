End‑to‑end pipeline to predict **clinical trial duration** from ClinicalTrials.gov data: ETL → feature engineering → (optional) text features via the TrialDura submodule → tree‑based regression with ablation studies and out‑of‑time evaluation. The main regression target is `duration_days`.

- **Background literature** Relevant literature PDFs for context under `Background Literature/`. 
- **ETL** scripts for collecting and flattening ClinicalTrials.gov records (`src/etl/…`). 
- **Feature engineering** entry point: `python -m src.features.make_features`  
  Reads `data/interim/ctgov_flat.parquet` and writes `data/processed/features_v6.parquet`. 
- **(Optional) Text features**: integrate HAN/transformer text embeddings with `src/features/merge_text_features.py`.
- **Models & analysis** (`src/models/…`). 

