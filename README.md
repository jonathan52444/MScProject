# Forecasting Clinical Trial Duration from Structured and Unstructured Protocol Data

Pipeline to predict **clinical trial duration** from ClinicalTrials.gov data: ETL → feature engineering → random-forest predictor. 

- **Background literature** Relevant literature PDFs for context under `Background Literature/`. 
- **ETL** scripts for collecting and flattening ClinicalTrials.gov records (`src/etl/…`). 
- **Feature engineering** entry point: `python -m src.features.make_features`  
  Reads `data/interim/ctgov_flat.parquet` and writes `data/processed/features_v6.parquet`. 
- **(Optional) Text features**: integrate HAN/transformer text embeddings with `src/features/merge_text_features.py`.
- **Models & analysis** (`src/models/…`). 

## Quick start
```bash
# clone
git clone https://github.com/jonathan52444/MScProject.git
cd MScProject

# Pull studies from ClinicalTrials.gov
python -m src.etl.pull_all

# Flatten JSON files to a single Parquet file
python -m src.etl.flatten_all

# Make features
python -m src.features.make_features

# (Optional) Generate HAN Embeddings and merge features
python -m src.text.train_han
python -m src.features.merge_text_features

# Run evaluation
python -m src.models.trial_duration_rf_oot_evaluation
