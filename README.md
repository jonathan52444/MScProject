# Forecasting Clinical Trial Duration from Structured and Unstructured Protocol Data

Pipeline to predict **clinical trial duration** from ClinicalTrials.gov data: ETL → feature engineering → random-forest predictor. 

- **Background literature** Relevant literature PDFs for context under `Background Literature/`. 
- **ETL** scripts for collecting and flattening ClinicalTrials.gov records (`src/etl/…`). 
- **Feature engineering** entry point: `python -m src.features.make_features`  
  Reads `data/interim/ctgov_flat.parquet` and writes `data/processed/features_v6.parquet`. 
- **(Optional) Text features**: integrate HAN/transformer text embeddings with `src/features/merge_text_features.py`.
- **Models & analysis** (`src/models/…`). 

## Abstract 

Reliable forecasts of clinical trial duration are essential for budgeting, vendor coordination, site planning, and timely patient access. We demonstrate that a compact model—taking as input structured clinical trial protocol metadata and a single short protocol paragraph (the Brief Summary)—achieves strong accuracy at low computational cost. Using a snapshot of ClinicalTrials.gov data (251,488 completed/terminated studies), a metadata-only random forest model attains a mean absolute error (MAE) of 1.390 years on an out-of-time split; adding a 256-dimensional Brief Summary embedding reduces MAE to 1.349 years (≈15 days reduction). Although our model analyses significantly less text than recent approaches that use full Eligibility Criteria and detailed embeddings of diseases or drugs, it narrows the gap toward state-of-the-art performance (1.175 years on the same validation set) while remaining straightforward, auditable, and deployable with modest computational resources.

Ablation analyses show that a focused set of design fields captures most of the predictive information in structured protocol metadata. To support clinical operations planning, we introduce three indices: a Complexity Score summarising procedural burden; a Novelty Score quantifying prior occurrence of similar trials; and an Attractiveness Score capturing recruitment appeal. These scores provide actionable guidance for protocol simplification and risk communication through their transparency and ease of interpretation by human experts.

On a small benchmark of 24 trials independently annotated by clinical trial experts at UCB through industry collaboration, our metadata-only model achieves an MAE of 0.763 years, approaching the human expert benchmark of 0.667 years. Collectively, these results suggest that a compact forecasting model—leveraging only structured protocol metadata and brief textual summaries—can provide useful early predictions efficiently, without the computational burden associated with analysing extensive protocol sections.

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
