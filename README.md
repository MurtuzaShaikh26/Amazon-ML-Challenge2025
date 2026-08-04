# Amazon ML Challenge 2026 - Research Repository

Modular, config-driven machine learning research framework built for the Amazon ML Challenge 2026 (using the 2025 dataset edition). The challenge targets product price prediction from product text catalog data (`catalog_content`) and product images (`image_link`). Performance is evaluated using Symmetric Mean Absolute Percentage Error (**SMAPE**).

## Two-Month Research Strategy

This repository will host dozens of iteration runs over two months spanning GBDTs (LightGBM, CatBoost), dense embeddings (E5, BGE), fine-tuned language models (DistilBERT, DeBERTa-v3), vision models (CLIP, SigLIP), multimodal LLMs (Qwen-VL), custom loss functions (approximate SMAPE, smooth L1), and post-processing calibration. The core architecture enforces complete decoupling: **adding Experiment N+1 requires only writing a config file and optional feature/model builder registration without altering any pipeline code**.

---

## Repository Structure

```
.
├── README.md                          # Repository documentation & leaderboard
├── NOTES.md                           # Research log & technical decisions
├── requirements.txt                   # Dependency definitions
├── .gitignore                         # Data/cache ignore rules
├── configs/
│   ├── base.yaml                      # Base defaults (paths, split ratios, target transforms)
│   └── run001_lgbm_tfidf.yaml         # Run 1 config (TF-IDF + Regex + LightGBM L1)
├── src/
│   └── amlc/                          # Core amlc package
│       ├── config.py                  # YAML inheritance & SHA1 hashing
│       ├── seed.py                    # Seeding utilities
│       ├── paths.py                   # Environment detection (Kaggle vs Local)
│       ├── logging_utils.py           # Logger configuration
│       ├── data/                      # Dataset loading, deterministic splitting, cleaning
│       ├── features/                  # Regex extraction, sparse TF-IDF, builder registry
│       ├── targets/                   # Invertible target transforms (Log1p, Log2, Log10)
│       ├── models/                    # BaseModel interface, LightGBM wrapper, model registry
│       ├── metrics/                   # SMAPE metric & decile slice diagnostics
│       ├── postprocess/               # Validation multiplicative calibration
│       ├── results/                   # Run artifact export & atomic leaderboard tracker
│       └── pipeline/                  # End-to-end execution narrative
├── kaggle_notebook/                  # Kaggle execution notebooks
│   └── run001_lgbm_tfidf.ipynb
├── scripts/
│   └── run_local.py                   # Local CLI runner
├── results/
│   ├── leaderboard.csv                # Master experiment leaderboard
│   └── runs/                          # Per-run predictions, metrics, and logs
└── tests/                             # Pytest suite
```

---

## How to Add a New Experiment (N+1)

1. **(Optional) Register a new Feature Builder**: Create a function in `src/amlc/features/` with signature `build(train_df, val_df, test_df, cfg) -> (X_train, X_val, X_test, artifacts)` and register it in `features/registry.py`.
2. **(Optional) Register a new Model**: Inherit from `BaseModel` in `src/amlc/models/` and register the class name in `models/registry.py`.
3. **Write Configuration**: Create `configs/run00X_<name>.yaml` extending `base.yaml`.
4. **Run or Create Notebook**: Execute via CLI or copy `kaggle_notebook/run001_lgbm_tfidf.ipynb` pointing to your new YAML config.

---

## Running Locally

1. Place dataset CSV in `./data/train.csv` (or any `.csv` inside `./data/`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
3. Run tests:
   ```bash
   pytest tests/
   ```
4. Execute experiment run:
   ```bash
   python scripts/run_local.py --config configs/run001_lgbm_tfidf.yaml
   ```

---

## Running on Kaggle

1. Push your changes to GitHub repository (`MurtuzaShaikh26/Amazon-ML-Challenge2025`).
2. On Kaggle, click **New Dataset** -> **Import from GitHub** -> enter repo `MurtuzaShaikh26/Amazon-ML-Challenge2025`.
3. Create a Kaggle Notebook and attach two datasets:
   - `suvroo/amazon-ml` (the dataset CSV)
   - `murtuzashaikh26/amazon-ml-challenge2025` (your uploaded code dataset)
4. Open or import `kaggle_notebook/run001_lgbm_tfidf.ipynb` and run all cells.
5. Download the output zip archive from `/kaggle/working/` and commit results back to Git.

---

## Experiment Leaderboard

| run_id | timestamp | description | model | features | target_transform | loss | n_train | val_smape | val_smape_calibrated | test_smape | test_smape_calibrated | calibration_multiplier | train_seconds | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `run001_lgbm_tfidf` | - | Baseline: LightGBM on TF-IDF + Regex | lgbm | tfidf_regex_v1 | log1p | regression_l1 | 50000 | - | - | - | - | - | - | - |
