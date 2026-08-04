# Research Journal & Design Notes

---

## Design Decisions & Reasonable Choices

Where the spec was silent, the following architectural choices were implemented:

1. **LightGBM Objective & Log Target Rationale**:
   - `objective: regression_l1` (MAE) in log-price space (`log1p`). MAE in log-space corresponds to median regression on relative errors ($|\ln \hat{y} - \ln y|$), which closely approximates the SMAPE loss objective ($2|\hat{y} - y| / (\hat{y} + y)$) far better than L2 regression.
   - `feature_fraction: 0.3` is selected due to high dimensionality of the combined TF-IDF word and character n-gram matrix (~300,000 sparse features).
2. **Regex Feature Extraction Base Units**:
   - Standardized unit normalization factor converts liquid volume (`fl oz`, `ml`, `liter`) and weight (`lb`, `kg`, `g`) into fluid/weight ounce equivalents (`oz`).
   - Categorical brand proxy (`first_token_hash`) uses deterministic `crc32(word) % 5000` to create a bounded integer bucket handled cleanly by LightGBM.
3. **Environment & Path Resolution**:
   - Automatic Kaggle detection checks `/kaggle/input`. When present, it globs all dataset directories under `/kaggle/input/*` for `.csv` files so mirror dataset name variations on Kaggle do not break data loading.
4. **PyTorch Optional Import**:
   - `seed.py` dynamically imports `torch` inside a `try/except` block, keeping CPU-only tabular runs lightweight and free of torch runtime overhead.
5. **Multiplicative Validation Calibration**:
   - Grid search sweeps $m \in [0.80, 1.06]$ on validation predictions in **original price space** (after target inversion), ensuring calibration directly minimizes SMAPE.

---

## Run 001: LightGBM on TF-IDF + Regex Features

- **Run ID**: `run001_lgbm_tfidf`
- **Date**: 2026-08-04
- **Status**: Scaffolded / Ready for Kaggle Execution

### Hypothesis
A lightweight GBDT baseline using sparse TF-IDF (word 1-2 ngrams + char 3-5 ngrams) and regex-extracted catalog attributes (`pack_count`, `parsed_value`, unit conversions, bullet counts) using L1 loss on `log1p(price)` will establish a reliable text-only holdout benchmark of ~50–52% SMAPE in under 20 minutes CPU runtime.

### Configuration Rationale
- **Target Transform**: `log1p` to compress severe right-skewed price distribution (max ~2796, 98.8% in [0.13, 139.92]).
- **Min Price Clipping**: `0.01` to prevent extreme SMAPE division spikes near zero.
- **Model Params**:
  - `num_leaves: 127`, `min_child_samples: 20` for deep tree capacity on text features.
  - `bagging_fraction: 0.8`, `bagging_freq: 1`, `lambda_l2: 1.0` to mitigate overfitting on sparse features.
  - `early_stopping_rounds: 100` monitored on 5,000 validation split.

### Result
*(To be updated after Kaggle notebook execution)*

- **Val SMAPE (Uncalibrated)**: `%`
- **Val SMAPE (Calibrated)**: `%`
- **Test SMAPE (Calibrated)**: `%`
- **Optimal Multiplier**: `m = `
- **Training Time**: ` seconds`

### What I Learned
*(To be updated post-run)*
