from pathlib import Path
from typing import Dict
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from amlc.paths import RESULTS_DIR
from amlc.logging_utils import get_logger

logger = get_logger("amlc.data.splits")

def make_splits(df: pd.DataFrame, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate deterministic stratified train/val/test split (50,000 / 5,000 / 20,000).
    Saves split metadata to results/splits/split_seed{seed}.json.
    """
    logger.info(f"Generating deterministic splits with seed={seed}")
    
    # 1. Sort by sample_id ascending first
    df_sorted = df.sort_values("sample_id").reset_index(drop=True)
    
    # 2. Log price transform and 20 quantile bins for stratification
    log_price = np.log1p(df_sorted["price"].values)
    bins = pd.qcut(log_price, q=20, labels=False, duplicates="drop")
    
    total_len = len(df_sorted)
    test_size = 20000
    val_size = 5000
    train_size = total_len - test_size - val_size
    
    # First stage: split off test set (20,000)
    df_rem, df_test = train_test_split(
        df_sorted,
        test_size=test_size,
        stratify=bins,
        random_state=seed,
        shuffle=True
    )
    
    # Re-bin remaining set for second stage split
    log_price_rem = np.log1p(df_rem["price"].values)
    bins_rem = pd.qcut(log_price_rem, q=20, labels=False, duplicates="drop")
    
    # Second stage: split remaining into train (50,000) and val (5,000)
    df_train, df_val = train_test_split(
        df_rem,
        test_size=val_size,
        stratify=bins_rem,
        random_state=seed,
        shuffle=True
    )
    
    train_ids = df_train["sample_id"].values
    val_ids = df_val["sample_id"].values
    test_ids = df_test["sample_id"].values
    
    # Assertions
    set_train, set_val, set_test = set(train_ids), set(val_ids), set(test_ids)
    assert len(set_train.intersection(set_val)) == 0, "Train and Val splits overlap!"
    assert len(set_train.intersection(set_test)) == 0, "Train and Test splits overlap!"
    assert len(set_val.intersection(set_test)) == 0, "Val and Test splits overlap!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == total_len, f"Split counts sum to {len(train_ids)+len(val_ids)+len(test_ids)}, expected {total_len}"
    
    # Stratification check on log price mean
    mean_train_log = np.mean(np.log1p(df_train["price"].values))
    mean_val_log = np.mean(np.log1p(df_val["price"].values))
    mean_test_log = np.mean(np.log1p(df_test["price"].values))
    
    diff_val = abs(mean_train_log - mean_val_log)
    diff_test = abs(mean_train_log - mean_test_log)
    
    logger.info(f"Log-price means - Train: {mean_train_log:.4f}, Val: {mean_val_log:.4f}, Test: {mean_test_log:.4f}")
    assert diff_val < 0.01, f"Validation log-price mean diff {diff_val:.4f} >= 0.01"
    assert diff_test < 0.01, f"Test log-price mean diff {diff_test:.4f} >= 0.01"
    
    # Write audit JSON
    splits_dir = RESULTS_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_json = splits_dir / f"split_seed{seed}.json"
    
    split_payload = {
        "seed": seed,
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "train_ids": [int(x) for x in train_ids],
        "val_ids": [int(x) for x in val_ids],
        "test_ids": [int(x) for x in test_ids],
    }
    
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(split_payload, f)
        logger.info(f"Saved split audit JSON to: {out_json}")
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not save split audit JSON: {e}")
        
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
