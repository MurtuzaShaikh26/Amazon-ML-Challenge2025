import numpy as np
import pandas as pd
import pytest
from amlc.data.splits import make_splits

def create_synthetic_df(n_rows: int = 75000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Generate right-skewed log-normal price distribution similar to spec
    log_prices = rng.normal(loc=3.0, scale=1.2, size=n_rows)
    prices = np.expm1(np.clip(log_prices, 0.1, 8.0)) + 0.13
    
    return pd.DataFrame({
        "sample_id": np.arange(1, n_rows + 1, dtype=np.int64),
        "catalog_content": [f"Item Name: Product {i} Value: {i % 100} Unit: Fl Oz" for i in range(n_rows)],
        "image_link": [f"https://images.amazon.com/item_{i}.jpg" for i in range(n_rows)],
        "price": prices
    })

def test_make_splits_sizes_and_disjoint():
    df = create_synthetic_df(75000, seed=42)
    splits = make_splits(df, seed=42)
    
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    
    assert len(train_ids) == 50000
    assert len(val_ids) == 5000
    assert len(test_ids) == 20000
    
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert len(s_train.intersection(s_val)) == 0
    assert len(s_train.intersection(s_test)) == 0
    assert len(s_val.intersection(s_test)) == 0

def test_make_splits_determinism():
    df = create_synthetic_df(75000, seed=42)
    splits1 = make_splits(df, seed=42)
    splits2 = make_splits(df, seed=42)
    
    np.testing.assert_array_equal(splits1["train"], splits2["train"])
    np.testing.assert_array_equal(splits1["val"], splits2["val"])
    np.testing.assert_array_equal(splits1["test"], splits2["test"])

def test_make_splits_log_price_balance():
    df = create_synthetic_df(75000, seed=42)
    splits = make_splits(df, seed=42)
    
    df_indexed = df.set_index("sample_id")
    
    mean_train = np.mean(np.log1p(df_indexed.loc[splits["train"], "price"].values))
    mean_val = np.mean(np.log1p(df_indexed.loc[splits["val"], "price"].values))
    mean_test = np.mean(np.log1p(df_indexed.loc[splits["test"], "price"].values))
    
    assert abs(mean_train - mean_val) < 0.01
    assert abs(mean_train - mean_test) < 0.01
