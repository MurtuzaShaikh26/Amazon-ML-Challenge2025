from pathlib import Path
from typing import Optional, Union
import os
import pandas as pd

from amlc.paths import find_train_csv
from amlc.logging_utils import get_logger

logger = get_logger("amlc.data.load")

REQUIRED_COLUMNS = ["sample_id", "catalog_content", "image_link", "price"]

def load_train(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Locate CSV, read it, validate columns, coerce dtypes, and log row count & nulls.
    Caches parsed DataFrame as parquet next to the source file on first load.
    """
    if csv_path is None:
        target_csv = find_train_csv()
    else:
        target_csv = Path(csv_path)

    parquet_cache = target_csv.parent / f"{target_csv.stem}_cache.parquet"

    # Read from cache if available
    if parquet_cache.exists():
        logger.info(f"Loading cached parquet dataset from: {parquet_cache}")
        df = pd.read_parquet(parquet_cache)
        logger.info(f"Loaded {len(df)} rows from parquet cache.")
        return df

    logger.info(f"Reading CSV dataset from: {target_csv}")
    df = pd.read_csv(target_csv)

    # Validate columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}. Found: {list(df.columns)}")

    # Coerce dtypes
    df["sample_id"] = df["sample_id"].astype("int64")
    df["catalog_content"] = df["catalog_content"].fillna("").astype("str")
    df["image_link"] = df["image_link"].fillna("").astype("str")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Logging metadata
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    logger.info(f"Null counts by column: {null_counts}")

    # Cache parquet if location is writable
    try:
        df.to_parquet(parquet_cache, index=False)
        logger.info(f"Cached parsed dataset to parquet at: {parquet_cache}")
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not cache parquet dataset due to permissions: {e}")

    return df
