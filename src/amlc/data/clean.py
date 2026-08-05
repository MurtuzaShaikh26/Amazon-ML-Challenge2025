import pandas as pd
from amlc.config import ConfigDict
from amlc.logging_utils import get_logger

logger = get_logger("amlc.data.clean")

def basic_clean(df: pd.DataFrame, cfg: ConfigDict) -> pd.DataFrame:
    """
    Config-gated dataset cleaning applied strictly to the training split.
    Logs row count reductions per step.
    """
    initial_rows = len(df)
    clean_cfg = cfg.data.clean if hasattr(cfg.data, "clean") else cfg.get("data", {}).get("clean", {})
    
    df_clean = df.copy()

    # Step 1: Drop null price
    if clean_cfg.get("drop_null_price", True):
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=["price"])
        logger.info(f"Clean step [drop_null_price]: removed {before - len(df_clean)} rows")

    # Step 2: Drop nonpositive price
    if clean_cfg.get("drop_nonpositive_price", True):
        before = len(df_clean)
        df_clean = df_clean[df_clean["price"] > 0]
        logger.info(f"Clean step [drop_nonpositive_price]: removed {before - len(df_clean)} rows")

    # Step 3: Trim price quantiles
    quantiles = clean_cfg.get("trim_price_quantiles", [0.0, 1.0])
    q_low, q_high = quantiles[0], quantiles[1]
    if q_low > 0.0 or q_high < 1.0:
        before = len(df_clean)
        low_val = df_clean["price"].quantile(q_low)
        high_val = df_clean["price"].quantile(q_high)
        df_clean = df_clean[(df_clean["price"] >= low_val) & (df_clean["price"] <= high_val)]
        logger.info(f"Clean step [trim_price_quantiles {q_low}-{q_high}]: removed {before - len(df_clean)} rows")

    # Step 4: Dedup exact text
    if clean_cfg.get("dedup_exact_text", False):
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["catalog_content"])
        logger.info(f"Clean step [dedup_exact_text]: removed {before - len(df_clean)} rows")

    total_removed = initial_rows - len(df_clean)
    logger.info(f"Basic cleaning complete. Initial: {initial_rows}, Final: {len(df_clean)}, Removed: {total_removed}")

    return df_clean
