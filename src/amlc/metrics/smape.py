from typing import Union
import numpy as np
import pandas as pd

def smape(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    eps: float = 1e-8
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE) in percentage terms (0 to 200%).
    Formula: 100/n * sum(2 * |y_pred - y_true| / (|y_pred| + |y_true| + eps))
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)}, len(y_pred)={len(yp)}")
    if len(yt) == 0:
        return 0.0

    numerator = 2.0 * np.abs(yp - yt)
    denominator = np.abs(yp) + np.abs(yt) + eps
    
    return float(100.0 * np.mean(numerator / denominator))

def smape_per_bucket(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    n_buckets: int = 10
) -> pd.DataFrame:
    """
    Calculate SMAPE per log-price decile/bucket.
    Returns a DataFrame with columns:
    [bucket, min_price, max_price, count, mean_true, mean_pred, smape]
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    df = pd.DataFrame({"y_true": yt, "y_pred": yp})
    df["log_true"] = np.log1p(df["y_true"])

    # Quantile binning by true log-price
    df["bucket"] = pd.qcut(df["log_true"], q=n_buckets, labels=False, duplicates="drop")

    records = []
    for b_id, group in df.groupby("bucket"):
        b_smape = smape(group["y_true"].values, group["y_pred"].values)
        records.append({
            "bucket": int(b_id),
            "min_price": float(group["y_true"].min()),
            "max_price": float(group["y_true"].max()),
            "count": int(len(group)),
            "mean_true": float(group["y_true"].mean()),
            "mean_pred": float(group["y_pred"].mean()),
            "smape": float(b_smape),
        })

    return pd.DataFrame(records)
