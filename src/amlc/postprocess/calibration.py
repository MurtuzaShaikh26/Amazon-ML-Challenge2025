from typing import Tuple, Union
import numpy as np
from amlc.metrics.smape import smape
from amlc.logging_utils import get_logger

logger = get_logger("amlc.postprocess.calibration")

def fit_multiplier(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    grid: Union[np.ndarray, list] = np.arange(0.80, 1.06, 0.005)
) -> Tuple[float, float]:
    """
    Sweep a multiplicative constant on validation predictions in original price space.
    Returns (best_multiplier, best_val_smape).
    """
    best_m = 1.0
    best_score = float("inf")

    for m in grid:
        m_float = float(m)
        score = smape(y_true_val, m_float * y_pred_val)
        if score < best_score:
            best_score = score
            best_m = m_float

    logger.info(f"Calibration grid search complete. Optimal multiplier: {best_m:.4f}, Val SMAPE: {best_score:.4f}%")
    return best_m, best_score

def apply_multiplier(y_pred: np.ndarray, multiplier: float) -> np.ndarray:
    """
    Apply multiplicative scalar to predictions.
    """
    return np.asarray(y_pred, dtype=np.float64) * float(multiplier)
