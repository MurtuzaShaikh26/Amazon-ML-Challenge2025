import numpy as np
import pytest
from amlc.metrics.smape import smape, smape_per_bucket

def test_smape_zero_for_identical():
    y = np.array([10.0, 50.0, 100.0, 250.0])
    assert smape(y, y) == pytest.approx(0.0, abs=1e-6)

def test_smape_hand_calculated():
    # 2 * |200 - 100| / (200 + 100) = 200 / 300 = 2/3 = 66.6666667%
    y_true = np.array([100.0])
    y_pred = np.array([200.0])
    assert smape(y_true, y_pred) == pytest.approx(66.66666666666667, abs=1e-5)

def test_smape_symmetry():
    a = np.array([15.0, 30.0, 45.0])
    b = np.array([12.0, 35.0, 40.0])
    assert smape(a, b) == pytest.approx(smape(b, a), abs=1e-6)

def test_smape_near_zero():
    y_true = np.array([0.0, 0.0, 1e-4])
    y_pred = np.array([0.0, 1e-4, 0.0])
    score = smape(y_true, y_pred)
    assert not np.isnan(score)
    assert not np.isinf(score)
    assert score >= 0.0

def test_smape_per_bucket():
    y_true = np.linspace(1.0, 100.0, 100)
    y_pred = y_true * 1.1
    df_bucket = smape_per_bucket(y_true, y_pred, n_buckets=5)
    assert len(df_bucket) == 5
    assert list(df_bucket.columns) == ["bucket", "min_price", "max_price", "count", "mean_true", "mean_pred", "smape"]
    assert (df_bucket["smape"] > 0).all()
