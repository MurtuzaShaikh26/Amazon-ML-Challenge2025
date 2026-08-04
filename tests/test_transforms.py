import numpy as np
import pytest
from amlc.targets.transforms import (
    IdentityTransform,
    Log1pTransform,
    Log2Transform,
    Log10Transform,
    get_target_transform
)

@pytest.mark.parametrize("name", ["identity", "log1p", "log2", "log10"])
def test_target_transforms_roundtrip(name: str):
    y_orig = np.array([0.5, 1.0, 10.5, 99.9, 1500.0], dtype=np.float64)
    transform = get_target_transform(name, min_price=0.01)
    
    z = transform.forward(y_orig)
    y_rec = transform.inverse(z)
    
    np.testing.assert_allclose(y_orig, y_rec, rtol=1e-5, atol=1e-5)

def test_target_transforms_clipping():
    transform = Log1pTransform(min_price=0.01)
    z_negative = np.array([-10.0, -1.0])
    y_rec = transform.inverse(z_negative)
    
    assert (y_rec >= 0.01).all()
