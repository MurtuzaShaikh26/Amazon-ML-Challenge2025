from typing import Protocol, Dict
import numpy as np

class TargetTransform(Protocol):
    def forward(self, y: np.ndarray) -> np.ndarray: ...
    def inverse(self, z: np.ndarray) -> np.ndarray: ...

class IdentityTransform:
    def __init__(self, min_price: float = 0.01):
        self.min_price = min_price

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(z, dtype=np.float64), self.min_price, None)

class Log1pTransform:
    def __init__(self, min_price: float = 0.01):
        self.min_price = min_price

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(np.asarray(y, dtype=np.float64))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        inv = np.expm1(np.asarray(z, dtype=np.float64))
        return np.clip(inv, self.min_price, None)

class Log2Transform:
    def __init__(self, min_price: float = 0.01):
        self.min_price = min_price

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.log2(np.clip(np.asarray(y, dtype=np.float64), 1e-8, None))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        inv = np.power(2.0, np.asarray(z, dtype=np.float64))
        return np.clip(inv, self.min_price, None)

class Log10Transform:
    def __init__(self, min_price: float = 0.01):
        self.min_price = min_price

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.log10(np.clip(np.asarray(y, dtype=np.float64), 1e-8, None))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        inv = np.power(10.0, np.asarray(z, dtype=np.float64))
        return np.clip(inv, self.min_price, None)

TARGET_TRANSFORMS = {
    "identity": IdentityTransform,
    "log1p": Log1pTransform,
    "log2": Log2Transform,
    "log10": Log10Transform,
}

def get_target_transform(name: str, min_price: float = 0.01) -> TargetTransform:
    """Instantiate a target transform class by name."""
    if name not in TARGET_TRANSFORMS:
        raise KeyError(f"Unknown target transform '{name}'. Available: {list(TARGET_TRANSFORMS.keys())}")
    return TARGET_TRANSFORMS[name](min_price=min_price)
