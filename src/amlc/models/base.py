from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union
import numpy as np
from amlc.config import ConfigDict

class BaseModel(ABC):
    """
    Abstract base interface for all models in the repository.
    
    Note: `y_train` and `y_val` passed to `fit` are already in target-transformed space
    (e.g., log1p). `predict` returns predictions in transformed space as well.
    The outer pipeline owns applying the inverse target transformation.
    """
    def __init__(self, cfg: ConfigDict):
        self.cfg = cfg

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any = None,
        y_val: np.ndarray = None
    ) -> None:
        """Fit model on training data with optional validation set for early stopping."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Generate predictions in target-transformed space."""
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint to disk."""
        raise NotImplementedError("Model save method not implemented.")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """Load model checkpoint from disk."""
        raise NotImplementedError("Model load method not implemented.")
