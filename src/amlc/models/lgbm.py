from pathlib import Path
from typing import Any, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from amlc.config import ConfigDict
from amlc.models.base import BaseModel
from amlc.logging_utils import get_logger

logger = get_logger("amlc.models.lgbm")

class LGBMModel(BaseModel):
    """
    LightGBM regressor wrapper implementing the BaseModel contract.
    """
    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        params = dict(cfg.model.params) if hasattr(cfg.model, "params") else {}
        self.model = lgb.LGBMRegressor(**params)
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any = None,
        y_val: np.ndarray = None
    ) -> None:
        logger.info(f"Fitting LightGBM model with params: {self.cfg.model.params}...")
        
        callbacks = []
        early_stopping_rounds = self.cfg.model.get("early_stopping_rounds", None)
        log_period = self.cfg.model.get("log_period", None)

        if X_val is not None and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
        if log_period and log_period > 0:
            callbacks.append(lgb.log_evaluation(period=log_period))

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_names"] = ["val"]
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(X_train, y_train, **fit_kwargs)
        logger.info("LightGBM fitting completed.")

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 50
    ) -> pd.DataFrame:
        """
        Extract top feature importances by gain.
        """
        importances = self.model.booster_.feature_importance(importance_type="gain")
        n_features = len(importances)
        
        if feature_names and len(feature_names) == n_features:
            names = feature_names
        else:
            names = [f"feature_{i}" for i in range(n_features)]
            
        df_imp = pd.DataFrame({
            "feature": names,
            "importance_gain": importances
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        
        return df_imp.head(top_k)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)
        logger.info(f"Saved LightGBM model to: {p}")

    @classmethod
    def load(cls, path: Union[str, Path], cfg: Optional[ConfigDict] = None) -> "LGBMModel":
        p = Path(path)
        if cfg is None:
            cfg = ConfigDict({"model": {"params": {}}})
        instance = cls(cfg)
        instance.model = joblib.load(p)
        logger.info(f"Loaded LightGBM model from: {p}")
        return instance
