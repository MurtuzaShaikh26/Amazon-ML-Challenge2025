from typing import Dict, Type
from amlc.models.base import BaseModel
from amlc.models.lgbm import LGBMModel

MODELS: Dict[str, Type[BaseModel]] = {
    "lgbm": LGBMModel,
}

def register_model(name: str, model_cls: Type[BaseModel]) -> None:
    """Register a new model class."""
    if name in MODELS:
        raise ValueError(f"Model class '{name}' is already registered.")
    MODELS[name] = model_cls

def get_model_class(name: str) -> Type[BaseModel]:
    """Retrieve registered model class by name."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODELS.keys())}")
    return MODELS[name]
