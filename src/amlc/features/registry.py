from typing import Callable, Dict, Tuple, Any
import pandas as pd
from scipy.sparse import csr_matrix
from amlc.config import ConfigDict
from amlc.features.tfidf import build_tfidf_regex_v1

FeatureBuilderFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, ConfigDict], Tuple[csr_matrix, csr_matrix, csr_matrix, Dict[str, Any]]]

FEATURE_BUILDERS: Dict[str, FeatureBuilderFn] = {
    "tfidf_regex_v1": build_tfidf_regex_v1,
}

def register_feature_builder(name: str, fn: FeatureBuilderFn) -> None:
    """Register a new feature builder function."""
    if name in FEATURE_BUILDERS:
        raise ValueError(f"Feature builder '{name}' is already registered.")
    FEATURE_BUILDERS[name] = fn

def get_feature_builder(name: str) -> FeatureBuilderFn:
    """Retrieve a registered feature builder function by name."""
    if name not in FEATURE_BUILDERS:
        raise KeyError(f"Unknown feature builder '{name}'. Registered: {list(FEATURE_BUILDERS.keys())}")
    return FEATURE_BUILDERS[name]
