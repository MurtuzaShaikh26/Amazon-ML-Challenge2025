from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from amlc.config import ConfigDict
from amlc.features.text_regex import extract_regex_features
from amlc.logging_utils import get_logger

logger = get_logger("amlc.features.tfidf")

def build_tfidf_regex_v1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: ConfigDict
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, Dict[str, Any]]:
    """
    Build TF-IDF word and char features combined with regex features.
    Vectorizers are fit ONLY on train_df and applied to val_df and test_df.
    Returns (X_train, X_val, X_test, artifacts_dict).
    """
    logger.info("Building TF-IDF + Regex feature pipeline v1...")
    
    # 1. Regex features
    use_regex = cfg.features.get("use_regex_features", True)
    if use_regex:
        logger.info("Extracting regex features from catalog_content...")
        regex_train = extract_regex_features(train_df["catalog_content"])
        regex_val = extract_regex_features(val_df["catalog_content"])
        regex_test = extract_regex_features(test_df["catalog_content"])
        
        regex_cols = list(regex_train.columns)
        sparse_regex_train = csr_matrix(regex_train.values)
        sparse_regex_val = csr_matrix(regex_val.values)
        sparse_regex_test = csr_matrix(regex_test.values)
    else:
        regex_cols = []
        sparse_regex_train = csr_matrix((len(train_df), 0))
        sparse_regex_val = csr_matrix((len(val_df), 0))
        sparse_regex_test = csr_matrix((len(test_df), 0))

    # 2. Word TF-IDF
    word_cfg = cfg.features.get("word_tfidf", {})
    logger.info(f"Fitting Word TF-IDF vectorizer with params: {word_cfg}...")
    word_vec = TfidfVectorizer(
        ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
        max_features=word_cfg.get("max_features", 200000),
        min_df=word_cfg.get("min_df", 3),
        sublinear_tf=word_cfg.get("sublinear_tf", True),
        strip_accents="unicode",
        lowercase=True
    )
    sparse_word_train = word_vec.fit_transform(train_df["catalog_content"])
    sparse_word_val = word_vec.transform(val_df["catalog_content"])
    sparse_word_test = word_vec.transform(test_df["catalog_content"])
    word_feature_names = [f"word_tfidf_{name}" for name in word_vec.get_feature_names_out()]

    # 3. Char TF-IDF
    char_cfg = cfg.features.get("char_tfidf", {})
    logger.info(f"Fitting Char TF-IDF vectorizer with params: {char_cfg}...")
    char_vec = TfidfVectorizer(
        analyzer=char_cfg.get("analyzer", "char_wb"),
        ngram_range=tuple(char_cfg.get("ngram_range", [3, 5])),
        max_features=char_cfg.get("max_features", 100000),
        min_df=char_cfg.get("min_df", 3),
        sublinear_tf=char_cfg.get("sublinear_tf", True)
    )
    sparse_char_train = char_vec.fit_transform(train_df["catalog_content"])
    sparse_char_val = char_vec.transform(val_df["catalog_content"])
    sparse_char_test = char_vec.transform(test_df["catalog_content"])
    char_feature_names = [f"char_tfidf_{name}" for name in char_vec.get_feature_names_out()]

    # Combine blocks
    blocks_train = [b for b in [sparse_regex_train, sparse_word_train, sparse_char_train] if b.shape[1] > 0]
    blocks_val = [b for b in [sparse_regex_val, sparse_word_val, sparse_char_val] if b.shape[1] > 0]
    blocks_test = [b for b in [sparse_regex_test, sparse_word_test, sparse_char_test] if b.shape[1] > 0]

    X_train = hstack(blocks_train).tocsr()
    X_val = hstack(blocks_val).tocsr()
    X_test = hstack(blocks_test).tocsr()

    feature_names = regex_cols + word_feature_names + char_feature_names

    # Sparsity stats
    non_zeros = X_train.nnz
    total_elements = X_train.shape[0] * X_train.shape[1]
    sparsity = 100.0 * (1.0 - (non_zeros / total_elements)) if total_elements > 0 else 0.0
    logger.info(f"Feature matrix complete. Train shape: {X_train.shape}, Sparsity: {sparsity:.2f}%")

    artifacts = {
        "word_vectorizer": word_vec,
        "char_vectorizer": char_vec,
        "feature_names": feature_names,
    }

    return X_train, X_val, X_test, artifacts
