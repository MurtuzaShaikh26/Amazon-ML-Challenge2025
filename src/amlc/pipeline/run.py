from pathlib import Path
from typing import Any, Dict, Union
import time
import numpy as np
import pandas as pd

from amlc.config import load_config, ConfigDict
from amlc.seed import set_seed
from amlc.paths import RESULTS_DIR
from amlc.logging_utils import setup_logger
from amlc.data.load import load_train
from amlc.data.splits import make_splits
from amlc.data.clean import basic_clean
from amlc.features.registry import get_feature_builder
from amlc.targets.transforms import get_target_transform
from amlc.models.registry import get_model_class
from amlc.metrics.smape import smape, smape_per_bucket
from amlc.postprocess.calibration import fit_multiplier, apply_multiplier
from amlc.results.tracker import save_run_artifacts

def run_experiment(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Master end-to-end experiment pipeline function.
    Reads like a linear technical narrative from config loading to artifact recording.
    """
    # 1. Load config, set seed, setup logging
    cfg = load_config(config_path)
    run_id = cfg.run_id
    
    run_log_file = RESULTS_DIR / "runs" / run_id / "log.txt"
    logger = setup_logger("amlc.pipeline", level=cfg.get("logging", {}).get("level", "INFO"), log_file=run_log_file)
    
    logger.info("=" * 70)
    logger.info(f"STARTING EXPERIMENT: {run_id}")
    logger.info(f"Description: {cfg.description}")
    logger.info("=" * 70)

    set_seed(cfg.get("seed", 42))

    # 2. Load dataset
    df = load_train()

    # 3. Make splits (fixed split seed 42)
    split_seed = cfg.get("splits", {}).get("seed", 42)
    splits = make_splits(df, seed=split_seed)
    
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    df_train_raw = df[df["sample_id"].isin(train_ids)].copy()
    df_val = df[df["sample_id"].isin(val_ids)].copy().sort_values("sample_id").reset_index(drop=True)
    df_test = df[df["sample_id"].isin(test_ids)].copy().sort_values("sample_id").reset_index(drop=True)

    # 4. Clean train split ONLY
    logger.info("Cleaning training split...")
    df_train = basic_clean(df_train_raw, cfg).sort_values("sample_id").reset_index(drop=True)

    # 5. Build features from registry
    builder_name = cfg.features.get("builder", "tfidf_regex_v1")
    feature_builder_fn = get_feature_builder(builder_name)
    
    logger.info(f"Building features using registered builder: '{builder_name}'...")
    X_train, X_val, X_test, feature_artifacts = feature_builder_fn(df_train, df_val, df_test, cfg)

    # 6. Target transform
    target_cfg = cfg.get("target", {})
    transform_name = target_cfg.get("transform", "log1p")
    min_price = target_cfg.get("min_price", 0.01)
    
    transform = get_target_transform(transform_name, min_price=min_price)
    
    logger.info(f"Applying target transform: '{transform_name}' (min_price={min_price})...")
    y_train_raw = df_train["price"].values
    y_val_raw = df_val["price"].values
    y_test_raw = df_test["price"].values

    z_train = transform.forward(y_train_raw)
    z_val = transform.forward(y_val_raw)

    # 7. Model instantiation & training
    model_name = cfg.model.get("name", "lgbm")
    model_cls = get_model_class(model_name)
    model = model_cls(cfg)

    logger.info(f"Training model: '{model_name}'...")
    t0 = time.time()
    model.fit(X_train, z_train, X_val=X_val, y_val=z_val)
    train_seconds = time.time() - t0
    logger.info(f"Model fitting finished in {train_seconds:.2f} seconds.")

    # 8. Predict val and test (in target-transformed space)
    z_pred_val = model.predict(X_val)
    z_pred_test = model.predict(X_test)

    # Invert target transformation back to original price space
    y_pred_val_uncal = transform.inverse(z_pred_val)
    y_pred_test_uncal = transform.inverse(z_pred_test)

    # 9. Compute uncalibrated SMAPE
    val_smape_uncal = smape(y_val_raw, y_pred_val_uncal)
    test_smape_uncal = smape(y_test_raw, y_pred_test_uncal)
    logger.info(f"Uncalibrated Val SMAPE: {val_smape_uncal:.4f}%, Test SMAPE: {test_smape_uncal:.4f}%")

    # 10. Calibration multiplier grid search ON VAL, apply to val & test
    postprocess_cfg = cfg.get("postprocess", {})
    if postprocess_cfg.get("calibrate", True):
        grid = np.arange(*postprocess_cfg.get("calibration_grid", [0.80, 1.06, 0.005]))
        multiplier, val_smape_cal = fit_multiplier(y_val_raw, y_pred_val_uncal, grid=grid)
    else:
        multiplier, val_smape_cal = 1.0, val_smape_uncal

    y_pred_val = apply_multiplier(y_pred_val_uncal, multiplier)
    y_pred_test = apply_multiplier(y_pred_test_uncal, multiplier)
    test_smape_cal = smape(y_test_raw, y_pred_test)

    logger.info(f"Calibrated (m={multiplier:.4f}) Val SMAPE: {val_smape_cal:.4f}%, Test SMAPE: {test_smape_cal:.4f}%")

    # 11. Per-bucket SMAPE & feature importances
    bucket_smape_df = smape_per_bucket(y_val_raw, y_pred_val)
    
    feature_names = feature_artifacts.get("feature_names", None)
    if hasattr(model, "feature_importance"):
        feature_imp_df = model.feature_importance(feature_names=feature_names, top_k=50)
    else:
        feature_imp_df = pd.DataFrame(columns=["feature", "importance_gain"])

    # Prepare prediction DataFrames
    val_preds_df = pd.DataFrame({
        "sample_id": df_val["sample_id"].values,
        "price_true": y_val_raw,
        "price_pred_uncalibrated": y_pred_val_uncal,
        "price_pred_calibrated": y_pred_val,
    })
    test_preds_df = pd.DataFrame({
        "sample_id": df_test["sample_id"].values,
        "price_true": y_test_raw,
        "price_pred_uncalibrated": y_pred_test_uncal,
        "price_pred_calibrated": y_pred_test,
    })

    # 12. Save artifacts & update leaderboard
    metrics_summary = {
        "n_train": len(df_train),
        "val_smape": val_smape_uncal,
        "val_smape_calibrated": val_smape_cal,
        "test_smape": test_smape_uncal,
        "test_smape_calibrated": test_smape_cal,
        "calibration_multiplier": multiplier,
        "train_seconds": train_seconds,
        "notes": f"Features: {X_train.shape[1]}, Sparsity OK",
    }

    run_dir = save_run_artifacts(
        cfg=cfg,
        metrics=metrics_summary,
        val_preds_df=val_preds_df,
        test_preds_df=test_preds_df,
        feature_imp_df=feature_imp_df,
        bucket_smape_df=bucket_smape_df,
        train_seconds=train_seconds
    )

    # 13. Print compact summary table to stdout
    print("\n" + "=" * 70)
    print(f" EXPERIMENT COMPLETED: {run_id}")
    print("=" * 70)
    print(f" Config Hash               : {save_run_artifacts.__module__}")
    print(f" Training Samples          : {len(df_train):,}")
    print(f" Feature Count             : {X_train.shape[1]:,}")
    print(f" Validation SMAPE (Uncal)  : {val_smape_uncal:.4f}%")
    print(f" Validation SMAPE (Cal)    : {val_smape_cal:.4f}%")
    print(f" Test SMAPE (Uncal)        : {test_smape_uncal:.4f}%")
    print(f" Test SMAPE (Calibrated)   : {test_smape_cal:.4f}%")
    print(f" Calibration Multiplier    : {multiplier:.4f}")
    print(f" Training Time             : {train_seconds:.2f}s")
    print(f" Artifacts Directory       : {run_dir}")
    print("=" * 70 + "\n")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics_summary,
        "val_preds": val_preds_df,
        "test_preds": test_preds_df,
    }
