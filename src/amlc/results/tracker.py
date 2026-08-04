from pathlib import Path
from typing import Any, Dict
import datetime
import json
import os
import platform
import sys
import yaml
import pandas as pd

from amlc.config import ConfigDict, config_hash
from amlc.paths import RESULTS_DIR
from amlc.logging_utils import get_logger

logger = get_logger("amlc.results.tracker")

LEADERBOARD_COLUMNS = [
    "run_id",
    "timestamp",
    "description",
    "model",
    "features",
    "target_transform",
    "loss",
    "n_train",
    "val_smape",
    "val_smape_calibrated",
    "test_smape",
    "test_smape_calibrated",
    "calibration_multiplier",
    "train_seconds",
    "config_hash",
    "notes"
]

def _get_git_commit() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

def _get_env_info() -> Dict[str, Any]:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _get_git_commit(),
    }
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "pyyaml"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not_installed"
    return info

def save_run_artifacts(
    cfg: ConfigDict,
    metrics: Dict[str, Any],
    val_preds_df: pd.DataFrame,
    test_preds_df: pd.DataFrame,
    feature_imp_df: pd.DataFrame,
    bucket_smape_df: pd.DataFrame,
    train_seconds: float = 0.0
) -> Path:
    """
    Save all experiment run artifacts in results/runs/{run_id}/ and update results/leaderboard.csv.
    """
    run_id = cfg.run_id
    run_dir = RESULTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving run artifacts to: {run_dir}")

    # 1. config.yaml
    config_file = run_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)

    # 2. metrics.json
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 3. predictions CSVs
    val_preds_df.to_csv(run_dir / "predictions_val.csv", index=False)
    test_preds_df.to_csv(run_dir / "predictions_test.csv", index=False)

    # 4. feature importance CSV
    feature_imp_df.to_csv(run_dir / "feature_importance.csv", index=False)

    # 5. smape by bucket CSV
    bucket_smape_df.to_csv(run_dir / "smape_by_bucket.csv", index=False)

    # 6. env.json
    env_file = run_dir / "env.json"
    with open(env_file, "w", encoding="utf-8") as f:
        json.dump(_get_env_info(), f, indent=2)

    # 7. Update leaderboard.csv
    update_leaderboard(cfg, metrics, train_seconds)

    return run_dir

def update_leaderboard(cfg: ConfigDict, metrics: Dict[str, Any], train_seconds: float) -> None:
    """
    Append experiment row to results/leaderboard.csv (creates with headers if absent).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lb_path = RESULTS_DIR / "leaderboard.csv"

    cfg_hash = config_hash(cfg)
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    loss_str = cfg.model.params.get("objective", "unknown") if hasattr(cfg.model, "params") else "unknown"

    row = {
        "run_id": cfg.run_id,
        "timestamp": now_str,
        "description": cfg.description,
        "model": cfg.model.get("name", "unknown"),
        "features": cfg.features.get("builder", "unknown"),
        "target_transform": cfg.target.get("transform", "unknown"),
        "loss": loss_str,
        "n_train": metrics.get("n_train", 0),
        "val_smape": round(metrics.get("val_smape", 0.0), 4),
        "val_smape_calibrated": round(metrics.get("val_smape_calibrated", 0.0), 4),
        "test_smape": round(metrics.get("test_smape", 0.0), 4),
        "test_smape_calibrated": round(metrics.get("test_smape_calibrated", 0.0), 4),
        "calibration_multiplier": round(metrics.get("calibration_multiplier", 1.0), 4),
        "train_seconds": round(train_seconds, 2),
        "config_hash": cfg_hash,
        "notes": metrics.get("notes", "")
    }

    if lb_path.exists():
        try:
            df_lb = pd.read_csv(lb_path)
            # Remove prior row if re-running same run_id
            df_lb = df_lb[df_lb["run_id"] != cfg.run_id]
            df_lb = pd.concat([df_lb, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            logger.warning(f"Error reading existing leaderboard ({e}), overwriting: {lb_path}")
            df_lb = pd.DataFrame([row], columns=LEADERBOARD_COLUMNS)
    else:
        df_lb = pd.DataFrame([row], columns=LEADERBOARD_COLUMNS)

    df_lb.to_csv(lb_path, index=False)
    logger.info(f"Updated leaderboard at: {lb_path}")
