import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amlc.pipeline.run import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run an Amazon ML Challenge experiment locally.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run001_lgbm_tfidf.yaml",
        help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    run_experiment(config_path)

if __name__ == "__main__":
    main()
