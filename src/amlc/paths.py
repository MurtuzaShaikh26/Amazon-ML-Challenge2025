from pathlib import Path
import os
import glob

# Detect repository root (src/amlc/paths.py -> parents[2])
REPO_ROOT = Path(__file__).resolve().parents[2]

# Detect Kaggle vs Local environment
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    # On Kaggle, locate input directory (search for amazon-ml or take first directory under /kaggle/input)
    kaggle_inputs = glob.glob("/kaggle/input/*")
    data_dir_candidate = None
    for inp in kaggle_inputs:
        if "amazon-ml" in inp.lower():
            data_dir_candidate = Path(inp)
            break
    if data_dir_candidate is None and len(kaggle_inputs) > 0:
        data_dir_candidate = Path(kaggle_inputs[0])
    elif data_dir_candidate is None:
        data_dir_candidate = Path("/kaggle/input")
    
    DATA_DIR = data_dir_candidate
    RESULTS_DIR = Path("/kaggle/working/results")
else:
    DATA_DIR = REPO_ROOT / "data"
    RESULTS_DIR = REPO_ROOT / "results"

def find_train_csv(search_dir: Path | str | None = None) -> Path:
    """
    Glob for the dataset CSV file within search_dir (or DATA_DIR).
    Looks for train.csv, dataset.csv, or any *.csv file containing the dataset.
    """
    target_dir = Path(search_dir) if search_dir else DATA_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {target_dir}")
    
    # Priority search
    priority_names = ["train.csv", "dataset.csv", "catalog.csv", "sample.csv"]
    for p_name in priority_names:
        matches = list(target_dir.glob(f"**/{p_name}"))
        if matches:
            return matches[0]
            
    csv_files = list(target_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {target_dir}")
    
    return csv_files[0]
