from pathlib import Path
import sys

# Ensure src/ directory is on sys.path for test discovery
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
