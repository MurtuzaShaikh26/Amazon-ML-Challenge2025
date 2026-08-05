from typing import List, Tuple
from pathlib import Path
import pandas as pd

def download_images(
    urls: List[str],
    out_dir: str | Path,
    n_threads: int = 16,
    size: Tuple[int, int] = (224, 224)
) -> pd.DataFrame:
    """
    STUB: Image downloading interface for multimodal models.
    Scheduled for full implementation in Run 4.
    """
    raise NotImplementedError(
        "Image downloading functionality is scheduled for activation in Run 4."
    )
