import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """
    Set seed for random, numpy, and torch (if installed).
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
