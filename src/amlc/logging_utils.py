from pathlib import Path
from typing import Optional, Union
import logging
import sys

def setup_logger(
    name: str = "amlc",
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Set up and return a configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Avoid adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_logger(name: str = "amlc") -> logging.Logger:
    """
    Get existing logger instance by name.
    """
    return logging.getLogger(name)
