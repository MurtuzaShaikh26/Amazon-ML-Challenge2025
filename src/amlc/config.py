from pathlib import Path
from typing import Any, Dict, Union
import json
import hashlib
import yaml

class ConfigDict(dict):
    """
    Dot-accessible dictionary class that preserves dict operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, dict) else value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert recursively back to plain dict."""
        res = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                res[k] = v.to_dict()
            else:
                res[k] = v
        return res

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load YAML configuration file with inheritance support (`extends: base.yaml`).
    Returns a dot-accessible ConfigDict.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "extends" in data:
        base_filename = data.pop("extends")
        base_path = path.parent / base_filename
        if not base_path.exists():
            # Check relative to REPO_ROOT/configs/
            from amlc.paths import REPO_ROOT
            base_path = REPO_ROOT / "configs" / base_filename
        
        base_cfg = load_config(base_path).to_dict()
        data = _deep_merge(base_cfg, data)

    cfg = ConfigDict(data)
    
    if "run_id" not in cfg:
        raise ValueError(f"Config at {path} must specify a 'run_id'")
    if "description" not in cfg:
        raise ValueError(f"Config at {path} must specify a 'description'")
        
    return cfg

def config_hash(cfg: ConfigDict) -> str:
    """
    Generate short SHA1 hash (10 chars) of canonical JSON representation of config.
    """
    d = cfg.to_dict() if isinstance(cfg, ConfigDict) else cfg
    canonical_json = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(canonical_json.encode("utf-8")).hexdigest()[:10]
