import re
import zlib
import numpy as np
import pandas as pd

# Module-level compiled regex patterns
RE_BULLET = re.compile(r"Bullet Point \d+:", re.IGNORECASE)
RE_ITEM_NAME_FLAG = re.compile(r"Item Name:", re.IGNORECASE)
RE_VALUE_FLAG = re.compile(r"Value:", re.IGNORECASE)
RE_UNIT_FLAG = re.compile(r"Unit:", re.IGNORECASE)

RE_PACK_COUNT = re.compile(r"(?:\(Pack of|Pack of)\s*(\d+)", re.IGNORECASE)
RE_PARSED_VALUE = re.compile(r"Value:\s*([\d.]+)", re.IGNORECASE)
RE_PARSED_UNIT = re.compile(r"Unit:\s*([A-Za-z ]+)", re.IGNORECASE)
RE_COUNT_PATTERN = re.compile(r"(\d+)\s*(?:count|ct|pack|pcs)", re.IGNORECASE)
RE_ITEM_NAME_VAL = re.compile(r"Item Name:\s*([^\s,;:]+)", re.IGNORECASE)

# Unit normalization factors to Oz / Fl Oz equivalent
UNIT_CONVERSIONS = {
    "fl oz": 1.0,
    "oz": 1.0,
    "ounce": 1.0,
    "ounces": 1.0,
    "fluid ounce": 1.0,
    "fluid ounces": 1.0,
    "lb": 16.0,
    "lbs": 16.0,
    "pound": 16.0,
    "pounds": 16.0,
    "g": 0.03527396,
    "gram": 0.03527396,
    "grams": 0.03527396,
    "kg": 35.27396,
    "kilogram": 35.27396,
    "kilograms": 35.27396,
    "ml": 0.033814,
    "milliliter": 0.033814,
    "milliliters": 0.033814,
    "l": 33.814,
    "liter": 33.814,
    "liters": 33.814,
}

KNOWN_UNITS = sorted(list(set(UNIT_CONVERSIONS.keys())))
UNIT_TO_CODE = {unit: float(idx + 1) for idx, unit in enumerate(KNOWN_UNITS)}
UNKNOWN_UNIT_CODE = 0.0

def _normalize_unit_value(value: float, unit_str: str) -> float:
    if np.isnan(value) or not unit_str:
        return np.nan
    u_clean = unit_str.strip().lower()
    factor = UNIT_CONVERSIONS.get(u_clean)
    if factor is not None:
        return value * factor
    return np.nan

def extract_regex_features(texts: pd.Series) -> pd.DataFrame:
    """
    Extract dense numeric and categorical features from catalog_content text series.
    Returns a DataFrame with float64 dtypes.
    """
    n_rows = len(texts)
    
    n_chars = np.zeros(n_rows, dtype=np.float64)
    n_words = np.zeros(n_rows, dtype=np.float64)
    n_digits = np.zeros(n_rows, dtype=np.float64)
    n_uppercase_words = np.zeros(n_rows, dtype=np.float64)
    n_bullet_points = np.zeros(n_rows, dtype=np.float64)
    
    has_item_name = np.zeros(n_rows, dtype=np.float64)
    has_value_field = np.zeros(n_rows, dtype=np.float64)
    has_unit_field = np.zeros(n_rows, dtype=np.float64)
    
    pack_count = np.ones(n_rows, dtype=np.float64)
    parsed_value = np.full(n_rows, np.nan, dtype=np.float64)
    parsed_unit_code = np.full(n_rows, UNKNOWN_UNIT_CODE, dtype=np.float64)
    value_normalized = np.full(n_rows, np.nan, dtype=np.float64)
    value_per_pack = np.full(n_rows, np.nan, dtype=np.float64)
    count_x_pattern = np.full(n_rows, np.nan, dtype=np.float64)
    first_token_hash = np.zeros(n_rows, dtype=np.float64)

    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text:
            continue
            
        n_chars[idx] = len(text)
        words = text.split()
        n_words[idx] = len(words)
        n_digits[idx] = sum(1 for c in text if c.isdigit())
        n_uppercase_words[idx] = sum(1 for w in words if len(w) > 1 and w.isupper())
        
        n_bullet_points[idx] = len(RE_BULLET.findall(text))
        has_item_name[idx] = 1.0 if RE_ITEM_NAME_FLAG.search(text) else 0.0
        has_value_field[idx] = 1.0 if RE_VALUE_FLAG.search(text) else 0.0
        has_unit_field[idx] = 1.0 if RE_UNIT_FLAG.search(text) else 0.0
        
        # Pack count
        pack_match = RE_PACK_COUNT.search(text)
        if pack_match:
            try:
                pack_count[idx] = float(pack_match.group(1))
            except (ValueError, TypeError):
                pack_count[idx] = 1.0
                
        # Value
        val_match = RE_PARSED_VALUE.search(text)
        val_float = np.nan
        if val_match:
            try:
                val_float = float(val_match.group(1))
                parsed_value[idx] = val_float
            except (ValueError, TypeError):
                pass
                
        # Unit
        unit_match = RE_PARSED_UNIT.search(text)
        unit_str = ""
        if unit_match:
            unit_str = unit_match.group(1).strip().lower()
            parsed_unit_code[idx] = UNIT_TO_CODE.get(unit_str, UNKNOWN_UNIT_CODE)
            
        # Value normalized & value per pack
        if not np.isnan(val_float):
            value_normalized[idx] = _normalize_unit_value(val_float, unit_str)
            if pack_count[idx] > 0:
                value_per_pack[idx] = val_float / pack_count[idx]
                
        # Count X pattern
        cnt_match = RE_COUNT_PATTERN.search(text)
        if cnt_match:
            try:
                count_x_pattern[idx] = float(cnt_match.group(1))
            except (ValueError, TypeError):
                pass
                
        # First token hash
        item_name_match = RE_ITEM_NAME_VAL.search(text)
        first_token = ""
        if item_name_match:
            first_token = item_name_match.group(1)
        elif words:
            first_token = words[0]
            
        if first_token:
            first_token_hash[idx] = float(zlib.crc32(first_token.lower().encode("utf-8")) % 5000)

    feature_dict = {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_digits": n_digits,
        "n_uppercase_words": n_uppercase_words,
        "n_bullet_points": n_bullet_points,
        "has_item_name": has_item_name,
        "has_value_field": has_value_field,
        "has_unit_field": has_unit_field,
        "pack_count": pack_count,
        "parsed_value": parsed_value,
        "parsed_unit": parsed_unit_code,
        "value_normalized": value_normalized,
        "value_per_pack": value_per_pack,
        "count_x_pattern": count_x_pattern,
        "first_token_hash": first_token_hash,
    }

    return pd.DataFrame(feature_dict)
