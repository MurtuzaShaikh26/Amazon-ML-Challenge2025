import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_dataset(n_rows: int = 75000, out_path: str = "data/train.csv"):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(42)
    log_prices = rng.normal(loc=3.0, scale=1.2, size=n_rows)
    prices = np.expm1(np.clip(log_prices, 0.1, 8.0)) + 0.13
    
    df = pd.DataFrame({
        "sample_id": np.arange(1, n_rows + 1, dtype=np.int64),
        "catalog_content": [
            f"Item Name: La Victoria Green Taco Sauce Mild, {12 * (i % 6 + 1)} Ounce (Pack of {i % 4 + 1}) Value: {72.0 + (i % 10)} Unit: Fl Oz Bullet Point 1: Great flavor {i}"
            for i in range(n_rows)
        ],
        "image_link": [f"https://images-na.ssl-images-amazon.com/images/I/product_{i}.jpg" for i in range(n_rows)],
        "price": prices
    })
    
    df.to_csv(path, index=False)
    print(f"Generated synthetic dataset with {len(df)} rows at: {path}")

if __name__ == "__main__":
    generate_synthetic_dataset()
