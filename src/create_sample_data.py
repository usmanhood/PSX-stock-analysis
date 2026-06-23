from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR, TOP_PSX_TICKERS


def create_sample_data(days: int = 520) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    rng = np.random.default_rng(42)

    for i, symbol in enumerate(TOP_PSX_TICKERS):
        base = 80 + i * 25
        returns = rng.normal(loc=0.0007, scale=0.018 + i * 0.001, size=len(dates))
        close = base * np.cumprod(1 + returns)
        open_price = close * (1 + rng.normal(0, 0.006, len(dates)))
        high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.018, len(dates)))
        low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.018, len(dates)))
        volume = rng.integers(150_000, 5_000_000, len(dates))

        df = pd.DataFrame(
            {
                "Symbol": symbol,
                "Date": dates.date,
                "Open": open_price.round(2),
                "High": high.round(2),
                "Low": low.round(2),
                "Close": close.round(2),
                "Volume": volume,
            }
        )
        df.to_csv(RAW_DIR / f"{symbol}.csv", index=False)


if __name__ == "__main__":
    create_sample_data()
    print("Sample PSX-style data created in data/raw.")
