"""Feature engineering and technical indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns, volatility, moving averages, and trend labels."""
    featured = df.sort_values("Date").copy()
    featured["Return"] = featured["Close"].pct_change()
    featured["Volatility_20D"] = featured["Return"].rolling(20).std()
    featured["MA20"] = featured["Close"].rolling(20).mean()
    featured["MA50"] = featured["Close"].rolling(50).mean()
    featured["MA200"] = featured["Close"].rolling(200).mean()
    featured["Trend"] = np.select(
        [featured["MA20"] > featured["MA50"], featured["MA20"] < featured["MA50"]],
        ["Bullish", "Bearish"],
        default="Sideways",
    )
    return featured


def make_supervised_frame(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """Convert closing prices into lagged features for next-day forecasting."""
    model_frame = df.sort_values("Date").copy()
    for lag in range(1, lookback_days + 1):
        model_frame[f"Close_Lag_{lag}"] = model_frame["Close"].shift(lag)
    model_frame["Target_Close"] = model_frame["Close"].shift(-1)
    return model_frame.dropna().reset_index(drop=True)
