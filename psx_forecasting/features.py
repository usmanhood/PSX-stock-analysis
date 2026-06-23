from __future__ import annotations

import numpy as np
import pandas as pd


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.drop_duplicates(subset=["Symbol", "Date"])
    cleaned = cleaned.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)
    cleaned["Volume"] = cleaned["Volume"].fillna(0)
    return cleaned


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = clean_price_data(df)
    data["Return"] = data["Close"].pct_change()
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["Volatility20"] = data["Return"].rolling(20).std()
    data["High_Low_Range"] = (data["High"] - data["Low"]) / data["Close"]
    data["Open_Close_Range"] = (data["Close"] - data["Open"]) / data["Open"]
    data["Volume_Change"] = data["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI14"] = 100 - (100 / (1 + rs))

    data["Target_Close"] = data["Close"].shift(-1)
    return data


def model_feature_columns() -> list[str]:
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return",
        "Log_Return",
        "MA20",
        "MA50",
        "MA200",
        "Volatility20",
        "High_Low_Range",
        "Open_Close_Range",
        "Volume_Change",
        "RSI14",
    ]


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_technical_indicators(df)
    cols = ["Date", "Symbol", *model_feature_columns(), "Target_Close"]
    return featured[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
