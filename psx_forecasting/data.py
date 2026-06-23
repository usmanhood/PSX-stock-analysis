"""Data extraction, storage, and cleaning helpers for PSX stocks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import PRICE_COLUMNS, PROCESSED_DIR, RAW_DIR, TOP_10_PSX_TICKERS


def ensure_data_dirs() -> None:
    """Create the local data directories used by the pipeline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_stock_history(symbol: str, yahoo_ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch historical OHLCV data for one stock from Yahoo Finance."""
    history = yf.Ticker(yahoo_ticker).history(period=period, auto_adjust=False)
    if history.empty:
        raise ValueError(f"No history returned for {symbol} ({yahoo_ticker}).")

    history = history.reset_index()
    history["Symbol"] = symbol
    history["YahooTicker"] = yahoo_ticker
    return history


def fetch_top_companies(period: str = "5y") -> dict[str, pd.DataFrame]:
    """Fetch historical prices for all configured top PSX companies."""
    ensure_data_dirs()
    frames: dict[str, pd.DataFrame] = {}
    for symbol, yahoo_ticker in TOP_10_PSX_TICKERS.items():
        frame = fetch_stock_history(symbol, yahoo_ticker, period=period)
        frame.to_csv(RAW_DIR / f"{symbol}.csv", index=False)
        frames[symbol] = frame
    return frames


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one raw OHLCV frame and standardize the schema."""
    cleaned = df.copy()
    if "Datetime" in cleaned.columns and "Date" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"Datetime": "Date"})

    cleaned["Date"] = pd.to_datetime(cleaned["Date"]).dt.tz_localize(None).dt.normalize()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    keep_columns = [*PRICE_COLUMNS, "Symbol", "YahooTicker"]
    cleaned = cleaned[[column for column in keep_columns if column in cleaned.columns]]
    cleaned = cleaned.dropna(subset=PRICE_COLUMNS)
    cleaned = cleaned.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    cleaned = cleaned.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return cleaned


def load_processed_history(symbol: str, directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load processed history for one symbol."""
    return pd.read_csv(directory / f"{symbol}.csv", parse_dates=["Date"])
