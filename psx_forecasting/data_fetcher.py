from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DEFAULT_PERIOD, RAW_DIR, TOP_PSX_TICKERS


@dataclass(frozen=True)
class FetchResult:
    symbol: str
    yahoo_ticker: str
    rows: int
    path: Path
    status: str


def _normalize_yahoo_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[col for col in keep if col in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date
    df["Symbol"] = symbol
    return df[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]


def fetch_stock(symbol: str, yahoo_ticker: str, period: str = DEFAULT_PERIOD) -> FetchResult:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{symbol}.csv"

    try:
        history = yf.Ticker(yahoo_ticker).history(period=period, auto_adjust=False)
        df = _normalize_yahoo_history(history, symbol)
        if df.empty:
            if path.exists():
                path.unlink()
            return FetchResult(symbol, yahoo_ticker, 0, path, "empty")
        df.to_csv(path, index=False)
        return FetchResult(symbol, yahoo_ticker, len(df), path, "ok")
    except Exception as exc:  # Network/data-provider failures should not stop all tickers.
        return FetchResult(symbol, yahoo_ticker, 0, path, f"error: {exc}")


def fetch_all(period: str = DEFAULT_PERIOD, tickers: dict[str, str] | None = None) -> list[FetchResult]:
    results = []
    for symbol, yahoo_ticker in (tickers or TOP_PSX_TICKERS).items():
        results.append(fetch_stock(symbol, yahoo_ticker, period))
    return results


def load_raw_stock(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No raw data found for {symbol}. Run data fetch first.")
    return pd.read_csv(path)
