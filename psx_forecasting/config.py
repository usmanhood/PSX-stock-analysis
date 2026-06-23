"""Project configuration constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions.csv"
VALIDATION_FILE = PREDICTIONS_DIR / "validation.csv"

# Yahoo Finance commonly exposes Karachi Stock Exchange tickers with the .KA suffix.
TOP_10_PSX_TICKERS: dict[str, str] = {
    "OGDC": "OGDC.KA",
    "MARI": "MARI.KA",
    "HUBC": "HUBC.KA",
    "ENGROH": "ENGROH.KA",
    "FFC": "FFC.KA",
    "UBL": "UBL.KA",
    "HBL": "HBL.KA",
    "LUCK": "LUCK.KA",
    "SYS": "SYS.KA",
    "PPL": "PPL.KA",
}

PRICE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
