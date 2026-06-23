from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

TOP_PSX_TICKERS = {
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

DEFAULT_PERIOD = "5y"
FORECAST_HORIZON_DAYS = 1
MIN_ROWS_FOR_MODEL = 120
