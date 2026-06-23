"""Prediction persistence and validation metrics."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import PREDICTIONS_FILE, VALIDATION_FILE


def append_predictions(predictions: list[dict]) -> pd.DataFrame:
    """Append prediction rows to the local prediction CSV."""
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(predictions)
    if PREDICTIONS_FILE.exists():
        existing = pd.read_csv(PREDICTIONS_FILE, parse_dates=["Prediction_Date"])
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = combined.drop_duplicates(subset=["Symbol", "Prediction_Date"], keep="last")
    combined.to_csv(PREDICTIONS_FILE, index=False)
    return combined


def validate_predictions(processed_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare stored predictions with actual closes when dates are available."""
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["Prediction_Date"])
    validation_rows: list[dict] = []
    for _, prediction in predictions.iterrows():
        symbol = prediction["Symbol"]
        actual_frame = processed_frames.get(symbol)
        if actual_frame is None:
            continue
        match = actual_frame[actual_frame["Date"] == prediction["Prediction_Date"]]
        if match.empty:
            continue
        actual_close = float(match.iloc[-1]["Close"])
        predicted_close = float(prediction["Predicted_Price"])
        error = actual_close - predicted_close
        validation_rows.append(
            {
                "Symbol": symbol,
                "Prediction_Date": prediction["Prediction_Date"],
                "Predicted_Price": predicted_close,
                "Actual_Close": actual_close,
                "Error": error,
                "Absolute_Error": abs(error),
                "Absolute_Percentage_Error": abs(error) / actual_close * 100 if actual_close else None,
            }
        )

    validation = pd.DataFrame(validation_rows)
    if not validation.empty:
        validation.to_csv(VALIDATION_FILE, index=False)
    return validation


def summarize_validation(validation: pd.DataFrame) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE from a validation table."""
    if validation.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    return {
        "MAE": float(mean_absolute_error(validation["Actual_Close"], validation["Predicted_Price"])),
        "RMSE": float(mean_squared_error(validation["Actual_Close"], validation["Predicted_Price"], squared=False)),
        "MAPE": float(validation["Absolute_Percentage_Error"].mean()),
    }
