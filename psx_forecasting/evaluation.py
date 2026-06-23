from __future__ import annotations

import numpy as np
import pandas as pd

from config import PREDICTIONS_DIR, RAW_DIR


def update_actuals() -> pd.DataFrame:
    predictions_path = PREDICTIONS_DIR / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("No predictions.csv found. Train models first.")

    predictions = pd.read_csv(predictions_path)
    if predictions.empty:
        return predictions

    predictions["Prediction_Date"] = pd.to_datetime(predictions["Prediction_Date"]).dt.date

    for idx, row in predictions.iterrows():
        if not pd.isna(row.get("Actual_Close")):
            continue

        raw_path = RAW_DIR / f"{row['Symbol']}.csv"
        if not raw_path.exists():
            continue

        prices = pd.read_csv(raw_path)
        prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
        match = prices[prices["Date"] == row["Prediction_Date"]]
        if match.empty:
            continue

        actual = float(match.iloc[-1]["Close"])
        predicted = float(row["Predicted_Close"])
        error = actual - predicted
        predictions.loc[idx, "Actual_Close"] = actual
        predictions.loc[idx, "Error"] = error
        predictions.loc[idx, "Error_Pct"] = np.nan if actual == 0 else abs(error / actual) * 100

    predictions.to_csv(predictions_path, index=False)
    return predictions
