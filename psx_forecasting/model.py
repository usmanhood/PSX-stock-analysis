from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MIN_ROWS_FOR_MODEL, MODELS_DIR, PREDICTIONS_DIR
from psx_forecasting.features import add_technical_indicators, build_model_frame, model_feature_columns


@dataclass(frozen=True)
class ModelResult:
    symbol: str
    rows: int
    model_name: str
    mae: float
    rmse: float
    mape: float
    predicted_close: float
    prediction_for_date: pd.Timestamp
    model_path: Path


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _candidate_models() -> dict[str, object]:
    return {
        "linear_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }


def _evaluate_time_series(model, x: pd.DataFrame, y: pd.Series) -> tuple[float, float, float]:
    splits = min(5, max(2, len(x) // 120))
    tscv = TimeSeriesSplit(n_splits=splits)
    actuals = []
    preds = []
    for train_idx, test_idx in tscv.split(x):
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        actuals.extend(y.iloc[test_idx].to_numpy())
        preds.extend(pred)

    actual = np.asarray(actuals)
    predicted = np.asarray(preds)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return mae, rmse, _mape(actual, predicted)


def train_and_predict(symbol: str, raw_df: pd.DataFrame) -> ModelResult:
    frame = build_model_frame(raw_df)
    if len(frame) < MIN_ROWS_FOR_MODEL:
        raise ValueError(f"{symbol} has only {len(frame)} usable rows; need at least {MIN_ROWS_FOR_MODEL}.")

    features = model_feature_columns()
    x = frame[features]
    y = frame["Target_Close"]

    best_name = ""
    best_model = None
    best_metrics = (float("inf"), float("inf"), float("inf"))
    for name, model in _candidate_models().items():
        mae, rmse, mape = _evaluate_time_series(model, x, y)
        if mae < best_metrics[0]:
            best_name = name
            best_model = model
            best_metrics = (mae, rmse, mape)

    if best_model is None:
        raise RuntimeError(f"No model could be trained for {symbol}.")

    best_model.fit(x, y)

    featured = add_technical_indicators(raw_df).replace([np.inf, -np.inf], np.nan)
    latest_row = featured.dropna(subset=features).iloc[-1]
    predicted_close = float(best_model.predict(latest_row[features].to_frame().T)[0])

    latest_date = pd.to_datetime(latest_row["Date"])
    prediction_for_date = latest_date + pd.tseries.offsets.BDay(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{symbol}_model.joblib"
    joblib.dump({"model": best_model, "features": features, "model_name": best_name}, model_path)

    result = ModelResult(
        symbol=symbol,
        rows=len(frame),
        model_name=best_name,
        mae=best_metrics[0],
        rmse=best_metrics[1],
        mape=best_metrics[2],
        predicted_close=predicted_close,
        prediction_for_date=prediction_for_date,
        model_path=model_path,
    )
    save_prediction(result, latest_close=float(latest_row["Close"]), latest_date=latest_date)
    return result


def save_prediction(result: ModelResult, latest_close: float, latest_date: pd.Timestamp) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / "predictions.csv"
    row = {
        "Symbol": result.symbol,
        "Source_Date": latest_date.date(),
        "Prediction_Date": result.prediction_for_date.date(),
        "Latest_Close": latest_close,
        "Predicted_Close": result.predicted_close,
        "Model": result.model_name,
        "Validation_MAE": result.mae,
        "Validation_RMSE": result.rmse,
        "Validation_MAPE": result.mape,
        "Actual_Close": np.nan,
        "Error": np.nan,
        "Error_Pct": np.nan,
    }

    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[
            ~(
                (existing["Symbol"] == row["Symbol"])
                & (existing["Source_Date"].astype(str) == str(row["Source_Date"]))
                & (existing["Prediction_Date"].astype(str) == str(row["Prediction_Date"]))
            )
        ]
        output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        output = pd.DataFrame([row])

    output.to_csv(path, index=False)
