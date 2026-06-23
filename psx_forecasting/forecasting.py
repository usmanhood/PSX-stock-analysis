"""Machine-learning forecasting utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import make_supervised_frame


@dataclass(frozen=True)
class ForecastResult:
    symbol: str
    prediction_date: pd.Timestamp
    predicted_price: float
    model_name: str
    mae: float | None
    rmse: float | None


def train_and_predict_next_close(
    df: pd.DataFrame,
    symbol: str,
    lookback_days: int = 30,
    model_name: str = "random_forest",
) -> ForecastResult:
    """Train a simple model and predict the next trading day's close."""
    supervised = make_supervised_frame(df, lookback_days=lookback_days)
    feature_columns = [column for column in supervised.columns if column.startswith("Close_Lag_")]
    if len(supervised) < 40:
        raise ValueError(f"Not enough rows to train {symbol}; need more than {lookback_days + 10} rows.")

    split_index = max(int(len(supervised) * 0.8), 1)
    train = supervised.iloc[:split_index]
    test = supervised.iloc[split_index:]

    if model_name == "linear_regression":
        model = LinearRegression()
        readable_model_name = "Linear Regression"
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
        readable_model_name = "Random Forest"

    model.fit(train[feature_columns], train["Target_Close"])

    mae = rmse = None
    if not test.empty:
        predictions = model.predict(test[feature_columns])
        mae = float(mean_absolute_error(test["Target_Close"], predictions))
        rmse = float(mean_squared_error(test["Target_Close"], predictions, squared=False))

    latest_features = supervised.iloc[[-1]][feature_columns]
    predicted_price = float(model.predict(latest_features)[0])
    last_date = pd.to_datetime(df["Date"].max())
    prediction_date = last_date + pd.offsets.BDay(1)

    return ForecastResult(symbol, prediction_date, predicted_price, readable_model_name, mae, rmse)
