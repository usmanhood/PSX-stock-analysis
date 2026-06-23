"""End-to-end ETL, analytics, forecasting, and validation pipeline."""
from __future__ import annotations

import argparse

from .config import PROCESSED_DIR
from .data import clean_price_data, fetch_top_companies
from .evaluation import append_predictions, validate_predictions
from .features import add_technical_indicators
from .forecasting import train_and_predict_next_close


def run_pipeline(period: str = "5y", lookback_days: int = 30, model_name: str = "random_forest") -> None:
    """Run the complete project flow for all configured PSX companies."""
    raw_frames = fetch_top_companies(period=period)
    processed_frames = {}
    prediction_rows = []

    for symbol, raw_frame in raw_frames.items():
        processed = add_technical_indicators(clean_price_data(raw_frame))
        processed.to_csv(PROCESSED_DIR / f"{symbol}.csv", index=False)
        processed_frames[symbol] = processed

        forecast = train_and_predict_next_close(processed, symbol, lookback_days, model_name)
        prediction_rows.append(
            {
                "Symbol": forecast.symbol,
                "Prediction_Date": forecast.prediction_date,
                "Predicted_Price": round(forecast.predicted_price, 4),
                "Model": forecast.model_name,
                "Backtest_MAE": forecast.mae,
                "Backtest_RMSE": forecast.rmse,
            }
        )

    append_predictions(prediction_rows)
    validate_predictions(processed_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSX stock forecasting pipeline.")
    parser.add_argument("--period", default="5y", help="Yahoo Finance history period, e.g. 1y, 5y, max.")
    parser.add_argument("--lookback-days", type=int, default=30, help="Number of past closes used as features.")
    parser.add_argument(
        "--model",
        choices=["random_forest", "linear_regression"],
        default="random_forest",
        help="Forecasting model to train.",
    )
    args = parser.parse_args()
    run_pipeline(period=args.period, lookback_days=args.lookback_days, model_name=args.model)


if __name__ == "__main__":
    main()
