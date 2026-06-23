from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import TOP_PSX_TICKERS
from psx_forecasting.data_fetcher import fetch_all, load_raw_stock
from psx_forecasting.evaluation import update_actuals
from psx_forecasting.model import train_and_predict
from psx_forecasting.reporting import generate_markdown_report


def run_pipeline(period: str, skip_fetch: bool = False) -> None:
    trainable_symbols = list(TOP_PSX_TICKERS)
    if not skip_fetch:
        print("Fetching latest PSX data from free Yahoo Finance tickers...")
        trainable_symbols = []
        for result in fetch_all(period=period):
            print(f"{result.symbol}: {result.status} ({result.rows} rows)")
            if result.status == "ok":
                trainable_symbols.append(result.symbol)

    print("\nTraining models and saving tomorrow predictions...")
    for symbol in trainable_symbols:
        try:
            raw_df = load_raw_stock(symbol)
            result = train_and_predict(symbol, raw_df)
            print(
                f"{symbol}: {result.model_name}, predicted close "
                f"{result.predicted_close:.2f} for {result.prediction_for_date.date()} "
                f"(MAPE {result.mape:.2f}%)"
            )
        except Exception as exc:
            print(f"{symbol}: skipped ({exc})")

    print("\nUpdating actual-vs-predicted records where actual prices are available...")
    try:
        update_actuals()
    except FileNotFoundError:
        pass

    generate_markdown_report()
    print("Done. Open the Streamlit dashboard with: streamlit run app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PSX forecasting pipeline.")
    parser.add_argument("--period", default="5y", help="Yahoo Finance period, e.g. 1y, 2y, 5y, max.")
    parser.add_argument("--skip-fetch", action="store_true", help="Use already downloaded CSV files.")
    args = parser.parse_args()
    run_pipeline(period=args.period, skip_fetch=args.skip_fetch)


if __name__ == "__main__":
    main()
