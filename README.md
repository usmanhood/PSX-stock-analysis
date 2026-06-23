# AI-Powered PSX Stock Forecasting and Analytics System

This is a free-resource Python project for fetching PSX stock prices, cleaning data, creating analytics charts, training machine learning forecasts, saving tomorrow predictions, checking actual prices later, and viewing everything in a Streamlit dashboard.

## Companies

The default top 10 list is:

- OGDC
- MARI
- HUBC
- ENGROH
- FFC
- UBL
- HBL
- LUCK
- SYS
- PPL

Yahoo Finance symbols use the `.KA` suffix, for example `OGDC.KA`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python3 src/run_pipeline.py
```

This will:

1. Fetch free historical data with `yfinance`.
2. Save raw CSV files in `data/raw`.
3. Clean data and calculate indicators.
4. Train Linear Regression and Random Forest models.
5. Save the better model for each stock.
6. Predict the next trading day's close.
7. Save predictions in `data/predictions/predictions.csv`.
8. Update actual-vs-predicted rows when actual prices become available.
9. Generate `reports/latest_report.md`.

If you want to test the app before downloading live data, create demo data first:

```bash
python3 src/create_sample_data.py
python3 src/run_pipeline.py --skip-fetch
```

## Open the dashboard

```bash
streamlit run app.py
```

## Daily automation

For a simple always-running local job:

```bash
python3 src/daily_job.py
```

For a cron job, schedule this command after PSX market close:

```bash
cd /path/to/project && .venv/bin/python src/run_pipeline.py
```

## Notes

Stock prediction is probabilistic. The model learns from historical prices and indicators, but news, macroeconomic changes, earnings, and market shocks can change prices suddenly.

If Yahoo Finance does not return data for a ticker, replace or extend that ticker in `config.py`, or add an official PSX scraper inside `psx_forecasting/data_fetcher.py`.
