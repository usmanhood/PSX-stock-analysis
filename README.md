# AI-Powered PSX Stock Forecasting and Analytics System

A university-project friendly Python system for fetching Pakistan Stock Exchange (PSX) stock data, cleaning it, performing exploratory analytics, building technical indicators, forecasting next-day close prices, validating predictions against actual closes, and presenting results in a Streamlit dashboard.

## Project Flow

```text
PSX Website/API
        ↓
Fetch Top 10 Companies Data
        ↓
Store Historical Prices
        ↓
Clean Data (Pandas)
        ↓
Exploratory Data Analysis
        ↓
Charts & Technical Indicators
        ↓
Machine Learning Forecast
        ↓
Predict Tomorrow's Price
        ↓
Next Day Fetch Actual Price
        ↓
Compare Prediction vs Actual
        ↓
Calculate Error %
        ↓
Dashboard + Report
```

## Features

- Free data-first workflow using `yfinance` tickers for PSX companies.
- Reproducible ETL pipeline for the top 10 sample companies: OGDC, MARI, HUBC, ENGROH, FFC, UBL, HBL, LUCK, SYS, and PPL.
- Pandas-based cleaning, duplicate removal, type conversion, and missing-value handling.
- Technical indicators: returns, volatility, MA20, MA50, MA200, and trend labels.
- Machine-learning forecasts using a rolling-window Random Forest model.
- Prediction storage and next-day actual-vs-predicted validation metrics.
- Streamlit dashboard with company selector, charts, latest prediction, and validation table.
- Daily automation entry point suitable for cron or scheduled jobs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fetch, Analyze, and Forecast

```bash
python -m psx_forecasting.pipeline --period 5y
```

This creates local CSV files under `data/`:

- `data/raw/`: uncleaned downloaded history.
- `data/processed/`: cleaned history with indicators.
- `data/predictions/predictions.csv`: latest generated predictions.
- `data/predictions/validation.csv`: actual-vs-predicted comparisons when actual closes are available.

## Run the Dashboard

```bash
streamlit run app.py
```

## Automate Daily Updates

Run once per day after market close:

```bash
python -m psx_forecasting.automation --period 5y
```

Example cron entry:

```cron
30 18 * * 1-5 cd /path/to/PSX-stock-analysis && /path/to/python -m psx_forecasting.automation --period 5y
```

## Cost

The default workflow uses free/open-source tools: Python, pandas, yfinance, Plotly, Streamlit, and scikit-learn. Paid APIs such as Alpha Vantage, Polygon.io, or Financial Modeling Prep can be added later if more reliable or higher-volume data is required.

## Important Note

Stock prediction is probabilistic, not certain. Forecasts are based on historical patterns and can be invalidated by news, macroeconomic changes, company announcements, liquidity, and unexpected market events.
