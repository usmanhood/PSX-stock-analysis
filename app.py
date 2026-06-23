"""Streamlit dashboard for PSX analytics and forecasts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from psx_forecasting.config import PREDICTIONS_FILE, PROCESSED_DIR, TOP_10_PSX_TICKERS, VALIDATION_FILE

st.set_page_config(page_title="PSX Forecasting Dashboard", layout="wide")
st.title("AI-Powered PSX Stock Forecasting and Analytics System")
st.caption("Fetch, clean, analyze, forecast, and validate top PSX company prices.")

symbol = st.sidebar.selectbox("Select Company", list(TOP_10_PSX_TICKERS.keys()))
history_path = PROCESSED_DIR / f"{symbol}.csv"

if not history_path.exists():
    st.warning("No processed data found. Run `python -m psx_forecasting.pipeline --period 5y` first.")
    st.stop()

history = pd.read_csv(history_path, parse_dates=["Date"])
latest = history.iloc[-1]

metric_cols = st.columns(4)
metric_cols[0].metric("Latest Close", f"{latest['Close']:.2f}")
metric_cols[1].metric("Latest Volume", f"{latest['Volume']:,.0f}")
metric_cols[2].metric("Trend", latest.get("Trend", "N/A"))
metric_cols[3].metric("20D Volatility", f"{latest.get('Volatility_20D', 0):.4f}")

price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=history["Date"], y=history["Close"], name="Close"))
for column in ["MA20", "MA50", "MA200"]:
    if column in history:
        price_fig.add_trace(go.Scatter(x=history["Date"], y=history[column], name=column))
price_fig.update_layout(title=f"{symbol} Close Price and Moving Averages", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(price_fig, use_container_width=True)

chart_cols = st.columns(2)
with chart_cols[0]:
    returns = history.dropna(subset=["Return"])
    st.plotly_chart(px.histogram(returns, x="Return", nbins=60, title="Daily Return Distribution"), use_container_width=True)
with chart_cols[1]:
    st.plotly_chart(px.bar(history.tail(90), x="Date", y="Volume", title="Last 90 Trading Days Volume"), use_container_width=True)

if PREDICTIONS_FILE.exists():
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["Prediction_Date"])
    company_predictions = predictions[predictions["Symbol"] == symbol].sort_values("Prediction_Date")
    st.subheader("Tomorrow Prediction")
    if company_predictions.empty:
        st.info("No prediction saved for this company yet.")
    else:
        st.dataframe(company_predictions.tail(10), use_container_width=True)
else:
    st.info("No predictions file found yet.")

if VALIDATION_FILE.exists():
    validation = pd.read_csv(VALIDATION_FILE, parse_dates=["Prediction_Date"])
    company_validation = validation[validation["Symbol"] == symbol]
    st.subheader("Actual vs Predicted")
    if company_validation.empty:
        st.info("No validated prediction exists for this company yet.")
    else:
        st.dataframe(company_validation.sort_values("Prediction_Date").tail(20), use_container_width=True)
