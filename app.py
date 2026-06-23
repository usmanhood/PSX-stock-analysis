from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PREDICTIONS_DIR, RAW_DIR, TOP_PSX_TICKERS
from psx_forecasting.features import add_technical_indicators


st.set_page_config(page_title="PSX Stock Forecasting", layout="wide")

st.title("AI-Powered PSX Stock Forecasting and Analytics")


@st.cache_data(show_spinner=False)
def load_stock(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    return add_technical_indicators(pd.read_csv(path))


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


symbol = st.sidebar.selectbox("Company", list(TOP_PSX_TICKERS.keys()))
df = load_stock(symbol)
predictions = load_predictions()

if df.empty:
    st.warning("No data found yet. Run `python src/run_pipeline.py` first.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])

latest = df.dropna(subset=["Close"]).iloc[-1]
symbol_predictions = predictions[predictions["Symbol"] == symbol].copy() if not predictions.empty else pd.DataFrame()
latest_prediction = symbol_predictions.tail(1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"{latest['Close']:.2f}")
col2.metric("Daily Return", f"{latest.get('Return', 0) * 100:.2f}%")
col3.metric("20-Day Volatility", f"{latest.get('Volatility20', 0) * 100:.2f}%")
if latest_prediction.empty:
    col4.metric("Next Prediction", "Not ready")
else:
    col4.metric("Next Prediction", f"{latest_prediction.iloc[-1]['Predicted_Close']:.2f}")

price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
price_fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))
price_fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))
price_fig.update_layout(title=f"{symbol} Close Price and Moving Averages", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(price_fig, use_container_width=True)

left, right = st.columns(2)
with left:
    returns = df.dropna(subset=["Return"])
    st.plotly_chart(
        px.histogram(returns, x="Return", nbins=60, title="Daily Return Distribution"),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        px.bar(df.tail(90), x="Date", y="Volume", title="Recent Volume"),
        use_container_width=True,
    )

if not predictions.empty:
    st.subheader("Prediction Log")
    st.dataframe(symbol_predictions.tail(20), use_container_width=True, hide_index=True)

available_frames = []
for candidate in TOP_PSX_TICKERS:
    stock = load_stock(candidate)
    if not stock.empty:
        available_frames.append(stock[["Date", "Symbol", "Close"]])

if len(available_frames) > 1:
    all_prices = pd.concat(available_frames)
    pivot = all_prices.pivot_table(index="Date", columns="Symbol", values="Close").pct_change()
    corr = pivot.corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Top Companies Return Correlation"), use_container_width=True)
