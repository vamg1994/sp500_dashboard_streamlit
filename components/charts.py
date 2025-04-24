import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import logging
# Import cached functions from utils
from utils import get_stock_info, get_stock_history
# Import export functions
from .export import download_plotly_figure, download_dataframe, get_download_filename

# Configure logging
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Technical Indicators (keep these functions here as they are specific to charts) ---

def calculate_garman_klass_volatility(data, window=10):
    """Calculate Garman-Klass volatility"""
    if not all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
        logging.warning("Missing required columns for Garman-Klass volatility calculation.")
        return pd.Series(index=data.index, dtype=float)
    # Avoid log(0) errors
    data = data.replace(0, np.nan)
    log_hl = (np.log(data['High']) - np.log(data['Low']))**2
    log_co = (np.log(data['Close']) - np.log(data['Open']))**2
    volatility = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    return volatility.rolling(window=window).mean()

def calculate_technical_indicators(data):
    """Calculate technical indicators for the stock data"""
    if data.empty or 'Close' not in data.columns or len(data['Close']) == 0:
         logging.warning("Data is empty or 'Close' column missing, skipping indicator calculation.")
         for col in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'GK_Volatility']:
             data[col] = np.nan
         return data
    data = data.copy()
    try:
        rsi_indicator = RSIIndicator(close=data['Close'], window=14, fillna=True)
        data['RSI'] = rsi_indicator.rsi()
        macd = MACD(close=data['Close'], fillna=True)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()
        sma_20 = SMAIndicator(close=data['Close'], window=20, fillna=True)
        sma_50 = SMAIndicator(close=data['Close'], window=50, fillna=True)
        data['SMA_20'] = sma_20.sma_indicator()
        data['SMA_50'] = sma_50.sma_indicator()
        data['GK_Volatility'] = calculate_garman_klass_volatility(data)
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}", exc_info=True)
        for col in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'GK_Volatility']:
            if col not in data.columns: data[col] = np.nan
        st.warning("Could not calculate some technical indicators.")
    return data

# --- Display Function ---

def display_stock_chart(stock, period):
    """Display interactive stock price chart with technical indicators"""
    ticker_symbol = getattr(stock, 'ticker', 'Unknown')

    # Get data using cached functions from utils
    hist = get_stock_history(stock, period) # Use utils function
    info = get_stock_info(stock)            # Use utils function

    if hist.empty:
        st.warning(f"No historical data available to plot for {ticker_symbol} for the selected period.")
        return

    eps = info.get('trailingEps', None) if info else None
    hist = calculate_technical_indicators(hist)

    if 'Close' not in hist.columns or hist['Close'].isnull().all():
        st.error(f"Could not process or plot price data for {ticker_symbol}.")
        return

    # --- Plotting Code ---
    st.subheader(f"Price Chart & Technical Indicators ({ticker_symbol})")
    fig = make_subplots(rows=4, cols=1, row_heights=[0.4, 0.2, 0.2, 0.2],
                        vertical_spacing=0.03, shared_xaxes=True,
                        subplot_titles=('Price & SMA', 'Volatility', 'RSI', 'MACD'))

    # Price & SMA
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    if 'SMA_20' in hist.columns: fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#ff7f0e', width=1)), row=1, col=1)
    if 'SMA_50' in hist.columns: fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#2ca02c', width=1)), row=1, col=1)
    if eps is not None: fig.add_hline(y=eps, line_dash="dash", line_color="#d62728", annotation_text=f"EPS: ${eps:.2f}", annotation_position="bottom right", row=1, col=1)

    # Volatility
    if 'GK_Volatility' in hist.columns: fig.add_trace(go.Scatter(x=hist.index, y=hist['GK_Volatility'], mode='lines', name='GK Volatility', line=dict(color='#9467bd', width=1)), row=2, col=1)

    # RSI
    if 'RSI' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI', line=dict(color='#8c564b', width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    if all(col in hist.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], mode='lines', name='MACD', line=dict(color='#e377c2', width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='#7f7f7f', width=1)), row=4, col=1)
        colors = ['#2ca02c' if val >= 0 else '#d62728' for val in hist['MACD_Hist']]
        fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='Histogram', marker_color=colors), row=4, col=1)

    # Layout
    fig.update_layout(height=800, autosize=True, margin=dict(l=40, r=40, t=50, b=40), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, zeroline=False, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Price", row=1, col=1, title_standoff=10)
    if 'GK_Volatility' in hist.columns: fig.update_yaxes(title_text="Volatility", row=2, col=1, title_standoff=10)
    if 'RSI' in hist.columns: fig.update_yaxes(title_text="RSI", row=3, col=1, title_standoff=10)
    if 'MACD' in hist.columns: fig.update_yaxes(title_text="MACD", row=4, col=1, title_standoff=10)

    st.plotly_chart(fig, use_container_width=True)

    # --- Export Buttons ---
    col1, col2 = st.columns(2)
    filename_base = get_download_filename(f"{ticker_symbol}_{period}", "")
    with col1:
        download_plotly_figure(fig, filename_base + "chart.html")
    with col2:
        export_hist = hist.copy().dropna(axis=1, how='all')
        download_dataframe(export_hist, filename_base + "data.csv")
