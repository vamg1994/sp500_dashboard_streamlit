import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from .export import download_plotly_figure, download_dataframe, get_download_filename


def calculate_garman_klass_volatility(data, window=10):
    """Calculate Garman-Klass volatility"""
    log_hl = (np.log(data['High']) - np.log(data['Low']))**2
    log_co = (np.log(data['Close']) - np.log(data['Open']))**2
    volatility = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    return volatility.rolling(window=window).mean()


def calculate_technical_indicators(data):
    """Calculate technical indicators for the stock data"""
    # Calculate RSI
    rsi_indicator = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi_indicator.rsi()

    # Calculate MACD
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()

    # Calculate SMA (20-day and 50-day)
    sma_20 = SMAIndicator(close=data['Close'], window=20)
    sma_50 = SMAIndicator(close=data['Close'], window=50)
    data['SMA_20'] = sma_20.sma_indicator()
    data['SMA_50'] = sma_50.sma_indicator()

    # Calculate Garman-Klass Volatility
    data['GK_Volatility'] = calculate_garman_klass_volatility(data)

    return data


def display_stock_chart(stock, period):
    """Display interactive stock price chart with technical indicators"""
    # Get historical data
    hist = stock.history(period=period)

    # Get EPS data
    info = stock.info
    eps = info.get('trailingEps', None)

    # Calculate technical indicators
    hist = calculate_technical_indicators(hist)

    # Create wrapper div for chart centering
    st.markdown("""
        <style>
        .chart-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 50%;
            max-width: 1000px;
            margin: 0 auto;
            padding: 0;
        }
        .chart-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            padding: 0;
        }
        @media screen and (max-width: 768px) {
            .chart-wrapper {
                padding: 0 10px;
            }
        }
        </style>
    """,
                unsafe_allow_html=True)

    # Create subplots with adjusted spacing
    fig = make_subplots(rows=4,
                        cols=1,
                        row_heights=[0.4, 0.2, 0.2, 0.2],
                        vertical_spacing=0.03,
                        shared_xaxes=True,
                        subplot_titles=('Price & SMA', 'Volatility', 'RSI',
                                        'MACD'))

    # Main price chart with SMA
    fig.add_trace(go.Scatter(x=hist.index,
                             y=hist['Close'],
                             mode='lines',
                             name='Stock Price',
                             line=dict(color='#ff8800', width=1.5)),
                  row=1,
                  col=1)

    # Add SMA lines
    fig.add_trace(go.Scatter(x=hist.index,
                             y=hist['SMA_20'],
                             mode='lines',
                             name='20-day SMA',
                             line=dict(color='#00ff00', width=1)),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=hist.index,
                             y=hist['SMA_50'],
                             mode='lines',
                             name='50-day SMA',
                             line=dict(color='#ff00ff', width=1)),
                  row=1,
                  col=1)

    # Add EPS line if available
    if eps is not None:
        fig.add_hline(y=eps,
                      line_dash="dash",
                      line_color="#00ffff",
                      annotation_text=f"EPS: ${eps:.2f}",
                      row=1,
                      col=1)

    # Garman-Klass Volatility
    fig.add_trace(go.Scatter(x=hist.index,
                             y=hist['GK_Volatility'],
                             mode='lines',
                             name='GK Volatility',
                             line=dict(color='#f6a709', width=1)),
                  row=2,
                  col=1)

    # RSI
    fig.add_trace(go.Scatter(x=hist.index,
                             y=hist['RSI'],
                             mode='lines',
                             name='RSI',
                             line=dict(color='#f6a709', width=1)),
                  row=3,
                  col=1)

    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=hist.index,
                             y=hist['MACD'],
                             mode='lines',
                             name='MACD',
                             line=dict(color='#ff8800', width=1)),
                  row=4,
                  col=1)

    fig.add_trace(go.Scatter(x=hist.index,
                             y=hist['MACD_Signal'],
                             mode='lines',
                             name='Signal Line',
                             line=dict(color='#f6a709', width=1)),
                  row=4,
                  col=1)

    fig.add_trace(go.Bar(x=hist.index,
                         y=hist['MACD_Hist'],
                         name='MACD Histogram',
                         marker_color=hist['MACD_Hist'].apply(
                             lambda x: '#ff8800' if x >= 0 else '#ff3322')),
                  row=4,
                  col=1)

    # Update layout with improved spacing and alignment
    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=900,  # Increased height to accommodate new subplot
        autosize=True,
        margin=dict(l=50, r=30, t=50, b=20, pad=5),
        showlegend=True,
        legend=dict(orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(26,28,35,0.8)'),
        plot_bgcolor='#1a1c23',
        paper_bgcolor='#1a1c23')

    # Update axes styling
    fig.update_xaxes(gridcolor='#2e3140',
                     zerolinecolor='#2e3140',
                     showgrid=True,
                     gridwidth=1)

    fig.update_yaxes(gridcolor='#2e3140',
                     zerolinecolor='#2e3140',
                     showgrid=True,
                     gridwidth=1,
                     automargin=True,
                     fixedrange=True)

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1, title_standoff=15)
    fig.update_yaxes(title_text="Volatility", row=2, col=1, title_standoff=15)
    fig.update_yaxes(title_text="RSI", row=3, col=1, title_standoff=15)
    fig.update_yaxes(title_text="MACD", row=4, col=1, title_standoff=15)

    # Create containers for proper centering
    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
    chart_container = st.container()
    with chart_container:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig,
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'responsive': True,
                            'scrollZoom': True,
                            'showAxisDragHandles': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        })
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Add export buttons in columns for better alignment
    col1, col2 = st.columns(2)
    with col1:
        download_plotly_figure(
            fig, get_download_filename(f"{stock.ticker}_chart", "html"))
    with col2:
        download_dataframe(
            hist, get_download_filename(f"{stock.ticker}_data", "csv"))
