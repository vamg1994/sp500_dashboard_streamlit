import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import logging
# Import functions from utils
from utils import format_number, get_stock_info, get_stock_history
# Import export functions
from .export import download_plotly_figure, download_dataframe, get_download_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_company_comparison(selected_symbols):
    """Display comparison charts and metrics using cached functions from utils."""
    if not selected_symbols:
        st.info("Please select companies from the sidebar to compare.")
        return
    if len(selected_symbols) < 2:
        st.warning("Please select at least two companies to compare.")
        return

    st.subheader("Performance & Metrics Comparison")

    # --- Step 1: Fetch Data Using Cached Functions from utils ---
    fetched_data = {}
    failed_symbols = []
    ticker_objects = {symbol: yf.Ticker(symbol) for symbol in selected_symbols}

    with st.spinner(f"Fetching data for {', '.join(selected_symbols)}..."):
        for symbol, stock_ticker in ticker_objects.items():
            logging.info(f"UTILS_CMP: Processing comparison data for {symbol}")
            # Use functions imported from utils
            info_data = get_stock_info(stock_ticker)
            hist_1y = get_stock_history(stock_ticker, '1y')
            hist_1mo = get_stock_history(stock_ticker, '1mo')

            if info_data is None or hist_1y.empty or 'Close' not in hist_1y.columns:
                logging.warning(f"UTILS_CMP: Failed to fetch complete data for {symbol}")
                failed_symbols.append(symbol)
                fetched_data[symbol] = {'info': info_data, 'hist_1y': hist_1y, 'hist_1mo': hist_1mo, 'valid': False}
            else:
                logging.info(f"UTILS_CMP: Successfully fetched data for {symbol}")
                fetched_data[symbol] = {'info': info_data, 'hist_1y': hist_1y, 'hist_1mo': hist_1mo, 'valid': True}

    valid_symbols = [s for s in selected_symbols if s not in failed_symbols]
    if failed_symbols:
        st.warning(f"Could not retrieve complete data for: {', '.join(failed_symbols)}. They may be excluded from parts of the comparison.")
    if not valid_symbols or len(valid_symbols) < 2:
        st.error("Insufficient data retrieved to perform comparison.")
        return

    # --- Step 2: Price Performance Comparison (% Change) ---
    price_fig = go.Figure()
    price_export_data = pd.DataFrame()
    st.markdown("#### Price Performance (1 Year, % Change)")
    with st.spinner("Generating price comparison chart..."):
        all_indices = pd.Index([])
        for symbol in valid_symbols:
             hist = fetched_data[symbol]['hist_1y']
             if not hist.empty: all_indices = all_indices.union(hist.index)

        for symbol in valid_symbols:
            hist = fetched_data[symbol]['hist_1y']
            if not hist.empty and 'Close' in hist.columns:
                hist_aligned = hist.reindex(all_indices).ffill()
                close_prices = hist_aligned['Close'].dropna()
                if not close_prices.empty:
                    first_valid_price = close_prices.iloc[0]
                    if pd.notna(first_valid_price) and first_valid_price != 0:
                        normalized_prices = (close_prices - first_valid_price) / first_valid_price * 100
                        price_fig.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices, mode='lines', name=symbol, hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Change: %{{y:.2f}}%<extra></extra>'))
                        price_export_data[f"{symbol}_Close"] = hist['Close']
                    else: logging.warning(f"Invalid first price for {symbol}, skipping price chart trace.")
                else: logging.warning(f"No valid close prices found for {symbol} after alignment.")

    price_fig.update_layout(xaxis_title='Date', yaxis_title='Percentage Change (%)', template='plotly_dark', hovermode="x unified", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(price_fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1: download_plotly_figure(price_fig, get_download_filename("price_comparison", "html"))
    with col2:
        price_export_data_aligned = price_export_data.reindex(all_indices).ffill()
        download_dataframe(price_export_data_aligned, get_download_filename("comparison_price_data", "csv"))


    # --- Step 3: Comparative Metrics ---
    st.markdown("#### Key Metrics Comparison")
    metrics_data = {}
    metrics_available = False
    with st.spinner("Gathering comparative metrics..."):
        metrics_definition = { # Define metrics centrally
            'Market Cap': 'marketCap', 'P/E Ratio': 'trailingPE', 'EPS': 'trailingEps',
            'Revenue': 'totalRevenue', 'Profit Margin %': 'profitMargins',
            'ROE %': 'returnOnEquity', 'Beta': 'beta', 'Debt/Equity': 'debtToEquity'
        }
        all_metrics = {key: {} for key in metrics_definition.keys()} # Store raw values

        for symbol in selected_symbols: # Iterate over all selected, even failed ones
            if symbol in fetched_data and fetched_data[symbol]['info']:
                info = fetched_data[symbol]['info']
                metrics_available = True # Mark metrics as available if at least one valid info exists
                for display_name, info_key in metrics_definition.items():
                    value = info.get(info_key)
                    fmt = '%' if '%' in display_name else ('.2f' if display_name in ['P/E Ratio', 'EPS', 'Beta', 'Debt/Equity'] else None)
                    all_metrics[display_name][symbol] = format_number(value, fmt) if value is not None else 'N/A'
            else: # Handle symbols where info fetch failed or was invalid
                 for display_name in metrics_definition.keys():
                     all_metrics[display_name][symbol] = 'N/A'


    if metrics_available:
        metrics_df = pd.DataFrame(all_metrics)
        # Reorder columns to match selected_symbols order if needed
        metrics_df = metrics_df[selected_symbols]
        st.dataframe(metrics_df, use_container_width=True)
        download_dataframe(metrics_df, get_download_filename("comparative_metrics", "csv"), label="Download Metrics")
    else:
        st.info("No comparative metrics could be generated as info fetching failed for all selected stocks.")


    # --- Step 4: Volume Comparison (1 Month) ---
    st.markdown("#### Trading Volume Comparison (1 Month)")
    volume_fig = go.Figure()
    volume_available = False
    with st.spinner("Generating volume comparison chart..."):
        for symbol in valid_symbols: # Only plot for valid symbols
            hist_1mo = fetched_data[symbol]['hist_1mo']
            if not hist_1mo.empty and 'Volume' in hist_1mo.columns and hist_1mo['Volume'].sum() > 0:
                volume_fig.add_trace(go.Bar(x=hist_1mo.index, y=hist_1mo['Volume'], name=symbol, hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Volume: %{{y:,.0f}}<extra></extra>'))
                volume_available = True
            else: logging.warning(f"No volume data available for {symbol} in the last month.")

    if volume_available:
        volume_fig.update_layout(template='plotly_dark', barmode='group', xaxis_title='Date', yaxis_title='Volume', height=400, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(volume_fig, use_container_width=True)
        download_plotly_figure(volume_fig, get_download_filename("volume_comparison", "html"))
    else:
        st.info("No trading volume data available for the selected companies in the last month.")
