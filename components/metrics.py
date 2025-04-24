import streamlit as st
# Import functions from utils
from utils import format_number, get_stock_info
import logging

# Configure logging
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def display_key_metrics(stock):
    """Display key financial metrics in a grid, using cached data from utils."""
    ticker_symbol = getattr(stock, 'ticker', 'Unknown')
    st.subheader(f"Key Financial Metrics ({ticker_symbol})")

    # Use the cached function from utils
    info = get_stock_info(stock)

    if info is None:
        st.warning(f"Financial metrics could not be loaded for {ticker_symbol}.")
        return

    # Define metrics and their formats
    metrics_definition = {
        'Market Cap': ('marketCap', None),
        'Trailing P/E': ('trailingPE', '.2f'),
        'Forward P/E': ('forwardPE', '.2f'), # Added Forward P/E
        'EPS': ('trailingEps', '.2f'),
        'Revenue': ('totalRevenue', None),
        'EBITDA': ('ebitda', None),
        'Gross Margin': ('grossMargins', '%'),
        'Operating Margin': ('operatingMargins', '%'),
        'Profit Margin': ('profitMargins', '%'),
        'ROE': ('returnOnEquity', '%'),
        'Debt/Equity': ('debtToEquity', '.2f'),
        'Current Ratio': ('currentRatio', '.2f'),
        'Beta': ('beta', '.2f'), # Added Beta
        # Add more metrics if needed: e.g., priceToBook, enterpriseValue
    }

    # Extract and format metrics
    metrics_data = {}
    for display_name, (info_key, fmt) in metrics_definition.items():
        value = info.get(info_key)
        if value is not None:
             metrics_data[display_name] = format_number(value, fmt) # Format here
        # else: Keep N/A out for cleaner display, or uncomment to show N/A
        #    metrics_data[display_name] = "N/A"

    if not metrics_data:
        st.info(f"No key financial metrics available for {ticker_symbol}.")
        return

    # Display metrics in a responsive grid (e.g., 3 or 4 columns)
    num_columns = 4
    cols = st.columns(num_columns)
    metric_items = list(metrics_data.items())

    for i, (label, formatted_value) in enumerate(metric_items):
        col_index = i % num_columns
        with cols[col_index]:
            st.metric(label=label, value=formatted_value)
