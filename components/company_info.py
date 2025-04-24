import streamlit as st
import logging
# Import the cached function from utils
from utils import get_stock_info, format_number # Also import format_number if used for stats

# Configure logging if needed specifically here, or rely on global config
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def display_company_info(stock):
    """Display company information and profile using responsive containers"""
    # Call the cached function (now imported from utils) to get info
    info = get_stock_info(stock)

    # If info is None (due to error or no data), display a message and exit
    if info is None:
        st.warning("Company information could not be loaded.") # Error displayed in get_stock_info
        return

    # Get symbol safely for keys etc.
    symbol = info.get('symbol', 'unknown')

    # Main container for company profile
    with st.container():
        st.subheader("Company Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        with col2:
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")

    # Business Summary section
    with st.container():
        st.markdown("#### Business Summary")
        business_summary = info.get('longBusinessSummary', 'No information available')
        st.markdown(f"<div style='max-height: 200px; overflow-y: auto; padding: 10px; background-color: {st.get_option('theme.secondaryBackgroundColor')}; border-radius: 0.5rem; margin-bottom: 1rem;'>{business_summary}</div>", unsafe_allow_html=True)
        # Removed the complex show more/less logic for simplicity, using scroll box instead

    # Key Statistics section
    with st.container():
        st.markdown("#### Key Statistics")
        stats = {
            'Market Cap': info.get('marketCap'),
            'Beta': info.get('beta'),
            '52 Week High': info.get('fiftyTwoWeekHigh'),
            '52 Week Low': info.get('fiftyTwoWeekLow'),
            'Forward P/E': info.get('forwardPE'),
            'Trailing P/E': info.get('trailingPE'),
            'Volume': info.get('volume'),
            'Avg Volume': info.get('averageVolume')
        }

        # Display stats in a responsive grid (e.g., 4 columns)
        num_columns = 4
        cols = st.columns(num_columns)
        stat_items = [(k, v) for k, v in stats.items() if v is not None] # Filter out None values

        if not stat_items:
            st.info("Key statistics are not available.")
        else:
            for i, (key, value) in enumerate(stat_items):
                 with cols[i % num_columns]:
                      # Determine format: default, or '.2f' for ratios
                      fmt = '.2f' if key in ['Beta', 'Forward P/E', 'Trailing P/E'] else None
                      st.metric(label=key, value=format_number(value, fmt)) # Use format_number
