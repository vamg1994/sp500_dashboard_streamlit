import pandas as pd
import yfinance as yf
import streamlit as st
import logging

def load_sp500_companies():
    """Load S&P 500 companies from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    
    # Print column names to debug
    print("Available columns:", df.columns.tolist())
    
    # Map expected column names to actual column names
    column_mapping = {
        'Symbol': 'Symbol',
        'Name': 'Security',  # Wikipedia uses 'Security' instead of 'Name'
        'Sector': 'GICS Sector'  # Wikipedia uses 'GICS Sector' instead of 'Sector'
    }
    
    # Select and rename columns
    return df[[column_mapping['Symbol'], column_mapping['Name'], column_mapping['Sector']]].rename(columns={
        column_mapping['Symbol']: 'Symbol',
        column_mapping['Name']: 'Name',
        column_mapping['Sector']: 'Sector'
    })

def format_number(number, percentage=False):
    """Format large numbers with K, M, B suffixes or as percentages"""
    if number is None or pd.isna(number):
        return "N/A"
    
    try:
        number = float(number)  # Convert to float for consistent handling
        
        if percentage:
            # Handle percentage values
            if abs(number) < 1:  # If number is already in decimal form (e.g., 0.15 for 15%)
                return f"{(number * 100):.2f}%"
            else:  # If number is already in percentage form (e.g., 15)
                return f"{number:.2f}%"
        
        if abs(number) >= 1e9:
            return f"{number/1e9:.2f}B"
        elif abs(number) >= 1e6:
            return f"{number/1e6:.2f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.2f}K"
        else:
            return f"{number:.2f}"
    except (ValueError, TypeError) as e:
        print(f"Error formatting number: {number}, Error: {str(e)}")
        return "N/A"

# Configure logging if not already done elsewhere, or ensure it's configured at app start
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Cached Data Fetching Functions ---

@st.cache_data(ttl=600) # Cache for 10 minutes (600 seconds)
def get_stock_info(_stock_ticker):
    """Fetches and caches stock information. To be called from utils."""
    # Ensure yfinance is imported if not already at the top of utils.py
    import yfinance as yf
    if not isinstance(_stock_ticker, yf.Ticker):
        logging.error(f"get_stock_info received invalid type: {type(_stock_ticker)}")
        return None

    ticker_symbol = getattr(_stock_ticker, 'ticker', 'unknown_ticker')
    try:
        logging.info(f"UTILS: Fetching info for {ticker_symbol}")
        info = _stock_ticker.info
        # Check if the returned info dict is empty or lacks essential data
        if not info or not info.get('symbol'):
            logging.warning(f"UTILS: Incomplete or empty info dictionary returned for ticker {ticker_symbol}.")
            # Basic check if the session seems valid - might indicate rate limit if headers are missing
            session_valid = hasattr(_stock_ticker, 'session') and hasattr(_stock_ticker.session, 'headers')
            if not session_valid or _stock_ticker.session.headers.get('User-Agent') is None:
                 st.error(f"⚠️ Could not fetch company info for {ticker_symbol}. Potential rate limit or API issue.")
                 logging.warning(f"UTILS: Potential rate limit suspected for {ticker_symbol} (info fetch).")
            else:
                 st.error(f"No data returned for ticker {ticker_symbol}. It might be delisted, invalid, or lack summary data.")
            return None
        logging.info(f"UTILS: Successfully fetched info for {ticker_symbol}")
        return info
    except Exception as e:
        logging.error(f"UTILS: Error fetching info for {ticker_symbol}: {e}", exc_info=True) # Log traceback
        st.error(f"⚠️ Could not fetch company info for {ticker_symbol}. An error occurred: {e}. The API might be temporarily unavailable or the ticker is invalid. Please try again later.")
        return None # Return None on error

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_stock_history(_stock_ticker, _period):
    """Fetches and caches historical stock data. To be called from utils."""
     # Ensure yfinance and pandas are imported if not already at the top of utils.py
    import yfinance as yf
    import pandas as pd
    if not isinstance(_stock_ticker, yf.Ticker):
        logging.error(f"get_stock_history received invalid type: {type(_stock_ticker)}")
        return pd.DataFrame()

    ticker_symbol = getattr(_stock_ticker, 'ticker', 'unknown_ticker')
    try:
        logging.info(f"UTILS: Fetching history for {ticker_symbol} with period {_period}")
        hist = _stock_ticker.history(period=_period)
        if hist.empty:
            st.warning(f"No historical data returned for {ticker_symbol} for period {_period}. The stock might be new, delisted, or have no data for this timeframe.")
            return pd.DataFrame() # Return empty DataFrame is safer than None for hist
        logging.info(f"UTILS: Successfully fetched history for {ticker_symbol} (period: {_period})")
        return hist
    except Exception as e:
        logging.error(f"UTILS: Error fetching history for {ticker_symbol} (period: {_period}): {e}", exc_info=True)
        st.error(f"⚠️ Could not fetch historical data for {ticker_symbol} (period: {_period}). API error or invalid ticker/period. Please try again later.")
        return pd.DataFrame() # Return empty DataFrame on error
