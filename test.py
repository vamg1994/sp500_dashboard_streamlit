import yfinance as yf
import pandas as pd
import logging
import json

# Configure basic logging to see output clearly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of tickers to test (include known problematic and known good ones)
tickers_to_test = ['MMM', 'ABBV', 'AAPL', 'GOOGL', 'INVALID_TICKER']

# --- Test Function Definitions ---

def test_fetch_info(ticker_symbol):
    """Attempts to fetch and print stock info."""
    logging.info(f"--- Testing .info for {ticker_symbol} ---")
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        if not info:
            logging.warning(f"`.info` returned an empty dictionary for {ticker_symbol}.")
        else:
            logging.info(f"Successfully fetched `.info` for {ticker_symbol}.")
            # Print a few key pieces of info
            print(f"  Symbol: {info.get('symbol', 'N/A')}")
            print(f"  Name: {info.get('shortName', info.get('longName', 'N/A'))}")
            print(f"  Sector: {info.get('sector', 'N/A')}")
            print(f"  Market Cap: {info.get('marketCap', 'N/A')}")
            # print(f"  Full info dict (first 500 chars): {str(info)[:500]}...") # Uncomment for more detail
        print("-" * 20)
        return True

    except requests.exceptions.HTTPError as http_err:
         logging.error(f"HTTP error fetching info for {ticker_symbol}: {http_err}")
         print(f"  Response Content: {http_err.response.text}") # Show error response from Yahoo
         print("-" * 20)
         return False
    except json.JSONDecodeError as json_err:
        logging.error(f"JSON decode error fetching info for {ticker_symbol}: {json_err}")
        # Try to get the raw text that caused the error (requires modification in yfinance or careful trapping)
        # This usually happens after a 4xx/5xx error where HTML/text is returned instead of JSON
        print(f"  Likely received non-JSON response from API for {ticker_symbol}.")
        print("-" * 20)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching info for {ticker_symbol}: {e}", exc_info=True)
        print("-" * 20)
        return False

def test_fetch_history(ticker_symbol, period='1y'):
    """Attempts to fetch and print stock history."""
    logging.info(f"--- Testing .history(period='{period}') for {ticker_symbol} ---")
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period)

        if hist.empty:
            logging.warning(f"`.history(period='{period}')` returned an empty DataFrame for {ticker_symbol}.")
        else:
            logging.info(f"Successfully fetched history for {ticker_symbol} (period: {period}).")
            print(f"  DataFrame shape: {hist.shape}")
            print(f"  Date range: {hist.index.min()} -> {hist.index.max()}")
            print(f"  Last 5 rows of data:")
            print(hist.tail().to_string()) # Print tail for verification
        print("-" * 20)
        return True

    except Exception as e:
        # yfinance often logs errors internally for history failures (e.g., "No data found")
        logging.error(f"An error occurred fetching history for {ticker_symbol} (period: {period}): {e}", exc_info=True)
        print("-" * 20)
        return False

# --- Run Tests ---

if __name__ == "__main__":
    logging.info("Starting yfinance data fetching test...")
    # Import requests here specifically to catch HTTPError
    import requests

    results = {}
    for ticker in tickers_to_test:
        print(f"\n===== Testing Ticker: {ticker} =====")
        info_success = test_fetch_info(ticker)
        hist_success = test_fetch_history(ticker)
        results[ticker] = {'info': info_success, 'history': hist_success}

    print("\n===== Test Summary =====")
    for ticker, status in results.items():
        print(f"{ticker}: Info Fetch={'Success' if status['info'] else 'Failed'}, History Fetch={'Success' if status['history'] else 'Failed'}")

    logging.info("Test finished.")
