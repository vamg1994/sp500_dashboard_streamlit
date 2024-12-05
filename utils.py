import pandas as pd
import yfinance as yf

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
