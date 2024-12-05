import streamlit as st
from utils import format_number

def display_key_metrics(stock):
    """Display key financial metrics in a 3-column grid"""
    info = stock.info
    
    metrics = {
        'Market Cap': info.get('marketCap'),
        'P/E Ratio': info.get('trailingPE'),
        'Revenue': info.get('totalRevenue'),
        'Gross Margin': info.get('grossMargins'),
        'Operating Margin': info.get('operatingMargins'),
        'Profit Margin': info.get('profitMargins'),
        'ROE': info.get('returnOnEquity'),
        'ROA': info.get('returnOnAssets'),
        'Current Ratio': info.get('currentRatio')
    }
    
    # Create metrics in groups of 3
    for i in range(0, len(metrics), 3):
        col1, col2, col3 = st.columns(3)
        metrics_slice = list(metrics.items())[i:i+3]
        
        for (metric, value), col in zip(metrics_slice, [col1, col2, col3]):
            with col:
                st.metric(
                    label=metric,
                    value=format_number(value)
                )
