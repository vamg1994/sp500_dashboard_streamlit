import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from .metrics import display_key_metrics
from utils import format_number
from .export import download_plotly_figure, download_dataframe, get_download_filename

def display_company_comparison(companies):
    """Display comparison charts and metrics for selected companies"""
    if len(companies) < 2:
        st.warning("Please select at least two companies to compare")
        return

    # Get stock data for all selected companies
    stocks_data = {}
    comparison_data = pd.DataFrame()
    
    print("\nFetching data for companies:", companies)
    for symbol in companies:
        try:
            stock = yf.Ticker(symbol)
            stocks_data[symbol] = stock
            hist = stock.history(period='1y')
            if hist.empty or 'Close' not in hist.columns:
                print(f"Warning: No data available for {symbol}")
                continue
            comparison_data[f"{symbol}_Close"] = hist['Close']
            comparison_data[f"{symbol}_Volume"] = hist['Volume']
            print(f"Successfully fetched data for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue

    print("\nCreating price comparison chart...")
    with st.container():
        # Create figure
        fig = go.Figure()
        
        for symbol, stock in stocks_data.items():
            hist = stock.history(period='1y')
            if hist.empty or 'Close' not in hist.columns:
                print(f"Warning: No valid data for {symbol}")
                continue
                
            # Normalize prices to percentage change from first day
            first_price = hist['Close'].iloc[0]
            if pd.isna(first_price) or first_price == 0:
                print(f"Warning: Invalid first price for {symbol}")
                continue
                
            normalized_prices = (hist['Close'] - first_price) / first_price * 100
            print(f"\nNormalized prices for {symbol}:")
            print(f"- Shape: {normalized_prices.shape}")
            print(f"- Range: [{normalized_prices.min():.2f}%, {normalized_prices.max():.2f}%]")
            
            # Add trace for valid data
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=normalized_prices,
                    mode='lines',
                    name=symbol,
                    hovertemplate=f'{symbol}<br>Change: %{{y:.2f}}%<extra></extra>'
                )
            )
            print(f"Added trace for {symbol}")

        # Minimal layout updates
        fig.update_layout(
            title='Price Performance Comparison (% Change)',
            xaxis_title='Date',
            yaxis_title='Percentage Change (%)',
            template='plotly_dark'
        )
        
        print("Rendering price comparison chart...")
        st.plotly_chart(fig, use_container_width=True)
        print("Price comparison chart rendered successfully")
    
    # Add export buttons for price comparison
    col1, col2 = st.columns(2)
    with col1:
        download_plotly_figure(fig, get_download_filename("price_comparison", "html"))
    with col2:
        download_dataframe(comparison_data, get_download_filename("comparison_data", "csv"))

    # Display comparative metrics
    st.subheader("Comparative Metrics")
    metrics_df = pd.DataFrame()

    print("\nProcessing comparative metrics...")
    for symbol, stock in stocks_data.items():
        try:
            info = stock.info
            metrics = {
                'Market Cap': format_number(info.get('marketCap')),
                'P/E Ratio': format_number(info.get('trailingPE')),
                'Revenue': format_number(info.get('totalRevenue')),
                'Profit Margin': format_number(info.get('profitMargins'), percentage=True),
                'ROE': format_number(info.get('returnOnEquity'), percentage=True),
                'Beta': format_number(info.get('beta'))
            }
            metrics_df[symbol] = pd.Series(metrics)
            print(f"Processed metrics for {symbol}")
        except Exception as e:
            print(f"Error getting metrics for {symbol}: {str(e)}")
            continue

    if not metrics_df.empty:
        print("\nMetrics DataFrame shape:", metrics_df.shape)
        st.dataframe(
            metrics_df.style.background_gradient(axis=1, cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Add export button for metrics
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=metrics_df.to_csv(),
            file_name=get_download_filename("comparative_metrics", "csv"),
            mime="text/csv"
        )

    # Volume comparison
    st.subheader("Trading Volume Comparison")
    volume_fig = go.Figure()
    
    print("\nCreating volume comparison chart...")
    for symbol, stock in stocks_data.items():
        hist = stock.history(period='1mo')  # Last month's volume
        if hist.empty or 'Volume' not in hist.columns:
            continue
            
        volume_fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name=symbol,
                hovertemplate=f'{symbol}<br>Volume: %{{y:,.0f}}<extra></extra>'
            )
        )

    volume_fig.update_layout(
        template='plotly_dark',
        barmode='group',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
    )

    st.plotly_chart(volume_fig, use_container_width=True)
    
    # Add export buttons for volume comparison
    col1, col2 = st.columns(2)
    with col1:
        download_plotly_figure(volume_fig, get_download_filename("volume_comparison", "html"))
