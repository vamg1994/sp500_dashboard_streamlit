import streamlit as st
import yfinance as yf
import pandas as pd
from components.charts import display_stock_chart
from components.metrics import display_key_metrics
from components.company_info import display_company_info
from components.comparison import display_company_comparison
from components.news import display_news_sentiment
from components.prediction import display_ml_prediction
from utils import load_sp500_companies

# Page configuration
st.set_page_config(
    page_title="VAM S&P 500 Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



def main():
    # Sidebar for mode selection
    with st.sidebar:
        st.title("VAM S&P 500 Dashboard")
        
        # Add profile picture and text
        st.markdown(
            '''
            <style>
            .circular-image {
                width: 150px;
                height: 150px;
                border-radius: 50%;
                object-fit: cover;
            }
            img {
                border-radius: 50%;
                object-fit: cover;
            }
            </style>
            ''',
            unsafe_allow_html=True
        )
        st.image("Foto VAM.png", width=150, use_container_width=False)
        st.markdown("""
        ### Virgilio Madrid
        Data Scientist
        """)
        
        st.markdown("---")  # Add a divider
        st.write("Data source: Yahoo Finance")
        mode = st.radio("Select Mode", ["Single Company Analysis", "Company Comparison"])
        
        # Load S&P 500 companies
        sp500_companies = load_sp500_companies()

    if mode == "Single Company Analysis":
        # Company selector in sidebar
        with st.sidebar:
            selected_company = st.selectbox(
                "Select a company",
                sp500_companies['Symbol'].tolist(),
                format_func=lambda x: f"{x} - {sp500_companies[sp500_companies['Symbol'] == x]['Name'].iloc[0]}"
            )
            
            # Time period selector with corrected period values
            periods = {
                '1D': '1d',
                '1W': '5d',
                '1M': '1mo',
                '3M': '3mo',
                'YTD': 'ytd',
                '1Y': '1y',
                '3Y': '2y',
                '5Y': '5y'
            }
            selected_period = st.select_slider('Select Time Period', options=list(periods.keys()))

        # Get stock data with loading animation
        with st.spinner(f'Loading data for {selected_company}...'):
            stock = yf.Ticker(selected_company)
            company_name = sp500_companies[sp500_companies['Symbol'] == selected_company]['Name'].iloc[0]

        

        # Create tabs for different sections
        overview_tab, charts_tab, financials_tab, prediction_tab, news_tab = st.tabs([
            "Company Overview", 
            "Price Charts", 
            "Financial Metrics",
            "Price Prediction",
            "News & Sentiment"
        ])

        with overview_tab:
            # Company information
            display_company_info(stock)
            

        with charts_tab:
            # Stock price chart with full width
            display_stock_chart(stock, periods[selected_period])

        with financials_tab:
            # Financial metrics
            display_key_metrics(stock)
            
        with prediction_tab:
            # ML price prediction
            display_ml_prediction(stock)
            
        with news_tab:
            # News sentiment analysis
            display_news_sentiment(selected_company, company_name)

    else:  # Company Comparison mode
        # Multiple company selector in sidebar
        with st.sidebar:
            selected_companies = st.multiselect(
                "Select companies to compare (2-4 recommended)",
                sp500_companies['Symbol'].tolist(),
                format_func=lambda x: f"{x} - {sp500_companies[sp500_companies['Symbol'] == x]['Name'].iloc[0]}",
                max_selections=4
            )

        # Create tabs for comparison sections
        if selected_companies:
            performance_tab = st.tabs([
                "Performance Comparison"
            ])
            
            with performance_tab:
                print(f"Main: Preparing to display comparison for companies: {selected_companies}")
                with st.spinner('Analyzing selected companies...'):
                    display_company_comparison(selected_companies)
                print("Main: Comparison display completed")
        else:
            st.info("Please select companies from the sidebar to start comparison")

if __name__ == "__main__":
    main()
