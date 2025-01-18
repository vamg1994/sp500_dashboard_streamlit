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
        st.image("vam_pp3.png", width=150, use_container_width=False)
        st.markdown("""
        ### Virgilio Madrid
        Data Scientist
        """)
        st.link_button("Portfolio 🌐", "https://portfolio-vam.vercel.app/")
        st.link_button("LinkedIn 💼", "https://www.linkedin.com/in/vamadrid/")
        st.link_button("E-Mail 📧", "mailto:virgiliomadrid1994@gmail.com")
        
        
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
        overview_tab, charts_tab, financials_tab, prediction_tab, news_tab, faq_tab = st.tabs([
            "Company Overview", 
            "Price Charts", 
            "Financial Metrics",
            "Price Prediction",
            "News & Sentiment",
            "FAQ"
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

        with faq_tab:
            st.header("Frequently Asked Questions")
            
            st.subheader("About This Dashboard")
            st.markdown("""
            This S&P 500 Stock Dashboard is a comprehensive tool for analyzing stocks from the S&P 500 index. 
            It provides real-time data analysis, technical charts, financial metrics, price predictions, and news sentiment analysis.
            """)
            
            # Explain each tab
            st.subheader("Tab Explanations")
            
            with st.expander("Company Overview"):
                st.markdown("""
                - Displays key company information and business summary
                - Shows sector, industry, and market cap details
                - Lists major shareholders and institutional holders
                - Tools used: yfinance API
                """)
                
            with st.expander("Price Charts"):
                st.markdown("""
                - Interactive stock price charts with multiple timeframes
                - Includes volume data and price indicators
                - Customizable chart periods (1D to 5Y)
                - Tools used: yfinance API, Plotly for interactive charts
                """)
                
            with st.expander("Financial Metrics"):
                st.markdown("""
                - Key financial ratios and metrics
                - Balance sheet highlights
                - Income statement analysis
                - Cash flow indicators
                - Tools used: yfinance API for financial data
                """)
                
            with st.expander("Price Prediction"):
                st.markdown("""
                - Machine learning-based price predictions
                - Historical price trend analysis
                - Prediction confidence intervals
                - Tools used: scikit-learn for machine learning predictions
                """)
                
            with st.expander("News & Sentiment"):
                st.markdown("""
                - Latest news articles about the company
                - Sentiment analysis of news headlines
                - News impact visualization
                - Tools used: NewsAPI for news gathering, NLTK for sentiment analysis
                """)
            
            # Tech Stack
            st.subheader("Technology Stack")
            st.markdown("""
            - **Frontend**: Streamlit (Python web framework)
            - **Data Sources**: 
                - Yahoo Finance API (stock data)
                - News API (company news)
            - **Analysis Tools**:
                - Pandas (data manipulation)
                - NumPy (numerical computations)
                - scikit-learn (machine learning)
                - NLTK (natural language processing)
            - **Visualization**:
                - Plotly (interactive charts)
                - Streamlit native charts
            """)
            
            # Additional Info
            st.subheader("Additional Information")
            st.markdown("""
            - Data is updated in real-time during market hours
            - Historical data may have up to 15 minutes delay
            - Predictions are for educational purposes only
            - News sentiment analysis uses basic NLP techniques
            """)
            
            # Disclaimer
            st.warning("""
            **Disclaimer**: This dashboard is for informational purposes only. 
            It is not intended to provide financial advice. Always conduct your own research 
            and consult with financial professionals before making investment decisions.
            """)

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
            performance_tab, metrics_tab = st.tabs([
                "Performance Comparison",
                "About"
            ])
            
            with performance_tab:
                print(f"Main: Preparing to display comparison for companies: {selected_companies}")
                with st.spinner('Analyzing selected companies...'):
                    display_company_comparison(selected_companies)
                print("Main: Comparison display completed")
                
            with metrics_tab:
                # Financial metrics
                st.write("This comparison shows the performance of the selected companies. You can select up to 4 companies.")
        else:
            st.info("Please select companies from the sidebar to start comparison")

if __name__ == "__main__":
    main()
