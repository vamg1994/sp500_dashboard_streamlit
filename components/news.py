from newsapi import NewsApiClient
import streamlit as st
from textblob import TextBlob
import os
from datetime import datetime, timedelta
import pandas as pd

def get_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    blob = TextBlob(text)
    # Get polarity score (-1 to 1)
    sentiment = blob.sentiment.polarity
    
    # Convert score to label
    if sentiment > 0.1:
        return 'Positive', sentiment
    elif sentiment < -0.1:
        return 'Negative', sentiment
    else:
        return 'Neutral', sentiment

def get_news_api_key():
    """Get News API key from Streamlit secrets"""
    try:
        return st.secrets["NEWS_API_KEY"]
    except KeyError:
        st.error("NEWS_API_KEY not found in secrets. Please configure it in your Streamlit deployment.")
        return None

def display_news_sentiment(symbol, company_name):
    """Display news and sentiment analysis for selected company"""
    st.subheader("Recent News & Sentiment Analysis")
    
    try:
        # Get API key from secrets
        NEWS_API_KEY = get_news_api_key()
        if not NEWS_API_KEY:
            return
            
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Get news from the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Search for news about the company
        news = newsapi.get_everything(
            q=f'"{company_name}" OR "{symbol}"',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt'
        )
        
        if not news['articles']:
            st.info(f"No recent news found for {company_name}")
            return
        
        # Create DataFrame for news and sentiment
        news_data = []
        for article in news['articles']:
            sentiment_label, sentiment_score = get_sentiment(article['title'] + ' ' + (article['description'] or ''))
            news_data.append({
                'Date': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d'),
                'Title': article['title'],
                'Sentiment': sentiment_label,
                'Score': sentiment_score,
                'URL': article['url']
            })
        
        df = pd.DataFrame(news_data)
        
        # Display sentiment summary
        col1, col2, col3 = st.columns(3)
        with col1:
            positive = len(df[df['Sentiment'] == 'Positive'])
            st.metric("Positive News", positive)
        with col2:
            neutral = len(df[df['Sentiment'] == 'Neutral'])
            st.metric("Neutral News", neutral)
        with col3:
            negative = len(df[df['Sentiment'] == 'Negative'])
            st.metric("Negative News", negative)
        
        # Calculate average sentiment
        avg_sentiment = df['Score'].mean()
        st.progress(
            (avg_sentiment + 1) / 2,  # Convert -1:1 to 0:1 range
            text=f"Overall Sentiment: {avg_sentiment:.2f} (-1 to 1 scale)"
        )
        
        # Display news table with sentiment
        st.markdown("### Recent News Articles")
        for _, row in df.iterrows():
            sentiment_color = {
                'Positive': 'green',
                'Neutral': 'gray',
                'Negative': 'red'
            }[row['Sentiment']]
            
            st.markdown(
                f"""
                <div style="border-left: 5px solid {sentiment_color}; padding-left: 10px; margin: 10px 0;">
                    <p style="margin: 0; color: gray;">{row['Date']}</p>
                    <a href="{row['URL']}" target="_blank" style="text-decoration: none;">
                        {row['Title']}
                    </a>
                    <p style="margin: 0; color: {sentiment_color};">
                        Sentiment: {row['Sentiment']} ({row['Score']:.2f})
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
