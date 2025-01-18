import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_features(df):
    """Create technical features for the model with proper lag to avoid data leakage"""
    df = df.copy()
    
    # Fill any missing values in OHLCV data using ffill() instead of fillna(method='ffill')
    df = df.ffill()
    
    # Price-based features with proper lag
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Volume-based features
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_std'] = df['Volume'].rolling(window=20).std()
    
    # Price range features
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Low']
    df['CO_PCT'] = (df['Close'] - df['Open']) / df['Open']
    
    # Trend features
    df['Trend_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Trend_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # Add cyclical time features
    df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
    df['Month'] = pd.to_datetime(df.index).month
    
    # Shift all features by 1 to avoid look-ahead bias
    feature_columns = ['Returns', 'Volatility', 'SMA_20', 'SMA_50', 
                      'Volume_MA', 'Volume_std', 'HL_PCT', 'CO_PCT',
                      'Trend_20', 'Trend_50']
    
    for col in feature_columns:
        df[f'{col}_lag1'] = df[col].shift(1)
    
    # Handle any remaining NaN values
    df = df.fillna(0)
    
    return df

def calculate_kelly_criterion(returns):
    """Calculate Kelly Criterion with improved risk adjustment"""
    win_probability = len(returns[returns > 0]) / len(returns)
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1

    if avg_loss == 0:
        return 0

    kelly = win_probability - ((1 - win_probability) / (avg_win / avg_loss))
    return max(0, min(kelly * 0.5, 1))  # Use half-Kelly for safety

def run_monte_carlo_simulation(hist_returns, current_price, days=30, simulations=1000):
    """Run Monte Carlo simulation with improved volatility scaling"""
    mu = hist_returns.mean()
    sigma = hist_returns.std()
    
    returns = np.random.normal(mu, sigma, (simulations, days))
    
    recent_vol = hist_returns[-20:].std()
    historical_vol = hist_returns.std()
    vol_scalar = recent_vol / historical_vol
    returns = returns * vol_scalar
    
    price_paths = np.zeros((simulations, days))
    price_paths[:, 0] = current_price
    
    for t in range(1, days):
        price_paths[:, t] = price_paths[:, t-1] * (1 + returns[:, t])
    
    return price_paths

def train_model(df):
    """Train Random Forest model with improved feature selection"""
    feature_columns = [col for col in df.columns if '_lag1' in col]
    feature_columns.extend(['DayOfWeek', 'Month'])
    
    target = 'Returns'
    
    df = df.dropna()
    if len(df) < 1:
        raise ValueError("Insufficient data for training after cleaning")
    
    X = df[feature_columns]
    y = df[target]
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    
    return model, feature_columns

def make_predictions(model, df, feature_columns, days=30):
    """Make future predictions with improved error handling"""
    last_date = df.index[-1]
    dates = pd.date_range(start=last_date + timedelta(days=1), 
                         periods=days, 
                         freq='B')
    
    future_data = []
    current_data = df.copy()
    
    try:
        for _ in range(days):
            next_row = current_data.iloc[-1:].copy()
            next_row.index = pd.DatetimeIndex([dates[len(future_data)]])
            
            next_row = create_features(current_data.tail(50))
            next_row = next_row.iloc[-1:]
            
            if not all(col in next_row.columns for col in feature_columns):
                raise ValueError("Missing required features for prediction")
            
            X = next_row[feature_columns]
            pred_return = model.predict(X)[0]
            
            last_price = current_data['Close'].iloc[-1]
            pred_price = last_price * (1 + pred_return)
            
            new_row = pd.DataFrame({
                'Close': pred_price,
                'Open': last_price,
                'High': max(pred_price, last_price),
                'Low': min(pred_price, last_price),
                'Volume': current_data['Volume'].mean()
            }, index=pd.DatetimeIndex([dates[len(future_data)]])
            )
            
            future_data.append(new_row)
            current_data = pd.concat([current_data, new_row])
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return pd.DataFrame()
    
    future_df = pd.concat(future_data)
    return future_df

def plot_predictions(historical_data, predictions):
    """Plot historical data and predictions"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=historical_data.index,
                   y=historical_data['Close'],
                   mode='lines',
                   name='Historical',
                   line=dict(color='#ff8800')))

    fig.add_trace(
        go.Scatter(x=predictions.index,
                   y=predictions['Close'],
                   mode='lines',
                   name='Predicted',
                   line=dict(color='#00ff00', dash='dash')))

    fig.update_layout(template='plotly_dark',
                      title='Price Prediction (Random Forest)',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      height=600,
                      showlegend=True,
                      margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h",
                                 yanchor="bottom",
                                 y=1.02,
                                 xanchor="right",
                                 x=1))

    return fig

def plot_monte_carlo(price_paths, historical_data):
    """Plot Monte Carlo simulation results"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=historical_data.index,
                   y=historical_data['Close'],
                   mode='lines',
                   name='Historical',
                   line=dict(color='#ff8800')))

    dates = pd.date_range(start=historical_data.index[-1],
                         periods=price_paths.shape[1] + 1,
                         freq='B')[1:]

    percentiles = np.percentile(price_paths, [10, 50, 90], axis=0)

    fig.add_trace(
        go.Scatter(x=dates,
                   y=percentiles[1],
                   mode='lines',
                   name='Median Forecast',
                   line=dict(color='#00ff00', dash='dash')))

    fig.add_trace(
        go.Scatter(x=dates,
                   y=percentiles[2],
                   mode='lines',
                   name='90th Percentile',
                   line=dict(color='rgba(0, 255, 0, 0.3)')))

    fig.add_trace(
        go.Scatter(x=dates,
                   y=percentiles[0],
                   mode='lines',
                   name='10th Percentile',
                   line=dict(color='rgba(255, 0, 0, 0.3)')))

    fig.update_layout(template='plotly_dark',
                      title='Monte Carlo Price Simulation',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      height=600,
                      showlegend=True,
                      margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h",
                                 yanchor="bottom",
                                 y=1.02,
                                 xanchor="right",
                                 x=1))

    return fig

def display_feature_importance(model, feature_columns):
    """Display feature importance plot"""
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig = go.Figure(
        go.Bar(x=importance_df['Feature'],
               y=importance_df['Importance'],
               marker_color='#ff8800'))

    fig.update_layout(template='plotly_dark',
                      title='Feature Importance in Prediction Model',
                      xaxis_title='Features',
                      yaxis_title='Importance',
                      height=400,
                      margin=dict(l=0, r=0, t=40, b=0))

    st.plotly_chart(fig, use_container_width=True)

def display_risk_metrics(hist_returns, kelly_fraction):
    """Display risk metrics including Kelly Criterion"""
    st.subheader("Risk Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Kelly Fraction",
                  f"{kelly_fraction:.2%}",
                  help="Suggested position size based on Kelly Criterion")

    with col2:
        sharpe = np.sqrt(252) * (hist_returns.mean() / hist_returns.std())
        st.metric("Sharpe Ratio",
                  f"{sharpe:.2f}",
                  help="Risk-adjusted return (higher is better)")

    with col3:
        max_drawdown = (hist_returns + 1).cumprod().div(
            (hist_returns + 1).cumprod().cummax()).min() - 1
        st.metric("Max Drawdown",
                  f"{max_drawdown:.2%}",
                  help="Largest peak-to-trough decline")

    with col4:
        vol = hist_returns.std() * np.sqrt(252)
        st.metric("Annual Volatility",
                  f"{vol:.2%}",
                  help="Annual price volatility")

def display_ml_prediction(stock):
    """Display machine learning predictions tab content"""
    st.subheader("Price Prediction & Risk Analysis")

    # Get historical data with explicit period
    hist = stock.history(period='max')  # Get maximum available data

    # Create timezone-aware timestamp for filtering
    one_year_ago = (pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1))
    hist = hist.loc[one_year_ago:]

    if len(hist) < 100:  # Reduce minimum required days to 100
        st.error("Insufficient historical data for analysis. Please try a different stock.")
        return

    # Calculate daily returns
    hist_returns = hist['Close'].pct_change().dropna()

    # Calculate Kelly Criterion
    kelly_fraction = calculate_kelly_criterion(hist_returns)

    # Display risk metrics
    display_risk_metrics(hist_returns, kelly_fraction)

    # Create features and handle missing data
    hist_with_features = create_features(hist)
    hist_with_features = hist_with_features.dropna()

    # Add tabs for different analyses
    prediction_tab, monte_carlo_tab = st.tabs(
        ["ML Prediction", "Monte Carlo Simulation"])

    with prediction_tab:
        with st.spinner('Training model...'):
            try:
                # Train model and make predictions
                model, feature_columns = train_model(hist_with_features)
                predictions = make_predictions(model, hist_with_features,
                                           feature_columns)

                # Plot predictions
                fig = plot_predictions(hist, predictions)
                st.plotly_chart(fig, use_container_width=True)

                # Display feature importance
                display_feature_importance(model, feature_columns)
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                return

    with monte_carlo_tab:
        with st.spinner('Running Monte Carlo simulation...'):
            # Run Monte Carlo simulation
            price_paths = run_monte_carlo_simulation(hist_returns,
                                                  hist['Close'].iloc[-1])

            # Plot Monte Carlo results
            fig_mc = plot_monte_carlo(price_paths, hist)
            st.plotly_chart(fig_mc, use_container_width=True)

            # Calculate and display probabilistic metrics
            final_prices = price_paths[:, -1]
            current_price = hist['Close'].iloc[-1]

            col1, col2, col3 = st.columns(3)

            with col1:
                prob_up = (final_prices > current_price).mean()
                st.metric("Probability of Price Increase", f"{prob_up:.1%}")

            with col2:
                expected_return = (final_prices.mean() / current_price - 1)
                st.metric("Expected Return", f"{expected_return:.1%}")

            with col3:
                var_95 = np.percentile((final_prices / current_price - 1), 5)
                st.metric("Value at Risk (95%)",
                         f"{var_95:.1%}",
                         help="Potential loss at 95% confidence level")

    # Add disclaimer
    st.info(
        "⚠️ These predictions and risk metrics are for educational purposes only. "
    )
