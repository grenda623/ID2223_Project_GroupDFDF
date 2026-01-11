"""
Electricity Price Prediction Streamlit Application (API Version)
Real-time data fetching from ENTSO-E API
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.api_config import (
    MODELS_DIR, FEATURES_DIR, DATA_DIR, DEFAULT_ZONE
)

# Page configuration
st.set_page_config(
    page_title="Electricity Price Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_name="xgboost"):
    """Load trained model"""
    model_path = MODELS_DIR / f"{model_name}_latest.pkl"
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.info("Please run: python models/train_models.py")
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_features():
    """Load feature data"""
    features_path = FEATURES_DIR / "features_latest.csv"
    if not features_path.exists():
        st.error(f"Features not found: {features_path}")
        st.info("Please run: python features/create_features.py")
        return None

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)

    # Filter to only use historical data (up to now)
    now = pd.Timestamp.now(tz='UTC')
    df = df[df.index <= now]

    return df


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_latest_data():
    """Load latest API data"""
    data_files = list(DATA_DIR.glob('entsoe_*.csv'))
    if not data_files:
        st.error("No API data files found!")
        st.info("Please run: python scripts/fetch_data.py")
        return None

    latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

    # Filter to only use historical data (up to now)
    now = pd.Timestamp.now(tz='UTC')
    df = df[df.index <= now]

    return df


def create_future_features(last_row, hours_ahead=24):
    """
    Create features for future prediction

    Args:
        last_row: Last row of historical data
        hours_ahead: Number of hours to predict

    Returns:
        DataFrame with future features
    """
    future_data = []

    for h in range(1, hours_ahead + 1):
        future_time = last_row.name + timedelta(hours=h)

        # Create basic calendar features
        hour = future_time.hour
        day_of_week = future_time.dayofweek
        day_of_month = future_time.day
        month = future_time.month
        week_of_year = future_time.isocalendar()[1]
        quarter = (month - 1) // 3 + 1

        # Cyclical encodings
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        is_weekend = 1 if day_of_week >= 5 else 0
        is_weekday = 1 - is_weekend

        # Use last known values for lag features (simplified)
        row = {
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_weekday': is_weekday,
            'day_of_month': day_of_month,
            'month': month,
            'quarter': quarter,
            'week_of_year': week_of_year,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_of_week_sin': day_sin,
            'day_of_week_cos': day_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
        }

        # Use historical lag features
        for col in last_row.index:
            if col not in row:
                row[col] = last_row[col]

        future_data.append(row)

    future_df = pd.DataFrame(future_data)
    future_df.index = [last_row.name + timedelta(hours=h) for h in range(1, hours_ahead + 1)]

    return future_df


def predict_future_prices(model, features_df, hours_ahead=24):
    """
    Predict future electricity prices

    Args:
        model: Trained model
        features_df: Feature DataFrame
        hours_ahead: Number of hours to predict

    Returns:
        DataFrame with predictions
    """
    last_row = features_df.iloc[-1]
    future_features = create_future_features(last_row, hours_ahead)

    # Ensure feature order consistency
    feature_cols = [col for col in features_df.columns if col != 'price_eur_mwh']
    future_features = future_features[feature_cols]

    # Predict
    predictions = model.predict(future_features)

    # Create result DataFrame
    pred_df = pd.DataFrame({
        'predicted_price': predictions
    }, index=future_features.index)

    return pred_df


def plot_forecast(historical_df, forecast_df):
    """Create interactive forecast plot"""
    # Get last 7 days of historical data
    cutoff_time = historical_df.index[-1] - pd.Timedelta(days=7)
    last_week = historical_df.loc[historical_df.index >= cutoff_time]

    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=last_week.index,
        y=last_week['price_eur_mwh'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2)
    ))

    # Predicted prices
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['predicted_price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='#ff7f0e', width=2)
    ))

    # Add forecast start line
    forecast_start_str = str(forecast_df.index[0])
    fig.add_shape(
        type="line",
        x0=forecast_start_str,
        x1=forecast_start_str,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )

    fig.add_annotation(
        x=forecast_start_str,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=10
    )

    fig.update_layout(
        title="Electricity Price Forecast (Last 7 Days History + 24h Prediction)",
        xaxis_title="Time",
        yaxis_title="Price (EUR/MWh)",
        hovermode='x unified',
        height=500
    )

    return fig


def plot_daily_pattern(forecast_df):
    """Plot daily price pattern"""
    fig = go.Figure()

    # Group by hour
    hourly_avg = forecast_df.groupby(forecast_df.index.hour)['predicted_price'].mean()

    fig.add_trace(go.Bar(
        x=hourly_avg.index,
        y=hourly_avg.values,
        marker=dict(
            color=hourly_avg.values,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Price<br>(EUR/MWh)")
        )
    ))

    fig.update_layout(
        title="Predicted Daily Price Pattern",
        xaxis_title="Hour",
        yaxis_title="Average Price (EUR/MWh)",
        xaxis=dict(tickmode='linear', dtick=1),
        height=400
    )

    return fig


def get_recommendations(forecast_df):
    """Generate electricity usage recommendations"""
    prices = forecast_df['predicted_price']

    # Find highest and lowest price periods
    lowest_idx = prices.idxmin()
    highest_idx = prices.idxmax()

    lowest_price = prices.min()
    highest_price = prices.max()
    avg_price = prices.mean()

    recommendations = []

    # Best usage time
    recommendations.append({
        'type': 'Best Usage Time',
        'time': lowest_idx.strftime('%Y-%m-%d %H:%M'),
        'price': f'{lowest_price:.2f} EUR/MWh',
        'description': 'Lowest price - ideal for high-energy appliances (washing machine, dryer, EV charging)'
    })

    # Avoid usage time
    recommendations.append({
        'type': 'Avoid Usage Time',
        'time': highest_idx.strftime('%Y-%m-%d %H:%M'),
        'price': f'{highest_price:.2f} EUR/MWh',
        'description': 'Highest price - avoid using high-energy appliances'
    })

    # Low price periods (below average)
    low_price_hours = forecast_df[forecast_df['predicted_price'] < avg_price]
    if len(low_price_hours) > 0:
        recommendations.append({
            'type': 'Low Price Periods',
            'time': f'{len(low_price_hours)} hours',
            'price': f'< {avg_price:.2f} EUR/MWh',
            'description': f'{len(low_price_hours)} hours with prices below average'
        })

    # Price volatility
    price_range = highest_price - lowest_price
    volatility = (price_range / avg_price) * 100

    recommendations.append({
        'type': 'Price Volatility',
        'time': f'{volatility:.1f}%',
        'price': f'{price_range:.2f} EUR/MWh',
        'description': f'Max-min difference: {price_range:.2f} EUR/MWh - optimize timing to save costs'
    })

    return recommendations


def main():
    """Main function"""
    # Title
    st.title("Electricity Price Predictor")
    st.markdown(f"**Data Source**: ENTSO-E Transparency Platform | **Region**: {DEFAULT_ZONE}")

    # Sidebar
    st.sidebar.header("Settings")

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["xgboost", "lightgbm"],
        index=0
    )

    # Prediction horizon
    hours_ahead = st.sidebar.slider(
        "Prediction Horizon (hours)",
        min_value=6,
        max_value=72,
        value=24,
        step=6
    )

    # Refresh data button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data and model
    with st.spinner("Loading data and model..."):
        model = load_model(model_choice)
        features_df = load_features()
        raw_data = load_latest_data()

    if model is None or features_df is None or raw_data is None:
        st.stop()

    # Display data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Info")
    st.sidebar.markdown(f"**Records**: {len(features_df)}")
    st.sidebar.markdown(f"**Time Range**: {features_df.index.min().date()} to {features_df.index.max().date()}")
    st.sidebar.markdown(f"**Last Update**: {features_df.index.max().strftime('%Y-%m-%d %H:%M')}")

    # Make predictions
    with st.spinner(f"Predicting prices for next {hours_ahead} hours..."):
        forecast_df = predict_future_prices(model, features_df, hours_ahead)

    # Main content
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Price",
            f"{features_df['price_eur_mwh'].iloc[-1]:.2f} EUR/MWh",
            delta=None
        )

    with col2:
        avg_forecast = forecast_df['predicted_price'].mean()
        st.metric(
            "Avg Predicted Price",
            f"{avg_forecast:.2f} EUR/MWh",
            delta=f"{avg_forecast - features_df['price_eur_mwh'].iloc[-1]:.2f}"
        )

    with col3:
        min_forecast = forecast_df['predicted_price'].min()
        st.metric(
            "Min Predicted Price",
            f"{min_forecast:.2f} EUR/MWh",
            delta=None
        )

    with col4:
        max_forecast = forecast_df['predicted_price'].max()
        st.metric(
            "Max Predicted Price",
            f"{max_forecast:.2f} EUR/MWh",
            delta=None
        )

    st.markdown("---")

    # Forecast chart
    st.subheader("Price Forecast")
    fig_forecast = plot_forecast(features_df, forecast_df)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Daily pattern
    if hours_ahead >= 24:
        st.subheader("Daily Price Pattern")
        fig_pattern = plot_daily_pattern(forecast_df)
        st.plotly_chart(fig_pattern, use_container_width=True)

    # Usage recommendations
    st.subheader("Usage Recommendations")
    recommendations = get_recommendations(forecast_df)

    for rec in recommendations:
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 5])
            with col1:
                st.markdown(f"**{rec['type']}**")
            with col2:
                st.markdown(f"`{rec['time']}`")
                st.markdown(f"*{rec['price']}*")
            with col3:
                st.markdown(rec['description'])
            st.markdown("---")

    # Detailed forecast data
    with st.expander("View Detailed Forecast Data"):
        forecast_display = forecast_df.copy()
        forecast_display['Time'] = forecast_display.index.strftime('%Y-%m-%d %H:%M')
        forecast_display['Predicted Price (EUR/MWh)'] = forecast_display['predicted_price'].round(2)
        st.dataframe(
            forecast_display[['Time', 'Predicted Price (EUR/MWh)']].reset_index(drop=True),
            use_container_width=True
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Data Source: ENTSO-E Transparency Platform | For reference only, not investment advice</p>
        <p>Last Updated: {}</p>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
