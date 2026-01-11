"""
API Version - Electricity Price Prediction Feature Engineering
Create features from data obtained from API
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config.api_config import (
    DATA_DIR, FEATURES_DIR, TIMEZONE,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)


class ElectricityPriceFeatureEngineer:
    """Electricity price prediction feature engineering class"""

    def __init__(self, price_df):
        """
        Initialize the feature engineer

        Args:
            price_df: DataFrame with timestamp index and 'price_eur_mwh' column
        """
        self.df = price_df.copy()
        logger.info(f"Initialization complete, {len(self.df)} price records in total")

    def create_lagged_features(self, lags=[1, 2, 3, 6, 12, 24, 48, 72]):
        """
        Create lagged features

        Args:
            lags: List of lag periods (hours)
        """
        logger.info(f"Creating {len(lags)} lagged features...")

        for lag in lags:
            col_name = f'price_lag_{lag}h'
            self.df[col_name] = self.df['price_eur_mwh'].shift(lag)

        logger.info(f"[OK] Created {len(lags)} lagged features")

    def create_rolling_features(self, windows=[6, 12, 24, 48, 72, 168]):
        """
        Create rolling statistical features

        Args:
            windows: List of window sizes (hours)
                    168 hours = 1 week
        """
        logger.info(f"Creating rolling statistical features for {len(windows)} windows...")

        for window in windows:
            # Rolling mean
            self.df[f'price_rolling_mean_{window}h'] = (
                self.df['price_eur_mwh'].shift(1).rolling(window=window, min_periods=1).mean()
            )

            # Rolling standard deviation (volatility)
            self.df[f'price_rolling_std_{window}h'] = (
                self.df['price_eur_mwh'].shift(1).rolling(window=window, min_periods=1).std()
            )

            # Rolling minimum
            self.df[f'price_rolling_min_{window}h'] = (
                self.df['price_eur_mwh'].shift(1).rolling(window=window, min_periods=1).min()
            )

            # Rolling maximum
            self.df[f'price_rolling_max_{window}h'] = (
                self.df['price_eur_mwh'].shift(1).rolling(window=window, min_periods=1).max()
            )

        logger.info(f"[OK] Created rolling statistical features (mean, std, min, max)")

    def create_calendar_features(self):
        """Create calendar features"""
        logger.info("Creating calendar features...")

        # Hour (0-23)
        self.df['hour'] = self.df.index.hour

        # Day of week (0=Monday, 6=Sunday)
        self.df['day_of_week'] = self.df.index.dayofweek

        # Is weekend
        self.df['is_weekend'] = (self.df.index.dayofweek >= 5).astype(int)

        # Is weekday
        self.df['is_weekday'] = (self.df.index.dayofweek < 5).astype(int)

        # Day of month
        self.df['day_of_month'] = self.df.index.day

        # Month
        self.df['month'] = self.df.index.month

        # Quarter
        self.df['quarter'] = self.df.index.quarter

        # Week of year
        self.df['week_of_year'] = self.df.index.isocalendar().week.astype(int)

        # Cyclical encoding (using sin and cos)
        # Hour cyclical encoding
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)

        # Day of week cyclical encoding
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

        # Month cyclical encoding
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        logger.info("[OK] Created calendar features")

    def create_interaction_features(self):
        """Create interaction features"""
        logger.info("Creating interaction features...")

        # Price and hour interaction
        if 'price_lag_24h' in self.df.columns and 'hour' in self.df.columns:
            self.df['price_hour_interaction'] = self.df['price_lag_24h'] * self.df['hour']

        # Weekend and hour interaction
        if 'is_weekend' in self.df.columns and 'hour' in self.df.columns:
            self.df['weekend_hour_interaction'] = self.df['is_weekend'] * self.df['hour']

        logger.info("[OK] Created interaction features")

    def create_price_change_features(self):
        """Create price change features"""
        logger.info("Creating price change features...")

        # Price difference
        for lag in [1, 2, 24]:
            self.df[f'price_diff_{lag}h'] = self.df['price_eur_mwh'].diff(lag)

        # Price percentage change
        for lag in [1, 24]:
            self.df[f'price_pct_change_{lag}h'] = self.df['price_eur_mwh'].pct_change(lag)

        logger.info("[OK] Created price change features")

    def create_statistical_features(self):
        """Create statistical features"""
        logger.info("Creating statistical features...")

        # Average price per hour
        hourly_avg = self.df.groupby('hour')['price_eur_mwh'].transform('mean')
        self.df['price_vs_hour_avg'] = self.df['price_eur_mwh'] / (hourly_avg + 1e-8)

        # Average price per day of week
        dow_avg = self.df.groupby('day_of_week')['price_eur_mwh'].transform('mean')
        self.df['price_vs_dow_avg'] = self.df['price_eur_mwh'] / (dow_avg + 1e-8)

        # Price deviation from 7-day mean
        if 'price_rolling_mean_168h' in self.df.columns:
            self.df['price_deviation_from_7d_mean'] = (
                self.df['price_eur_mwh'] - self.df['price_rolling_mean_168h']
            ) / (self.df['price_rolling_mean_168h'] + 1e-8)

        logger.info("[OK] Created statistical features")

    def create_all_features(self):
        """Create all features"""
        logger.info("=" * 60)
        logger.info("Starting to create all features...")
        logger.info("=" * 60)

        self.create_lagged_features()
        self.create_rolling_features()
        self.create_calendar_features()
        self.create_price_change_features()
        self.create_statistical_features()
        self.create_interaction_features()

        logger.info("=" * 60)
        logger.info("Feature creation complete!")
        logger.info("=" * 60)

        # Count the number of features
        feature_cols = [col for col in self.df.columns if col != 'price_eur_mwh']
        logger.info(f"Total features: {len(feature_cols)}")

        return self.df

    def get_feature_dataframe(self):
        """
        Get the feature DataFrame

        Returns:
            Feature DataFrame
        """
        return self.df

    def save_features(self, filename=None):
        """
        Save features to CSV file

        Args:
            filename: File name (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'features_{timestamp}.csv'

        filepath = FEATURES_DIR / filename

        # Remove rows containing NaN
        df_clean = self.df.dropna()

        df_clean.to_csv(filepath)
        logger.info(f"Features saved to: {filepath}")
        logger.info(f"Saved {len(df_clean)} records (removed {len(self.df) - len(df_clean)} records containing NaN)")

        return filepath


def load_latest_data():
    """
    Load the latest API data

    Returns:
        DataFrame
    """
    # Find the latest data file
    data_files = list(DATA_DIR.glob('entsoe_*.csv'))

    if not data_files:
        logger.error("API data file not found!")
        logger.error(f"Please run fetch_data.py first to obtain data")
        return None

    # Get the latest file
    latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading data file: {latest_file}")

    # Read data
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Time range: {df.index.min()} to {df.index.max()}")

    return df


def main():
    """Main function"""
    print("=" * 60)
    print("Electricity Price Prediction Feature Engineering (API Version)")
    print("=" * 60)

    # Load data
    print("\nStep 1/3: Loading API data...")
    df = load_latest_data()

    if df is None:
        print("\n[ERROR] Data loading failed!")
        print("Please run first: python scripts/fetch_data.py")
        return

    print(f"[OK] Successfully loaded {len(df)} records")

    # Create features
    print("\nStep 2/3: Creating features...")
    print("-" * 60)

    engineer = ElectricityPriceFeatureEngineer(df)
    features_df = engineer.create_all_features()

    print("-" * 60)
    print(f"[OK] Created {len(features_df.columns) - 1} features")

    # Save features
    print("\nStep 3/3: Saving features...")
    filepath = engineer.save_features('features_latest.csv')

    # Display summary
    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)
    print(f"Input data: {len(df)} records")
    print(f"Output features: {len(features_df.columns) - 1}")
    print(f"Valid records: {len(features_df.dropna())}")
    print(f"Saved to: {filepath}")
    print("=" * 60)

    # Display feature examples
    print("\nFeature examples (first 5 rows):")
    print(features_df.dropna().head())


if __name__ == '__main__':
    main()
