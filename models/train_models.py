"""
API Version - Electricity Price Prediction Model Training
Train baseline models and machine learning models
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import pickle

from config.api_config import (
    FEATURES_DIR, MODELS_DIR,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Import ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Please run: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Please run: pip install lightgbm")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaselineModel:
    """Baseline model base class"""

    def __init__(self, name):
        self.name = name
        self.predictions = None

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }


class PersistenceModel(BaselineModel):
    """Persistence model: Predict the price from 24 hours ago"""

    def __init__(self):
        super().__init__("Persistence (24h lag)")

    def predict(self, X):
        """Use price_lag_24h as prediction"""
        return X['price_lag_24h'].values


class HistoricalAverageModel(BaselineModel):
    """Historical average model: Based on average price per hour"""

    def __init__(self):
        super().__init__("Historical Average (by hour)")
        self.hourly_avg = None

    def fit(self, X, y):
        """Calculate average price per hour"""
        df = pd.DataFrame({'hour': X['hour'], 'price': y})
        self.hourly_avg = df.groupby('hour')['price'].mean().to_dict()

    def predict(self, X):
        """Predict using hour-specific average values"""
        return X['hour'].map(self.hourly_avg).values


class RollingAverageModel(BaselineModel):
    """Rolling average model: Using 24-hour rolling average"""

    def __init__(self):
        super().__init__("Rolling Average (24h)")

    def predict(self, X):
        """Use 24-hour rolling average as prediction"""
        return X['price_rolling_mean_24h'].values


def prepare_train_test_split(df, test_days=7):
    """
    Split data into training and test sets

    Args:
        df: Feature DataFrame
        test_days: Number of days for testing (default: 7)

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Data split: Last {test_days} days for testing")

    # Data cleaning: Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    logger.info(f"Data cleaning complete")

    # Calculate split point
    split_date = df.index.max() - pd.Timedelta(days=test_days)

    # Split
    train_df = df[df.index <= split_date]
    test_df = df[df.index > split_date]

    logger.info(f"Training set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

    # Separate features and target
    target_col = 'price_eur_mwh'

    # Exclude non-feature columns
    feature_cols = [col for col in df.columns if col != target_col]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    logger.info(f"Number of features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train and evaluate baseline models"""
    logger.info("=" * 70)
    logger.info("Training baseline models")
    logger.info("=" * 70)

    results = {}

    # 1. Persistence model
    logger.info("\n1. Persistence model (24-hour lag)...")
    persistence = PersistenceModel()
    y_pred_persistence = persistence.predict(X_test)
    results['Persistence'] = persistence.evaluate(y_test.values, y_pred_persistence)
    logger.info(f"   MAE: {results['Persistence']['mae']:.2f}, R2: {results['Persistence']['r2']:.4f}")

    # 2. Historical average model
    logger.info("\n2. Historical average model (by hour)...")
    hist_avg = HistoricalAverageModel()
    hist_avg.fit(X_train, y_train)
    y_pred_hist = hist_avg.predict(X_test)
    results['Historical_Average'] = hist_avg.evaluate(y_test.values, y_pred_hist)
    logger.info(f"   MAE: {results['Historical_Average']['mae']:.2f}, R2: {results['Historical_Average']['r2']:.4f}")

    # 3. Rolling average model
    logger.info("\n3. Rolling average model (24 hours)...")
    rolling_avg = RollingAverageModel()
    y_pred_rolling = rolling_avg.predict(X_test)
    results['Rolling_Average'] = rolling_avg.evaluate(y_test.values, y_pred_rolling)
    logger.info(f"   MAE: {results['Rolling_Average']['mae']:.2f}, R2: {results['Rolling_Average']['r2']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("[OK] Baseline model training complete")
    logger.info("=" * 70)

    return results


def train_xgboost_model(X_train, X_test, y_train, y_test):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not installed, skipping")
        return None, None

    logger.info("\nTraining XGBoost model...")

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-8))) * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

    logger.info(f"[OK] XGBoost - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")

    return model, metrics


def train_lightgbm_model(X_train, X_test, y_train, y_test):
    """Train LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM not installed, skipping")
        return None, None

    logger.info("\nTraining LightGBM model...")

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-8))) * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

    logger.info(f"[OK] LightGBM - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")

    return model, metrics


def save_model(model, model_name, metrics):
    """
    Save model and metrics

    Args:
        model: Trained model
        model_name: Model name
        metrics: Dictionary of evaluation metrics
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_filename = f'{model_name.lower()}_{timestamp}.pkl'
    model_path = MODELS_DIR / model_filename

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Model saved: {model_path}")

    # Save metrics
    metrics_filename = f'{model_name.lower()}_metrics_{timestamp}.json'
    metrics_path = MODELS_DIR / metrics_filename

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics saved: {metrics_path}")

    # Also save a "latest" version
    latest_model_path = MODELS_DIR / f'{model_name.lower()}_latest.pkl'
    with open(latest_model_path, 'wb') as f:
        pickle.dump(model, f)

    latest_metrics_path = MODELS_DIR / f'{model_name.lower()}_metrics_latest.json'
    with open(latest_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return model_path, metrics_path


def load_latest_features():
    """
    Load the latest feature file

    Returns:
        DataFrame
    """
    # Find the latest feature file
    feature_files = list(FEATURES_DIR.glob('features_*.csv'))

    if not feature_files:
        logger.error("Feature file not found!")
        logger.error(f"Please run features/create_features.py first")
        return None

    # Get the latest file
    latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading feature file: {latest_file}")

    # Read data
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

    logger.info(f"Loaded {len(df)} records, {len(df.columns)} features")

    return df


def main():
    """Main function"""
    print("=" * 70)
    print("Electricity Price Prediction Model Training (API Version)")
    print("=" * 70)

    # Load features
    print("\nStep 1/4: Loading feature data...")
    df = load_latest_features()

    if df is None:
        print("\n[ERROR] Feature loading failed!")
        print("Please run first: python features/create_features.py")
        return

    print(f"[OK] Successfully loaded {len(df)} records, {len(df.columns)} features")

    # Prepare train/test split
    print("\nStep 2/4: Preparing train/test split...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, test_days=7)
    print(f"[OK] Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Train baseline models
    print("\nStep 3/4: Training baseline models...")
    print("-" * 70)
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test)

    # Train ML models
    print("\nStep 4/4: Training machine learning models...")
    print("-" * 70)

    all_results = baseline_results.copy()

    # XGBoost
    xgb_model, xgb_metrics = train_xgboost_model(X_train, X_test, y_train, y_test)
    if xgb_model is not None:
        all_results['XGBoost'] = xgb_metrics
        save_model(xgb_model, 'xgboost', xgb_metrics)

    # LightGBM
    lgb_model, lgb_metrics = train_lightgbm_model(X_train, X_test, y_train, y_test)
    if lgb_model is not None:
        all_results['LightGBM'] = lgb_metrics
        save_model(lgb_model, 'lightgbm', lgb_metrics)

    # Results summary
    print("\n" + "=" * 70)
    print("Model training complete!")
    print("=" * 70)

    # Create results table
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values('r2', ascending=False)

    print("\nModel performance comparison:")
    print(results_df.to_string())

    # Save results
    results_path = MODELS_DIR / 'model_comparison_latest.csv'
    results_df.to_csv(results_path)
    print(f"\nResults saved to: {results_path}")

    # Find best model
    best_model = results_df.index[0]
    best_r2 = results_df.loc[best_model, 'r2']
    best_mae = results_df.loc[best_model, 'mae']

    print("\n" + "=" * 70)
    print(f"Best model: {best_model}")
    print(f"R2: {best_r2:.4f}, MAE: {best_mae:.2f} EUR/MWh")
    print("=" * 70)


if __name__ == '__main__':
    main()
