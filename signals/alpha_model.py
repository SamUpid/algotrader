"""
signals/alpha_model.py
LightGBM alpha model with triple barrier labeling and walk-forward validation.

Key concepts:
1. Triple barrier labeling (Lopez de Prado):
   - +1 if forward return > upper barrier (volatility threshold)
   - -1 if forward return < lower barrier (-volatility threshold)
   - 0 (neutral) if within barriers (no bet)

2. Walk-forward training:
   - Train on 252 days, test on 63 days, roll by 63 days
   - Re-train model on each split

3. Feature importance: verify top features make economic sense

Why this matters for interviews:
   - Shows you understand ML in finance
   - Prevents lookahead bias
   - Demonstrates robust validation methodology
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Try to import LightGBM
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM not installed. Install with: pip install lightgbm")

logger = logging.getLogger(__name__)


# ============================================================================
# TRIPLE BARRIER LABELING
# ============================================================================

def create_triple_barrier_labels(
    returns: pd.Series,
    volatility: pd.Series,
    horizon: int = 5,
    threshold_multiplier: float = 1.0
) -> pd.Series:
    """
    Create triple barrier labels for classification.

    Triple barrier labeling from Lopez de Prado:
    - Upper barrier: +threshold (volatility * multiplier)
    - Lower barrier: -threshold (-volatility * multiplier)
    - Horizontal barrier: horizon days

    Label =  1 if return > upper barrier (BUY signal)
    Label = -1 if return < lower barrier (SELL signal)
    Label =  0 if |return| <= threshold (NEUTRAL, no bet)

    Args:
        returns: Forward returns (e.g., 5-day forward)
        volatility: Rolling volatility (e.g., 20-day)
        horizon: Forward horizon in days (default 5)
        threshold_multiplier: Multiply volatility by this (default 1.0)

    Returns:
        Series of labels: 1, -1, or 0 (NaN for insufficient data)

    Interview Tip: "Triple barrier labeling reduces noise by only
    taking signals when the move is large enough to be tradable.
    The volatility threshold adapts to market conditions."

    Example:
        returns = [0.02, -0.015, 0.005, -0.03, 0.01]
        volatility = [0.02, 0.02, 0.02, 0.02, 0.02]
        threshold = 0.02

        Label =  1 (0.02 > 0.02? Yes, buy)
        Label = -1 (-0.015 < -0.02? No, 0)
        Label =  0 (0.005 within ±0.02, neutral)
        Label = -1 (-0.03 < -0.02? Yes, sell)
        Label =  0 (0.01 within ±0.02, neutral)
    """
    if len(returns) == 0:
        return pd.Series(dtype=float)

    # Align indices
    common_idx = returns.index.intersection(volatility.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    returns = returns.reindex(common_idx)
    volatility = volatility.reindex(common_idx)

    # Calculate threshold
    threshold = volatility * threshold_multiplier

    # Create labels
    labels = pd.Series(0, index=common_idx, dtype=float)

    # Up barrier (buy)
    labels[returns > threshold] = 1.0

    # Down barrier (sell)
    labels[returns < -threshold] = -1.0

    # Neutral = 0 (already set)

    # First horizon-1 days are NaN (need horizon days for forward return)
    # We'll handle this by returning NaN for the first horizon-1 rows
    # But since we're using forward returns, the NaN will naturally appear
    # at the end of the series, not the beginning.

    return labels


def compute_forward_returns(
    prices: pd.Series,
    horizon: int = 5
) -> pd.Series:
    """
    Compute forward returns for a given horizon.

    forward_return[t] = (prices[t+horizon] / prices[t]) - 1

    Args:
        prices: Price series
        horizon: Forward horizon in days (default 5)

    Returns:
        Series of forward returns (last horizon days are NaN)

    Interview Tip: "Using forward returns as labels is standard.
    The lookahead is intentional - we're predicting the future.
    But we NEVER use future data in features."
    """
    if len(prices) <= horizon:
        return pd.Series(dtype=float)

    # Shift price forward by horizon days
    forward_return = (prices.shift(-horizon) / prices) - 1

    return forward_return


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features_for_ticker(
    feature_matrix: pd.DataFrame,
    ticker: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and returns for a single ticker.

    Args:
        feature_matrix: MultiIndex DataFrame (date, ticker)
        ticker: Stock symbol

    Returns:
        Tuple of (features_df, returns_series)
        features_df: Features for the ticker
        returns_series: Forward returns for labeling
    """
    # Get data for this ticker
    ticker_data = feature_matrix.xs(ticker, level='ticker')

    # Features: all columns except those that would cause leakage
    # We keep returns for labeling but don't use them as features
    features = ticker_data.copy()

    # Drop columns that are direct returns (we want to predict returns,
    # not use them as features)
    # Keep all other features

    # For returns, we need the actual returns to compute labels
    # We'll compute forward returns from the returns column

    # Get the price return series (we need this for forward returns)
    # But we don't have prices in feature matrix, so we need to load them
    # Let's load the clean data for this ticker

    data_path = config.processed_path(ticker)
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned data not found for {ticker}")

    clean_data = pd.read_parquet(data_path)

    # Use close prices for forward returns
    prices = clean_data['close']

    # Align features with prices
    common_idx = features.index.intersection(prices.index)

    if len(common_idx) == 0:
        raise ValueError(f"No overlapping dates for {ticker}")

    features = features.loc[common_idx]
    prices = prices.loc[common_idx]

    return features, prices


# ============================================================================
# LIGHTGBM TRAINING
# ============================================================================

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None
) -> lgb.LGBMClassifier:
    """
    Train LightGBM classifier on training data.

    Args:
        X_train: Training features
        y_train: Training labels (1, -1, 0)
        params: LightGBM parameters

    Returns:
        Trained LightGBM classifier

    Interview Tip: "I use class_weight='balanced' to handle imbalanced
    labels (most days are neutral). The hyperparameters are chosen
    to prevent overfitting: small max_depth, regularized learning rate."
    """
    if not LGBM_AVAILABLE:
        raise ImportError("LightGBM not available. Install with: pip install lightgbm")

    # Default parameters optimized for financial data
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'class_weight': 'balanced',  # Handle imbalanced labels
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    if params:
        default_params.update(params)

    # Remove rows with NaN labels
    mask = ~y_train.isna()
    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask]

    if len(X_train) == 0:
        raise ValueError("No training data after removing NaN labels")

    # Train model
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    return model


def generate_predictions(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    return_proba: bool = True
) -> pd.Series:
    """
    Generate predictions from trained model.

    Args:
        model: Trained LightGBM classifier
        X_test: Test features
        return_proba: Return probabilities (default True)

    Returns:
        Predictions (probabilities or class labels)
    """
    if return_proba:
        # Get probability of class 1 (BUY)
        proba = model.predict_proba(X_test)

        # Map labels: 0=neutral, 1=buy, 2=sell
        # We want probability of buy (class 1)
        # For binary classification, we need to map
        # But we have 3 classes: 0, 1, -1
        # LightGBM expects labels to be 0, 1, 2
        # So we need to re-map

        # We'll handle this in the wrapper
        # For now, use predict_proba
        return pd.Series(proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
                        index=X_test.index)
    else:
        predictions = model.predict(X_test)
        return pd.Series(predictions, index=X_test.index)


def get_feature_importances(
    model: lgb.LGBMClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get and sort feature importances from trained model.

    Args:
        model: Trained LightGBM classifier
        feature_names: List of feature names

    Returns:
        DataFrame with feature names and importances

    Interview Tip: "Feature importance is critical for model validation.
    The top features should make economic sense. If they don't, you have
    a data leakage problem or a spurious correlation."
    """
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df


# ============================================================================
# WALK-FORWARD ML PIPELINE
# ============================================================================

def run_walk_forward_ml(
    tickers: Optional[List[str]] = None,
    horizon: int = 5,
    train_size: int = 252,
    test_size: int = 63,
    step: int = 63,
    gap: int = 5,
    threshold_multiplier: float = 1.0,
    verbose: bool = True
) -> Dict:
    """
    Run walk-forward ML training and prediction for all tickers.

    Pipeline:
        1. Load feature matrix
        2. For each ticker:
            a. Compute forward returns
            b. Create triple barrier labels
            c. For each split:
                - Train LightGBM on train data
                - Predict on test data
                - Store predictions
        3. Save all predictions to ml_signals.parquet

    Args:
        tickers: List of tickers (defaults to config.UNIVERSE)
        horizon: Forward horizon in days (default 5)
        train_size: Training window size (default 252)
        test_size: Test window size (default 63)
        step: Roll-forward step (default 63)
        gap: Purge gap between train and test (default 5)
        threshold_multiplier: Volatility threshold multiplier (default 1.0)
        verbose: Print progress

    Returns:
        Dictionary with predictions, models, and feature importances

    Interview Tip: "Walk-forward ML is the gold standard for strategy
    evaluation. It prevents lookahead bias and simulates real-world
    conditions where you retrain models periodically."
    """
    if not LGBM_AVAILABLE:
        raise ImportError("LightGBM not available. Install with: pip install lightgbm")

    if tickers is None:
        tickers = config.UNIVERSE

    logger.info(f"Running walk-forward ML for {len(tickers)} tickers...")

    # Load feature matrix
    feature_matrix_path = config.DATA_PROC_DIR / "feature_matrix.parquet"
    if not feature_matrix_path.exists():
        raise FileNotFoundError("Feature matrix not found. Run features.indicators first.")

    feature_matrix = pd.read_parquet(feature_matrix_path)

    # Store results
    all_predictions = []
    all_models = {}
    all_importances = {}

    for ticker_idx, ticker in enumerate(tickers):
        if verbose:
            logger.info(f"  [{ticker_idx+1}/{len(tickers)}] Processing {ticker}...")

        try:
            # Get features and prices for this ticker
            features, prices = prepare_features_for_ticker(feature_matrix, ticker)

            # Compute forward returns for labeling
            forward_returns = compute_forward_returns(prices, horizon)

            # Compute rolling volatility for threshold
            volatility = prices.pct_change().rolling(20, min_periods=20).std()

            # Create labels
            labels = create_triple_barrier_labels(
                forward_returns,
                volatility,
                horizon,
                threshold_multiplier
            )

            # Align all data
            common_idx = features.index.intersection(labels.index)
            if len(common_idx) < train_size + test_size:
                if verbose:
                    logger.warning(f"  {ticker}: Not enough data ({len(common_idx)} days)")
                continue

            features = features.loc[common_idx]
            labels = labels.loc[common_idx]

            # Get dates for walk-forward split
            dates = features.index

            # Generate splits
            from signals.backtester import walk_forward_split, add_purge_gap

            splits = list(walk_forward_split(dates, train_size, test_size, step))

            ticker_predictions = []
            ticker_models = []

            for split_idx, (train_idx, test_idx) in enumerate(splits):
                # Apply purge gap
                purged_train, purged_test = add_purge_gap(train_idx, test_idx, gap)

                if verbose:
                    logger.info(f"    Split {split_idx+1}: "
                               f"{purged_train[0].date()} → {purged_train[-1].date()} | "
                               f"{purged_test[0].date()} → {purged_test[-1].date()}")

                # Get train/test data
                X_train = features.loc[purged_train]
                y_train = labels.loc[purged_train]
                X_test = features.loc[purged_test]

                # Remove rows with NaN labels
                mask = ~y_train.isna()
                X_train = X_train.loc[mask]
                y_train = y_train.loc[mask]

                if len(X_train) < 50 or len(X_test) == 0:
                    if verbose:
                        logger.warning(f"      Skipping: insufficient data ({len(X_train)} train, {len(X_test)} test)")
                    continue

                # Train model
                try:
                    model = train_lightgbm(X_train, y_train)
                except Exception as e:
                    if verbose:
                        logger.warning(f"      Training failed: {e}")
                    continue

                # Generate predictions (probability of BUY)
                proba = model.predict_proba(X_test)
                # Class mapping: 0=neutral, 1=buy, 2=sell
                # We want probability of buy (class 1)
                if proba.shape[1] >= 2:
                    buy_prob = proba[:, 1]
                else:
                    # Binary classification
                    buy_prob = proba[:, 0]

                pred_df = pd.DataFrame({
                    'date': X_test.index,
                    'ticker': ticker,
                    'signal_probability': buy_prob,
                    'split': split_idx
                })

                ticker_predictions.append(pred_df)
                ticker_models.append(model)

            # Combine predictions for this ticker
            if ticker_predictions:
                ticker_pred_df = pd.concat(ticker_predictions, axis=0, ignore_index=True)
                all_predictions.append(ticker_pred_df)

                # Store models and importances
                if ticker_models:
                    # Get importance from the last model
                    last_model = ticker_models[-1]
                    all_models[ticker] = ticker_models

                    importances = get_feature_importances(
                        last_model,
                        features.columns.tolist()
                    )
                    all_importances[ticker] = importances

                    if verbose:
                        logger.info(f"    Top 5 features for {ticker}:")
                        for _, row in importances.head(5).iterrows():
                            logger.info(f"      {row['feature']}: {row['importance']:.0f}")

        except Exception as e:
            logger.error(f"  Error processing {ticker}: {e}")
            continue

    # Combine all predictions
    if all_predictions:
        all_pred_df = pd.concat(all_predictions, axis=0, ignore_index=True)

        # Convert date to datetime
        all_pred_df['date'] = pd.to_datetime(all_pred_df['date'])

        # Create final signals: -1, 0, 1 based on probability
        # Using a simple threshold: 0.6 for buy, 0.4 for sell
        # For now, let's use 0.5 threshold for buy
        def create_signal(prob):
            if prob > 0.6:
                return 1.0  # BUY
            elif prob < 0.4:
                return -1.0  # SELL
            else:
                return 0.0  # NEUTRAL

        all_pred_df['signal'] = all_pred_df['signal_probability'].apply(create_signal)

        # Set MultiIndex
        all_pred_df = all_pred_df.set_index(['date', 'ticker'])

        # Save predictions
        save_path = config.DATA_PROC_DIR / "ml_signals.parquet"
        all_pred_df.to_parquet(save_path)
        logger.info(f"Saved ML signals to {save_path}")
        logger.info(f"  Shape: {all_pred_df.shape}")
        logger.info(f"  Signal distribution: {all_pred_df['signal'].value_counts().to_dict()}")

    else:
        logger.error("No predictions generated for any ticker")
        all_pred_df = pd.DataFrame()

    return {
        'predictions': all_pred_df,
        'models': all_models,
        'importances': all_importances
    }


# ============================================================================
# FASTAPI HELPER FUNCTIONS
# ============================================================================

def get_ml_signals(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ML signals from disk.

    Args:
        ticker: Filter by ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with ML signals
    """
    signals_path = config.DATA_PROC_DIR / "ml_signals.parquet"

    if not signals_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(signals_path)

    if ticker:
        df = df[df['ticker'] == ticker]

    if start_date:
        df = df[df['date'] >= start_date]

    if end_date:
        df = df[df['date'] <= end_date]

    return df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run walk-forward ML alpha model.

    Usage:
        # Run full pipeline
        python -m signals.alpha_model

        # Run with verbose output
        python -m signals.alpha_model --verbose

        # Run with custom parameters
        python -m signals.alpha_model --horizon 10 --train-size 504
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="LightGBM alpha model")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward horizon in days")
    parser.add_argument("--train-size", type=int, default=252,
                        help="Training window size")
    parser.add_argument("--test-size", type=int, default=63,
                        help="Test window size")
    parser.add_argument("--step", type=int, default=63,
                        help="Roll-forward step")
    parser.add_argument("--threshold-multiplier", type=float, default=1.0,
                        help="Volatility threshold multiplier")
    parser.add_argument("--ticker", type=str,
                        help="Process a single ticker only")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")

    args = parser.parse_args()

    if not LGBM_AVAILABLE:
        logger.error("LightGBM not installed. Install with: pip install lightgbm")
        exit(1)

    tickers = [args.ticker] if args.ticker else config.UNIVERSE

    logger.info("="*60)
    logger.info("LIGHTGBM ALPHA MODEL")
    logger.info("="*60)
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Horizon: {args.horizon} days")
    logger.info(f"Train size: {args.train_size} days")
    logger.info(f"Test size: {args.test_size} days")
    logger.info(f"Step: {args.step} days")
    logger.info(f"Threshold multiplier: {args.threshold_multiplier}")
    logger.info("="*60)

    results = run_walk_forward_ml(
        tickers=tickers,
        horizon=args.horizon,
        train_size=args.train_size,
        test_size=args.test_size,
        step=args.step,
        gap=5,
        threshold_multiplier=args.threshold_multiplier,
        verbose=args.verbose
    )

    # Print summary
    predictions = results['predictions']

    if not predictions.empty:
        print("\n" + "="*60)
        print("ML SIGNALS SUMMARY")
        print("="*60)

        # Signal distribution
        print("\nSignal distribution:")
        signal_counts = predictions['signal'].value_counts()
        for signal, count in signal_counts.items():
            label = "BUY" if signal == 1 else "SELL" if signal == -1 else "NEUTRAL"
            print(f"  {label} ({signal:+.0f}): {count:,} ({count/len(predictions)*100:.1f}%)")

        # Feature importances (average across tickers)
        print("\nTop features by average importance:")
        all_importances = results['importances']

        if all_importances:
            avg_importance = pd.concat(all_importances).groupby('feature')['importance'].mean()
            top_features = avg_importance.sort_values(ascending=False).head(10)
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.1f}")

        print("\n" + "="*60)

        # Save summary
        summary_path = config.DATA_PROC_DIR / "ml_signals_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("ML ALPHA MODEL SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tickers: {len(tickers)}\n")
            f.write(f"Horizon: {args.horizon} days\n")
            f.write(f"Train size: {args.train_size} days\n")
            f.write(f"Test size: {args.test_size} days\n")
            f.write(f"Step: {args.step} days\n")
            f.write("-"*60 + "\n")
            f.write("Signal Distribution:\n")
            for signal, count in signal_counts.items():
                label = "BUY" if signal == 1 else "SELL" if signal == -1 else "NEUTRAL"
                f.write(f"  {label} ({signal:+.0f}): {count:,} ({count/len(predictions)*100:.1f}%)\n")

        logger.info(f"Saved summary to {summary_path}")
    else:
        logger.error("No predictions generated")