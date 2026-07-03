"""
signals/ensemble.py
Signal ensemble and composite score generation.

Combines multiple signal sources into a single trading signal:
1. Momentum z-score (cross_sectional_rank)
2. Mean-reversion RSI signal (RSI < 30 → buy, RSI > 70 → sell)
3. LightGBM probability (ML alpha model)

Outputs:
- Composite signals for all tickers
- Cross-sectional ranks
- Long-short portfolio selection
- Information Coefficient (IC) analysis
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
from scipy.stats import spearmanr

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# SIGNAL GENERATORS
# ============================================================================

def get_momentum_signal(
    feature_matrix: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    Generate momentum signal from cross-sectional rank of returns.

    Uses the pre-computed cs_rank_returns from feature matrix.
    This is a pure momentum signal - stocks with high recent returns.

    Args:
        feature_matrix: MultiIndex DataFrame with cs_rank_returns
        window: Lookback period (not used, but kept for consistency)

    Returns:
        Momentum signal series (-1 to +1)
    """
    # cs_rank_returns is already normalized to [-1, +1]
    # Higher = better recent performance
    signal = feature_matrix['cs_rank_returns'].copy()

    return signal


def get_rsi_signal(
    feature_matrix: pd.DataFrame,
    oversold: float = 30,
    overbought: float = 70
) -> pd.Series:
    """
    Generate mean-reversion signal from RSI.

    - RSI < 30 → BUY (+1) - oversold, expect bounce
    - RSI > 70 → SELL (-1) - overbought, expect drop
    - 30 <= RSI <= 70 → NEUTRAL (0)

    This is a pure mean-reversion signal.

    Args:
        feature_matrix: MultiIndex DataFrame with rsi_14
        oversold: RSI threshold for buy signal (default 30)
        overbought: RSI threshold for sell signal (default 70)

    Returns:
        RSI signal series (-1, 0, +1)

    Interview Tip: "Mean-reversion signals work because markets overreact.
    When RSI is extreme, it often reverses. This is the classic
    'buy when others are fearful, sell when others are greedy' signal."
    """
    rsi = feature_matrix['rsi_14'].copy()

    signal = pd.Series(0, index=rsi.index, dtype=float)
    signal[rsi < oversold] = 1.0      # BUY (oversold)
    signal[rsi > overbought] = -1.0   # SELL (overbought)

    return signal


def get_ml_signal(
    ml_signals_path: Optional[Path] = None
) -> pd.Series:
    """
    Load ML signals from alpha_model.py.

    Args:
        ml_signals_path: Path to ml_signals.parquet

    Returns:
        ML signal series (probability of BUY)

    Interview Tip: "The ML signal is the 'smart' signal.
    It captures non-linear relationships that simple indicators miss.
    The probability output allows for position sizing."
    """
    if ml_signals_path is None:
        ml_signals_path = config.DATA_PROC_DIR / "ml_signals.parquet"

    if not ml_signals_path.exists():
        logger.error("ML signals not found. Run signals.alpha_model first.")
        return pd.Series(dtype=float)

    ml_signals = pd.read_parquet(ml_signals_path)

    # Use signal_probability as the signal
    # Range: 0 to 1, where > 0.5 suggests BUY
    signal = ml_signals['signal_probability'].copy()

    # Normalize to [-1, +1] range
    # probability 0.0 → -1 (strong SELL)
    # probability 0.5 → 0 (NEUTRAL)
    # probability 1.0 → +1 (strong BUY)
    signal = (signal - 0.5) * 2  # Maps 0.5 → 0, 1.0 → 1, 0.0 → -1

    return signal


# ============================================================================
# ENSEMBLE COMBINATION
# ============================================================================

def compute_composite_signal(
    feature_matrix: pd.DataFrame,
    ml_signals_path: Optional[Path] = None,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Combine multiple signal sources into a composite signal.

    Signal sources:
    1. Momentum (cs_rank_returns)
    2. Mean-reversion (RSI)
    3. ML probability (LightGBM)

    Args:
        feature_matrix: MultiIndex DataFrame with features
        ml_signals_path: Path to ML signals
        weights: Dictionary of weights for each signal source
                 Default: {'momentum': 0.33, 'rsi': 0.33, 'ml': 0.34}

    Returns:
        Composite signal series (-1 to +1)

    Interview Tip: "Ensemble signals are more robust than single signals.
    Momentum and mean-reversion capture different market regimes.
    ML adds the 'smart' component. Equal weighting is a good starting point."
    """
    if weights is None:
        weights = {
            'momentum': 1/3,
            'rsi': 1/3,
            'ml': 1/3
        }

    # Step 1: Get individual signals
    momentum_signal = get_momentum_signal(feature_matrix)
    rsi_signal = get_rsi_signal(feature_matrix)
    ml_signal = get_ml_signal(ml_signals_path)

    # Step 2: Align all signals
    # ML signal may have different date range, so we need to align
    common_idx = momentum_signal.index.intersection(rsi_signal.index)

    if len(ml_signal) > 0:
        common_idx = common_idx.intersection(ml_signal.index)

    if len(common_idx) == 0:
        logger.error("No overlapping dates between signals")
        return pd.Series(dtype=float)

    momentum_signal = momentum_signal.reindex(common_idx)
    rsi_signal = rsi_signal.reindex(common_idx)

    if len(ml_signal) > 0:
        ml_signal = ml_signal.reindex(common_idx)
    else:
        # If ML not available, use only momentum and RSI
        logger.warning("ML signals not available. Using only momentum and RSI.")
        weights = {
            'momentum': 0.5,
            'rsi': 0.5,
            'ml': 0.0
        }
        # Re-normalize weights
        total = weights['momentum'] + weights['rsi']
        weights['momentum'] = weights['momentum'] / total
        weights['rsi'] = weights['rsi'] / total

    # Step 3: Compute composite signal
    composite = pd.Series(0.0, index=common_idx, dtype=float)

    if weights.get('momentum', 0) > 0:
        composite += weights['momentum'] * momentum_signal

    if weights.get('rsi', 0) > 0:
        composite += weights['rsi'] * rsi_signal

    if weights.get('ml', 0) > 0 and len(ml_signal) > 0:
        composite += weights['ml'] * ml_signal

    # Step 4: Clip to [-1, +1] range
    composite = composite.clip(-1, 1)

    return composite


def cross_sectional_rank_signals(
    composite_signal: pd.Series,
    normalize: bool = True
) -> pd.Series:
    """
    Rank composite signal cross-sectionally within each date.

    This is the key step for long-short strategies:
    - Stocks with high composite signal get positive ranks (buy)
    - Stocks with low composite signal get negative ranks (sell)

    Args:
        composite_signal: MultiIndex Series (date, ticker) of signals
        normalize: Normalize to [-1, +1] (default True)

    Returns:
        Cross-sectional ranks (-1 to +1)
    """
    # Pivot to wide format (dates × tickers)
    wide = composite_signal.unstack(level='ticker')

    # Rank across tickers (axis=1)
    ranks = wide.rank(axis=1, method='average', ascending=True)

    if normalize:
        n_cols = wide.shape[1]
        # Normalize to [-1, +1]
        normalized = 2 * (ranks - 1) / (n_cols - 1) - 1
    else:
        normalized = ranks

    # Stack back to long format
    ranked_signals = normalized.stack()

    return ranked_signals


def get_long_short_portfolio(
    ranked_signals: pd.Series,
    long_count: int = 2,
    short_count: int = 2
) -> pd.DataFrame:
    """
    Select long and short positions based on ranked signals.

    Args:
        ranked_signals: Cross-sectional ranks (-1 to +1)
        long_count: Number of stocks to go long (default 2)
        short_count: Number of stocks to go short (default 2)

    Returns:
        DataFrame with columns: ticker, signal, position

    Interview Tip: "A 2-long, 2-short portfolio is market-neutral.
    Equal dollar amounts long and short means you're immune to
    broad market moves. This is the standard hedge fund approach."
    """
    # Reset index to work with date groups
    df = ranked_signals.reset_index()
    df.columns = ['date', 'ticker', 'signal']

    # Group by date and select top/long positions
    positions = []

    for date, group in df.groupby('date'):
        # Sort by signal descending
        sorted_group = group.sort_values('signal', ascending=False)

        # Top 'long_count' are long
        longs = sorted_group.head(long_count).copy()
        longs['position'] = 1.0  # LONG

        # Bottom 'short_count' are short
        shorts = sorted_group.tail(short_count).copy()
        shorts['position'] = -1.0  # SHORT

        # Combine
        day_positions = pd.concat([longs, shorts], axis=0)
        day_positions['date'] = date

        positions.append(day_positions)

    # Combine all dates
    portfolio = pd.concat(positions, axis=0, ignore_index=True)

    # Set MultiIndex
    portfolio = portfolio.set_index(['date', 'ticker'])

    return portfolio


# ============================================================================
# INFORMATION COEFFICIENT (IC) ANALYSIS
# ============================================================================

def compute_information_coefficient(
    signals: pd.Series,
    forward_returns: pd.Series,
    horizon: int = 5
) -> float:
    """
    Compute Information Coefficient for a given horizon.

    IC = Spearman rank correlation between signal[t] and forward_return[t+horizon]

    Args:
        signals: Signal series at time t
        forward_returns: Forward returns at time t+horizon
        horizon: Forward horizon in days

    Returns:
        IC value (-1 to +1)

    Interview Tip: "IC measures the predictive power of your signal.
    An IC of 0.03-0.05 is considered good in quantitative finance.
    IC > 0.1 is exceptional."
    """
    # Align indices
    common_idx = signals.index.intersection(forward_returns.index)

    if len(common_idx) < 10:
        return np.nan

    signals = signals.reindex(common_idx)
    forward_returns = forward_returns.reindex(common_idx)

    # Remove NaN values
    mask = ~(signals.isna() | forward_returns.isna())
    signals = signals[mask]
    forward_returns = forward_returns[mask]

    if len(signals) < 10:
        return np.nan

    # Spearman rank correlation
    ic, p_value = spearmanr(signals, forward_returns)

    return ic


def compute_ic_decay(
    ranked_signals: pd.Series,
    returns_matrix: pd.DataFrame,
    horizons: List[int] = [1, 5, 10, 21]
) -> pd.DataFrame:
    """
    Compute IC decay for different holding periods.

    Shows how signal predictive power decays over time.

    Args:
        ranked_signals: Cross-sectional ranks (-1 to +1)
        returns_matrix: Wide DataFrame (dates × tickers) of returns
        horizons: List of horizons to test

    Returns:
        DataFrame with IC values for each horizon
    """
    # Reset index to long format
    df = ranked_signals.reset_index()
    df.columns = ['date', 'ticker', 'signal']

    results = []

    for horizon in horizons:
        # Compute forward returns using price matrix
        # We need prices, not returns, to compute forward returns
        # Let's load the price matrix
        price_path = config.DATA_PROC_DIR / "price_matrix.parquet"
        if not price_path.exists():
            logger.error("Price matrix not found. Cannot compute forward returns.")
            results.append({'horizon': horizon, 'ic': np.nan, 'count': 0})
            continue

        price_matrix = pd.read_parquet(price_path)
        
        # Align dates
        common_dates = price_matrix.index.intersection(returns_matrix.index)
        price_matrix = price_matrix.loc[common_dates]
        
        # Compute forward returns for each ticker
        # Forward return: price[t+horizon] / price[t] - 1
        forward_returns = price_matrix.shift(-horizon) / price_matrix - 1
        
        # Stack to long format
        forward_long = forward_returns.stack()
        forward_long.name = 'forward_return'
        
        # Reset index for merging
        forward_df = forward_long.reset_index()
        forward_df.columns = ['date', 'ticker', 'forward_return']
        
        # Merge with signals
        merged = df.merge(
            forward_df,
            on=['date', 'ticker'],
            how='inner'
        )
        
        if len(merged) == 0:
            results.append({'horizon': horizon, 'ic': np.nan, 'count': 0})
            continue
        
        # Remove NaN values
        merged = merged.dropna(subset=['signal', 'forward_return'])
        
        if len(merged) < 10:
            results.append({'horizon': horizon, 'ic': np.nan, 'count': len(merged)})
            continue
        
        # Compute IC - Spearman rank correlation
        try:
            ic, p_value = spearmanr(merged['signal'], merged['forward_return'])
        except Exception as e:
            logger.warning(f"Error computing IC for horizon {horizon}: {e}")
            ic = np.nan
        
        results.append({
            'horizon': horizon,
            'ic': ic,
            'count': len(merged)
        })
        
        logger.info(f"  Horizon {horizon}: IC = {ic:.4f} (n={len(merged):,})")

    return pd.DataFrame(results)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_ensemble_pipeline(
    feature_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    ml_signals_path: Optional[Path] = None,
    weights: Optional[Dict[str, float]] = None,
    long_count: int = 2,
    short_count: int = 2,
    save: bool = True
) -> Dict:
    """
    Run the full ensemble signal pipeline.

    Steps:
    1. Compute composite signal
    2. Cross-sectional rank
    3. Select long-short portfolio
    4. Compute IC analysis

    Args:
        feature_matrix: MultiIndex DataFrame with features
        returns_matrix: Wide DataFrame (dates × tickers) of returns
        ml_signals_path: Path to ML signals
        weights: Signal weights
        long_count: Number of long positions
        short_count: Number of short positions
        save: Save results to disk

    Returns:
        Dictionary with all results
    """
    logger.info("Running ensemble signal pipeline...")

    # Step 1: Compute composite signal
    logger.info("Computing composite signal...")
    composite = compute_composite_signal(
        feature_matrix,
        ml_signals_path,
        weights
    )

    if len(composite) == 0:
        logger.error("No composite signal generated")
        return {}

    logger.info(f"Composite signal shape: {composite.shape}")

    # Step 2: Cross-sectional rank
    logger.info("Computing cross-sectional ranks...")
    ranked_signals = cross_sectional_rank_signals(composite)

    logger.info(f"Ranked signals shape: {ranked_signals.shape}")

    # Step 3: Select long-short portfolio
    logger.info(f"Selecting {long_count} long, {short_count} short positions...")
    portfolio = get_long_short_portfolio(
        ranked_signals,
        long_count,
        short_count
    )

    logger.info(f"Portfolio shape: {portfolio.shape}")

    # Step 4: Compute IC analysis
    logger.info("Computing Information Coefficient analysis...")
    horizons = [1, 5, 10, 21]
    ic_decay = compute_ic_decay(
        ranked_signals,
        returns_matrix,
        horizons
    )

    logger.info(f"IC decay:\n{ic_decay}")

    # Step 5: Save results
    if save:
        # Save composite signals - convert Series to DataFrame first
        composite_df = composite.to_frame('composite_signal')
        composite_path = config.DATA_PROC_DIR / "composite_signals.parquet"
        composite_df.to_parquet(composite_path)
        logger.info(f"Saved composite signals → {composite_path}")

        # Save ranked signals - convert Series to DataFrame first
        ranked_df = ranked_signals.to_frame('ranked_signal')
        ranked_path = config.DATA_PROC_DIR / "ranked_signals.parquet"
        ranked_df.to_parquet(ranked_path)
        logger.info(f"Saved ranked signals → {ranked_path}")

        # Save portfolio
        portfolio_path = config.DATA_PROC_DIR / "long_short_portfolio.parquet"
        portfolio.to_parquet(portfolio_path)
        logger.info(f"Saved long-short portfolio → {portfolio_path}")

        # Save IC analysis
        ic_path = config.DATA_PROC_DIR / "ic_analysis.parquet"
        ic_decay.to_parquet(ic_path)
        logger.info(f"Saved IC analysis → {ic_path}")

    return {
        'composite': composite,
        'ranked_signals': ranked_signals,
        'portfolio': portfolio,
        'ic_decay': ic_decay
    }


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run ensemble signal pipeline.

    Usage:
        python -m signals.ensemble
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("="*60)
    logger.info("SIGNAL ENSEMBLE PIPELINE")
    logger.info("="*60)

    # Load feature matrix
    feature_path = config.DATA_PROC_DIR / "feature_matrix.parquet"
    if not feature_path.exists():
        logger.error("Feature matrix not found. Run features.indicators first.")
        exit(1)

    feature_matrix = pd.read_parquet(feature_path)

    # Load returns matrix
    returns_path = config.DATA_PROC_DIR / "returns_matrix.parquet"
    if not returns_path.exists():
        logger.error("Returns matrix not found. Run data.universe first.")
        exit(1)

    returns_matrix = pd.read_parquet(returns_path)

    # Run pipeline
    results = run_ensemble_pipeline(
        feature_matrix=feature_matrix,
        returns_matrix=returns_matrix,
        ml_signals_path=config.DATA_PROC_DIR / "ml_signals.parquet",
        weights={'momentum': 1/3, 'rsi': 1/3, 'ml': 1/3},
        long_count=2,
        short_count=2,
        save=True
    )

    # Print summary
    if results:
        print("\n" + "="*60)
        print("ENSEMBLE SIGNAL SUMMARY")
        print("="*60)

        composite = results['composite']
        ranked = results['ranked_signals']
        portfolio = results['portfolio']
        ic_decay = results['ic_decay']

        print(f"\nComposite Signals:")
        print(f"  Total: {len(composite):,}")
        print(f"  Range: {composite.min():.3f} to {composite.max():.3f}")
        print(f"  Mean: {composite.mean():.3f}")

        print(f"\nRanked Signals:")
        print(f"  Total: {len(ranked):,}")
        print(f"  Range: {ranked.min():.3f} to {ranked.max():.3f}")
        print(f"  Mean: {ranked.mean():.3f}")

        print(f"\nLong-Short Portfolio:")
        print(f"  Total positions: {len(portfolio):,}")
        long_positions = portfolio[portfolio['position'] == 1]
        short_positions = portfolio[portfolio['position'] == -1]
        print(f"  Long positions: {len(long_positions):,}")
        print(f"  Short positions: {len(short_positions):,}")

        print(f"\nInformation Coefficient Decay:")
        for _, row in ic_decay.iterrows():
            print(f"  {int(row['horizon'])}-day horizon: IC = {row['ic']:.4f} (n={row['count']:,})")

        print("="*60)
    else:
        logger.error("Pipeline failed - no results generated")