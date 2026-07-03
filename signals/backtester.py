"""
signals/backtester.py
Vectorized backtester for algorithmic trading strategies.

Key design principles:
1. NO lookahead bias - signals are shifted by 1 day
2. Transaction costs on position changes only
3. Vectorized operations for speed (no loops over positions)
4. Equal-weighted portfolio by default, but supports custom weights

Why vectorized backtesting matters:
- 100x faster than event-driven backtesters
- Easier to debug (all returns computed in one operation)
- Standard in quant research (all academic papers use vectorized)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# CORE BACKTESTING FUNCTIONS
# ============================================================================

def compute_positions(
    signals: pd.Series,
    shift: int = 1
) -> pd.Series:
    """
    Convert signals to positions with execution delay.

    CRITICAL: Signals use today's close, but earliest execution is tomorrow's open.
    We shift signals by 1 day to simulate this delay.

    Position sizing:
    - Long: positive signal → +1 (buy 1 unit)
    - Short: negative signal → -1 (sell 1 unit)  
    - Neutral: zero → 0 (no position)

    In production, you'd scale positions by volatility, ATR, or Kelly criterion.

    Args:
        signals: Signal values (-1 to +1 or any range)
        shift: Execution delay in days (default 1)

    Returns:
        Positions series (same index as signals, shifted)

    Interview Tip: "We shift signals by 1 day because you can't trade
    at today's close using today's signal. You execute at tomorrow's open.
    This is the most common source of lookahead bias in backtests."
    """
    # Shift signals forward by 1 day
    positions = signals.shift(shift)

    # Clip to [-1, 1] range (if signals are beyond that)
    positions = positions.clip(-1, 1)

    # First position is NaN (no signal on day 0)
    return positions


def compute_returns(
    positions: pd.Series,
    price_returns: pd.Series
) -> pd.Series:
    """
    Compute strategy returns from positions and asset returns.

    Formula:
        strategy_return[t] = position[t] * asset_return[t]

    This assumes you can trade fractional shares (equal weights).

    Args:
        positions: Positions at time t (from compute_positions)
        price_returns: Asset returns at time t

    Returns:
        Strategy returns series

    Interview Tip: "Position at time t is applied to asset return at time t.
    This is correct because you enter at open[t] and exit at close[t]."
    """
    # Align indices (just in case)
    common_idx = positions.index.intersection(price_returns.index)

    if len(common_idx) == 0:
        logger.warning("No overlapping indices between positions and returns")
        return pd.Series(dtype=float)

    positions = positions.reindex(common_idx)
    price_returns = price_returns.reindex(common_idx)

    return positions * price_returns


def apply_transaction_costs(
    positions: pd.Series,
    prices: pd.Series,
    spread_bps: float = 10,
    commission: float = 0.005
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply transaction costs on position changes.

    Cost is paid on EVERY position change (turnover).

    Cost components:
    1. Bid-ask spread: spread_bps / 10000 * |Δposition| * price
    2. Commission: commission * |Δposition| (per share)

    Why cost on position changes only:
    - Holding positions doesn't incur costs (no rebalancing fees)
    - Only when you trade (enter or exit)

    Args:
        positions: Positions series
        prices: Asset prices (for dollar cost calculation)
        spread_bps: Bid-ask spread in basis points (1 bp = 0.01%)
        commission: Commission per share (dollars)

    Returns:
        Tuple of (net_returns, costs_series)

    Interview Tip: "Transaction costs are the silent killer of strategies.
    A 10 bp spread means you lose 10 bps on every round-trip trade.
    A strategy that looks good at 0 bps but collapses at 10 bps is marginal."
    """
    # Calculate position changes (turnover)
    position_changes = positions.diff().abs()

    # Align with prices
    common_idx = position_changes.index.intersection(prices.index)
    if len(common_idx) == 0:
        logger.warning("No overlapping indices for transaction costs")
        return positions * 0, pd.Series(0, index=positions.index)

    position_changes = position_changes.reindex(common_idx)
    prices_aligned = prices.reindex(common_idx)

    # Spread cost: spread_bps / 10000 * |Δposition| * price
    # 1 bp = 0.0001, so spread_bps / 10000 = fraction
    spread_cost = (spread_bps / 10000) * position_changes * prices_aligned

    # Commission cost: commission * |Δposition|
    commission_cost = commission * position_changes

    # Total cost
    total_cost = spread_cost + commission_cost

    # Convert costs to returns (as fraction of position value)
    # Cost / (position * price) = cost per dollar traded
    # But since position changes may be zero, we need to handle carefully
    position_value = positions.reindex(common_idx).abs() * prices_aligned

    # Avoid division by zero
    cost_as_return = total_cost / position_value.replace(0, np.nan)
    cost_as_return = cost_as_return.fillna(0)

    return cost_as_return, total_cost


def compute_portfolio_returns(
    strategy_returns: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    weight_type: str = 'equal'
) -> pd.Series:
    """
    Aggregate individual strategy returns into portfolio returns.

    Args:
        strategy_returns: DataFrame of returns for each strategy/asset
        weights: Weight for each column (if None, uses equal weights)
        weight_type: 'equal', 'vol_target', or 'custom'

    Returns:
        Portfolio returns series

    Interview Tip: "Equal weighting is the most common benchmark in quant research.
    It's simple, transparent, and avoids overfitting on weights.
    In production, we'd use volatility targeting or risk parity."
    """
    if weights is None:
        if weight_type == 'equal':
            weights = pd.Series(1 / strategy_returns.shape[1],
                                index=strategy_returns.columns)
        elif weight_type == 'vol_target':
            # Volatility target: 1 / volatility
            vol = strategy_returns.std()
            weights = (1 / vol) / (1 / vol).sum()
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

    # Align weights with columns
    weights = weights.reindex(strategy_returns.columns, fill_value=0)

    # Compute weighted returns
    portfolio_returns = (strategy_returns * weights).sum(axis=1)

    return portfolio_returns


def compute_equity_curve(
    portfolio_returns: pd.Series,
    initial_capital: float = 1_000_000
) -> pd.Series:
    """
    Compute equity curve from portfolio returns.

    Formula:
        Equity[t] = Equity[t-1] * (1 + return[t])

    Args:
        portfolio_returns: Daily portfolio returns
        initial_capital: Starting capital (default $1,000,000)

    Returns:
        Equity curve series (cumulative wealth)

    Interview Tip: "Starting with $1M is standard for institutional backtests.
    It allows for realistic position sizing and transaction costs."
    """
    # Cumulative product of (1 + returns)
    equity = initial_capital * (1 + portfolio_returns).cumprod()

    return equity


# ============================================================================
# PERFORMANCE METRICS (Core - used during backtest)
# ============================================================================

def compute_sharpe(
    returns: pd.Series,
    rf: float = 0.00015,
    periods_per_year: int = 252
) -> float:
    """
    Sharpe ratio: (mean_return - rf) / std_return * sqrt(periods)

    Args:
        returns: Return series
        rf: Risk-free rate (daily, default 0.015% ≈ 3.8% annual)
        periods_per_year: Trading days per year (default 252)

    Returns:
        Sharpe ratio (annualized)

    Interview Tip: "We use 252 trading days per year (not 365).
    Sharpe ratio is the most widely used risk-adjusted return metric."
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - rf
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0

    return mean_excess / std_excess * np.sqrt(periods_per_year)


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown: peak-to-trough decline.

    Formula:
        DD[t] = (Equity[t] - Peak[t]) / Peak[t]
        Max DD = min(DD)

    Args:
        returns: Return series

    Returns:
        Maximum drawdown (as negative fraction, e.g., -0.25 for 25% loss)
    """
    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak

    return drawdown.min()


def compute_calmar(returns: pd.Series) -> float:
    """
    Calmar ratio: Annualized return / |Max Drawdown|

    Args:
        returns: Return series

    Returns:
        Calmar ratio

    Interview Tip: "Calmar ratio is more conservative than Sharpe.
    It penalizes strategies with large drawdowns, even if they have high returns."
    """
    annual_return = returns.mean() * 252
    max_dd = compute_max_drawdown(returns)

    if max_dd == 0:
        return np.inf

    return annual_return / abs(max_dd)


# ============================================================================
# MAIN BACKTESTER CLASS
# ============================================================================

class VectorizedBacktester:
    """
    Main backtesting engine for single or multi-asset strategies.

    Usage:
        # Single asset
        bt = VectorizedBacktester()
        bt.set_returns(price_returns)
        positions = bt.compute_positions(signals)
        results = bt.backtest(positions, prices)

        # Multi-asset
        bt = VectorizedBacktester()
        bt.set_returns(returns_matrix)
        positions_df = bt.compute_positions(signals_df)
        results = bt.backtest(positions_df, prices_df)
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.returns = None
        self.prices = None
        self.results = {}

    def set_returns(self, returns: Union[pd.Series, pd.DataFrame]):
        """Set the asset returns data."""
        self.returns = returns
        return self

    def set_prices(self, prices: Union[pd.Series, pd.DataFrame]):
        """Set the asset prices data."""
        self.prices = prices
        return self

    def compute_positions(
        self,
        signals: Union[pd.Series, pd.DataFrame],
        shift: int = 1
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Convert signals to positions with execution delay.

        Args:
            signals: Signal values (single asset or multi-asset)
            shift: Execution delay in days

        Returns:
            Positions (same shape as signals)
        """
        if isinstance(signals, pd.Series):
            return compute_positions(signals, shift)
        else:
            # Multi-asset: apply to each column
            positions = signals.apply(lambda col: compute_positions(col, shift))
            return positions

    def compute_transaction_costs(
        self,
        positions: Union[pd.Series, pd.DataFrame],
        prices: Union[pd.Series, pd.DataFrame],
        spread_bps: float = 10,
        commission: float = 0.005
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute transaction costs for positions.

        Args:
            positions: Positions
            prices: Asset prices
            spread_bps: Bid-ask spread in basis points
            commission: Commission per share

        Returns:
            Transaction costs as returns
        """
        if isinstance(positions, pd.Series):
            return apply_transaction_costs(positions, prices, spread_bps, commission)[0]
        else:
            # Multi-asset: apply to each column
            costs = pd.DataFrame(index=positions.index)
            for col in positions.columns:
                if col in prices.columns:
                    cost, _ = apply_transaction_costs(
                        positions[col],
                        prices[col],
                        spread_bps,
                        commission
                    )
                    costs[col] = cost
            return costs

    def backtest(
        self,
        positions: Union[pd.Series, pd.DataFrame],
        prices: Union[pd.Series, pd.DataFrame],
        weights: Optional[pd.Series] = None,
        spread_bps: float = 10,
        commission: float = 0.005,
        rf: float = 0.00015
    ) -> Dict:
        """
        Run full backtest.

        Returns a dictionary with:
            - equity_curve: Cumulative wealth
            - returns: Daily strategy returns
            - costs: Daily transaction costs
            - net_returns: Returns after costs
            - metrics: Performance metrics (Sharpe, drawdown, etc.)

        Args:
            positions: Positions (from compute_positions)
            prices: Asset prices
            weights: Portfolio weights (if multi-asset)
            spread_bps: Bid-ask spread in basis points
            commission: Commission per share
            rf: Risk-free rate

        Returns:
            Dictionary of backtest results
        """
        # Compute strategy returns
        if isinstance(positions, pd.Series):
            # Single asset
            strategy_returns = compute_returns(positions, self.returns)
            strategy_returns = strategy_returns.to_frame('strategy')
            costs = self.compute_transaction_costs(positions, prices, spread_bps, commission)
            costs = costs.to_frame('costs')
            net_returns = strategy_returns['strategy'] - costs['costs']
        else:
            # Multi-asset
            strategy_returns = pd.DataFrame(index=positions.index)
            costs = pd.DataFrame(index=positions.index)

            for col in positions.columns:
                if col in self.returns.columns:
                    strategy_returns[col] = compute_returns(
                        positions[col],
                        self.returns[col]
                    )

                if col in prices.columns:
                    cost = self.compute_transaction_costs(
                        positions[col],
                        prices[col],
                        spread_bps,
                        commission
                    )
                    costs[col] = cost

            # Compute portfolio returns
            net_strategy_returns = strategy_returns - costs
            portfolio_returns = compute_portfolio_returns(net_strategy_returns, weights)

            net_returns = portfolio_returns

        # Compute equity curve
        equity_curve = compute_equity_curve(net_returns, self.initial_capital)

        # Compute metrics
        metrics = {
            'sharpe': compute_sharpe(net_returns, rf),
            'max_drawdown': compute_max_drawdown(net_returns),
            'calmar': compute_calmar(net_returns),
            'annual_return': net_returns.mean() * 252,
            'annual_volatility': net_returns.std() * np.sqrt(252),
            'total_return': (equity_curve.iloc[-1] / self.initial_capital - 1) * 100,
            'start_date': net_returns.index[0],
            'end_date': net_returns.index[-1],
            'num_days': len(net_returns)
        }

        self.results = {
            'equity_curve': equity_curve,
            'returns': net_returns,
            'gross_returns': strategy_returns,
            'costs': costs,
            'metrics': metrics
        }

        return self.results

    def cost_sensitivity_test(
        self,
        positions: Union[pd.Series, pd.DataFrame],
        prices: Union[pd.Series, pd.DataFrame],
        spread_bps_list: list = [0, 5, 10, 20],
        commission: float = 0.005
    ) -> Dict:
        """
        Test how Sharpe ratio changes with transaction costs.

        Args:
            positions: Positions
            prices: Asset prices
            spread_bps_list: List of spread bps to test
            commission: Commission per share

        Returns:
            Dictionary of Sharpe ratios for each spread level

        Interview Tip: "This is the 'acid test' for a strategy.
        If Sharpe drops significantly at 10 bps, the strategy is marginal.
        Institutional strategies should survive 20 bps transaction costs."
        """
        results = {}

        for spread_bps in spread_bps_list:
            bt_results = self.backtest(
                positions=positions,
                prices=prices,
                spread_bps=spread_bps,
                commission=commission
            )
            results[spread_bps] = bt_results['metrics']['sharpe']

        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def simple_strategy_backtest(
    signals: Union[pd.Series, pd.DataFrame],
    returns: Union[pd.Series, pd.DataFrame],
    prices: Union[pd.Series, pd.DataFrame],
    initial_capital: float = 1_000_000,
    spread_bps: float = 10,
    commission: float = 0.005,
    rf: float = 0.00015
) -> Dict:
    """
    Quick one-shot backtest for a single strategy.

    Args:
        signals: Strategy signals
        returns: Asset returns
        prices: Asset prices
        initial_capital: Starting capital
        spread_bps: Bid-ask spread in basis points
        commission: Commission per share

    Returns:
        Backtest results dictionary
    """
    bt = VectorizedBacktester(initial_capital)
    bt.set_returns(returns).set_prices(prices)

    positions = bt.compute_positions(signals)
    results = bt.backtest(
        positions=positions,
        prices=prices,
        spread_bps=spread_bps,
        commission=commission,
        rf=rf
    )

    return results


# ============================================================================
# WALK-FORWARD VALIDATION (Thursday)
# ============================================================================

def walk_forward_split(
    dates: pd.DatetimeIndex,
    train_size: int = 252,
    test_size: int = 63,
    step: int = 63
):
    """
    Generate walk-forward train/test splits.

    Walk-forward validation simulates rolling retraining:
    - Train on first 'train_size' days
    - Test on next 'test_size' days
    - Roll forward by 'step' days
    - Repeat

    Args:
        dates: Full date index
        train_size: Number of days in training set (default 252 ≈ 1 year)
        test_size: Number of days in test set (default 63 ≈ 1 quarter)
        step: Step size for rolling window (default 63)

    Yields:
        Tuple of (train_idx, test_idx) for each split

    Example:
        dates = [2018-01-01, 2018-01-02, ..., 2024-12-31]
        train=252, test=63, step=63

        Split 1: train [0:252], test [252:315]
        Split 2: train [63:315], test [315:378]
        Split 3: train [126:378], test [378:441]
        ...

    Interview Tip: "Walk-forward validation is the gold standard.
    It simulates how you'd actually trade: retrain periodically
    with the most recent data, then test on unseen future data."
    """
    n_dates = len(dates)

    if n_dates < train_size + test_size:
        raise ValueError(
            f"Not enough data: {n_dates} days, need {train_size + test_size}"
        )

    start = 0
    split_num = 0

    while start + train_size + test_size <= n_dates:
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = train_end + test_size

        train_idx = dates[train_start:train_end]
        test_idx = dates[test_start:test_end]

        split_num += 1
        yield train_idx, test_idx

        start += step


def add_purge_gap(
    train_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    gap: int = 5
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Add a purge gap between train and test sets.

    Removes the last 'gap' observations from training to prevent
    label leakage between train and test sets.

    Why this matters:
        If your label is a 5-day forward return, the last 5 days of
        training data overlap with the first 5 days of test data.
        This leaks future information into the training set!

    Args:
        train_idx: Training dates
        test_idx: Test dates
        gap: Number of days to remove from end of training set

    Returns:
        Tuple of (purged_train_idx, test_idx)

    Example:
        train = [Jan 1, Jan 2, Jan 3, Jan 4, Jan 5, Jan 6, Jan 7]
        test = [Jan 8, Jan 9, Jan 10]
        gap = 2

        Purged train = [Jan 1, Jan 2, Jan 3, Jan 4, Jan 5]  # removed Jan 6, 7
        Test stays the same = [Jan 8, Jan 9, Jan 10]

    Interview Tip: "The purge gap is critical for preventing leakage
    when using forward returns as labels. Without it, your model is
    effectively cheating by using data from the test period during training."
    """
    if gap <= 0:
        return train_idx, test_idx

    if len(train_idx) <= gap:
        logger.warning(
            f"Train set has {len(train_idx)} days, gap={gap} days. "
            f"Cannot purge {gap} days."
        )
        return train_idx, test_idx

    # Remove last 'gap' days from training
    purged_train = train_idx[:-gap]

    return purged_train, test_idx


def walk_forward_backtest(
    signals_generator,
    returns: Union[pd.Series, pd.DataFrame],
    prices: Union[pd.Series, pd.DataFrame],
    train_size: int = 252,
    test_size: int = 63,
    step: int = 63,
    gap: int = 5,
    initial_capital: float = 1_000_000,
    spread_bps: float = 10,
    commission: float = 0.005,
    rf: float = 0.00015,
    verbose: bool = True
) -> Dict:
    """
    Run a walk-forward backtest with purge gap.

    This simulates how you'd actually trade in production:
    1. Train model on historical data (train_size days)
    2. Generate signals for the next period (test_size days)
    3. Roll forward and retrain

    Args:
        signals_generator: Function that takes train_data and test_dates
                           and returns signals for test period
        returns: Asset returns (full series or DataFrame)
        prices: Asset prices (full series or DataFrame)
        train_size: Training window size (default 252)
        test_size: Test window size (default 63)
        step: Roll-forward step (default 63)
        gap: Purge gap between train and test (default 5)
        initial_capital: Starting capital
        spread_bps: Transaction cost in bps
        commission: Commission per share
        rf: Risk-free rate
        verbose: Print progress

    Returns:
        Dictionary with:
            - oos_returns: Out-of-sample returns (concatenated)
            - oos_equity: Out-of-sample equity curve
            - split_results: List of results per split
            - metrics: Performance metrics on OOS returns
            - splits: List of (train_dates, test_dates) for each split

    Interview Tip: "Walk-forward backtesting is the most rigorous way
    to evaluate a strategy. It simulates real-world conditions where
    you retrain models periodically and test on unseen data."
    """
    # Get full date range
    if isinstance(returns, pd.Series):
        all_dates = returns.index
    else:
        all_dates = returns.index

    # Generate splits
    splits = list(walk_forward_split(all_dates, train_size, test_size, step))

    if not splits:
        raise ValueError(
            f"No splits generated. Need at least {train_size + test_size} days."
        )

    logger.info(f"Running walk-forward backtest with {len(splits)} splits...")

    # Store results from each split
    split_results = []
    oos_returns_list = []
    oos_equity_list = []

    # Track for debugging
    split_info = []

    for i, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            logger.info(
                f"  Split {i+1}/{len(splits)}: "
                f"{train_idx[0].date()} → {train_idx[-1].date()} | "
                f"{test_idx[0].date()} → {test_idx[-1].date()}"
            )

        # Apply purge gap
        purged_train, purged_test = add_purge_gap(train_idx, test_idx, gap)

        if verbose:
            logger.info(
                f"    Purged train: {purged_train[0].date()} → {purged_train[-1].date()} "
                f"(removed last {gap} days)"
            )

        # Get train and test data
        if isinstance(returns, pd.Series):
            train_returns = returns.loc[purged_train]
            test_returns = returns.loc[purged_test]
            train_prices = prices.loc[purged_train]
            test_prices = prices.loc[purged_test]
        else:
            train_returns = returns.loc[purged_train]
            test_returns = returns.loc[purged_test]
            train_prices = prices.loc[purged_train]
            test_prices = prices.loc[purged_test]

        # Generate signals for test period using the signals generator
        # signals_generator takes (train_returns, train_prices, test_returns, test_prices)
        # and returns signals for the test period
        try:
            test_signals = signals_generator(
                train_returns=train_returns,
                train_prices=train_prices,
                test_returns=test_returns,
                test_prices=test_prices,
                test_dates=purged_test
            )
        except Exception as e:
            logger.error(f"  Error generating signals for split {i+1}: {e}")
            continue

        # Convert signals to positions (shift by 1 day within test period)
        if isinstance(test_signals, pd.Series):
            positions = compute_positions(test_signals, shift=1)
        else:
            positions = pd.DataFrame(index=test_signals.index)
            for col in test_signals.columns:
                positions[col] = compute_positions(test_signals[col], shift=1)

        # Run backtest on this split
        bt = VectorizedBacktester(initial_capital)
        bt.set_returns(test_returns).set_prices(test_prices)

        results = bt.backtest(
            positions=positions,
            prices=test_prices,
            spread_bps=spread_bps,
            commission=commission,
            rf=rf
        )

        # Store results
        split_results.append(results)
        oos_returns_list.append(results['returns'])
        oos_equity_list.append(results['equity_curve'])
        split_info.append({
            'split': i,
            'train_start': purged_train[0].date(),
            'train_end': purged_train[-1].date(),
            'test_start': purged_test[0].date(),
            'test_end': purged_test[-1].date(),
            'sharpe': results['metrics']['sharpe']
        })

    # Combine all out-of-sample returns
    if oos_returns_list:
        # Concatenate returns
        oos_returns = pd.concat(oos_returns_list)

        # Compute combined metrics
        equity_curve = compute_equity_curve(oos_returns, initial_capital)

        metrics = {
            'sharpe': compute_sharpe(oos_returns, rf),
            'max_drawdown': compute_max_drawdown(oos_returns),
            'calmar': compute_calmar(oos_returns),
            'annual_return': oos_returns.mean() * 252,
            'annual_volatility': oos_returns.std() * np.sqrt(252),
            'total_return': (equity_curve.iloc[-1] / initial_capital - 1) * 100,
            'num_days': len(oos_returns),
            'num_splits': len(split_results),
            'avg_split_sharpe': np.mean([s['sharpe'] for s in split_info])
        }

        logger.info(f"Walk-forward complete: {metrics['num_days']} OOS days, "
                   f"Sharpe: {metrics['sharpe']:.3f}")

        return {
            'oos_returns': oos_returns,
            'oos_equity': equity_curve,
            'split_results': split_results,
            'split_info': split_info,
            'metrics': metrics
        }
    else:
        logger.error("No valid splits found")
        return None


# ============================================================================
# EXAMPLE SIGNAL GENERATOR (for testing walk-forward)
# ============================================================================

def random_signal_generator(
    train_returns: pd.Series,
    train_prices: pd.Series,
    test_returns: pd.Series,
    test_prices: pd.Series,
    test_dates: pd.DatetimeIndex,
    seed: int = 42
) -> pd.Series:
    """
    Simple random signal generator for testing walk-forward.

    In practice, this would be your ML model or trading strategy.
    """
    np.random.seed(seed)
    # Generate random signals between -1 and 1
    signals = pd.Series(
        np.random.uniform(-1, 1, len(test_dates)),
        index=test_dates
    )
    return signals


# ============================================================================
# PRINT WALK-FORWARD RESULTS
# ============================================================================

def print_walk_forward_summary(results: Dict) -> None:
    """
    Print a summary of walk-forward backtest results.

    Args:
        results: Results from walk_forward_backtest()
    """
    if results is None:
        print("No results to display")
        return

    metrics = results['metrics']
    split_info = results['split_info']

    print("\n" + "="*70)
    print("WALK-FORWARD BACKTEST SUMMARY")
    print("="*70)

    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Out-of-Sample Days:      {metrics['num_days']:,}")
    print(f"  Number of Splits:        {metrics['num_splits']}")
    print(f"  Annual Return:           {metrics['annual_return']:.2%}")
    print(f"  Annual Volatility:       {metrics['annual_volatility']:.2%}")
    print(f"  Sharpe Ratio:            {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown:            {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:            {metrics['calmar']:.3f}")
    print(f"  Total Return:            {metrics['total_return']:.1f}%")
    print(f"  Avg Split Sharpe:        {metrics['avg_split_sharpe']:.3f}")

    print(f"\n📋 SPLIT DETAILS:")
    print(f"  {'Split':>6} | {'Train Start':>12} | {'Train End':>12} | "
          f"{'Test Start':>12} | {'Test End':>12} | {'Sharpe':>8}")
    print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*8}")

    for info in split_info:
        print(f"  {info['split']+1:>6} | {str(info['train_start']):>12} | "
              f"{str(info['train_end']):>12} | {str(info['test_start']):>12} | "
              f"{str(info['test_end']):>12} | {info['sharpe']:>8.3f}")

    print("="*70)


# ============================================================================
# UPDATED SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Test walk-forward backtest with synthetic data.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("Testing walk-forward backtest with synthetic data...")

    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1500, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)
    prices = 100 * (1 + returns).cumprod()

    # Run walk-forward backtest
    results = walk_forward_backtest(
        signals_generator=random_signal_generator,
        returns=returns,
        prices=prices,
        train_size=252,
        test_size=63,
        step=63,
        gap=5,
        initial_capital=1_000_000,
        spread_bps=10,
        commission=0.005,
        verbose=True
    )

    # Print results
    if results:
        print_walk_forward_summary(results)

        # Show equity curve sample
        print(f"\n📈 Equity Curve Sample (first 5 days):")
        print(results['oos_equity'].head())

        print(f"\n📈 Equity Curve Sample (last 5 days):")
        print(results['oos_equity'].tail())