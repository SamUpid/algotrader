"""
signals/metrics.py
Performance metrics for trading strategies - all implemented from scratch.

This module provides the complete set of metrics used in institutional
quant research. No black-box libraries for the core calculations.

Metrics included:
    - Sharpe Ratio
    - Sortino Ratio (downside deviation only)
    - Calmar Ratio (return / max_drawdown)
    - Maximum Drawdown
    - Maximum Drawdown Duration
    - Hit Rate (Win Rate)
    - Profit Factor
    - Monthly Returns Table
    - Rolling Sharpe
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# CORE METRICS
# ============================================================================

def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.00015,
    periods_per_year: int = 252
) -> float:
    """
    Sharpe Ratio: (Mean Return - Risk-Free) / Std Deviation

    Measures return per unit of total volatility (risk).

    Args:
        returns: Daily return series
        rf: Daily risk-free rate (default 0.015% ≈ 3.8% annual)
        periods_per_year: Trading days per year (default 252)

    Returns:
        Annualized Sharpe ratio

    Interview Tip: "Sharpe is the most widely used risk-adjusted metric.
    A Sharpe > 1 is good, > 2 is excellent, > 3 is exceptional."

    Example:
        returns = [0.01, -0.005, 0.02, -0.01]
        mean = 0.00375, std = 0.0129
        Sharpe = (0.00375 - 0.00015) / 0.0129 * sqrt(252) = 4.43
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - rf
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    return (mean_excess / std_excess) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    rf: float = 0.00015,
    periods_per_year: int = 252
) -> float:
    """
    Sortino Ratio: (Mean Return - Risk-Free) / Downside Deviation

    KEY DIFFERENCE: Only penalizes downside volatility (negative returns),
    not upside volatility.

    Why this matters:
        - Trend-following strategies have positive skew (big wins, small losses)
        - Sharpe penalizes the big wins (high total volatility)
        - Sortino recognizes that upside volatility is GOOD

    Args:
        returns: Daily return series
        rf: Daily risk-free rate
        periods_per_year: Trading days per year

    Returns:
        Annualized Sortino ratio

    Interview Tip: "Sortino is more relevant than Sharpe for strategies
    with asymmetric returns. It doesn't penalize you for having huge wins."

    Example:
        returns = [0.10, -0.01, 0.08, -0.01, 0.09, -0.01]
        Only downside returns: [-0.01, -0.01, -0.01]
        Downside std = 0.0 (if all negative returns are equal)
        Sortino is very high (no downside volatility)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - rf
    mean_excess = excess_returns.mean()

    # Downside deviation: only consider returns below the risk-free rate
    # (or below 0 for simplicity)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) < 2:
        # No downside volatility - Sortino is infinite
        return np.inf if mean_excess > 0 else 0.0

    downside_std = downside_returns.std()

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return (mean_excess / downside_std) * np.sqrt(periods_per_year)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calmar Ratio: Annualized Return / |Maximum Drawdown|

    Measures return per unit of worst-case loss.

    Why this matters:
        - Penalizes catastrophic events (large drawdowns)
        - More conservative than Sharpe
        - Critical for risk-averse investors

    Args:
        returns: Daily return series
        periods_per_year: Trading days per year

    Returns:
        Calmar ratio

    Interview Tip: "Calmar is the 'worst-case scenario' metric.
    It tells you how much return you get for each unit of maximum loss.
    A Calmar below 0.5 means you're taking too much risk."

    Example:
        Annual Return = 15%, Max Drawdown = -20%
        Calmar = 0.15 / 0.20 = 0.75
    """
    if len(returns) < 2:
        return 0.0

    annual_return = returns.mean() * periods_per_year
    max_dd = max_drawdown(returns)

    if max_dd == 0 or np.isnan(max_dd):
        return np.inf if annual_return > 0 else 0.0

    return annual_return / abs(max_dd)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown: Largest peak-to-trough decline.

    Formula:
        DD[t] = (Equity[t] - Peak[t]) / Peak[t]
        Max DD = min(DD)

    Args:
        returns: Daily return series

    Returns:
        Maximum drawdown as negative fraction (e.g., -0.25 for 25% loss)

    Interview Tip: "A 50% drawdown requires a 100% gain to recover.
    This is why large drawdowns are so devastating."

    Example:
        Equity: [100, 102, 101, 105, 100, 95, 98, 110]
        Peak:   [100, 102, 102, 105, 105, 105, 105, 110]
        DD:     [0,   0,   -0.0098, 0,  -0.0476, -0.0952, -0.0667, 0]
        Max DD = -9.52%
    """
    if len(returns) < 2:
        return 0.0

    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak

    return drawdown.min()


def max_drawdown_duration(returns: pd.Series) -> int:
    """
    Maximum Drawdown Duration: Days between peak and recovery.

    Measures how long the worst drawdown lasted.

    Args:
        returns: Daily return series

    Returns:
        Number of days from peak to recovery

    Interview Tip: "Drawdown duration matters as much as magnitude.
    A 20% drawdown that lasts 2 years is worse than a 30% drawdown
    that recovers in 3 months. It's about endurance."

    Example:
        Equity: [100, 102, 101, 105, 100, 95, 98, 110]
        Peak:   [100, 102, 102, 105, 105, 105, 105, 110]
        DD:     [0,   0,   -1%,  0,  -5%, -10%, -7%, 0]
        Duration from -10% (day 6) to recovery (day 8) = 2 days
    """
    if len(returns) < 2:
        return 0

    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak

    # Find the longest period between a new peak and recovery
    max_duration = 0
    current_duration = 0
    in_drawdown = False

    for i in range(len(equity)):
        if drawdown.iloc[i] < 0:
            if not in_drawdown:
                in_drawdown = True
                current_duration = 0
            else:
                current_duration += 1
        else:
            if in_drawdown:
                # Recovered!
                max_duration = max(max_duration, current_duration)
                in_drawdown = False
                current_duration = 0

    return max_duration


def hit_rate(returns: pd.Series) -> float:
    """
    Hit Rate (Win Rate): Fraction of positive return days.

    Formula:
        Hit Rate = #Positive Days / #Total Days

    Args:
        returns: Daily return series

    Returns:
        Hit rate as fraction (0 to 1)

    Interview Tip: "Hit rate is the most intuitive metric.
    A 60% hit rate means you're right 6 out of 10 days.
    But a low hit rate can still be profitable if winners are big."
    """
    if len(returns) == 0:
        return 0.0

    positive_days = (returns > 0).sum()
    return positive_days / len(returns)


def profit_factor(returns: pd.Series) -> float:
    """
    Profit Factor: Gross Profit / Gross Loss

    Formula:
        Gross Profit = sum of all positive returns
        Gross Loss = absolute sum of all negative returns
        Profit Factor = Gross Profit / |Gross Loss|

    > 1.0 = Profitable
    > 1.5 = Good
    > 2.0 = Excellent

    Args:
        returns: Daily return series

    Returns:
        Profit factor (1.0 is break-even)

    Interview Tip: "Profit factor tells you if your winners are bigger
    than your losers. A strategy with 40% hit rate but profit factor 2.0
    is better than 60% hit rate with profit factor 0.8."

    Example:
        Returns: [+0.10, -0.05, +0.08, -0.02, +0.12]
        Gross Profit = 0.10 + 0.08 + 0.12 = 0.30
        Gross Loss = 0.05 + 0.02 = 0.07
        Profit Factor = 0.30 / 0.07 = 4.29 (Excellent!)
    """
    if len(returns) == 0:
        return 0.0

    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()

    if losses == 0:
        return np.inf if gains > 0 else 0.0

    return gains / abs(losses)


# ============================================================================
# ROLLING METRICS
# ============================================================================

def rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    rf: float = 0.00015
) -> pd.Series:
    """
    Rolling Sharpe Ratio over a fixed window.

    Shows how Sharpe evolves over time. Useful for identifying
    periods of strong vs weak performance.

    Args:
        returns: Daily return series
        window: Rolling window size (default 60 days ≈ 3 months)
        rf: Daily risk-free rate

    Returns:
        Series of rolling Sharpe ratios

    Interview Tip: "Rolling Sharpe shows you the strategy's stability.
    If the rolling Sharpe is volatile, the strategy is inconsistent.
    If it's consistently positive, the strategy is robust."
    """
    if len(returns) < window:
        return pd.Series(index=returns.index, dtype=float)

    rolling_sharpe = pd.Series(index=returns.index, dtype=float)

    for i in range(window - 1, len(returns)):
        window_returns = returns.iloc[i - window + 1:i + 1]
        sharpe = sharpe_ratio(window_returns, rf)
        rolling_sharpe.iloc[i] = sharpe

    return rolling_sharpe


# ============================================================================
# MONTHLY RETURNS TABLE
# ============================================================================

def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Create a monthly returns table (12 columns × N years).

    Format:
        Columns: Jan, Feb, Mar, ..., Dec
        Rows: Year 1, Year 2, ..., Year N

    This is used for heatmaps and visualization.

    Args:
        returns: Daily return series

    Returns:
        DataFrame with years as rows and months as columns

    Interview Tip: "Monthly returns heatmaps are the industry standard
    for visualising performance. They show seasonality and consistency
    at a glance."

    Example:
                Jan    Feb    Mar    ...    Dec
        2020    2.1%   1.5%   0.8%    ...   2.3%
        2021    1.2%   -0.5%  3.2%    ...   1.8%
    """
    if len(returns) == 0:
        return pd.DataFrame()

    # Resample to monthly returns
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # Extract year and month
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })

    # Pivot to wide format
    table = monthly_df.pivot(index='year', columns='month', values='return')

    # Name months
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    table.columns = month_names

    return table


# ============================================================================
# COMPLETE METRICS DICTIONARY
# ============================================================================

def compute_all_metrics(
    returns: pd.Series,
    rf: float = 0.00015
) -> Dict:
    """
    Compute ALL metrics for a return series.

    Returns a dictionary with all metrics, ready for JSON serialization.

    Args:
        returns: Daily return series
        rf: Daily risk-free rate

    Returns:
        Dictionary with all metrics

    Interview Tip: "This is the one-stop-shop for performance evaluation.
    Portfolio managers want to see all these metrics in one place."
    """
    if len(returns) == 0:
        return {
            'error': 'No returns data available'
        }

    metrics = {
        'sharpe_ratio': sharpe_ratio(returns, rf),
        'sortino_ratio': sortino_ratio(returns, rf),
        'calmar_ratio': calmar_ratio(returns),
        'max_drawdown': max_drawdown(returns),
        'max_drawdown_duration': max_drawdown_duration(returns),
        'hit_rate': hit_rate(returns),
        'profit_factor': profit_factor(returns),
        'annual_return': returns.mean() * 252,
        'annual_volatility': returns.std() * np.sqrt(252),
        'total_return': (1 + returns).prod() - 1,
        'num_days': len(returns),
        'start_date': returns.index[0].strftime('%Y-%m-%d') if len(returns) > 0 else None,
        'end_date': returns.index[-1].strftime('%Y-%m-%d') if len(returns) > 0 else None,
    }

    # Add rolling Sharpe (last 60 days)
    if len(returns) >= 60:
        rolling = rolling_sharpe(returns, 60, rf)
        metrics['rolling_sharpe_last_60'] = rolling.iloc[-1] if not rolling.isna().all() else None

    return metrics


def format_metrics_for_print(metrics: Dict) -> str:
    """
    Format metrics dictionary for pretty printing.

    Args:
        metrics: Dictionary from compute_all_metrics

    Returns:
        Formatted string

    Interview Tip: "This is what you'd present in a pitch deck
    or portfolio review meeting."
    """
    if 'error' in metrics:
        return f"Error: {metrics['error']}"

    lines = []
    lines.append("=" * 60)
    lines.append("PERFORMANCE METRICS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Start Date:           {metrics['start_date']}")
    lines.append(f"End Date:             {metrics['end_date']}")
    lines.append(f"Total Days:           {metrics['num_days']:,}")
    lines.append("")
    lines.append("RETURNS:")
    lines.append(f"  Annual Return:       {metrics['annual_return']:.2%}")
    lines.append(f"  Annual Volatility:   {metrics['annual_volatility']:.2%}")
    lines.append(f"  Total Return:        {metrics['total_return']:.2%}")
    lines.append("")
    lines.append("RISK-ADJUSTED:")
    lines.append(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    lines.append(f"  Sortino Ratio:       {metrics['sortino_ratio']:.3f}")
    lines.append(f"  Calmar Ratio:        {metrics['calmar_ratio']:.3f}")
    lines.append("")
    lines.append("RISK:")
    lines.append(f"  Max Drawdown:        {metrics['max_drawdown']:.2%}")
    lines.append(f"  Max DD Duration:     {metrics['max_drawdown_duration']} days")
    lines.append("")
    lines.append("TRADE STATISTICS:")
    lines.append(f"  Hit Rate:            {metrics['hit_rate']:.2%}")
    lines.append(f"  Profit Factor:       {metrics['profit_factor']:.3f}")
    lines.append("=" * 60)

    if 'rolling_sharpe_last_60' in metrics and metrics['rolling_sharpe_last_60'] is not None:
        lines.append(f"Rolling Sharpe (60d): {metrics['rolling_sharpe_last_60']:.3f}")

    return "\n".join(lines)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Test metrics with synthetic data.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("Testing performance metrics...")

    # Create synthetic returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)

    # Compute metrics
    metrics = compute_all_metrics(returns)

    # Print formatted metrics
    print(format_metrics_for_print(metrics))

    # Show monthly returns table
    print("\nMonthly Returns Table:")
    monthly = monthly_returns_table(returns)
    if not monthly.empty:
        print(monthly.round(4))