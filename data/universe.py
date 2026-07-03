"""
data/universe.py
────────────────
Aligns individual ticker DataFrames into a single price matrix.

Why alignment matters:
  If AAPL has 1,510 trading days and MSFT has 1,508 (two holidays
  where one exchange was open), a naive join leaves NaN gaps that
  will contaminate any cross-sectional calculation (z-scores, ranks,
  covariance matrices). We align on a common trading calendar.

Survivorship bias note:
  We're using a static universe of 20 established companies.
  A production system would use point-in-time index membership
  (e.g. the S&P 500 constituents as they existed on each date).
  For this project, document this limitation in your README — showing
  you're *aware* of survivorship bias is more impressive than pretending
  you've solved it.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_price_matrix(
    universe_data: dict[str, pd.DataFrame],
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Stack individual ticker DataFrames into a wide price matrix.

    Returns DataFrame: rows = dates, columns = tickers
    Only includes dates where ALL tickers have data (inner join).
    """
    frames = {
        ticker: df[price_col].rename(ticker)
        for ticker, df in universe_data.items()
        if price_col in df.columns
    }

    # Outer join first to see coverage, then report before inner join
    wide_outer = pd.concat(frames, axis=1)
    wide_inner = wide_outer.dropna(how="any")

    dropped_days = len(wide_outer) - len(wide_inner)
    if dropped_days > 0:
        log.info(
            f"Alignment: dropped {dropped_days} days with incomplete data "
            f"({len(wide_inner)} common trading days remain)"
        )

    return wide_inner


def build_returns_matrix(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from price matrix. Shape: same as price_matrix."""
    return price_matrix.pct_change().dropna()


def build_log_returns_matrix(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """Log returns. Additive over time, better statistical properties."""
    return np.log(price_matrix / price_matrix.shift(1)).dropna()


def build_volume_matrix(universe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dollar volume matrix — used for ADV calculation in market impact model."""
    frames = {
        ticker: df["dollar_volume"].rename(ticker)
        for ticker, df in universe_data.items()
        if "dollar_volume" in df.columns
    }
    wide = pd.concat(frames, axis=1)
    return wide.dropna(how="any")


def compute_adv(
    volume_matrix: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Average Daily (Dollar) Volume over rolling window.

    ADV is used in position sizing and market impact:
      max_position_notional = adv_fraction × ADV
    Typical quant limit: don't trade more than 5–10% of ADV in one day.
    """
    return volume_matrix.rolling(window=window, min_periods=window).mean()


def universe_summary(universe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a summary table of coverage, return stats, and liquidity.
    Useful for a first sanity check and for your README.
    """
    rows = []
    for ticker, df in universe_data.items():
        ret = df["return"] if "return" in df.columns else df["close"].pct_change()
        rows.append({
            "ticker":         ticker,
            "start":          str(df.index.min().date()),
            "end":            str(df.index.max().date()),
            "trading_days":   len(df),
            "ann_return":     round((1 + ret.mean()) ** 252 - 1, 4),
            "ann_volatility": round(ret.std() * np.sqrt(252), 4),
            "sharpe_raw":     round(ret.mean() / ret.std() * np.sqrt(252), 2),
            "max_drawdown":   round(_max_drawdown(df["close"]), 4),
            "avg_dollar_vol_M": round(df["dollar_volume"].mean() / 1e6, 1)
                                 if "dollar_volume" in df.columns else None,
        })
    return pd.DataFrame(rows).set_index("ticker")


def _max_drawdown(prices: pd.Series) -> float:
    """Peak-to-trough maximum drawdown."""
    rolling_max = prices.cummax()
    drawdown    = (prices - rolling_max) / rolling_max
    return drawdown.min()
