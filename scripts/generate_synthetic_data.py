"""
scripts/generate_synthetic_data.py
────────────────────────────────────
Generates realistic synthetic OHLCV data for pipeline testing.

Uses Geometric Brownian Motion (GBM) — the same process that underlies
Black-Scholes. Each stock has:
  - Its own annualised drift (μ) and volatility (σ)
  - Market factor loading (β) — correlation with a shared market return
  - Realistic volume with daily variation

This lets you test the full pipeline without internet access.
Run this first, then run_data_pipeline.py will use the cached data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from config import UNIVERSE, BENCHMARK_TICKER, DATA_RAW_DIR, DATA_PROC_DIR, raw_path

np.random.seed(42)  # Reproducible

# ── Stock parameters (μ, σ, β, avg_volume_M) ────────────────────────────────
PARAMS = {
    "AAPL": (0.28, 0.28, 1.2, 80),    "MSFT": (0.30, 0.24, 1.1, 25),
    "GOOGL":(0.22, 0.27, 1.1, 20),    "NVDA": (0.55, 0.50, 1.5, 40),
    "META": (0.25, 0.38, 1.2, 20),    "JPM":  (0.15, 0.22, 1.0, 12),
    "BAC":  (0.12, 0.25, 1.1, 45),    "GS":   (0.14, 0.23, 1.0, 3),
    "MS":   (0.16, 0.24, 1.0, 8),     "BLK":  (0.18, 0.22, 0.9, 1),
    "JNJ":  (0.08, 0.14, 0.6, 7),     "UNH":  (0.20, 0.18, 0.7, 3),
    "PFE":  (0.05, 0.20, 0.7, 25),    "ABBV": (0.14, 0.22, 0.7, 8),
    "MRK":  (0.12, 0.17, 0.6, 10),    "AMZN": (0.28, 0.32, 1.2, 35),
    "WMT":  (0.10, 0.15, 0.5, 6),     "HD":   (0.18, 0.22, 0.9, 4),
    "BA":   (0.05, 0.38, 1.0, 5),     "XOM":  (0.08, 0.28, 0.8, 15),
    "SPY":  (0.12, 0.16, 1.0, 80),
}

START = "2018-01-01"
END   = "2024-12-31"
dt    = 1 / 252  # Daily time step

def gbm_ohlcv(ticker: str, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate GBM OHLCV with realistic intraday variation."""
    mu, sigma, beta, avg_vol_M = PARAMS.get(ticker, (0.10, 0.20, 1.0, 5))
    n = len(trading_days)

    # Market factor: shared component driving correlations
    market_shocks = np.random.normal(0, sigma * 0.7, n)

    # Idiosyncratic shocks
    idio_shocks = np.random.normal(0, sigma * np.sqrt(1 - 0.5), n)

    # Daily log returns: drift + market exposure + idio
    daily_log_ret = (mu - 0.5 * sigma**2) * dt \
                    + beta * market_shocks * np.sqrt(dt) \
                    + idio_shocks * np.sqrt(dt)

    # Close prices via cumulative sum of log returns
    S0 = 100.0
    close = S0 * np.exp(np.cumsum(daily_log_ret))

    # Intraday variation: open/high/low relative to close
    intraday_vol = sigma * np.sqrt(dt) * 0.5
    open_  = close * np.exp(np.random.normal(0, intraday_vol, n))
    high_offset = np.abs(np.random.normal(0, intraday_vol, n))
    low_offset  = np.abs(np.random.normal(0, intraday_vol, n))
    high  = np.maximum(close, open_) * (1 + high_offset)
    low   = np.minimum(close, open_) * (1 - low_offset)

    # Volume: log-normal with day-of-week effect
    base_vol = avg_vol_M * 1e6
    dow_factor = np.array([1.1, 1.05, 1.0, 1.0, 0.85])  # Mon high, Fri low
    dow_weights = np.array([dow_factor[d.weekday()] for d in trading_days])
    volume = (base_vol * dow_weights
              * np.random.lognormal(0, 0.3, n)).astype(int)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=trading_days)
    df.index.name = "date"
    return df.round(4)


def main():
    trading_days = pd.bdate_range(START, END)
    tickers = UNIVERSE + [BENCHMARK_TICKER]

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic OHLCV for {len(tickers)} tickers "
          f"({len(trading_days)} trading days)...")

    for ticker in tickers:
        df = gbm_ohlcv(ticker, trading_days)
        out = raw_path(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        print(f"  {ticker:6s}: {len(df)} rows, "
              f"close ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")

    print(f"\nSynthetic data written to data/raw/")
    print("Now run: python scripts/run_data_pipeline.py")


if __name__ == "__main__":
    main()
