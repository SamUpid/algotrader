"""
Central configuration for the algo trading platform.
All magic numbers live here — never hardcode them elsewhere.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_RAW_DIR   = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR  = ROOT_DIR / "data" / "processed"
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"

# ── Universe ───────────────────────────────────────────────────────────────
# Start with a manageable 20-stock universe across sectors.
# Easy to expand later. Avoid tickers that were added to S&P 500 recently
# (survivorship bias) — these are long-established names.
UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    # Financials
    "JPM", "BAC", "GS", "MS", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    # Consumer / Industrial
    "AMZN", "WMT", "HD", "BA", "XOM",
]

# ── Data parameters ────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2018-01-01"   # 6+ years of history
DEFAULT_END_DATE   = "2024-12-31"   # fixed end for reproducible backtests
BENCHMARK_TICKER   = "SPY"          # S&P 500 ETF — used for alpha/beta calcs

# Intervals supported by yfinance for historical data
VALID_INTERVALS = ["1d", "1wk", "1mo"]
DEFAULT_INTERVAL = "1d"

# ── Data quality thresholds ────────────────────────────────────────────────
MAX_DAILY_RETURN   = 0.25   # flag any |return| > 25% in a single day
MIN_DAILY_VOLUME   = 100_000  # ignore days with suspiciously low volume
MAX_MISSING_RATIO  = 0.05   # error if >2% of trading days missing per ticker

# ── Trading calendar ───────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252

# ── Paths helpers ──────────────────────────────────────────────────────────
def raw_path(ticker: str, interval: str = "1d") -> Path:
    """Where raw downloaded data lives."""
    return DATA_RAW_DIR / f"{ticker}_{interval}.parquet"

def processed_path(ticker: str, interval: str = "1d") -> Path:
    """Where cleaned data lives."""
    return DATA_PROC_DIR / f"{ticker}_{interval}.parquet"
