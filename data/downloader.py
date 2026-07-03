"""
data/downloader.py
──────────────────
Downloads and caches OHLCV data from Yahoo Finance.

Key design decisions:
- Always use 'auto_adjust=True' → prices reflect splits + dividends.
  This is critical for backtesting: without adjustment, a 2:1 split
  looks like a 50% overnight drop, destroying your signal.
- Cache to Parquet on disk. Parquet is columnar (fast reads), typed
  (no dtype guessing), and 10× smaller than CSV.
- Validate immediately after download — bad data in = bad signals out.
"""

import logging
import time
import asyncio
from data.database import upsert_ohlcv
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    DATA_RAW_DIR, DATA_PROC_DIR,
    MAX_DAILY_RETURN, MIN_DAILY_VOLUME, MAX_MISSING_RATIO,
    raw_path, processed_path,
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_INTERVAL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Download ────────────────────────────────────────────────────────────────

def download_ticker(
    ticker: str,
    start: str = DEFAULT_START_DATE,
    end: str   = DEFAULT_END_DATE,
    interval: str = DEFAULT_INTERVAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker.

    Uses disk cache — won't re-download unless force_refresh=True.
    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (timezone-naive, date only)
    """
    cache_file = raw_path(ticker, interval)

    if cache_file.exists() and not force_refresh:
        log.info(f"  Cache hit: {ticker} ← {cache_file.name}")
        return pd.read_parquet(cache_file)

    log.info(f"  Downloading: {ticker} ({start} → {end}, {interval})")

    try:
        # auto_adjust=True applies split/dividend adjustments to all OHLCV.
        # This gives you the "true" economic return, which is what you want.
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,  # flatten multi-index for single ticker
        )
    except Exception as e:
        log.error(f"  Download failed for {ticker}: {e}")
        raise

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Standardise column names to lowercase
    raw.columns = [c.lower() for c in raw.columns]

    # Keep only the columns we need
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    raw = raw[[c for c in ohlcv_cols if c in raw.columns]].copy()

    # Strip timezone from index (keeps it as date, not datetime)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"

    # Drop rows where ALL price columns are NaN (occasionally happens at
    # the very start/end of a ticker's trading history)
    raw = raw.dropna(subset=["open", "high", "low", "close"], how="all")

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_file)
    log.info(f"  Saved {len(raw)} rows → {cache_file.name}")

    return raw

def _write_to_db(df: pd.DataFrame, ticker: str) -> None:
    """Convert a clean OHLCV DataFrame to rows and write to TimescaleDB in chunks."""
    rows = []
    for ts, row in df.iterrows():
        try:
            vol = int(row["volume"]) if pd.notna(row["volume"]) else 0
        except (ValueError, TypeError):
            vol = 0
        rows.append({
            "time": ts.tz_localize("UTC").to_pydatetime(),
            "symbol": ticker,
            "open":  float(row["open"]) if pd.notna(row["open"]) else None,
            "high":  float(row["high"]) if pd.notna(row["high"]) else None,
            "low":   float(row["low"])  if pd.notna(row["low"])  else None,
            "close": float(row["close"]) if pd.notna(row["close"]) else None,
            "volume": vol,
            "return": float(row["return"]) if pd.notna(row["return"]) else None,
            "dollar_volume": float(row["dollar_volume"]) if pd.notna(row["dollar_volume"]) else None,
        })
    chunk_size = 500
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        total += asyncio.run(upsert_ohlcv(chunk))
    log.info(f"  [{ticker}] → TimescaleDB: {total} rows inserted")

def download_universe(
    tickers: list[str],
    start: str = DEFAULT_START_DATE,
    end: str   = DEFAULT_END_DATE,
    interval: str = DEFAULT_INTERVAL,
    force_refresh: bool = False,
    sleep_between: float = 0.3,   # be polite to Yahoo's servers
) -> dict[str, pd.DataFrame]:
    """
    Download an entire universe of tickers.
    Returns dict: {ticker: DataFrame}
    Failed tickers are skipped with a warning, not an exception.
    """
    results = {}
    failed  = []

    for i, ticker in enumerate(tickers):
        try:
            df = download_ticker(ticker, start, end, interval, force_refresh)
            results[ticker] = df
        except Exception as e:
            log.warning(f"  Skipping {ticker}: {e}")
            failed.append(ticker)

        if i < len(tickers) - 1:
            time.sleep(sleep_between)

    log.info(f"\nDownload complete: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        log.warning(f"Failed tickers: {failed}")

    return results


# ── Validation ──────────────────────────────────────────────────────────────

class DataQualityError(Exception):
    """Raised when data fails quality checks and cannot be used safely."""
    pass


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> dict:
    """
    Run quality checks on OHLCV data. Returns a report dict.
    Raises DataQualityError if data is unusable.

    Checks:
    1. OHLC sanity: high >= low, high >= open/close, low <= open/close
    2. No negative prices
    3. Extreme return spikes (possible bad data or unadjusted splits)
    4. Missing data ratio
    5. Volume anomalies
    """
    report = {"ticker": ticker, "rows": len(df), "issues": [], "warnings": []}

    # ── 1. OHLC sanity ──────────────────────────────────────────────────────
    bad_high = (df["high"] < df["low"]).sum()
    if bad_high > 0:
        report["issues"].append(f"{bad_high} rows where high < low")

    bad_close_vs_high = (df["close"] > df["high"] * 1.001).sum()
    if bad_close_vs_high > 0:
        report["issues"].append(f"{bad_close_vs_high} rows where close > high")

    bad_close_vs_low = (df["close"] < df["low"] * 0.999).sum()
    if bad_close_vs_low > 0:
        report["issues"].append(f"{bad_close_vs_low} rows where close < low")

    # ── 2. Negative prices ──────────────────────────────────────────────────
    neg_prices = (df[["open", "high", "low", "close"]] < 0).any(axis=1).sum()
    if neg_prices > 0:
        report["issues"].append(f"{neg_prices} rows with negative prices")

    # ── 3. Return spikes ────────────────────────────────────────────────────
    daily_returns = df["close"].pct_change()
    spikes = (daily_returns.abs() > MAX_DAILY_RETURN).sum()
    if spikes > 0:
        # This could be a missed split adjustment or genuinely extreme event
        spike_dates = daily_returns[daily_returns.abs() > MAX_DAILY_RETURN].index.tolist()
        report["warnings"].append(
            f"{spikes} return spikes >25% on: "
            + ", ".join(str(d.date()) for d in spike_dates[:5])
            + (" ..." if spikes > 5 else "")
        )

    # ── 4. Missing data ─────────────────────────────────────────────────────
    # Build a complete trading day calendar from the date range
    expected_days = pd.bdate_range(df.index.min(), df.index.max())
    actual_days   = df.index
    missing_days  = expected_days.difference(actual_days)
    missing_ratio = len(missing_days) / len(expected_days) if len(expected_days) > 0 else 0

    report["missing_days"]  = len(missing_days)
    report["missing_ratio"] = round(missing_ratio, 4)

    if missing_ratio > MAX_MISSING_RATIO:
        report["issues"].append(
            f"Missing {len(missing_days)} trading days "
            f"({missing_ratio:.1%} of expected) — check for halts or delistings"
        )
    elif len(missing_days) > 0:
        report["warnings"].append(
            f"{len(missing_days)} missing days (within tolerance)"
        )

    # ── 5. Volume anomalies ─────────────────────────────────────────────────
    low_vol_days = (df["volume"] < MIN_DAILY_VOLUME).sum()
    zero_vol_days = (df["volume"] == 0).sum()
    if zero_vol_days > 0:
        report["warnings"].append(f"{zero_vol_days} days with zero volume")
    if low_vol_days > 0:
        report["warnings"].append(f"{low_vol_days} days with volume < {MIN_DAILY_VOLUME:,}")

    # ── 6. Date range coverage ─────────────────────────────────────────────
    report["date_start"] = str(df.index.min().date())
    report["date_end"]   = str(df.index.max().date())

    # Raise if any hard issues found
    if report["issues"]:
        raise DataQualityError(
            f"{ticker} failed quality checks:\n  " + "\n  ".join(report["issues"])
        )

    return report


# ── Cleaning ────────────────────────────────────────────────────────────────

def clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean raw OHLCV data. Returns cleaned DataFrame.

    Steps:
    1. Forward-fill missing prices (max 3 days — market closure, not delist)
    2. Remove rows with zero volume (typically non-trading days that slipped in)
    3. Add derived columns: daily return, log return, dollar volume
    4. Sort by date (should already be sorted, but verify)
    """
    df = df.copy().sort_index()

    # Forward-fill prices for short gaps (e.g. market holiday data quirks)
    # Limit=3 means we won't fill more than 3 consecutive missing days
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].ffill(limit=3)

    # Remove rows where we still have NaN prices after ffill
    df = df.dropna(subset=["close"])

    # Remove zero-volume rows (non-trading days that leaked through)
    df = df[df["volume"] > 0]

    # ── Add derived columns ─────────────────────────────────────────────────
    # Simple return: (Pₜ - Pₜ₋₁) / Pₜ₋₁
    df["return"] = df["close"].pct_change()

    # Log return: ln(Pₜ / Pₜ₋₁)
    # Log returns are additive over time and better-behaved statistically.
    # Most academic papers use log returns. Simple returns are used for
    # portfolio P&L calculation (can't aggregate log returns across assets).
    import numpy as np
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Dollar volume: how much $ traded. Better liquidity proxy than raw volume.
    # Used in market impact model: ΔP ∝ √(order_size / ADV)
    df["dollar_volume"] = df["close"] * df["volume"]

    # Drop the first row (NaN returns from pct_change)
    df = df.dropna(subset=["return"])

    return df


# ── Pipeline: download + validate + clean ──────────────────────────────────

def get_clean_data(
    ticker: str,
    start: str = DEFAULT_START_DATE,
    end:   str = DEFAULT_END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline for one ticker:
    download → validate → clean → save processed → return.

    This is the function you'll call from everywhere else.
    """
    proc_file = processed_path(ticker)

    if proc_file.exists() and not force_refresh:
        clean = pd.read_parquet(proc_file)
        _write_to_db(clean, ticker)
        return clean

    # Download raw
    raw = download_ticker(ticker, start, end, force_refresh=force_refresh)

    # Validate (raises DataQualityError on hard failures)
    try:
        report = validate_ohlcv(raw, ticker)
        if report["warnings"]:
            for w in report["warnings"]:
                log.warning(f"  [{ticker}] {w}")
        log.info(
            f"  [{ticker}] OK — {report['rows']} rows, "
            f"{report['date_start']} → {report['date_end']}, "
            f"{report['missing_days']} missing days"
        )
    except DataQualityError as e:
        log.error(str(e))
        raise

    # Clean
    clean = clean_ohlcv(raw, ticker)

    # Save processed
    proc_file.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(proc_file)
    # Write to TimescaleDB
    _write_to_db(clean, ticker)
    return clean


def get_clean_universe(
    tickers: list[str],
    start: str = DEFAULT_START_DATE,
    end:   str = DEFAULT_END_DATE,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run the full pipeline for every ticker in the universe.
    Returns {ticker: clean_df}. Failed tickers logged, not raised.
    """
    log.info(f"Processing {len(tickers)} tickers...")
    results = {}
    failed  = []

    for ticker in tickers:
        try:
            df = get_clean_data(ticker, start, end, force_refresh)
            results[ticker] = df
        except Exception as e:
            log.error(f"  [{ticker}] FAILED: {e}")
            failed.append((ticker, str(e)))

    log.info(
        f"\nPipeline complete: {len(results)} succeeded, {len(failed)} failed"
    )
    if failed:
        for t, err in failed:
            log.warning(f"  {t}: {err}")

    return results
