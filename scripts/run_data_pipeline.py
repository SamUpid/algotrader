"""
scripts/run_data_pipeline.py
────────────────────────────
Entry point: download → validate → clean → build matrices.

Run from the project root:
  python scripts/run_data_pipeline.py

This is your "day 1" script. By the end of running this you have:
- Clean OHLCV parquet files for all 20 tickers
- A price matrix ready for feature engineering
- A universe summary with key stats
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from config import UNIVERSE, BENCHMARK_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE
from data.downloader import get_clean_universe, get_clean_data
from data.universe import (
    build_price_matrix,
    build_returns_matrix,
    build_volume_matrix,
    compute_adv,
    universe_summary,
)


def main():
    print("=" * 60)
    print("  Algo Trading Platform — Data Pipeline")
    print("=" * 60)

    # ── Step 1: Download + clean all tickers ────────────────────────────────
    print(f"\n[1/4] Downloading universe ({len(UNIVERSE)} tickers)...")
    print(f"      Date range: {DEFAULT_START_DATE} → {DEFAULT_END_DATE}\n")

    universe_data = get_clean_universe(
        UNIVERSE,
        start=DEFAULT_START_DATE,
        end=DEFAULT_END_DATE,
    )

    # Also fetch benchmark for later alpha/beta calculations
    print(f"\n      Fetching benchmark ({BENCHMARK_TICKER})...")
    benchmark = get_clean_data(BENCHMARK_TICKER)

    print(f"\n[1/4] Done. {len(universe_data)}/{len(UNIVERSE)} tickers loaded.")

    # ── Step 2: Build aligned matrices ──────────────────────────────────────
    print("\n[2/4] Building aligned price and returns matrices...")

    prices  = build_price_matrix(universe_data)
    returns = build_returns_matrix(prices)
    volumes = build_volume_matrix(universe_data)
    adv     = compute_adv(volumes, window=20)

    print(f"      Price matrix shape:   {prices.shape}  (dates × tickers)")
    print(f"      Returns matrix shape: {returns.shape}")
    print(f"      Date range:  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"      Tickers:     {list(prices.columns)}")

    # Save matrices
    from config import DATA_PROC_DIR
    prices.to_parquet(DATA_PROC_DIR / "price_matrix.parquet")
    returns.to_parquet(DATA_PROC_DIR / "returns_matrix.parquet")
    volumes.to_parquet(DATA_PROC_DIR / "volume_matrix.parquet")
    adv.to_parquet(DATA_PROC_DIR / "adv_matrix.parquet")
    print("\n      Matrices saved to data/processed/")

    # ── Step 3: Universe summary ─────────────────────────────────────────────
    print("\n[3/4] Universe summary:")
    summary = universe_summary(universe_data)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(summary.to_string())

    # ── Step 4: Sanity checks ────────────────────────────────────────────────
    print("\n[4/4] Sanity checks:")

    # Check 1: Correlation matrix (should be positive, equities tend to move together)
    corr = returns.corr()
    avg_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).mean().mean()
    print(f"      Average pairwise correlation: {avg_corr:.3f}  (expected: 0.3–0.7 for equities)")

    # Check 2: Annualised vol range
    ann_vols = returns.std() * np.sqrt(252)
    print(f"      Annualised vol range: {ann_vols.min():.1%} – {ann_vols.max():.1%}  (expected: 15%–60%)")

    # Check 3: Any NaN in the return matrix?
    nan_count = returns.isna().sum().sum()
    status = "PASS" if nan_count == 0 else f"WARNING: {nan_count} NaN values"
    print(f"      NaN in returns matrix: {status}")

    # Check 4: Benchmark correlation
    bench_ret = benchmark["return"].reindex(returns.index).dropna()
    bench_corr = returns.corrwith(bench_ret).mean()
    print(f"      Avg correlation with {BENCHMARK_TICKER}: {bench_corr:.3f}  (expected: 0.5–0.9)")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Data ready for feature engineering.")
    print("  Next step: features/indicators.py")
    print("=" * 60)

    return {
        "universe_data": universe_data,
        "prices":        prices,
        "returns":       returns,
        "volumes":       volumes,
        "adv":           adv,
        "benchmark":     benchmark,
        "summary":       summary,
    }


if __name__ == "__main__":
    main()
