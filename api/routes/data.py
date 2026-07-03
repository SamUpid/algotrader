"""
api/routes/data.py
OHLCV + quote endpoints. All DB calls use raw asyncpg (same pattern as database.py).
"""
import asyncpg
import os
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from api.models import (
    OHLCVBar, OHLCVResponse,
    QuoteTick, QuoteResponse,
    SymbolListResponse, SummaryResponse, SymbolSummary,
    DataQualityResponse, DataQualityItem, DataQualityMetadata,
)

router = APIRouter(prefix="/data", tags=["data"])

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:algotrader123@localhost:5432/algotrader",
)


async def get_conn():
    return await asyncpg.connect(DATABASE_URL)


# ── GET /data/symbols ────────────────────────────────────────────────────────

@router.get("/symbols", response_model=SymbolListResponse)
async def get_symbols():
    """Return all distinct ticker symbols in the ohlcv table."""
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
        )
        symbols = [r["symbol"] for r in rows]
        return SymbolListResponse(count=len(symbols), symbols=symbols)
    finally:
        await conn.close()


# ── GET /data/ohlcv ──────────────────────────────────────────────────────────

@router.get("/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str = Query(..., description="Ticker symbol e.g. AAPL"),
    days:   int = Query(30,  description="Number of most recent trading days", ge=1, le=1760),
):
    """Return the last N trading days of OHLCV data for a symbol."""
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT time, symbol, open, high, low, close, volume,
                   "return", dollar_volume
            FROM ohlcv
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT $2
            """,
            symbol.upper(), days,
        )
        if not rows:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        bars = [
            OHLCVBar(
                time=r["time"],
                symbol=r["symbol"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
                ret=r["return"],
                dollar_volume=r["dollar_volume"],
            )
            for r in rows
        ]
        return OHLCVResponse(symbol=symbol.upper(), days=days, count=len(bars), data=bars)
    finally:
        await conn.close()


# ── GET /data/quotes ─────────────────────────────────────────────────────────

@router.get("/quotes", response_model=QuoteResponse)
async def get_quotes(
    symbol: str = Query(..., description="Ticker symbol e.g. AAPL"),
    limit:  int = Query(100, description="Max ticks to return", ge=1, le=10000),
):
    """Return the most recent quote ticks for a symbol."""
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT time, symbol, bid, ask, mid, spread_bps
            FROM quotes
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT $2
            """,
            symbol.upper(), limit,
        )
        ticks = [
            QuoteTick(
                time=r["time"],
                symbol=r["symbol"],
                bid=r["bid"],
                ask=r["ask"],
                mid=r["mid"],
                spread_bps=r["spread_bps"],
            )
            for r in rows
        ]
        return QuoteResponse(symbol=symbol.upper(), count=len(ticks), data=ticks)
    finally:
        await conn.close()


# ── GET /data/summary ────────────────────────────────────────────────────────

@router.get("/summary", response_model=SummaryResponse)
async def get_summary():
    """Return per-symbol statistics: date range, annualised return + vol, last close."""
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT
                symbol,
                COUNT(*)                    AS trading_days,
                MIN(time)::date::text       AS date_start,
                MAX(time)::date::text       AS date_end,
                EXP(SUM(LN(1 + "return"))) - 1   AS total_return,
                STDDEV("return") * SQRT(252)      AS ann_volatility,
                (array_agg(close ORDER BY time DESC))[1] AS last_close
            FROM ohlcv
            WHERE symbol != 'TEST'
            GROUP BY symbol
            ORDER BY symbol
            """
        )
        data = []
        for r in rows:
            tr = r["total_return"] or 0
            days = r["trading_days"] or 1
            years = days / 252
            ann_ret = (1 + tr) ** (1 / years) - 1 if years > 0 else 0
            data.append(SymbolSummary(
                symbol=r["symbol"],
                trading_days=r["trading_days"],
                date_start=r["date_start"],
                date_end=r["date_end"],
                ann_return=round(ann_ret, 4),
                ann_volatility=round(r["ann_volatility"] or 0, 4),
                last_close=round(r["last_close"] or 0, 2),
            ))
        return SummaryResponse(count=len(data), data=data)
    finally:
        await conn.close()


# ── GET /data/quality ───────────────────────────────────────────────────────

@router.get("/quality", response_model=DataQualityResponse)
async def get_quality_report(
    limit: Optional[int] = Query(None, description="Limit number of symbols returned", ge=1, le=100)
):
    """
    Return the data quality report.
    
    This reads the quality_report.csv generated by data/quality.py.
    Shows quality metrics for each symbol including:
    - Missing bars ratio
    - Return spike counts
    - Zero volume days
    - OHLC sanity violations
    - Price continuity gaps
    - Overall quality score (0-100)
    
    Used by the dashboard to monitor data feed health.
    """
    from pathlib import Path
    import pandas as pd
    
    from config import DATA_PROC_DIR
    
    quality_path = DATA_PROC_DIR / "quality_report.csv"
    
    if not quality_path.exists():
        try:
            from data.quality import generate_quality_report
            report_df = generate_quality_report()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Quality report not found and generation failed: {str(e)}"
            )
    else:
        report_df = pd.read_csv(quality_path)
    
    if limit:
        report_df = report_df.head(limit)
    
    symbols = []
    for _, row in report_df.iterrows():
        symbols.append(DataQualityItem(
            symbol=row['symbol'],
            missing_ratio=row['missing_ratio'],
            missing_days=int(row['missing_days']),
            spike_count=int(row['spike_count']),
            max_return=row['max_return'],
            min_return=row['min_return'],
            zero_volume_days=int(row['zero_volume_days']),
            zero_volume_ratio=row['zero_volume_ratio'],
            ohlc_errors=int(row['ohlc_errors']),
            ohlc_passed=row['ohlc_passed'],
            large_gaps=int(row['large_gaps']),
            max_gap_days=int(row['max_gap_days']),
            has_large_gap=row['has_large_gap'],
            missing_ratio_pass=row['missing_ratio_pass'],
            quality_score=row['quality_score']
        ))
    
    metadata = DataQualityMetadata(
        generated_at=report_df.attrs.get('generated_at', datetime.now().isoformat()),
        total_symbols=len(report_df),
        avg_quality_score=float(report_df['quality_score'].mean()),
        min_quality_score=float(report_df['quality_score'].min()),
        max_quality_score=float(report_df['quality_score'].max()),
        symbols_with_ohlc_errors=int(len(report_df[report_df['ohlc_errors'] > 0])),
        symbols_with_large_gaps=int(len(report_df[report_df['large_gaps'] > 0]))
    )
    
    return DataQualityResponse(symbols=symbols, metadata=metadata)
