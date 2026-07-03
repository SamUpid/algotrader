# data/database.py
import os
import asyncpg
from datetime import timezone

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:algotrader123@localhost:5432/algotrader",
)


async def upsert_ohlcv(rows: list[dict]) -> int:
    """Bulk upsert OHLCV rows using raw asyncpg."""
    if not rows:
        return 0
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        records = [
            (
                r["time"], r["symbol"],
                r["open"], r["high"], r["low"], r["close"],
                r["volume"], r["return"], r["dollar_volume"],
            )
            for r in rows
        ]
        await conn.executemany(
            """
            INSERT INTO ohlcv (time, symbol, open, high, low, close, volume, return, dollar_volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (time, symbol) DO NOTHING
            """,
            records,
        )
        return len(records)
    finally:
        await conn.close()


async def upsert_quotes(rows: list[dict]) -> int:
    """Bulk upsert quote ticks using raw asyncpg."""
    if not rows:
        return 0
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        records = [
            (r["time"], r["symbol"], r["bid"], r["ask"], r["mid"], r["spread_bps"])
            for r in rows
        ]
        await conn.executemany(
            """
            INSERT INTO quotes (time, symbol, bid, ask, mid, spread_bps)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (time, symbol) DO NOTHING
            """,
            records,
        )
        return len(records)
    finally:
        await conn.close()


async def close_engine():
    pass  # no pool to close with per-call connections