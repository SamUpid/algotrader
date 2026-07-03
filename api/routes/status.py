"""
api/routes/status.py
Health-check endpoint — confirms DB connectivity and live feed activity.
"""
import asyncpg
import os
from fastapi import APIRouter
from api.models import StatusResponse

router = APIRouter(prefix="/status", tags=["status"])

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:algotrader123@localhost:5432/algotrader",
)


@router.get("", response_model=StatusResponse)
async def get_status():
    """Check DB connection, row counts, and last live tick timestamp."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            ohlcv_rows = await conn.fetchval("SELECT COUNT(*) FROM ohlcv WHERE symbol != 'TEST'")
            quote_ticks = await conn.fetchval("SELECT COUNT(*) FROM quotes WHERE symbol NOT IN ('AAPL','MSFT') OR time > NOW() - INTERVAL '1 hour'")
            last_tick = await conn.fetchrow(
                "SELECT symbol, time FROM quotes ORDER BY time DESC LIMIT 1"
            )
        finally:
            await conn.close()

        live_active = False
        last_symbol = None
        last_time = None
        if last_tick:
            from datetime import timezone
            from datetime import datetime, timedelta
            last_time = last_tick["time"]
            last_symbol = last_tick["symbol"]
            # Consider feed "active" if a tick arrived in the last 60 seconds
            now = datetime.now(timezone.utc)
            live_active = (now - last_time.replace(tzinfo=timezone.utc)).seconds < 60

        return StatusResponse(
            status="ok",
            db_connected=True,
            ohlcv_rows=ohlcv_rows,
            quote_ticks=quote_ticks,
            last_tick_symbol=last_symbol,
            last_tick_time=last_time,
            live_feed_active=live_active,
        )
    except Exception as e:
        return StatusResponse(
            status=f"degraded: {e}",
            db_connected=False,
            ohlcv_rows=0,
            quote_ticks=0,
            live_feed_active=False,
        )
