"""
data/live_feed.py - Alpaca WebSocket live quote feed.
Runs StockDataStream in a thread (it manages its own event loop),
while the asyncio flush loop writes buffered ticks to TimescaleDB.
"""
import asyncio
import logging
import os
import threading
from datetime import datetime, timezone

from dotenv import load_dotenv
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from data.database import upsert_quotes
from config import UNIVERSE

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

_buffer: list[dict] = []
FLUSH_INTERVAL = 5  # seconds


async def handle_quote(quote) -> None:
    bid = float(quote.bid_price or 0)
    ask = float(quote.ask_price or 0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return
    mid = (bid + ask) / 2.0
    spread_bps = (ask - bid) / mid * 10_000
    ts = quote.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    _buffer.append({
        "time":       ts,
        "symbol":     quote.symbol,
        "bid":        bid,
        "ask":        ask,
        "mid":        mid,
        "spread_bps": spread_bps,
    })
    log.debug(f"{quote.symbol} mid={mid:.4f} spread={spread_bps:.1f}bps")


async def flush_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL)
        if not _buffer:
            continue
        batch = _buffer.copy()
        _buffer.clear()
        try:
            n = await upsert_quotes(batch)
            log.info(f"Flushed {n} ticks to TimescaleDB")
        except Exception as e:
            log.error(f"DB flush failed: {e}")
            _buffer[:0] = batch


def _run_stream(tickers):
    """Run the Alpaca stream in a background thread (it has its own event loop)."""
    while True:
        try:
            log.info(f"Connecting to Alpaca IEX feed for {len(tickers)} tickers...")
            stream = StockDataStream(
                api_key=API_KEY,
                secret_key=API_SECRET,
                feed=DataFeed.IEX,
            )
            stream.subscribe_quotes(handle_quote, *tickers)
            stream.run()  # blocking — runs until disconnected
        except Exception as e:
            log.warning(f"Stream error: {e}. Reconnecting in 5s...")
            import time; time.sleep(5)


async def main():
    tickers = [t for t in UNIVERSE if t != "SPY"]
    t = threading.Thread(target=_run_stream, args=(tickers,), daemon=True)
    t.start()
    await flush_loop()  # runs forever


async def _mock_test():
    fake = [
        {"time": datetime.now(timezone.utc), "symbol": "AAPL",
         "bid": 189.50, "ask": 189.52, "mid": 189.51, "spread_bps": 1.05},
        {"time": datetime.now(timezone.utc), "symbol": "MSFT",
         "bid": 415.20, "ask": 415.23, "mid": 415.215, "spread_bps": 0.72},
    ]
    n = await upsert_quotes(fake)
    print(f"Mock inserted {n} rows")


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_API_SECRET not set in .env")
    asyncio.run(main())
    # asyncio.run(_mock_test())
