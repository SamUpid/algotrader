"""
api/models.py
Pydantic response models — define the shape of every API response.
FastAPI uses these for automatic validation, serialization, and /docs schema.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class OHLCVBar(BaseModel):
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    ret: Optional[float] = None        # "return" is a Python keyword — alias it
    dollar_volume: Optional[float] = None

    class Config:
        populate_by_name = True


class OHLCVResponse(BaseModel):
    symbol: str
    days: int
    count: int
    data: list[OHLCVBar]


class QuoteTick(BaseModel):
    time: datetime
    symbol: str
    bid: float
    ask: float
    mid: float
    spread_bps: Optional[float] = None


class QuoteResponse(BaseModel):
    symbol: str
    count: int
    data: list[QuoteTick]


class SymbolSummary(BaseModel):
    symbol: str
    trading_days: int
    date_start: str
    date_end: str
    ann_return: Optional[float] = None
    ann_volatility: Optional[float] = None
    last_close: Optional[float] = None


class SymbolListResponse(BaseModel):
    count: int
    symbols: list[str]


class SummaryResponse(BaseModel):
    count: int
    data: list[SymbolSummary]


class StatusResponse(BaseModel):
    status: str                        # "ok" or "degraded"
    db_connected: bool
    ohlcv_rows: int
    quote_ticks: int
    last_tick_symbol: Optional[str] = None
    last_tick_time: Optional[datetime] = None
    live_feed_active: bool

# ── Data Quality Models ──────────────────────────────────────────────────────

class DataQualityItem(BaseModel):
    """Quality metrics for a single symbol"""
    symbol: str
    missing_ratio: float
    missing_days: int
    spike_count: int
    max_return: float
    min_return: float
    zero_volume_days: int
    zero_volume_ratio: float
    ohlc_errors: int
    ohlc_passed: bool
    large_gaps: int
    max_gap_days: int
    has_large_gap: bool
    missing_ratio_pass: bool
    quality_score: float


class DataQualityMetadata(BaseModel):
    """Metadata for the quality report"""
    generated_at: str
    total_symbols: int
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    symbols_with_ohlc_errors: int
    symbols_with_large_gaps: int


class DataQualityResponse(BaseModel):
    """Response model for /data/quality endpoint"""
    symbols: list[DataQualityItem]
    metadata: DataQualityMetadata
