"""
api/main.py
FastAPI application entry point.
"""
import os
import base64
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import signals
from api.routes.data import router as data_router
from api.routes.status import router as status_router
from api.routes.portfolio import router as portfolio_router 

# Import scheduler
from api.scheduler import start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Algo Trading Platform API",
    description="OHLCV data, live quotes, and system status for the algotrader portfolio project.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:5174",
        "https://algotrader.vercel.app",
        "https://algotrader-git-main.vercel.app",
        "https://algotrader-*.vercel.app",
        "https://frontend-two-phi-84.vercel.app",
        "https://*.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(status_router)
app.include_router(signals.router)
app.include_router(portfolio_router) 


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Algo Trading API — see /docs"}


@app.on_event("startup")
async def startup_event():
    """
    Run on application startup:
    1. Load trade ledger from environment variable (if available)
    2. Start the scheduler
    """
    # Step 1: Load trade ledger from environment variable
    encoded = os.getenv("TRADE_LEDGER_BASE64")
    if encoded:
        try:
            decoded = base64.b64decode(encoded).decode('utf-8')
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/trade_ledger.csv", "w") as f:
                f.write(decoded)
            logger.info("✅ Trade ledger loaded from environment")
        except Exception as e:
            logger.error(f"❌ Failed to load trade ledger: {e}")
    else:
        logger.info("ℹ️ No TRADE_LEDGER_BASE64 environment variable found")
    
    # Step 2: Start scheduler
    start_scheduler()
    logger.info("✅ Application started with scheduler")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the scheduler on application shutdown."""
    stop_scheduler()
    logger.info("🔴 Application shutting down")