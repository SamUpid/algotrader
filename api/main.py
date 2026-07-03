"""
api/main.py
FastAPI application entry point.
"""
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
        "https://algotrader.vercel.app",  # Vercel preview
        "https://algotrader-git-main.vercel.app",
        "https://algotrader-*.vercel.app",  # All preview deployments
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
    """Start the scheduler on application startup."""
    start_scheduler()
    logger.info("Application started with scheduler")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the scheduler on application shutdown."""
    stop_scheduler()
    logger.info("Application shutting down")