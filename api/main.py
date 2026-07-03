"""
api/main.py
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import signals
from api.routes.data import router as data_router
from api.routes.status import router as status_router
from api.routes.portfolio import router as portfolio_router 

app = FastAPI(
    title="Algo Trading Platform API",
    description="OHLCV data, live quotes, and system status for the algotrader portfolio project.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # Vite ports
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