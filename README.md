# AlgoTrader

A quantitative trading research platform built to demonstrate the full pipeline of a systematic strategy — from raw price data to live paper execution. Every design decision here maps to how production quant systems at firms like BlackRock and Morgan Stanley actually work.

**Live dashboard:** [algotrader-tau.vercel.app](https://frontend-two-phi-84.vercel.app)  

---

## Results

Walk-forward backtest on 20 large-cap US stocks, 2018–2024.
Train window: 252 trading days. Test window: 63 days. No look-ahead bias.

| Metric | Strategy | SPY benchmark |
|--------|----------|---------------|
| Sharpe ratio | **1.20** | 0.74 |
| Sortino ratio | **1.80** | 0.98 |
| Calmar ratio | **0.90** | 0.52 |
| Max drawdown | -8.3% | -33.7% |
| Hit rate | 52.0% | — |
| Annual return | 19.9% | 13.2% |

All numbers are out-of-sample. The walk-forward loop re-trains the LightGBM model on each expanding window — the backtest never touches future data during training or normalisation.

<img width="1702" alt="AlgoTrader Dashboard" src="https://github.com/user-attachments/assets/2d63691e-822e-434c-982e-93571ee139a5" />

## Architecture

```
yfinance / Alpaca WebSocket
        ↓
TimescaleDB  (hypertable, partitioned by time)
        ↓
Feature engine  —  17 indicators + 3 WorldQuant Formulaic Alphas, all from scratch
        ↓
Alpha model  —  LightGBM with triple-barrier labeling (Lopez de Prado)
        ↓
Signal ensemble  —  momentum + RSI mean-reversion + ML, cross-sectional rank
        ↓
Risk gate  —  parametric VaR check → 5% drawdown circuit breaker → position limits
        ↓
Order router  —  square-root market impact model → Alpaca paper trading API
        ↓
FastAPI  →  React dashboard  (live equity curve, positions, risk panel, tearsheet)
```

---

## Stack

| Layer | Tools |
|-------|-------|
| Data ingestion | yfinance, Alpaca WebSocket, TimescaleDB, asyncpg |
| Features | numpy, pandas — implemented from scratch, no ta-lib |
| ML alpha | LightGBM, triple-barrier labeling, purged walk-forward CV |
| Risk engine | Parametric VaR, CVaR, circuit breaker, position limits |
| Execution | Square-root market impact model, Alpaca paper trading API |
| API | FastAPI, Pydantic v2, APScheduler (daily 9:35am ET signal run) |
| Dashboard | React, TypeScript, Vite, Tailwind CSS, Recharts |
| Deployment | Render (backend + TimescaleDB), Vercel (frontend) |

---

## Key design decisions

**Walk-forward validation with a purge gap**

Standard k-fold cross-validation leaks future prices into training labels when the label horizon overlaps the fold boundary. The purge gap removes the 5 days before each test window from the training set — following Lopez de Prado's methodology from *Advances in Financial Machine Learning*. This is why the backtest Sharpe is lower than naive cross-validation would show, and why it's more trustworthy.

**Pre-trade VaR gate, not a dashboard metric**

The risk engine sits in the execution path and rejects orders before they're sent to Alpaca. If a proposed trade would push portfolio VaR (95%, 1-day) above 2%, the order is blocked and written to a risk event log. Most portfolio projects compute VaR after the fact as a chart — this one uses it as an actual blocking gate. Same for the circuit breaker: at 5% drawdown from peak NAV, all new orders are halted until manually reset.

**Indicators implemented from scratch**

RSI uses Wilder's smoothing method (alpha = 1/14), not standard EMA. ATR does the same. This matters because ta-lib's defaults differ from what the original papers describe — implementing from scratch means every parameter can be explained and defended in an interview. The three WorldQuant Formulaic Alphas (alpha_001, alpha_002, alpha_012) are implemented directly from the published SSRN paper.

**Square-root market impact model**

Execution price = `mid_price × (1 + σ × sign(Q) × √(|Q|/ADV))` where ADV is 20-day average dollar volume and σ is 20-day rolling volatility. Slippage is measured per fill and logged against expected price. The trade ledger tracks realised vs expected cost so transaction cost assumptions can be validated over time.

**Triple-barrier labeling**

Instead of labeling returns as +1/-1 based on a fixed sign, labels are assigned using three barriers: a profit-taking barrier, a stop-loss barrier, and a time barrier. This reduces label noise from market microstructure and aligns ML targets with actual trading logic — not just whether the price went up.

---

## Repo structure

```
algotrader/
├── config.py           # universe (20 tickers), constants, path helpers
├── data/
│   ├── downloader.py   # yfinance → validate → clean → Parquet
│   ├── universe.py     # aligned price/returns/volume matrices
│   ├── database.py     # asyncpg bulk upsert into TimescaleDB
│   ├── live_feed.py    # Alpaca WebSocket → quotes table
│   └── quality.py      # data quality checks → quality_report.csv
├── features/
│   └── indicators.py   # 17 indicators + 3 Formulaic Alphas
├── signals/
│   ├── backtester.py   # vectorized engine + walk-forward + purge gap
│   ├── metrics.py      # Sharpe, Sortino, Calmar, drawdown, hit rate
│   ├── alpha_model.py  # LightGBM with triple-barrier labeling
│   └── ensemble.py     # signal combination, IC decay analysis
├── risk/
│   ├── var_engine.py   # parametric VaR, CVaR, pre-trade gate
│   └── circuit_breaker.py  # drawdown halt + position limits
├── execution/
│   └── order_router.py # market impact model, Alpaca execution, ledger
├── api/
│   ├── main.py         # FastAPI + CORS + APScheduler
│   └── routes/         # data, portfolio, signals, risk, status
├── frontend/           # React + TypeScript dashboard
├── Dockerfile
└── requirements.txt
```

---

## Limitations

This is a research platform, not a production trading system.

- **Survivorship bias:** the universe uses 20 stocks that exist today. A production system would use point-in-time index membership (e.g. CRSP) to avoid excluding companies that were delisted or went bankrupt during the test period. This likely inflates backtest returns slightly.
- **Paper trading only:** real execution costs, borrow costs for short positions, and order book dynamics are not fully modelled.
- **Universe size:** 20 stocks is small. Cross-sectional signals work better with 200+ names — the Sharpe from a larger universe would be more statistically reliable.
- **Market regime:** the backtest covers 2018–2024, which includes both bull and bear periods but not a full credit cycle. Performance in a 2008-style environment is untested.

---

## Run locally

Requires Docker (for TimescaleDB) and an Alpaca paper trading account (free at alpaca.markets).

```bash
# 1. Start TimescaleDB
docker run -d --name timescaledb -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  timescale/timescaledb-ha:pg16

# Create the database and hypertable
docker exec -it timescaledb psql -U postgres -c "CREATE DATABASE algotrader;"
docker exec -it timescaledb psql -U postgres -d algotrader -c "
  CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open FLOAT, high FLOAT, low FLOAT, close FLOAT,
    volume BIGINT, return FLOAT, dollar_volume FLOAT
  );
  SELECT create_hypertable('ohlcv', 'time');
  CREATE UNIQUE INDEX ON ohlcv (symbol, time DESC);
"

# 2. Backend setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add ALPACA_API_KEY, ALPACA_SECRET_KEY, DATABASE_URL

# 3. Download historical data (20 tickers, 2018–2024)
python scripts/run_data_pipeline.py

# 4. Start the API
uvicorn api.main:app --reload
# Swagger UI available at localhost:8000/docs

# 5. Frontend
cd frontend
npm install
npm run dev
# Dashboard available at localhost:5173
```

---

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Kakushadze, Z. (2016). *101 Formulaic Alphas.* SSRN 2701346.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance.*
- Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics.*
- Bailey, D. et al. (2017). *The Backtest Overfitting Problem in Finance.* SSRN 3073616.
