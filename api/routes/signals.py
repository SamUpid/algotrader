"""
api/routes/signals.py
FastAPI endpoints for strategy signals and metrics.
"""

import logging
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from signals.metrics import compute_all_metrics, format_metrics_for_print
from signals.tearsheet import generate_tearsheet
from signals.factor_attribution import compute_factor_exposure, get_fama_french_factors 

# Try to import quantstats
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

router = APIRouter(prefix="/signals", tags=["signals"])
logger = logging.getLogger(__name__)


# ============================================================================
# METRICS ENDPOINTS (Specific routes FIRST, then catch-all)
# ============================================================================

@router.get("/metrics")
async def get_metrics(
    force_refresh: bool = False
) -> dict:
    """
    Get all performance metrics for the strategy.

    Returns:
        JSON with all performance metrics
    """
    try:
        # Load the backtest returns
        returns_file = config.DATA_PROC_DIR / "backtest_returns.parquet"

        if not returns_file.exists():
            return {
                "status": "no_data",
                "message": "No backtest results found. Run a backtest first."
            }

        # Load the returns data
        df = pd.read_parquet(returns_file)
        
        # Check if it's empty
        if len(df) == 0:
            return {
                "status": "empty",
                "message": "Backtest returns file is empty."
            }

        # Extract the returns series
        if 'returns' in df.columns:
            returns = df['returns']
        else:
            returns = df.iloc[:, 0]

        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]

        metrics = compute_all_metrics(returns)

        return {
            "status": "success",
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Error in /signals/metrics: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/metrics/text")
async def get_metrics_text() -> dict:
    """
    Get formatted text version of metrics (for CLI/console).

    Returns:
        JSON with formatted text
    """
    try:
        returns_file = config.DATA_PROC_DIR / "backtest_returns.parquet"

        if not returns_file.exists():
            return {
                "status": "error",
                "message": "No backtest results found."
            }

        df = pd.read_parquet(returns_file)
        
        if len(df) == 0:
            return {
                "status": "error",
                "message": "Backtest returns file is empty."
            }

        if 'returns' in df.columns:
            returns = df['returns']
        else:
            returns = df.iloc[:, 0]

        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]

        metrics = compute_all_metrics(returns)
        formatted = format_metrics_for_print(metrics)

        return {
            "status": "success",
            "formatted_text": formatted
        }

    except Exception as e:
        logger.error(f"Error in /signals/metrics/text: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/metrics/monthly-returns")
async def get_monthly_returns() -> dict:
    """
    Get monthly returns table for heatmap.
    """
    try:
        returns_file = config.DATA_PROC_DIR / "backtest_returns.parquet"
        
        if not returns_file.exists():
            return {
                "status": "error",
                "message": "No backtest results found."
            }
        
        df = pd.read_parquet(returns_file)
        
        if len(df) == 0:
            return {
                "status": "error",
                "message": "Backtest returns file is empty."
            }
        
        # Extract returns
        if 'returns' in df.columns:
            returns = df['returns']
        else:
            returns = df.iloc[:, 0]
        
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        
        # Ensure index is datetime
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        # Compute monthly returns
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        # Create table
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })
        
        table = monthly_df.pivot(index='year', columns='month', values='return')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        table.columns = month_names
        
        result = {
            'years': table.index.tolist(),
            'months': month_names,
            'data': table.fillna(0).values.tolist()
        }
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error in /signals/metrics/monthly-returns: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/metrics/factor-exposure")
async def get_factor_exposure() -> dict:
    """
    Get Fama-French factor exposure for the strategy.
    
    Returns:
        Dict with alpha, factor betas, and R-squared
    """
    try:
        returns_file = config.DATA_PROC_DIR / "backtest_returns.parquet"
        
        if not returns_file.exists():
            return {
                "status": "error",
                "message": "No backtest results found. Run a backtest first."
            }
        
        df = pd.read_parquet(returns_file)
        
        if len(df) == 0:
            return {
                "status": "error",
                "message": "Backtest returns file is empty."
            }
        
        if 'returns' in df.columns:
            returns = df['returns']
        else:
            returns = df.iloc[:, 0]
        
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        factor_data = get_fama_french_factors()
        exposure = compute_factor_exposure(returns, factor_data)
        
        if 'error' in exposure:
            return {
                "status": "error",
                "message": exposure['error']
            }
        
        return {
            "status": "success",
            "exposure": exposure
        }
        
    except Exception as e:
        logger.error(f"Error in /signals/metrics/factor-exposure: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/metrics/{strategy}")
async def get_strategy_metrics(strategy: str) -> dict:
    """
    Get performance metrics for a specific strategy.
    
    Args:
        strategy: 'momentum', 'rsi', 'ml', or 'composite'
    """
    try:
        # Load different returns based on strategy
        strategy_file = config.DATA_PROC_DIR / f"{strategy}_returns.parquet"
        
        if strategy_file.exists():
            df = pd.read_parquet(strategy_file)
        else:
            # Fallback: load composite returns
            returns_file = config.DATA_PROC_DIR / "backtest_returns.parquet"
            if not returns_file.exists():
                return {"status": "error", "message": "No backtest results found."}
            df = pd.read_parquet(returns_file)
        
        if len(df) == 0:
            return {"status": "error", "message": "Returns file is empty."}
        
        if 'returns' in df.columns:
            returns = df['returns']
        else:
            returns = df.iloc[:, 0]
        
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        
        metrics = compute_all_metrics(returns)
        
        return {
            "status": "success",
            "strategy": strategy,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error in /signals/metrics/{strategy}: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# TEARSHEET ENDPOINT
# ============================================================================

@router.post("/tearsheet")
async def generate_tearsheet_endpoint(
    title: str = "Strategy Tearsheet",
    include_benchmark: bool = False
) -> dict:
    """
    Generate a professional HTML tearsheet.

    Args:
        title: Title for the tearsheet
        include_benchmark: Include SPY benchmark?

    Returns:
        JSON with tearsheet file path
    """
    if not QUANTSTATS_AVAILABLE:
        return {
            "status": "error",
            "message": "QuantStats not available. Install with: pip install quantstats"
        }

    try:
        returns_file = config.DATA_PROC_DIR / "backtest_returns.parquet"

        if not returns_file.exists():
            return {
                "status": "error",
                "message": "No backtest results found. Run a backtest first."
            }

        df = pd.read_parquet(returns_file)
        
        if len(df) == 0:
            return {
                "status": "error",
                "message": "Backtest returns file is empty."
            }

        if 'returns' in df.columns:
            returns = df['returns']
        else:
            returns = df.iloc[:, 0]

        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]

        benchmark = None
        if include_benchmark:
            benchmark_file = config.DATA_PROC_DIR / "spy_returns.parquet"
            if benchmark_file.exists():
                benchmark_df = pd.read_parquet(benchmark_file)
                if 'returns' in benchmark_df.columns:
                    benchmark = benchmark_df['returns']
                else:
                    benchmark = benchmark_df.iloc[:, 0]

        output_file = config.ROOT_DIR / "reports" / "tearsheet.html"
        result = generate_tearsheet(
            returns=returns,
            benchmark=benchmark,
            output_file=str(output_file),
            title=title
        )

        if result:
            return {
                "status": "success",
                "file_path": str(result),
                "message": f"Tearsheet generated: {result}"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate tearsheet"
            }

    except Exception as e:
        logger.error(f"Error in /signals/tearsheet: {e}")
        return {
            "status": "error",
            "message": str(e)
        }