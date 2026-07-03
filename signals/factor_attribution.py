"""
signals/factor_attribution.py
Fama-French 3-factor regression for strategy evaluation.

Computes:
1. Jensen's Alpha: excess return not explained by market
2. Factor Betas: market, size (SMB), value (HML)
3. R-squared: how much variance is explained by factors
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
import statsmodels.api as sm

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def get_fama_french_factors() -> pd.DataFrame:
    """Get Fama-French 3-factor data from Ken French Data Library."""
    cache_path = config.DATA_PROC_DIR / "fama_french_daily.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded Fama-French factors from cache")
        return df

    logger.warning("Fama-French data not found. Generating synthetic factors for demo.")
    return generate_synthetic_factors()


def generate_synthetic_factors() -> pd.DataFrame:
    """Generate synthetic Fama-French factors for demo purposes."""
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='D')
    
    # Clean generation - no NaN values
    mkt = np.random.normal(0.0004, 0.01, len(dates))
    smb = np.random.normal(0.0001, 0.005, len(dates))
    hml = np.random.normal(0.0002, 0.006, len(dates))
    rf = np.random.normal(0.00015, 0.0001, len(dates))
    
    # Ensure no NaN values
    mkt = np.nan_to_num(mkt, nan=0.0)
    smb = np.nan_to_num(smb, nan=0.0)
    hml = np.nan_to_num(hml, nan=0.0)
    rf = np.nan_to_num(rf, nan=0.0)
    
    df = pd.DataFrame({
        'Mkt-RF': mkt,
        'SMB': smb,
        'HML': hml,
        'RF': rf
    }, index=dates)
    
    return df


def compute_factor_exposure(
    strategy_returns: pd.Series,
    factor_data: pd.DataFrame,
    rf: float = 0.00015
) -> Dict:
    """
    Run Fama-French 3-factor regression.
    """
    # Align data
    common_idx = strategy_returns.index.intersection(factor_data.index)
    
    if len(common_idx) < 30:
        return {'error': f'Insufficient overlapping data: {len(common_idx)} days'}
    
    returns = strategy_returns.reindex(common_idx)
    factors = factor_data.reindex(common_idx)
    
    # Clean data - remove any remaining NaN
    returns = returns.fillna(0)
    factors = factors.fillna(0)
    
    # Prepare regression data
    y = returns - rf
    X = factors[['Mkt-RF', 'SMB', 'HML']].copy()
    X = sm.add_constant(X)
    
    # Remove rows with infinite or NaN values
    mask = ~(np.isinf(y) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isnan(X).any(axis=1))
    y = y[mask]
    X = X[mask]
    
    if len(y) < 30:
        return {'error': f'Insufficient data after cleaning: {len(y)} days'}
    
    # Run regression
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Extract results
    alpha = results.params.get('const', 0.0)
    beta_mkt = results.params.get('Mkt-RF', 0.0)
    beta_smb = results.params.get('SMB', 0.0)
    beta_hml = results.params.get('HML', 0.0)
    
    return {
        'alpha_daily': float(alpha),
        'alpha_annual': float(alpha * 252),
        'beta_mkt': float(beta_mkt),
        'beta_smb': float(beta_smb),
        'beta_hml': float(beta_hml),
        'r_squared': float(results.rsquared),
        'adj_r_squared': float(results.rsquared_adj),
        'num_observations': int(len(y)),
        'residual_std': float(results.mse_resid ** 0.5)
    }
