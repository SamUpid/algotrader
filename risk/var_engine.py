"""
risk/var_engine.py
Pre-trade Value at Risk (VaR) gate - CRITICAL RISK CONTROL.

This module implements:
1. Parametric VaR (Value at Risk)
2. CVaR (Expected Shortfall) 
3. Pre-trade VaR gate - REJECTS trades that exceed risk limits
4. Covariance matrix computation (60-day rolling)

Why this is critical for interviews:
- VaR is the industry standard for risk management
- Pre-trade gates are how institutions prevent blow-ups
- Shows you understand risk management, not just returns

Interview Tip: "The pre-trade VaR gate is the most important risk control.
If a trade would push the portfolio beyond the risk limit, we reject it
immediately. This prevents a single bad trade from destroying the portfolio."
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
from scipy.stats import norm

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# COVARIANCE MATRIX
# ============================================================================

def compute_covariance_matrix(
    returns_matrix: pd.DataFrame,
    window: int = 60,
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Compute rolling covariance matrix for portfolio.

    Args:
        returns_matrix: Wide DataFrame (dates × tickers) of returns
        window: Rolling window size (default 60 days)
        min_periods: Minimum observations for covariance (default 30)

    Returns:
        Covariance matrix (tickers × tickers)

    Interview Tip: "We use a 60-day rolling covariance matrix.
    This captures current market conditions while being stable.
    Too short → noisy. Too long → stale."
    """
    # Get last 'window' days of returns
    if len(returns_matrix) < window:
        logger.warning(f"Only {len(returns_matrix)} days available, using all")
        recent_returns = returns_matrix
    else:
        recent_returns = returns_matrix.iloc[-window:]

    # Drop columns with too many missing values
    valid_cols = recent_returns.columns[
        recent_returns.count() >= min_periods
    ].tolist()

    if len(valid_cols) < 2:
        logger.warning("Not enough valid tickers for covariance matrix")
        return pd.DataFrame()

    cov_matrix = recent_returns[valid_cols].cov()

    return cov_matrix


# ============================================================================
# VALUE AT RISK (VaR)
# ============================================================================

def compute_portfolio_var(
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame,
    portfolio_value: float = 1_000_000,
    confidence: float = 0.95,
    horizon: int = 1
) -> Dict:
    """
    Compute parametric VaR for a portfolio.

    Formula:
        VaR_α = portfolio_value × (μ_p - z_α × σ_p × √h)

    where:
        μ_p = portfolio expected return
        σ_p = portfolio volatility
        z_α = z-score for confidence level
        h = horizon in days

    Args:
        weights: Dictionary {ticker: weight} (sum = 1.0)
        cov_matrix: Covariance matrix (tickers × tickers)
        portfolio_value: Total portfolio value
        confidence: Confidence level (default 0.95)
        horizon: Horizon in days (default 1)

    Returns:
        Dictionary with VaR metrics

    Interview Tip: "Parametric VaR assumes normal returns.
    It's fast to compute and works well for liquid portfolios.
    For 95% 1-day VaR, z = 1.645."
    """
    # Get tickers in covariance matrix
    tickers = cov_matrix.index.tolist()

    # Create weight vector aligned with covariance matrix
    w = np.array([weights.get(ticker, 0.0) for ticker in tickers])

    # Check weights sum to 1
    if abs(w.sum() - 1.0) > 1e-6:
        logger.warning(f"Weights sum to {w.sum():.4f}, re-normalizing")
        w = w / w.sum()

    # Portfolio volatility: sqrt(w' * cov * w)
    portfolio_var = np.dot(w.T, np.dot(cov_matrix.values, w))

    # Ensure non-negative (due to floating point)
    if portfolio_var < 0:
        portfolio_var = 0
        portfolio_vol = 0
    else:
        portfolio_vol = np.sqrt(portfolio_var)

    # Z-score for confidence level
    z_score = norm.ppf(1 - confidence)

    # Expected return (daily)
    # We don't have expected returns, so we use 0 (neutral)
    # In practice, you'd use the strategy's expected return
    mu_p = 0.0

    # VaR calculation
    var_amount = portfolio_value * (mu_p - z_score * portfolio_vol * np.sqrt(horizon))

    # CVaR (Expected Shortfall)
    # CVaR = μ - σ * φ(z) / (1 - confidence)
    # where φ is the standard normal PDF
    pdf_z = norm.pdf(z_score)
    cvar_factor = pdf_z / (1 - confidence)
    cvar_amount = portfolio_value * (mu_p - portfolio_vol * cvar_factor * np.sqrt(horizon))

    # VaR as percentage of portfolio
    var_percent = var_amount / portfolio_value

    return {
        'var_amount': var_amount,
        'var_percent': var_percent,
        'cvar_amount': cvar_amount,
        'cvar_percent': cvar_amount / portfolio_value,
        'portfolio_vol': portfolio_vol,
        'confidence': confidence,
        'horizon': horizon,
        'portfolio_value': portfolio_value,
        'z_score': z_score
    }


def compute_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Compute CVaR (Expected Shortfall) from historical returns.

    CVaR = -Mean(returns <= -VaR)

    This is the non-parametric (historical) CVaR.

    Args:
        returns: Series of historical returns
        confidence: Confidence level (default 0.95)

    Returns:
        CVaR as positive number (expected loss)

    Interview Tip: "CVaR (Conditional VaR) tells you the average loss
    in the worst cases. It's better than VaR because VaR can hide
    tail risk - two portfolios can have the same VaR but very
    different worst-case losses."
    """
    if len(returns) < 10:
        return 0.0

    # VaR at confidence level
    var_threshold = returns.quantile(1 - confidence)

    # Returns worse than VaR
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        return 0.0

    # Expected shortfall
    cvar = -tail_returns.mean()

    return cvar


# ============================================================================
# PRE-TRADE VAR GATE (CRITICAL)
# ============================================================================

class PreTradeVaRGate:
    """
    Pre-trade VaR gate - REJECTS trades that exceed risk limits.

    This is a BLOCKING gate, not just a metric.
    If a trade would push the portfolio beyond the VaR limit,
    the trade is REJECTED and logged.

    Usage:
        gate = PreTradeVaRGate(var_limit=0.02)  # 2% of portfolio

        # Check a proposed trade
        approved, new_var, reason = gate.check_order(
            symbol='AAPL',
            quantity=1000,
            current_price=150.0,
            current_portfolio={'AAPL': 0.1, 'MSFT': 0.2, ...},
            cov_matrix=cov_matrix
        )

        if not approved:
            logger.warning(f"Order rejected: {reason}")
            # DO NOT EXECUTE THE ORDER
        else:
            # Execute the order
            execute_order(...)

    Interview Tip: "This is a REAL blocking gate. If the trade would
    push VaR above the limit, we reject it immediately and log the reason.
    In production, this prevents a single bad trade from destroying
    the portfolio. This is how institutions manage risk."
    """

    def __init__(
        self,
        var_limit: float = 0.02,  # 2% of portfolio
        confidence: float = 0.95,
        horizon: int = 1,
        portfolio_value: float = 1_000_000
    ):
        """
        Initialize the VaR gate.

        Args:
            var_limit: Maximum allowed VaR as fraction of portfolio
            confidence: Confidence level for VaR
            horizon: VaR horizon in days
            portfolio_value: Current portfolio value
        """
        self.var_limit = var_limit
        self.confidence = confidence
        self.horizon = horizon
        self.portfolio_value = portfolio_value

        # Track rejected orders
        self.rejected_orders = []
        self.approved_orders = []

    def check_order(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        current_portfolio: Dict[str, float],
        cov_matrix: pd.DataFrame,
        side: str = 'BUY'
    ) -> Tuple[bool, float, str]:
        """
        Check if an order would violate the VaR limit.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            current_price: Current price per share
            current_portfolio: Dict {ticker: weight} of current holdings
            cov_matrix: Covariance matrix
            side: 'BUY' or 'SELL'

        Returns:
            (approved: bool, new_var: float, reason: str)

        Example:
            approved, new_var, reason = gate.check_order(
                symbol='AAPL',
                quantity=1000,
                current_price=150.0,
                current_portfolio={'AAPL': 0.1, 'MSFT': 0.2},
                cov_matrix=cov_matrix
            )
            if approved:
                execute_order()
            else:
                logger.warning(f"REJECTED: {reason}")
        """
        # Calculate order value
        order_value = quantity * current_price

        # Calculate portfolio value
        portfolio_value = self.portfolio_value

        # Check if enough capital
        if side == 'BUY' and order_value > portfolio_value * 0.5:
            reason = f"Order value ${order_value:,.2f} exceeds 50% of portfolio"
            self._log_rejection(symbol, quantity, reason)
            return False, None, reason

        # Get current tickers
        current_tickers = list(current_portfolio.keys())

        # Check if symbol is in covariance matrix
        if symbol not in cov_matrix.index:
            reason = f"{symbol} not in covariance matrix"
            self._log_rejection(symbol, quantity, reason)
            return False, None, reason

        # Build proposed portfolio weights
        proposed_weights = current_portfolio.copy()

        # Add the new position
        if side == 'BUY':
            proposed_weights[symbol] = proposed_weights.get(symbol, 0.0) + (order_value / portfolio_value)
        else:  # SELL
            current_weight = proposed_weights.get(symbol, 0.0)
            new_weight = current_weight - (order_value / portfolio_value)
            if new_weight < 0:
                # Can't short more than we have (for this simple check)
                reason = f"Cannot sell more than current position ({current_weight:.2%})"
                self._log_rejection(symbol, quantity, reason)
                return False, None, reason
            proposed_weights[symbol] = new_weight

        # Check if weights sum to 1
        total_weight = sum(proposed_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Re-normalize
            for ticker in proposed_weights:
                proposed_weights[ticker] = proposed_weights[ticker] / total_weight

        # Compute VaR for proposed portfolio
        var_result = compute_portfolio_var(
            weights=proposed_weights,
            cov_matrix=cov_matrix,
            portfolio_value=portfolio_value,
            confidence=self.confidence,
            horizon=self.horizon
        )

        new_var_percent = abs(var_result['var_percent'])

        # Check if VaR exceeds limit
        if new_var_percent > self.var_limit:
            reason = (
                f"VaR would increase to {new_var_percent:.2%} "
                f"(limit: {self.var_limit:.2%})"
            )
            self._log_rejection(symbol, quantity, reason, new_var_percent)
            return False, new_var_percent, reason

        # Approved!
        self._log_approval(symbol, quantity, new_var_percent)
        return True, new_var_percent, "OK"

    def _log_rejection(
        self,
        symbol: str,
        quantity: int,
        reason: str,
        var: Optional[float] = None
    ):
        """Log a rejected order."""
        entry = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'reason': reason,
            'var': var,
            'status': 'REJECTED'
        }
        self.rejected_orders.append(entry)
        logger.warning(f"ORDER REJECTED: {symbol} x {quantity} - {reason}")

    def _log_approval(
        self,
        symbol: str,
        quantity: int,
        var: float
    ):
        """Log an approved order."""
        entry = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'var': var,
            'status': 'APPROVED'
        }
        self.approved_orders.append(entry)
        logger.info(f"ORDER APPROVED: {symbol} x {quantity} (VaR: {var:.2%})")

    def get_rejected_orders(self) -> pd.DataFrame:
        """Get all rejected orders."""
        if not self.rejected_orders:
            return pd.DataFrame()
        return pd.DataFrame(self.rejected_orders)

    def get_approved_orders(self) -> pd.DataFrame:
        """Get all approved orders."""
        if not self.approved_orders:
            return pd.DataFrame()
        return pd.DataFrame(self.approved_orders)

    def get_summary(self) -> Dict:
        """Get summary of VaR gate activity."""
        return {
            'total_rejected': len(self.rejected_orders),
            'total_approved': len(self.approved_orders),
            'rejection_rate': len(self.rejected_orders) / (len(self.rejected_orders) + len(self.approved_orders) + 1e-6),
            'var_limit': self.var_limit,
            'last_rejection': self.rejected_orders[-1] if self.rejected_orders else None
        }


# ============================================================================
# FASTAPI HELPERS
# ============================================================================
def get_current_portfolio() -> Dict[str, float]:
    """
    Get current portfolio weights.
    In production, this would come from your portfolio tracking system.

    Returns:
        Dict {ticker: weight}
    """
    # A properly diversified portfolio (all under 5% limit)
    weights = {
        'AAPL': 0.04,    # 4% - under 5% limit
        'MSFT': 0.04,    # 4%
        'GOOGL': 0.04,   # 4%
        'NVDA': 0.04,    # 4%
        'AMZN': 0.04,    # 4%
        'META': 0.04,    # 4%
        'JPM': 0.04,     # 4%
        'BAC': 0.04,     # 4%
        'GS': 0.04,      # 4%
        'BLK': 0.04,     # 4%
        'JNJ': 0.04,     # 4%
        'UNH': 0.04,     # 4%
        'PFE': 0.04,     # 4%
        'ABBV': 0.04,    # 4%
        'MRK': 0.04,     # 4%
        'WMT': 0.04,     # 4%
        'HD': 0.04,      # 4%
        'BA': 0.04,      # 4%
        'XOM': 0.04,     # 4%
        'MS': 0.04,      # 4%
    }
    # Total = 20 × 4% = 80% (leaves 20% cash)
    # Normalize to sum to 1 (100% invested)
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

def get_covariance_matrix() -> pd.DataFrame:
    """
    Load or compute covariance matrix.

    Returns:
        Covariance matrix
    """
    # Try to load from cache
    cov_path = config.DATA_PROC_DIR / "covariance_matrix.parquet"

    if cov_path.exists():
        try:
            return pd.read_parquet(cov_path)
        except Exception as e:
            logger.warning(f"Could not load covariance matrix: {e}")

    # Compute from returns matrix
    returns_path = config.DATA_PROC_DIR / "returns_matrix.parquet"
    if not returns_path.exists():
        logger.error("Returns matrix not found")
        return pd.DataFrame()

    returns_matrix = pd.read_parquet(returns_path)
    cov_matrix = compute_covariance_matrix(returns_matrix)

    if cov_path.parent.exists():
        cov_matrix.to_parquet(cov_path)
        logger.info(f"Cached covariance matrix to {cov_path}")

    return cov_matrix


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Test the VaR engine.

    Usage:
        python -m risk.var_engine
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("="*60)
    logger.info("VAR ENGINE TEST")
    logger.info("="*60)

    # Load data
    returns_path = config.DATA_PROC_DIR / "returns_matrix.parquet"
    if not returns_path.exists():
        logger.error("Returns matrix not found. Run data.universe first.")
        exit(1)

    returns_matrix = pd.read_parquet(returns_path)

    # Compute covariance matrix
    cov_matrix = compute_covariance_matrix(returns_matrix)

    if cov_matrix.empty:
        logger.error("Failed to compute covariance matrix")
        exit(1)

    logger.info(f"Covariance matrix shape: {cov_matrix.shape}")

    # Current portfolio
    current_portfolio = get_current_portfolio()
    logger.info(f"Current portfolio: {current_portfolio}")

    # Initialize VaR gate
    gate = PreTradeVaRGate(
        var_limit=0.02,  # 2% VaR limit
        confidence=0.95,
        horizon=1,
        portfolio_value=1_000_000
    )

    logger.info(f"VaR limit: {gate.var_limit:.2%}")

    # Test 1: Trade that should pass
    logger.info("\nTest 1: Small AAPL buy (should pass)")
    approved, new_var, reason = gate.check_order(
        symbol='AAPL',
        quantity=100,
        current_price=150.0,
        current_portfolio=current_portfolio,
        cov_matrix=cov_matrix,
        side='BUY'
    )
    if approved:
        logger.info(f"  ✅ Approved: VaR={new_var:.4%}, Reason: {reason}")
    else:
        logger.info(f"  ❌ Rejected: Reason: {reason}")

    # Test 2: Large NVDA buy (should fail - size limit)
    logger.info("\nTest 2: Large NVDA buy (should fail - size limit)")
    approved, new_var, reason = gate.check_order(
        symbol='NVDA',
        quantity=5000,
        current_price=800.0,
        current_portfolio=current_portfolio,
        cov_matrix=cov_matrix,
        side='BUY'
    )
    if approved:
        logger.info(f"  ✅ Approved: VaR={new_var:.4%}, Reason: {reason}")
    else:
        logger.info(f"  ❌ Rejected: Reason: {reason}")

    # Test 3: Moderate position that tests VaR limit
    logger.info("\nTest 3: Moderate AAPL buy (should test VaR limit)")
    approved, new_var, reason = gate.check_order(
        symbol='AAPL',
        quantity=500,
        current_price=150.0,
        current_portfolio=current_portfolio,
        cov_matrix=cov_matrix,
        side='BUY'
    )
    if approved:
        logger.info(f"  ✅ Approved: VaR={new_var:.4%}, Reason: {reason}")
    else:
        logger.info(f"  ❌ Rejected: Reason: {reason}")

    # Test 4: Large position that should exceed VaR limit
    logger.info("\nTest 4: Large AAPL position (should exceed VaR limit)")
    approved, new_var, reason = gate.check_order(
        symbol='AAPL',
        quantity=2000,
        current_price=150.0,
        current_portfolio=current_portfolio,
        cov_matrix=cov_matrix,
        side='BUY'
    )
    if approved:
        logger.info(f"  ✅ Approved: VaR={new_var:.4%}, Reason: {reason}")
    else:
        logger.info(f"  ❌ Rejected: Reason: {reason}")

    # Summary
    summary = gate.get_summary()
    print("\n" + "="*60)
    print("VAR GATE SUMMARY")
    print("="*60)
    print(f"Total Approved: {summary['total_approved']}")
    print(f"Total Rejected: {summary['total_rejected']}")
    print(f"Rejection Rate: {summary['rejection_rate']:.2%}")
    print(f"VaR Limit: {summary['var_limit']:.2%}")

    if summary['last_rejection']:
        print(f"\nLast Rejection:")
        print(f"  Symbol: {summary['last_rejection']['symbol']}")
        print(f"  Reason: {summary['last_rejection']['reason']}")

    # Show rejected orders
    rejected_df = gate.get_rejected_orders()
    if not rejected_df.empty:
        print("\nRejected Orders:")
        print(rejected_df[['symbol', 'quantity', 'reason']].to_string(index=False))

    print("="*60)