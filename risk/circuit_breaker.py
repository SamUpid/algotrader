"""
risk/circuit_breaker.py
Circuit breaker and position limit controls - CRITICAL RISK CONTROLS.

Components:
1. DrawdownCircuitBreaker - Halts trading when drawdown > 5%
2. PositionLimits - Caps single-stock (5%) and sector (25%) weights
3. RiskEventLog - Tracks all risk events

Why this matters for interviews:
- Circuit breakers prevent catastrophic losses during market crashes
- Position limits prevent over-concentration
- Shows you understand practical risk management

Interview Tip: "Circuit breakers are the last line of defense.
When drawdown exceeds 5%, we halt all trading until we can
investigate what went wrong. This prevents the strategy from
digging a deeper hole during market stress."
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# SECTOR MAPPING
# ============================================================================

# Sector mapping for each ticker
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'NVDA': 'Technology',
    'META': 'Technology',
    
    # Financials
    'JPM': 'Financials',
    'BAC': 'Financials',
    'GS': 'Financials',
    'MS': 'Financials',
    'BLK': 'Financials',
    
    # Healthcare
    'JNJ': 'Healthcare',
    'UNH': 'Healthcare',
    'PFE': 'Healthcare',
    'ABBV': 'Healthcare',
    'MRK': 'Healthcare',
    
    # Consumer / Industrial
    'AMZN': 'Consumer',
    'WMT': 'Consumer',
    'HD': 'Consumer',
    'BA': 'Industrial',
    'XOM': 'Energy',
}


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    return SECTOR_MAP.get(ticker, 'Other')


def get_tickers_in_sector(sector: str) -> List[str]:
    """Get all tickers in a sector."""
    return [ticker for ticker, sec in SECTOR_MAP.items() if sec == sector]


# ============================================================================
# DRAWDOWN CIRCUIT BREAKER
# ============================================================================

class DrawdownCircuitBreaker:
    """
    Circuit breaker that halts trading when drawdown exceeds threshold.

    When current drawdown > 5%, all orders are REJECTED.
    Manual reset required after investigation.

    Usage:
        breaker = DrawdownCircuitBreaker(max_drawdown=0.05)

        # Check if trading is allowed
        if not breaker.is_trading_allowed():
            logger.warning("Circuit breaker triggered - trading halted")
            return

        # Execute order
        execute_order(...)

        # Update NAV daily
        breaker.update_nav(current_nav)

    Interview Tip: "Circuit breakers are the last line of defense.
    When drawdown exceeds 5%, we halt all trading until we can
    investigate what went wrong. This prevents the strategy from
    digging a deeper hole during market stress."
    """

    def __init__(
        self,
        max_drawdown: float = 0.05,
        initial_nav: float = 1_000_000,
        reset_required: bool = True
    ):
        """
        Initialize the circuit breaker.

        Args:
            max_drawdown: Maximum allowed drawdown (default 5%)
            initial_nav: Starting NAV
            reset_required: Require manual reset after trigger (default True)
        """
        self.max_drawdown = max_drawdown
        self.initial_nav = initial_nav
        self.peak_nav = initial_nav
        self.current_nav = initial_nav
        self.trading_halted = False
        self.reset_required = reset_required
        self.trigger_date = None
        self.trigger_drawdown = None

        # Track history
        self.nav_history = [{'date': datetime.now(), 'nav': initial_nav, 'peak': initial_nav}]
        self.rejection_history = []

    def update_nav(self, current_nav: float, date: Optional[datetime] = None) -> Dict:
        """
        Update current NAV and check drawdown.

        Args:
            current_nav: Current portfolio value
            date: Date of update (defaults to now)

        Returns:
            Dict with drawdown status

        Example:
            status = breaker.update_nav(950_000)
            # status = {'drawdown': -0.05, 'trading_halted': True}
        """
        if date is None:
            date = datetime.now()

        self.current_nav = current_nav

        # Update peak NAV
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav

        # Calculate drawdown
        current_drawdown = (current_nav - self.peak_nav) / self.peak_nav

        # Store history
        self.nav_history.append({
            'date': date,
            'nav': current_nav,
            'peak': self.peak_nav,
            'drawdown': current_drawdown
        })

        # Check if drawdown exceeds threshold
        if current_drawdown < -self.max_drawdown and not self.trading_halted:
            self.trading_halted = True
            self.trigger_date = date
            self.trigger_drawdown = current_drawdown
            logger.warning(
                f"🔴 CIRCUIT BREAKER TRIGGERED: Drawdown {current_drawdown:.2%} "
                f"exceeds limit {self.max_drawdown:.2%}"
            )
            self._log_event('CIRCUIT_BREAKER_TRIGGERED', {
                'drawdown': current_drawdown,
                'nav': current_nav,
                'peak': self.peak_nav
            })

        return {
            'drawdown': current_drawdown,
            'peak_nav': self.peak_nav,
            'current_nav': current_nav,
            'trading_halted': self.trading_halted,
            'trigger_date': self.trigger_date,
            'trigger_drawdown': self.trigger_drawdown
        }

    def is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed.

        Returns:
            True if trading allowed, False if halted
        """
        return not self.trading_halted

    def check_order(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """
        Check if an order can be executed.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share

        Returns:
            (approved: bool, reason: str)
        """
        if not self.is_trading_allowed():
            reason = f"Circuit breaker triggered on {self.trigger_date} (drawdown: {self.trigger_drawdown:.2%})"
            self._log_rejection(symbol, quantity, reason)
            return False, reason

        return True, "OK"

    def reset_breaker(self, reason: str) -> Dict:
        """
        Manually reset the circuit breaker.

        Args:
            reason: Reason for reset

        Returns:
            Dict with reset status

        Example:
            status = breaker.reset_breaker("Strategy review complete - issues resolved")
        """
        if not self.trading_halted:
            return {'reset': False, 'message': 'Breaker not triggered'}

        self.trading_halted = False
        self.trigger_date = None
        self.trigger_drawdown = None

        logger.info(f"🔵 CIRCUIT BREAKER RESET: {reason}")
        self._log_event('CIRCUIT_BREAKER_RESET', {'reason': reason})

        return {
            'reset': True,
            'reason': reason,
            'peak_nav': self.peak_nav,
            'current_nav': self.current_nav
        }

    def get_status(self) -> Dict:
        """
        Get current circuit breaker status.

        Returns:
            Dict with status information
        """
        current_drawdown = (self.current_nav - self.peak_nav) / self.peak_nav

        return {
            'trading_halted': self.trading_halted,
            'peak_nav': self.peak_nav,
            'current_nav': self.current_nav,
            'drawdown': current_drawdown,
            'drawdown_limit': self.max_drawdown,
            'trigger_date': self.trigger_date,
            'trigger_drawdown': self.trigger_drawdown,
            'rejection_count': len(self.rejection_history)
        }

    def _log_event(self, event_type: str, details: Dict):
        """Log a risk event."""
        # Not needed for now, but could save to database
        pass

    def _log_rejection(self, symbol: str, quantity: int, reason: str):
        """Log an order rejection."""
        entry = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'reason': reason,
            'type': 'CIRCUIT_BREAKER_REJECTION'
        }
        self.rejection_history.append(entry)
        logger.warning(f"ORDER REJECTED (circuit breaker): {symbol} x {quantity} - {reason}")

    def get_rejection_history(self) -> pd.DataFrame:
        """Get all rejection history."""
        if not self.rejection_history:
            return pd.DataFrame()
        return pd.DataFrame(self.rejection_history)


# ============================================================================
# POSITION LIMITS
# ============================================================================

class PositionLimits:
    """
    Enforce position limits:
    - Max single-stock weight: 5% of NAV
    - Max sector weight: 25% of NAV

    Usage:
        limits = PositionLimits(max_stock_weight=0.05, max_sector_weight=0.25)

        # Check a proposed order
        approved, reason = limits.check_order(
            symbol='AAPL',
            quantity=1000,
            price=150.0,
            current_portfolio={'AAPL': 0.1, 'MSFT': 0.2, ...},
            nav=1_000_000
        )

        if not approved:
            logger.warning(f"Order rejected: {reason}")
            return

    Interview Tip: "Position limits prevent over-concentration.
    A 5% single-stock limit means we never put more than 5% of
    the portfolio in one stock. A 25% sector limit prevents
    sector-specific crashes from destroying the portfolio."
    """

    def __init__(
        self,
        max_stock_weight: float = 0.05,
        max_sector_weight: float = 0.25
    ):
        """
        Initialize position limits.

        Args:
            max_stock_weight: Maximum weight per stock (default 5%)
            max_sector_weight: Maximum weight per sector (default 25%)
        """
        self.max_stock_weight = max_stock_weight
        self.max_sector_weight = max_sector_weight
        self.rejection_history = []

    def check_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        current_portfolio: Dict[str, float],
        nav: float
    ) -> Tuple[bool, str]:
        """
        Check if an order violates position limits.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            current_portfolio: Dict {ticker: weight} of current holdings
            nav: Current NAV

        Returns:
            (approved: bool, reason: str)
        """
        order_value = quantity * price
        proposed_weight = order_value / nav

        # Check 1: Single-stock weight limit
        current_weight = current_portfolio.get(symbol, 0.0)
        new_weight = current_weight + proposed_weight

        if new_weight > self.max_stock_weight:
            reason = (
                f"Single-stock weight would be {new_weight:.2%} "
                f"(limit: {self.max_stock_weight:.2%})"
            )
            self._log_rejection(symbol, quantity, reason)
            return False, reason

        # Check 2: Sector weight limit
        sector = get_sector(symbol)
        if sector == 'Other':
            # No sector limit for unknown sectors
            return True, "OK"

        # Calculate current sector weight
        current_sector_weight = 0.0
        for ticker, weight in current_portfolio.items():
            if get_sector(ticker) == sector:
                current_sector_weight += weight

        # Add proposed position
        new_sector_weight = current_sector_weight + proposed_weight

        if new_sector_weight > self.max_sector_weight:
            reason = (
                f"Sector {sector} weight would be {new_sector_weight:.2%} "
                f"(limit: {self.max_sector_weight:.2%})"
            )
            self._log_rejection(symbol, quantity, reason)
            return False, reason

        return True, "OK"

    def _log_rejection(self, symbol: str, quantity: int, reason: str):
        """Log a rejection."""
        entry = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'reason': reason,
            'type': 'POSITION_LIMIT_REJECTION'
        }
        self.rejection_history.append(entry)
        logger.warning(f"ORDER REJECTED (position limit): {symbol} x {quantity} - {reason}")

    def get_rejection_history(self) -> pd.DataFrame:
        """Get all rejection history."""
        if not self.rejection_history:
            return pd.DataFrame()
        return pd.DataFrame(self.rejection_history)

    def get_portfolio_summary(self, current_portfolio: Dict[str, float]) -> Dict:
        """
        Get summary of current portfolio weights.

        Args:
            current_portfolio: Dict {ticker: weight}

        Returns:
            Dict with single-stock and sector weights
        """
        # Single-stock weights
        stock_weights = current_portfolio.copy()

        # Sector weights
        sector_weights = {}
        for ticker, weight in current_portfolio.items():
            sector = get_sector(ticker)
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        # Check limits
        stock_exceeded = {}
        for ticker, weight in stock_weights.items():
            if weight > self.max_stock_weight:
                stock_exceeded[ticker] = weight

        sector_exceeded = {}
        for sector, weight in sector_weights.items():
            if weight > self.max_sector_weight:
                sector_exceeded[sector] = weight

        return {
            'stock_weights': stock_weights,
            'sector_weights': sector_weights,
            'stock_exceeded': stock_exceeded,
            'sector_exceeded': sector_exceeded,
            'max_stock_weight': self.max_stock_weight,
            'max_sector_weight': self.max_sector_weight,
            'within_limits': len(stock_exceeded) == 0 and len(sector_exceeded) == 0
        }


# ============================================================================
# RISK EVENT LOG
# ============================================================================

class RiskEventLog:
    """
    Log all risk events to CSV.

    Events logged:
    - Circuit breaker triggers and resets
    - VaR rejections
    - Position limit rejections
    - Order rejections

    Usage:
        log = RiskEventLog()
        log.log_event('CIRCUIT_BREAKER_TRIGGERED', {'drawdown': -0.06})
        log.save()

    Interview Tip: "Risk event logging is critical for post-trade analysis.
    When things go wrong, you need a record of exactly what happened
    and why. This helps prevent the same mistake twice."
    """

    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize risk event log.

        Args:
            save_path: Path to save CSV (defaults to data/processed/risk_events.csv)
        """
        if save_path is None:
            save_path = config.DATA_PROC_DIR / "risk_events.csv"

        self.save_path = save_path
        self.events = []

    def log_event(self, event_type: str, details: Dict):
        """
        Log a risk event.

        Args:
            event_type: Type of event (e.g., 'CIRCUIT_BREAKER_TRIGGERED')
            details: Dictionary with event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **details
        }
        self.events.append(event)

    def log_rejection(self, symbol: str, quantity: int, reason: str, rejection_type: str):
        """
        Log an order rejection.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            reason: Rejection reason
            rejection_type: Type of rejection (e.g., 'VAR', 'POSITION_LIMIT')
        """
        self.log_event('ORDER_REJECTED', {
            'symbol': symbol,
            'quantity': quantity,
            'reason': reason,
            'rejection_type': rejection_type
        })

    def save(self):
        """Save all events to CSV."""
        if not self.events:
            logger.info("No events to save")
            return

        df = pd.DataFrame(self.events)

        # Ensure directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing file if it exists
        if self.save_path.exists():
            existing = pd.read_csv(self.save_path)
            df = pd.concat([existing, df], axis=0, ignore_index=True)

        df.to_csv(self.save_path, index=False)
        logger.info(f"Saved {len(df)} risk events to {self.save_path}")

    def get_events(self, event_type: Optional[str] = None) -> pd.DataFrame:
        """Get all events, optionally filtered by type."""
        if not self.events:
            return pd.DataFrame()

        df = pd.DataFrame(self.events)

        if event_type:
            df = df[df['event_type'] == event_type]

        return df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Test circuit breaker and position limits.

    Usage:
        python -m risk.circuit_breaker
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("="*60)
    logger.info("CIRCUIT BREAKER + POSITION LIMITS TEST")
    logger.info("="*60)

    # Test 1: Circuit Breaker
    logger.info("\n" + "-"*40)
    logger.info("Test 1: Circuit Breaker")
    logger.info("-"*40)

    breaker = DrawdownCircuitBreaker(
        max_drawdown=0.05,
        initial_nav=1_000_000
    )

    # Check initial state
    status = breaker.get_status()
    logger.info(f"Initial - Trading allowed: {breaker.is_trading_allowed()}")

    # Simulate drawdown
    logger.info("\nSimulating drawdown...")
    for nav in [980_000, 960_000, 940_000]:
        status = breaker.update_nav(nav)
        logger.info(f"NAV: ${nav:,.0f}, Drawdown: {status['drawdown']:.2%}, "
                   f"Trading halted: {status['trading_halted']}")

    # Check order
    logger.info("\nChecking order during halt...")
    approved, reason = breaker.check_order('AAPL', 100, 150)
    logger.info(f"Order approved: {approved}, Reason: {reason}")

    # Reset breaker
    logger.info("\nResetting breaker...")
    reset_status = breaker.reset_breaker("Test reset - everything is fine")
    logger.info(f"Reset status: {reset_status}")

    # Check after reset
    status = breaker.get_status()
    logger.info(f"After reset - Trading allowed: {breaker.is_trading_allowed()}")

    # Test 2: Position Limits
    logger.info("\n" + "-"*40)
    logger.info("Test 2: Position Limits")
    logger.info("-"*40)

    limits = PositionLimits(
        max_stock_weight=0.05,
        max_sector_weight=0.25
    )

    # Current portfolio
    current_portfolio = {
        'AAPL': 0.10,  # Already at 10% - exceeds 5% limit
        'MSFT': 0.04,
        'GOOGL': 0.03,
        'NVDA': 0.02
    }

    logger.info(f"Current portfolio: {current_portfolio}")

    # Check order that violates single-stock limit
    logger.info("\nChecking AAPL order (violates single-stock limit)...")
    approved, reason = limits.check_order(
        symbol='AAPL',
        quantity=100,
        price=150.0,
        current_portfolio=current_portfolio,
        nav=1_000_000
    )
    logger.info(f"Approved: {approved}, Reason: {reason}")

    # Check order that is within limits
    logger.info("\nChecking MSFT order (within limits)...")
    approved, reason = limits.check_order(
        symbol='MSFT',
        quantity=50,
        price=200.0,
        current_portfolio=current_portfolio,
        nav=1_000_000
    )
    logger.info(f"Approved: {approved}, Reason: {reason}")

    # Portfolio summary
    summary = limits.get_portfolio_summary(current_portfolio)
    print("\n" + "-"*40)
    print("Portfolio Summary:")
    print(f"Within limits: {summary['within_limits']}")
    print(f"Stock weights exceeded: {summary['stock_exceeded']}")
    print(f"Sector weights exceeded: {summary['sector_exceeded']}")

    # Test 3: Risk Event Log
    logger.info("\n" + "-"*40)
    logger.info("Test 3: Risk Event Log")
    logger.info("-"*40)

    log = RiskEventLog()
    log.log_event('CIRCUIT_BREAKER_TRIGGERED', {'drawdown': -0.06, 'nav': 940_000})
    log.log_event('CIRCUIT_BREAKER_RESET', {'reason': 'Test reset'})
    log.log_rejection('AAPL', 100, 'Circuit breaker triggered', 'CIRCUIT_BREAKER')
    log.save()

    events_df = log.get_events()
    logger.info(f"Logged {len(events_df)} events")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)