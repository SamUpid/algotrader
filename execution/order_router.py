"""
execution/order_router.py
Order routing and trade execution with market impact modeling.

This is the final step in the trading pipeline:
1. Signal → 2. Position Sizing → 3. Risk Gates → 4. Execution → 5. Ledger

Key components:
1. Square-root market impact model (industry standard)
2. Pre-trade risk checks (VaR, circuit breaker, position limits)
3. Trade ledger with full P&L tracking
4. Paper trading simulation

Why this matters for interviews:
- Shows you understand execution costs (not just returns)
- Demonstrates practical trading system design
- Market impact modeling is a key quant skill

Interview Tip: "The square-root market impact model is the industry standard.
It captures the diminishing returns of trading larger orders - impact grows
with the square root of order size. A $100M order doesn't cost 10x more than
a $10M order, it costs about 3x more."
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Import risk modules
from risk.var_engine import PreTradeVaRGate, get_covariance_matrix, get_current_portfolio
from risk.circuit_breaker import DrawdownCircuitBreaker, PositionLimits, RiskEventLog

logger = logging.getLogger(__name__)


# ============================================================================
# MARKET IMPACT MODEL
# ============================================================================

def compute_market_impact(
    quantity: int,
    adv: float,
    sigma: float,
    side: str = 'BUY'
) -> float:
    """
    Compute square-root market impact.

    Formula:
        ΔP = σ × sign(Q) × √(|Q| / ADV)

    Where:
        σ = 20-day rolling volatility of the stock
        Q = order size in shares
        ADV = 20-day average daily volume

    Args:
        quantity: Number of shares to trade
        adv: Average daily volume (20-day)
        sigma: Daily volatility (20-day rolling)
        side: 'BUY' or 'SELL'

    Returns:
        Market impact as a fraction (e.g., 0.001 = 10 bps)

    Interview Tip: "The square-root impact model is empirically validated.
    It shows that impact grows with the square root of order size.
    This is because as you trade more, you're trading deeper into
    the order book where there's less liquidity."

    Example:
        quantity = 10,000 shares
        adv = 1,000,000 shares  → 1% of ADV
        sigma = 0.02 (2% daily vol)

        impact = 0.02 × √(0.01) = 0.02 × 0.1 = 0.002 (20 bps)
    """
    if adv <= 0:
        logger.warning(f"ADV <= 0, using fallback impact estimate")
        return 0.001 * (quantity / 10000)  # 10 bps per 10k shares

    if sigma <= 0:
        logger.warning(f"Sigma <= 0, using fallback impact estimate")
        sigma = 0.02  # 2% default

    # Order size relative to ADV
    order_pct = abs(quantity) / adv

    # Square-root impact
    impact = sigma * np.sqrt(order_pct)

    # Apply sign based on side
    if side == 'BUY':
        impact = impact  # Positive impact (price goes up)
    else:  # SELL
        impact = -impact  # Negative impact (price goes down)

    return impact


def compute_execution_price(
    current_price: float,
    quantity: int,
    adv: float,
    sigma: float,
    side: str = 'BUY'
) -> float:
    """
    Compute the execution price including market impact.

    Args:
        current_price: Current mid-price
        quantity: Number of shares
        adv: Average daily volume
        sigma: Daily volatility
        side: 'BUY' or 'SELL'

    Returns:
        Execution price (current_price × (1 + impact))

    Example:
        current_price = $100
        impact = 0.001 (10 bps)

        BUY execution price = $100 × (1 + 0.001) = $100.10
        SELL execution price = $100 × (1 - 0.001) = $99.90
    """
    impact = compute_market_impact(quantity, adv, sigma, side)

    # For buys, we pay the impact (price goes up)
    # For sells, we receive the impact (price goes down)
    execution_price = current_price * (1 + impact)

    return execution_price


# ============================================================================
# ORDER EXECUTION
# ============================================================================

@dataclass
class Order:
    """Represents a trade order."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_price: float  # Limit price or market price
    order_type: str = 'MARKET'  # 'MARKET' or 'LIMIT'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Fill:
    """Represents a filled trade."""
    symbol: str
    side: str
    quantity: int
    price: float  # Execution price
    expected_price: float  # Price without impact
    market_impact: float  # Impact as fraction
    market_impact_bps: float  # Impact in bps
    slippage_bps: float  # Additional slippage
    commission: float  # Commission paid
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None


class TradeLedger:
    """
    Trade ledger tracking all fills with costs and P&L.

    Stores every fill with:
        - symbol, side, quantity
        - expected_price, fill_price
        - slippage_bps, market_impact_bps
        - commission
        - timestamp

    Usage:
        ledger = TradeLedger()
        ledger.record_fill(fill)
        ledger.save()
        pnl = ledger.get_pnl()
    """

    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize the trade ledger.

        Args:
            save_path: Path to save CSV (defaults to data/processed/trade_ledger.csv)
        """
        if save_path is None:
            save_path = config.DATA_PROC_DIR / "trade_ledger.csv"

        self.save_path = save_path
        self.fills = []

    def record_fill(self, fill: Fill) -> None:
        """
        Record a fill in the ledger.

        Args:
            fill: Fill object from execute_order
        """
        self.fills.append(fill)
        logger.info(f"📊 TRADE RECORDED: {fill.symbol} {fill.side} {fill.quantity} @ ${fill.price:.2f}")

    def get_fills(self) -> pd.DataFrame:
        """Get all fills as DataFrame."""
        if not self.fills:
            return pd.DataFrame()

        fill_dicts = [asdict(f) for f in self.fills]
        return pd.DataFrame(fill_dicts)

    def get_pnl(self) -> Dict:
        """
        Compute P&L from all fills.

        Returns:
            Dictionary with total P&L and costs

        Interview Tip: "PnL attribution is critical for strategy evaluation.
        We track gross PnL, impact costs, and slippage separately so we
        can see exactly where the strategy is losing money."
        """
        df = self.get_fills()
        if df.empty:
            return {
                'gross_pnl': 0.0,
                'total_impact_cost': 0.0,
                'total_commission': 0.0,
                'net_pnl': 0.0,
                'total_volume': 0,
                'num_trades': 0
            }

        # For buys: PnL = (current_price - fill_price) × quantity
        # For sells: PnL = (fill_price - current_price) × quantity
        # Since we're in paper trading, we use the expected_price as "fair value"

        # Simplified PnL: for a buy, we lost the market impact
        # For a sell, we lost the market impact
        # In reality, you'd track against benchmark

        # Total costs
        total_impact = (df['market_impact_bps'] * df['quantity'] / 10000).sum()
        total_commission = df['commission'].sum()

        # Gross PnL from price movement (simplified)
        # We use expected_price as the "fair value" benchmark
        pnl_by_trade = []
        for _, row in df.iterrows():
            if row['side'] == 'BUY':
                # Buy: profit if price goes up, but we pay impact
                pnl = (row['price'] - row['expected_price']) * row['quantity']
            else:  # SELL
                # Sell: profit if price goes down, but we pay impact
                pnl = (row['expected_price'] - row['price']) * row['quantity']

            pnl_by_trade.append(pnl)

        gross_pnl = sum(pnl_by_trade)

        return {
            'gross_pnl': gross_pnl,
            'total_impact_cost': total_impact,
            'total_commission': total_commission,
            'net_pnl': gross_pnl - total_impact - total_commission,
            'total_volume': df['quantity'].sum(),
            'num_trades': len(df)
        }

    def save(self) -> None:
        """Save ledger to CSV."""
        if not self.fills:
            logger.info("No fills to save")
            return

        df = self.get_fills()

        # Ensure directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing file if it exists
        if self.save_path.exists():
            existing = pd.read_csv(self.save_path)
            df = pd.concat([existing, df], axis=0, ignore_index=True)

        df.to_csv(self.save_path, index=False)
        logger.info(f"Saved {len(df)} trades to {self.save_path}")


# ============================================================================
# ORDER ROUTER - MAIN EXECUTION ENGINE
# ============================================================================

class OrderRouter:
    """
    Main order routing engine.

    Pipeline:
        1. Receive order
        2. Check VaR gate
        3. Check circuit breaker
        4. Check position limits
        5. Compute market impact
        6. Execute order
        7. Record in ledger

    Usage:
        router = OrderRouter()
        router.set_var_gate(va_gate)
        router.set_circuit_breaker(breaker)
        router.set_position_limits(limits)
        router.set_ledger(ledger)

        fill = router.execute_order(
            symbol='AAPL',
            side='BUY',
            quantity=1000,
            current_price=150.0,
            adv=50_000_000,
            sigma=0.02
        )

    Interview Tip: "The order router is the final gatekeeper.
    Every order must pass through all risk checks before execution.
    This ensures no order is executed without proper risk controls."
    """

    def __init__(self):
        """Initialize the order router."""
        self.var_gate = None
        self.circuit_breaker = None
        self.position_limits = None
        self.ledger = None
        self.risk_log = None
        self.current_portfolio = {}
        self.nav = 1_000_000

    def set_var_gate(self, var_gate: PreTradeVaRGate):
        """Set the VaR gate."""
        self.var_gate = var_gate

    def set_circuit_breaker(self, circuit_breaker: DrawdownCircuitBreaker):
        """Set the circuit breaker."""
        self.circuit_breaker = circuit_breaker

    def set_position_limits(self, position_limits: PositionLimits):
        """Set the position limits."""
        self.position_limits = position_limits

    def set_ledger(self, ledger: TradeLedger):
        """Set the trade ledger."""
        self.ledger = ledger

    def set_risk_log(self, risk_log: RiskEventLog):
        """Set the risk event log."""
        self.risk_log = risk_log

    def set_portfolio(self, portfolio: Dict[str, float], nav: float):
        """
        Set current portfolio and NAV.

        Args:
            portfolio: Dict {ticker: weight}
            nav: Current portfolio value
        """
        self.current_portfolio = portfolio
        self.nav = nav

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        adv: float,
        sigma: float,
        commission_per_share: float = 0.005,
        order_type: str = 'MARKET'
    ) -> Dict:
        """
        Execute an order through the full pipeline.

        Pipeline:
            1. Check VaR gate
            2. Check circuit breaker
            3. Check position limits
            4. Compute market impact
            5. Execute order
            6. Record in ledger

        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            current_price: Current mid-price
            adv: Average daily volume
            sigma: Daily volatility
            commission_per_share: Commission per share
            order_type: 'MARKET' or 'LIMIT'

        Returns:
            Dictionary with execution result

        Example:
            result = router.execute_order(
                symbol='AAPL',
                side='BUY',
                quantity=1000,
                current_price=150.0,
                adv=50_000_000,
                sigma=0.02
            )
        """
        order_value = quantity * current_price

        logger.info(f"📈 ORDER REQUEST: {side} {symbol} x {quantity} @ ${current_price:.2f} (${order_value:,.2f})")

        # Step 1: Check VaR gate
        if self.var_gate is not None:
            approved, var, reason = self.var_gate.check_order(
                symbol=symbol,
                quantity=quantity,
                current_price=current_price,
                current_portfolio=self.current_portfolio,
                cov_matrix=get_covariance_matrix(),
                side=side
            )

            if not approved:
                self._log_rejection(symbol, quantity, reason, 'VAR_GATE')
                return {
                    'approved': False,
                    'reason': reason,
                    'gate': 'VAR_GATE'
                }

        # Step 2: Check circuit breaker
        if self.circuit_breaker is not None:
            approved, reason = self.circuit_breaker.check_order(
                symbol=symbol,
                quantity=quantity,
                price=current_price
            )

            if not approved:
                self._log_rejection(symbol, quantity, reason, 'CIRCUIT_BREAKER')
                return {
                    'approved': False,
                    'reason': reason,
                    'gate': 'CIRCUIT_BREAKER'
                }

        # Step 3: Check position limits
        if self.position_limits is not None:
            approved, reason = self.position_limits.check_order(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                current_portfolio=self.current_portfolio,
                nav=self.nav
            )

            if not approved:
                self._log_rejection(symbol, quantity, reason, 'POSITION_LIMIT')
                return {
                    'approved': False,
                    'reason': reason,
                    'gate': 'POSITION_LIMIT'
                }

        # Step 4: Compute market impact
        impact_fraction = compute_market_impact(quantity, adv, sigma, side)
        impact_bps = impact_fraction * 10000
        execution_price = compute_execution_price(
            current_price, quantity, adv, sigma, side
        )

        logger.info(f"  Market impact: {impact_bps:.1f} bps (${execution_price:.2f})")

        # Step 5: Compute commission
        commission = commission_per_share * quantity

        # Step 6: Create fill record
        fill = Fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            expected_price=current_price,
            market_impact=impact_fraction,
            market_impact_bps=impact_bps,
            slippage_bps=impact_bps * 0.1,  # 10% of impact as slippage
            commission=commission,
            timestamp=datetime.now(),
            order_id=f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Step 7: Record in ledger
        if self.ledger is not None:
            self.ledger.record_fill(fill)
            self.ledger.save()

        # Step 8: Update portfolio (for testing)
        # In production, this would come from a portfolio tracker
        order_value_exec = quantity * execution_price
        new_weight = order_value_exec / self.nav
        if side == 'BUY':
            self.current_portfolio[symbol] = self.current_portfolio.get(symbol, 0.0) + new_weight
        else:
            self.current_portfolio[symbol] = self.current_portfolio.get(symbol, 0.0) - new_weight

        logger.info(f"✅ ORDER EXECUTED: {symbol} x {quantity} @ ${execution_price:.2f}")

        return {
            'approved': True,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'fill_price': execution_price,
            'expected_price': current_price,
            'market_impact_bps': impact_bps,
            'commission': commission,
            'order_value': order_value_exec,
            'gate': 'EXECUTED'
        }

    def _log_rejection(self, symbol: str, quantity: int, reason: str, gate: str):
        """Log an order rejection."""
        logger.warning(f"❌ ORDER REJECTED ({gate}): {symbol} x {quantity} - {reason}")

        if self.risk_log is not None:
            self.risk_log.log_rejection(symbol, quantity, reason, gate)
            self.risk_log.save()

    def execute_orders(
        self,
        orders: List[Dict]
    ) -> List[Dict]:
        """
        Execute multiple orders.

        Args:
            orders: List of order dicts with keys:
                symbol, side, quantity, current_price, adv, sigma

        Returns:
            List of execution results
        """
        results = []
        for order in orders:
            result = self.execute_order(**order)
            results.append(result)
        return results


# ============================================================================
# FASTAPI HELPERS
# ============================================================================

def get_router() -> OrderRouter:
    """
    Get a configured order router for testing.
    """
    router = OrderRouter()

    # Set VaR gate
    var_gate = PreTradeVaRGate(
        var_limit=0.02,
        confidence=0.95,
        horizon=1,
        portfolio_value=1_000_000
    )
    router.set_var_gate(var_gate)

    # Set circuit breaker
    breaker = DrawdownCircuitBreaker(
        max_drawdown=0.05,
        initial_nav=1_000_000
    )
    router.set_circuit_breaker(breaker)

    # Set position limits
    limits = PositionLimits(
        max_stock_weight=0.05,
        max_sector_weight=0.25
    )
    router.set_position_limits(limits)

    # Set ledger
    ledger = TradeLedger()
    router.set_ledger(ledger)

    # Set risk log
    risk_log = RiskEventLog()
    router.set_risk_log(risk_log)

    # Set portfolio
    router.set_portfolio(get_current_portfolio(), 1_000_000)

    return router


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Test the full execution pipeline.

    Usage:
        python -m execution.order_router
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("="*60)
    logger.info("ORDER ROUTER TEST")
    logger.info("="*60)

    # --- Setup ---
    # Create a fresh, diversified portfolio (all well under 5%)
    test_portfolio = {
        'AAPL': 0.02,   # 2% - well under 5% limit
        'MSFT': 0.02,
        'GOOGL': 0.02,
        'NVDA': 0.02,
        'AMZN': 0.02,
        'META': 0.02,
        'JPM': 0.02,
        'BAC': 0.02,
        'GS': 0.02,
        'BLK': 0.02,
        'JNJ': 0.02,
        'UNH': 0.02,
        'PFE': 0.02,
        'ABBV': 0.02,
        'MRK': 0.02,
        'WMT': 0.02,
        'HD': 0.02,
        'BA': 0.02,
        'XOM': 0.02,
        'MS': 0.02,
    }
    # Keep as raw weights (not normalized)
    # Total = 40%, rest is cash
    NAV = 1_000_000

    def create_router():
        router = OrderRouter()
        
        # VaR gate with 2% limit
        var_gate = PreTradeVaRGate(
            var_limit=0.02,
            confidence=0.95,
            horizon=1,
            portfolio_value=NAV
        )
        router.set_var_gate(var_gate)
        
        # Circuit breaker
        breaker = DrawdownCircuitBreaker(
            max_drawdown=0.05,
            initial_nav=NAV
        )
        router.set_circuit_breaker(breaker)
        
        # Position limits: 5% stock, 25% sector
        limits = PositionLimits(
            max_stock_weight=0.05,
            max_sector_weight=0.25
        )
        router.set_position_limits(limits)
        
        # Ledger and risk log
        ledger = TradeLedger()
        router.set_ledger(ledger)
        
        risk_log = RiskEventLog()
        router.set_risk_log(risk_log)
        
        # Set portfolio with 2% weights (not normalized)
        router.set_portfolio(test_portfolio.copy(), NAV)
        
        return router

    # --- Test 1: Small AAPL buy (should PASS) ---
    logger.info("\n" + "-"*40)
    logger.info("Test 1: Small AAPL buy (100 shares - should PASS)")
    logger.info("-"*40)

    router = create_router()
    result = router.execute_order(
        symbol='AAPL',
        side='BUY',
        quantity=100,
        current_price=150.0,
        adv=50_000_000,
        sigma=0.02
    )

    if result['approved']:
        logger.info(f"✅ PASSED: Market impact: {result.get('market_impact_bps', 0):.1f} bps")
    else:
        logger.info(f"❌ REJECTED: {result['reason']}")

    # --- Test 2: Medium AAPL buy (should FAIL - position limit) ---
    logger.info("\n" + "-"*40)
    logger.info("Test 2: Medium AAPL buy (500 shares - should FAIL - position limit)")
    logger.info("-"*40)

    router = create_router()
    result = router.execute_order(
        symbol='AAPL',
        side='BUY',
        quantity=500,
        current_price=150.0,
        adv=50_000_000,
        sigma=0.02
    )

    if result['approved']:
        logger.info(f"✅ PASSED: Market impact: {result.get('market_impact_bps', 0):.1f} bps")
    else:
        logger.info(f"❌ REJECTED: {result['reason']}")

    # --- Test 3: Large AAPL buy (should FAIL - position limit) ---
    logger.info("\n" + "-"*40)
    logger.info("Test 3: Large AAPL buy (5000 shares - should FAIL - position limit)")
    logger.info("-"*40)

    router = create_router()
    result = router.execute_order(
        symbol='AAPL',
        side='BUY',
        quantity=5000,
        current_price=150.0,
        adv=50_000_000,
        sigma=0.02
    )

    if result['approved']:
        logger.info(f"✅ PASSED: Market impact: {result.get('market_impact_bps', 0):.1f} bps")
    else:
        logger.info(f"❌ REJECTED: {result['reason']}")

    # --- Test 4: Huge NVDA order (should FAIL - VaR gate) ---
    logger.info("\n" + "-"*40)
    logger.info("Test 4: Huge NVDA order (10000 shares - should FAIL - VaR gate)")
    logger.info("-"*40)

    router = create_router()
    result = router.execute_order(
        symbol='NVDA',
        side='BUY',
        quantity=10000,
        current_price=800.0,
        adv=20_000_000,
        sigma=0.035
    )

    if result['approved']:
        logger.info(f"✅ PASSED: Market impact: {result.get('market_impact_bps', 0):.1f} bps")
    else:
        logger.info(f"❌ REJECTED: {result['reason']}")

    # --- Summary ---
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    # Show risk events from the last router
    if router.risk_log:
        events = router.risk_log.get_events()
        if not events.empty:
            print("\nRisk Events (last test):")
            print(events[['event_type', 'symbol', 'reason']].to_string(index=False))

    print("="*60)
    logger.info("Test complete!")