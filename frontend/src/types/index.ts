export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  weight_percent: number;
}

export interface EquityData {
  dates: string[];
  equity: number[];
  returns: number[];
}

export interface PortfolioSummary {
  nav: number;
  unrealized_pnl: number;
  realized_pnl: number;
  total_pnl: number;
  positions_count: number;
  last_updated: string;
}

export interface RiskData {
  var_95_1d: number;
  current_drawdown: number;
  drawdown_limit: number;
  circuit_breaker_triggered: boolean;
  last_trigger: string | null;
  trading_halted: boolean;
}
