"""
signals/tearsheet.py
HTML tearsheets using QuantStats.

QuantStats is the industry standard for tearsheet generation.
It provides:
    - Professional styling (looks like Bloomberg terminal)
    - All key visualizations
    - Interactive plots (Plotly)
    - 50+ performance metrics


Usage:
    from signals.tearsheet import generate_tearsheet
    generate_tearsheet(returns, output_file='tearsheet.html')
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Try to import quantstats
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    logger.warning("quantstats not installed. Install with: pip install quantstats")


def generate_tearsheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    output_file: str = 'tearsheet.html',
    title: str = 'Strategy Tearsheet',
    rf: float = 0.00015
) -> str:
    """
    Generate a professional HTML tearsheet using QuantStats.

    Args:
        returns: Daily return series
        benchmark: Optional benchmark returns (e.g., SPY)
        output_file: Output HTML file path
        title: Title for the tearsheet
        rf: Daily risk-free rate

    Returns:
        Path to generated HTML file

    Interview Tip: "QuantStats tearsheets are the industry standard.
    They're what you'd present to a portfolio manager or client.
    The HTML format is interactive and looks professional."

    Example:
        returns = pd.read_parquet('data/returns.parquet')
        generate_tearsheet(returns, output_file='reports/tearsheet.html')
    """
    if not QUANTSTATS_AVAILABLE:
        logger.error("quantstats not available. Install with: pip install quantstats")
        return None

    # Ensure we have enough data
    if len(returns) < 20:
        logger.error(f"Not enough returns data: {len(returns)} days")
        return None

    # Make output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating tearsheet: {output_path}")

    try:
        # Configure QuantStats
        qs.extend_pandas()

        # Create the tearsheet
        qs.reports.html(
            returns,
            benchmark=benchmark,
            title=title,
            rf=rf,
            output=output_file,
            compounded=True
        )

        logger.info(f"Tearsheet saved to: {output_file}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error generating tearsheet: {e}")
        return None


def generate_basic_report(
    returns: pd.Series,
    metrics: dict,
    output_file: str = 'tearsheet_basic.html'
) -> str:
    """
    Generate a basic HTML report without QuantStats.

    Fallback for when QuantStats is not available.

    Args:
        returns: Return series
        metrics: Metrics dictionary from compute_all_metrics
        output_file: Output file path

    Returns:
        Path to generated HTML file
    """
    if 'error' in metrics:
        logger.error(f"Cannot generate report: {metrics['error']}")
        return None

    # Create a simple HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 16px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f5f5f5;
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                margin: 8px 0;
            }}
            .metric-label {{
                color: #666;
                font-size: 14px;
            }}
            .positive {{ color: #2ecc71; }}
            .negative {{ color: #e74c3c; }}
            hr {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Strategy Performance Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics['sharpe_ratio']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sortino Ratio</div>
                <div class="metric-value">{metrics['sortino_ratio']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Calmar Ratio</div>
                <div class="metric-value">{metrics['calmar_ratio']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Hit Rate</div>
                <div class="metric-value">{metrics['hit_rate']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{metrics['profit_factor']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{metrics['max_drawdown']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Annual Return</div>
                <div class="metric-value">{metrics['annual_return']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{metrics['total_return']:.1%}</div>
            </div>
        </div>

        <hr>

        <h2>Additional Metrics</h2>
        <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>Start Date</strong></td><td>{metrics['start_date']}</td></tr>
            <tr><td><strong>End Date</strong></td><td>{metrics['end_date']}</td></tr>
            <tr><td><strong>Total Days</strong></td><td>{metrics['num_days']:,}</td></tr>
            <tr><td><strong>Annual Volatility</strong></td><td>{metrics['annual_volatility']:.1%}</td></tr>
            <tr><td><strong>Max Drawdown Duration</strong></td><td>{metrics['max_drawdown_duration']} days</td></tr>
        </table>

        <hr>
        <p style="color: #999; font-size: 12px;">
            Generated by Algorithmic Trading Platform
        </p>
    </body>
    </html>
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"Basic report saved to: {output_file}")
    return str(output_path)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Test tearsheet generation with synthetic data.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("Testing tearsheet generation...")

    # Create synthetic returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=750, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)

    # Generate tearsheet
    if QUANTSTATS_AVAILABLE:
        output = generate_tearsheet(
            returns,
            output_file='reports/tearsheet.html',
            title='Synthetic Strategy Tearsheet'
        )
        if output:
            logger.info(f"✅ Tearsheet generated: {output}")
    else:
        logger.warning("QuantStats not available. Generating basic report...")
        from signals.metrics import compute_all_metrics
        metrics = compute_all_metrics(returns)
        output = generate_basic_report(returns, metrics, 'reports/tearsheet_basic.html')
        if output:
            logger.info(f"✅ Basic report generated: {output}")