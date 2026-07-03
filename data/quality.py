
"""
data/quality.py - Production data quality validation suite
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import using your config's exact variable names
from config import (
    DATA_RAW_DIR,
    DATA_PROC_DIR,
    UNIVERSE,
    MAX_DAILY_RETURN,
    MAX_MISSING_RATIO,
    raw_path,
    processed_path
)

logger = logging.getLogger(__name__)

def load_price_matrix() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed price and returns matrices from Parquet files."""
    price_path = DATA_PROC_DIR / "price_matrix.parquet"
    returns_path = DATA_PROC_DIR / "returns_matrix.parquet"
    
    if not price_path.exists():
        raise FileNotFoundError(
            f"Price matrix not found at {price_path}. "
            "Run data/downloader.py first."
        )
    
    price_df = pd.read_parquet(price_path)
    returns_df = pd.read_parquet(returns_path)
    
    # Load volume matrix
    volume_path = DATA_PROC_DIR / "volume_matrix.parquet"
    if not volume_path.exists():
        logger.warning("Volume matrix not found, computing from raw files...")
        volume_df = build_volume_matrix()
    else:
        volume_df = pd.read_parquet(volume_path)
    
    logger.info(f"Loaded {len(price_df)} dates × {len(price_df.columns)} symbols")
    return price_df, returns_df, volume_df

def build_volume_matrix() -> pd.DataFrame:
    """Build volume matrix from raw Parquet files."""
    all_volumes = {}
    for ticker in UNIVERSE:
        file_path = raw_path(ticker)
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df = df.set_index('date')
            all_volumes[ticker] = df['volume']
    
    volume_df = pd.DataFrame(all_volumes)
    volume_df.index = pd.to_datetime(volume_df.index)
    volume_df = volume_df.sort_index()
    volume_df.to_parquet(DATA_PROC_DIR / "volume_matrix.parquet")
    logger.info(f"Saved volume_matrix.parquet with {len(volume_df)} rows")
    return volume_df

def check_missing_bars(price_matrix: pd.DataFrame, threshold: float = None) -> Dict:
    """Check for missing trading days per ticker."""
    if threshold is None:
        threshold = MAX_MISSING_RATIO
    
    logger.info(f"Checking missing bars per ticker (threshold: {threshold:.1%})...")
    full_date_range = pd.date_range(
        start=price_matrix.index.min(),
        end=price_matrix.index.max(),
        freq='B'
    )
    
    results = {}
    all_passed = True
    
    for symbol in price_matrix.columns:
        non_null = price_matrix[symbol].dropna().index
        null_count = len(full_date_range) - len(non_null)
        null_ratio = null_count / len(full_date_range) if len(full_date_range) > 0 else 0
        passed = null_ratio <= threshold
        if not passed:
            all_passed = False
        results[symbol] = {
            'total_expected_days': len(full_date_range),
            'actual_days': len(non_null),
            'missing_days': null_count,
            'missing_ratio': null_ratio,
            'passed': passed,
            'warning': f"Missing {null_count}/{len(full_date_range)} days ({null_ratio:.2%})"
        }
    
    results['_summary'] = {
        'total_symbols': len(price_matrix.columns),
        'passed_count': sum(1 for k, v in results.items() if k != '_summary' and v['passed']),
        'failed_count': sum(1 for k, v in results.items() if k != '_summary' and not v['passed']),
        'overall_pass': all_passed
    }
    
    logger.info(f"Missing bar check complete: {results['_summary']['passed_count']}/{results['_summary']['total_symbols']} passed")
    return results

def check_return_spikes(returns_matrix: pd.DataFrame, threshold: float = None) -> Dict:
    """Detect extreme daily returns."""
    if threshold is None:
        threshold = MAX_DAILY_RETURN
    
    logger.info(f"Checking return spikes > {threshold*100}%...")
    results = {}
    total_spikes = 0
    
    for symbol in returns_matrix.columns:
        returns = returns_matrix[symbol].dropna()
        spikes = returns[abs(returns) > threshold]
        spike_dates = spikes.index.strftime('%Y-%m-%d').tolist() if len(spikes) > 0 else []
        results[symbol] = {
            'spike_count': len(spikes),
            'spike_dates': spike_dates,
            'max_return': float(spikes.max() if len(spikes) > 0 else 0),
            'min_return': float(spikes.min() if len(spikes) > 0 else 0),
        }
        if len(spikes) > 0:
            total_spikes += len(spikes)
            for date, ret in spikes.items():
                logger.warning(f"{symbol} {date.strftime('%Y-%m-%d')}: {ret:.2%} ({'down' if ret < 0 else 'up'}) - Keep in dataset")
        else:
            logger.debug(f"{symbol}: No return spikes detected")
    
    results['_summary'] = {
        'total_symbols': len(returns_matrix.columns),
        'total_spikes': total_spikes,
        'spikes_per_symbol': total_spikes / len(returns_matrix.columns),
        'threshold_used': threshold
    }
    
    logger.info(f"Return spike check complete: {total_spikes} total spikes detected")
    return results

def check_zero_volume(volume_matrix: pd.DataFrame) -> Dict:
    """Count days with zero volume per ticker."""
    logger.info("Checking zero volume days...")
    results = {}
    total_zero_days = 0
    
    for symbol in volume_matrix.columns:
        volumes = volume_matrix[symbol]
        zero_days = volumes[volumes == 0]
        zero_count = len(zero_days)
        if zero_count > 0:
            total_zero_days += zero_count
            logger.warning(f"{symbol}: {zero_count} zero volume days - check data quality")
            first_dates = zero_days.index[:3].strftime('%Y-%m-%d').tolist()
            results[symbol] = {
                'zero_volume_count': zero_count,
                'first_3_dates': first_dates,
                'zero_ratio': zero_count / len(volumes.dropna())
            }
        else:
            results[symbol] = {
                'zero_volume_count': 0,
                'first_3_dates': [],
                'zero_ratio': 0.0
            }
            logger.debug(f"{symbol}: No zero volume days")
    
    results['_summary'] = {
        'total_symbols': len(volume_matrix.columns),
        'total_zero_volume_days': total_zero_days,
        'symbols_with_zeros': sum(1 for k, v in results.items() if k != '_summary' and v['zero_volume_count'] > 0)
    }
    
    logger.info(f"Zero volume check complete: {total_zero_days} total zero days")
    return results

def check_ohlc_sanity(price_matrix: pd.DataFrame) -> Dict:
    """Verify OHLC relationships."""
    logger.info("Checking OHLC sanity from raw Parquet files...")
    results = {}
    all_passed = True
    total_errors = 0
    
    for symbol in UNIVERSE:
        file_path = raw_path(symbol)
        if not file_path.exists():
            logger.warning(f"{symbol} raw file not found - skipping OHLC check")
            continue
        df = pd.read_parquet(file_path)
        high_low_violations = df[df['high'] < df['low']]
        high_low_count = len(high_low_violations)
        close_below_low = df[df['close'] < df['low']]
        close_above_high = df[df['close'] > df['high']]
        close_violations = len(close_below_low) + len(close_above_high)
        errors = high_low_count + close_violations
        if errors > 0:
            all_passed = False
            total_errors += errors
            logger.error(f"{symbol}: {errors} OHLC violations - {high_low_count} high<low, {close_violations} close outside [low, high]")
        else:
            logger.debug(f"{symbol}: OHLC sanity OK")
        results[symbol] = {
            'high_lt_low_count': high_low_count,
            'close_outside_range': close_violations,
            'total_errors': errors,
            'passed': errors == 0
        }
    
    results['_summary'] = {
        'total_checked': len([r for r in results.keys() if r != '_summary']),
        'passed_count': sum(1 for k, v in results.items() if k != '_summary' and v['passed']),
        'total_errors': total_errors,
        'overall_pass': all_passed
    }
    
    logger.info(f"OHLC sanity check complete: {results['_summary']['passed_count']} symbols passed")
    return results

def check_price_continuity(price_matrix: pd.DataFrame, max_gap: int = 3) -> Dict:
    """Detect gaps > 3 consecutive missing business days."""
    logger.info(f"Checking price continuity (max gap: {max_gap} business days)...")
    full_date_range = pd.date_range(start=price_matrix.index.min(), end=price_matrix.index.max(), freq='B')
    date_set = set(full_date_range)
    results = {}
    total_gaps_found = 0
    
    for symbol in price_matrix.columns:
        present_dates = set(price_matrix[symbol].dropna().index)
        missing_dates = sorted(date_set - present_dates)
        if not missing_dates:
            results[symbol] = {'gap_count': 0, 'max_gap_days': 0, 'gap_intervals': [], 'has_large_gap': False}
            continue
        gaps = []
        current_gap = [missing_dates[0]]
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i-1]).days == 1:
                current_gap.append(missing_dates[i])
            else:
                if len(current_gap) > 0:
                    gaps.append(current_gap)
                current_gap = [missing_dates[i]]
        if current_gap:
            gaps.append(current_gap)
        large_gaps = [g for g in gaps if len(g) > max_gap]
        total_gaps_found += len(large_gaps)
        if large_gaps:
            gap_intervals = [{'start': g[0].strftime('%Y-%m-%d'), 'end': g[-1].strftime('%Y-%m-%d'), 'days': len(g)} for g in large_gaps]
            logger.warning(f"{symbol}: {len(large_gaps)} gaps > {max_gap} days, longest: {max(len(g) for g in large_gaps)} days")
        else:
            gap_intervals = []
            logger.debug(f"{symbol}: No large gaps detected")
        results[symbol] = {'gap_count': len(large_gaps), 'max_gap_days': max([len(g) for g in large_gaps]) if large_gaps else 0, 'gap_intervals': gap_intervals, 'has_large_gap': len(large_gaps) > 0}
    
    results['_summary'] = {
        'total_symbols': len(price_matrix.columns),
        'symbols_with_large_gaps': sum(1 for k, v in results.items() if k != '_summary' and v['has_large_gap']),
        'total_large_gaps': total_gaps_found,
        'max_gap_threshold': max_gap
    }
    
    logger.info(f"Price continuity check complete: {results['_summary']['symbols_with_large_gaps']} symbols with large gaps")
    return results

def generate_quality_report() -> pd.DataFrame:
    """Run all quality checks and generate a comprehensive report."""
    logger.info("=" * 60)
    logger.info("Starting comprehensive data quality report")
    logger.info("=" * 60)
    
    try:
        price_df, returns_df, volume_df = load_price_matrix()
        missing_results = check_missing_bars(price_df)
        spike_results = check_return_spikes(returns_df)
        zero_vol_results = check_zero_volume(volume_df)
        ohlc_results = check_ohlc_sanity(price_df)
        continuity_results = check_price_continuity(price_df)
        
        report_data = []
        for symbol in UNIVERSE:
            missing = missing_results.get(symbol, {})
            spikes = spike_results.get(symbol, {})
            zero_vol = zero_vol_results.get(symbol, {})
            ohlc = ohlc_results.get(symbol, {})
            continuity = continuity_results.get(symbol, {})
            report_data.append({
                'symbol': symbol,
                'missing_ratio': missing.get('missing_ratio', 0.0),
                'missing_days': missing.get('missing_days', 0),
                'spike_count': spikes.get('spike_count', 0),
                'max_return': spikes.get('max_return', 0.0),
                'min_return': spikes.get('min_return', 0.0),
                'zero_volume_days': zero_vol.get('zero_volume_count', 0),
                'zero_volume_ratio': zero_vol.get('zero_ratio', 0.0),
                'ohlc_errors': ohlc.get('total_errors', 0),
                'ohlc_passed': ohlc.get('passed', False),
                'large_gaps': continuity.get('gap_count', 0),
                'max_gap_days': continuity.get('max_gap_days', 0),
                'has_large_gap': continuity.get('has_large_gap', False),
                'missing_ratio_pass': missing.get('passed', True),
                'quality_score': 100.0 - (
                    (missing.get('missing_ratio', 0) * 100) * 0.3 +
                    (spikes.get('spike_count', 0) / 10) * 0.2 +
                    zero_vol.get('zero_volume_count', 0) * 0.5 +
                    ohlc.get('total_errors', 0) * 5.0 +
                    continuity.get('gap_count', 0) * 2.0
                )
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.attrs['generated_at'] = datetime.now().isoformat()
        report_df.attrs['num_symbols'] = len(UNIVERSE)
        output_path = DATA_PROC_DIR / "quality_report.csv"
        report_df.to_csv(output_path, index=False)
        logger.info(f"Quality report saved to {output_path}")
        
        logger.info("=" * 60)
        logger.info("QUALITY REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Generated at: {report_df.attrs['generated_at']}")
        logger.info(f"Symbols checked: {len(report_df)}")
        logger.info(f"Average quality score: {report_df['quality_score'].mean():.1f}/100")
        logger.info(f"Min quality score: {report_df['quality_score'].min():.1f}")
        logger.info(f"Max quality score: {report_df['quality_score'].max():.1f}")
        logger.info(f"Failed missing ratio (>5%): {len(report_df[report_df['missing_ratio_pass'] == False])}")
        logger.info(f"Symbols with OHLC errors: {len(report_df[report_df['ohlc_errors'] > 0])}")
        logger.info(f"Symbols with large gaps: {len(report_df[report_df['has_large_gap'] == True])}")
        logger.info("=" * 60)
        return report_df
    except Exception as e:
        logger.error(f"Quality report generation failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    report = generate_quality_report()
    print("Quality report preview:")
    print(report[['symbol', 'missing_ratio', 'spike_count', 'quality_score']].head())
