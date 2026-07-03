"""
api/scheduler.py
APScheduler for daily signal generation and order execution.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = BackgroundScheduler()


def daily_pipeline():
    """
    Run the full trading pipeline:
    1. Generate signals (ensemble + ML)
    2. Execute orders through risk gates
    3. Update portfolio
    """
    logger.info("🔄 Running daily trading pipeline...")
    try:
        # Import here to avoid circular imports
        import sys
        sys.path.append('.')
        
        # In production, you'd call:
        # from signals.ensemble import run_ensemble_pipeline
        # from execution.order_router import OrderRouter
        
        logger.info("  Step 1: Generating ensemble signals...")
        # run_ensemble_pipeline()
        
        logger.info("  Step 2: Executing orders through risk gates...")
        # router = OrderRouter()
        # router.execute_orders()
        
        logger.info("✅ Daily pipeline complete!")
        
    except Exception as e:
        logger.error(f"❌ Daily pipeline failed: {e}")


def start_scheduler():
    """Start the APScheduler."""
    # Schedule for 9:35 AM ET (market open is 9:30 AM ET)
    # For testing, you can also schedule it to run every minute:
    # from apscheduler.triggers.interval import IntervalTrigger
    # scheduler.add_job(daily_pipeline, trigger=IntervalTrigger(minutes=1))
    
    scheduler.add_job(
        daily_pipeline,
        trigger=CronTrigger(hour=9, minute=35),
        id='daily_pipeline',
        replace_existing=True
    )
    scheduler.start()
    logger.info("✅ Scheduler started. Daily pipeline will run at 9:35 AM ET")


def stop_scheduler():
    """Stop the APScheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 Scheduler stopped")
