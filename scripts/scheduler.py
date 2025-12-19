"""Automated scheduler for data ingestion."""
import schedule
import time
import subprocess
from pathlib import Path
from loguru import logger
import yaml


def run_ingestion():
    """Run data ingestion pipeline."""
    logger.info("Running scheduled data ingestion...")
    try:
        result = subprocess.run(
            ["python", "-m", "src.ingestion.main"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        if result.returncode == 0:
            logger.info("Data ingestion completed successfully")
        else:
            logger.error(f"Data ingestion failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running ingestion: {e}")


def main():
    """Main scheduler."""
    # Load configuration
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        schedule_config = config.get("ingestion", {}).get("schedule", {})
        frequency = schedule_config.get("frequency", "daily")
        time_str = schedule_config.get("time", "02:00")
        
        if frequency == "daily":
            schedule.every().day.at(time_str).do(run_ingestion)
        elif frequency == "hourly":
            schedule.every().hour.do(run_ingestion)
        elif frequency == "weekly":
            schedule.every().week.at(time_str).do(run_ingestion)
        
        logger.info(f"Scheduled data ingestion: {frequency} at {time_str}")
    else:
        # Default: daily at 2 AM
        schedule.every().day.at("02:00").do(run_ingestion)
        logger.info("Using default schedule: daily at 02:00")
    
    # Run once immediately (optional)
    # run_ingestion()
    
    # Keep running
    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()

