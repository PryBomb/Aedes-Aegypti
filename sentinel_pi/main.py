import logging
import sys
from pathlib import Path

from sentinel.logger import setup_logging
from sentinel.config import load_config
from sentinel.storage.db import DatabaseManager

logger = setup_logging()

def main():
    logger.info("Starting MosquitoSentinel...")
    
    try:
        config = load_config("config.yaml")
        logger.info(f"Loaded config for monitoring point: {config.site.monitoring_point}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Initialize Database
    db_path = Path("data/sentinel.db")
    schema_path = Path("sentinel/storage/schema.sql")
    
    db_manager = DatabaseManager(db_path, schema_path)
    logger.info(f"Database initialized at {db_path.resolve()}")
    
    logger.info("Phase 1 skeleton complete. Pipeline and Dashboard not yet started.")
    # TODO: Orchestrate FrameSource, Detector, EventBuilder, Dashboard, etc.

if __name__ == "__main__":
    main()
