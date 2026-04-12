import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(
        name: str,
        log_dir: Optional[Path] = None,
        console_level: Optional[int] = None,
        file_level: Optional[int] = None
    ) -> logging.Logger:
    """
    Creates a logger that outputs to both Console and File.

    Params
    ------
    `name`: str
        Name of the logger (usually __name__).
    `log_dir`: Optional[Path]
         Directory where log files will be saved. If None, file logging is disabled.
    `console_level`: Optional[int]
        Logging level for console output (e.g., logging.INFO). If None, defaults to logging.INFO.
    `file_level`: Optional[int]
        Logging level for file output (e.g., logging.DEBUG). If None, defaults to logging.DEBUG.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Lazy import to avoid circular dependencies
    from idssp.sonk import config

    # Use config defaults if not provided
    if log_dir is None:
        log_dir = config.LOG_DIR
    if console_level is None:
        console_level = config.LOG_LEVEL_CONSOLE
    if file_level is None:
        file_level = config.LOG_LEVEL_FILE

    logger = logging.getLogger(name)
    # CRITICAL: Set the logger itself to the LOWEST level (DEBUG)
    # This ensures the logger accepts all messages, and the Handlers do the filtering.
    logger.setLevel(logging.DEBUG)

    # Prevent adding handlers multiple times if imported repeatedly
    if logger.handlers:
        return logger

    # Format: [Time] [Level] [Module] Message
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (stdout, filtered by console_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level) # e.g., INFO
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (if log_dir is provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"training_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level) 
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.info("Logging initialized. File: {%s}", log_file)

    return logger
