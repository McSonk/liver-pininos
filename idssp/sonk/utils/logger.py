import logging
import sys
from pathlib import Path
from typing import Optional

def get_logger(
        name: str,
        log_dir: Optional[Path] = None,
        level: Optional[int] = None
    ) -> logging.Logger:
    """
    Creates a logger that outputs to both Console and File.

    Params
    ------
    `name`: str
        Name of the logger (usually __name__).
    `log_dir`: Optional[Path]
         Directory where log files will be saved. If None, file logging is disabled.
    `level`: Optional[int]
        Logging level (e.g., logging.INFO). If None, defaults to logging.INFO.

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
    if level is None:
        level = config.LOG_LEVEL


    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times if imported repeatedly
    if logger.handlers:
        return logger

    # Format: [Time] [Level] [Module] Message
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (if log_dir is provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to file: {%s}", log_file)

    return logger
