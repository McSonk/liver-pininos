import faulthandler
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import torch

# -----------------------------------------------------------------------------
# Module-level variable to ensure consistent timestamp across all loggers
# -----------------------------------------------------------------------------
_RUN_TIMESTAMP: Optional[str] = None
_RUN_LOG_FILE: Optional[Path] = None

def _get_run_timestamp() -> str:
    """
    Returns a consistent timestamp for the entire Python process run.
    Generated once on first call, then cached.
    """
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_TIMESTAMP

def get_active_log_file() -> Optional[Path]:
    """
    Returns the path to the current run's log file, or None if file logging is disabled.
    Thread-safe: the value is set once during logger initialisation.
    """
    return _RUN_LOG_FILE


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
    # Lazy import to avoid circular dependencies (such as config.py importing this logger)
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
    # To prevent duplicate in case other libraries also use logging (e.g., MONAI),
    # we set propagate to False to avoid messages being passed to the root logger
    logger.propagate = False

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
        timestamp = _get_run_timestamp()
        log_file = log_dir / f"training_{timestamp}.log"
        global _RUN_LOG_FILE
        if _RUN_LOG_FILE is None:  # Only set once per process run
            _RUN_LOG_FILE = log_file

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level) 
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.debug("Logging initialized. File: %s", log_file)

    return logger

def log_memory_usage(logger: logging.Logger, prefix: str = ""):
    """
    Logs the current GPU and CPU memory usage.

    Params
    ------
    `logger`: logging.Logger
        The logger instance to use for logging memory usage.
    `prefix`: str
        Optional prefix to add to the log message for context.
    """

    # GPU Memory Usage
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB

        # get_device_properties returns a named tuple with 'total_memory' in bytes
        device_props = torch.cuda.get_device_properties(0)
        total_gb = device_props.total_memory / (1024 ** 3)

        # Calculate Free & Utilization
        free_gb = total_gb - gpu_memory_reserved
        utilization_pct = (gpu_memory_reserved / total_gb) * 100

        logger.info("%sGPU Memory - Alloc: %.2f GB | Reserv: %.2f GB | Total: %.2f GB "
                    "| Free: %.2f GB (%.1f%% used)",
                    prefix,
                    gpu_memory_allocated,
                    gpu_memory_reserved,
                    total_gb,
                    free_gb,
                    utilization_pct)

    # CPU Memory Usage (using psutil if available)
    try:
        cpu_memory = psutil.virtual_memory()
        cpu_memory_used = cpu_memory.used / (1024 ** 3)  # GB
        cpu_memory_total = cpu_memory.total / (1024 ** 3)  # GB
        logger.info("%sCPU Memory - Used: %.2f GB, Total: %.2f GB",
                    prefix, cpu_memory_used, cpu_memory_total)
    except ImportError:
        logger.info("%spsutil not installed. CPU memory usage not logged.", prefix)

def install_global_exception_handlers(logger: logging.Logger) -> None:
    """
    Install global exception hooks to ensure ALL unhandled Python-level errors
    are logged to your file handler.

    MUST be called AFTER your main logger (with FileHandler) is initialised.

    Params
    ------
    `logger`: logging.Logger
        Your fully-configured main logger (with file handler attached).
    """
    # 1. Main process unhandled exceptions
    original_sys_hook = sys.excepthook
    def _global_excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            original_sys_hook(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "UNHANDLED EXCEPTION — PROCESS EXITING",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        # Still print to stderr for immediate visibility
        original_sys_hook(exc_type, exc_value, exc_traceback)
    sys.excepthook = _global_excepthook

    # 2. Unhandled exceptions in background threads
    original_thread_hook = threading.excepthook
    def _global_thread_hook(args):
        logger.critical(
            "UNHANDLED EXCEPTION IN THREAD '%s'", args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )
        original_thread_hook(args)
    threading.excepthook = _global_thread_hook

    # 3. Optional: faulthandler for C-level crashes (writes to stderr)
    if not faulthandler.is_enabled():
        faulthandler.enable()
        logger.debug("faulthandler enabled for C-level crash diagnostics")

    logger.info("Global exception handlers installed.")
