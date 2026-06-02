import faulthandler
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import psutil
import torch

from idssp.sonk.config import get_container_usage
from idssp.sonk.config import Config

# -----------------------------------------------------------------------------
# Module-level variable to ensure consistent timestamp across all loggers
# -----------------------------------------------------------------------------
_RUN_LOG_FILE: Optional[Path] = None

def get_active_log_file() -> Optional[Path]:
    """
    Returns the path to the current run's log file, or None if file logging is disabled.
    """
    return _RUN_LOG_FILE


def configure_logging(config: Config) -> logging.Logger:
    """
    Configure shared console and file handlers on the root logger.

    All module loggers created with logging.getLogger(__name__) will inherit these
    handlers through propagation.
    
    Params
    ------
    `log_dir`: Optional[Path]
        Directory where the log file will be written. If None, file logging is disabled.
    `console_level`: int | str
        Logging level for console output (e.g., logging.INFO or 'INFO'). Defaults to INFO.
    `file_level`: int | str
        Logging level for file output (e.g., logging.DEBUG or 'DEBUG'). Defaults to DEBUG.
    
    Returns
    -------
    logging.Logger
        The configured root logger instance.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return root_logger

    # Set root logger to the lowest level to ensure all messages are processed by handlers
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (stdout, filtered by console_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.LOG_LEVEL_CONSOLE)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. File Handler 
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / "training.log"

    if _RUN_LOG_FILE is None:
        globals()["_RUN_LOG_FILE"] = log_file

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.LOG_LEVEL_FILE)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.debug("Logging initialized. File: %s", log_file)

    return root_logger


def get_logger(
        name: str
    ) -> logging.Logger:
    """
    Return a named logger with optional shared logging bootstrap.

    Params
    ------
    `name`: str
        Name of the logger (usually __name__).

    Returns
    -------
    logging.Logger
        A named logger instance.
    """
    return logging.getLogger(name)

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

    # CPU Memory Usage
    try:
        limit, usage, free, percentage = get_container_usage()
        if limit <= 0:
            cpu_memory = psutil.virtual_memory()
            cpu_memory_used = cpu_memory.used / (1024 ** 3)  # GB
            cpu_memory_total = cpu_memory.total / (1024 ** 3)  # GB
            logger.info("%sCPU (total) Memory - Used: %.2f GB, Total: %.2f GB",
                        prefix, cpu_memory_used, cpu_memory_total)
        else:
            logger.info("%sCPU (container) Memory - Used: %.2f GB, Limit: %.2f GB, "
                        "Free: %.2f GB (%.1f%% of limit used)",
                        prefix, usage, limit, free, percentage)
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("%sCould not retrieve CPU memory usage: %s", prefix, str(e))

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
