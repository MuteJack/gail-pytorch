# utils/logger.py

""" Import Library """
# Standard library imports
import logging
import sys
import os
import warnings
from datetime import datetime


# Add custom log level for FutureWarning details
FUTURE_WARNING_LEVEL = 19
logging.addLevelName(FUTURE_WARNING_LEVEL, "FUTUREWARN")


class LevelFilter(logging.Filter):
    """Filter to log only specific level range"""
    def __init__(self, min_level, max_level):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        return self.min_level <= record.levelno <= self.max_level


def setup_logger(name="GAIL", level=logging.INFO, log_dir="logs", console=True, file=True):
    """
    Setup logger with console and/or file output

    Log routing:
    - Console: INFO (20) and above
    - train_{timestamp}.log: INFO (20) and above
    - train_{timestamp}_debug.log: Below INFO (< 20), typically DEBUG

    Custom levels are supported based on their numeric value:
    - Level < 20 (e.g., TRACE=5, DEBUG=10, VERBOSE=15) -> debug.log only
    - Level >= 20 (e.g., INFO=20, SUCCESS=25, WARNING=30) -> train.log + console

    Args:
        name: Logger name
        level: Logging level for console (default: INFO)
        log_dir: Directory to save log files
        console: Enable console output
        file: Enable file output
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Avoid adding multiple handlers
    if logger.handlers:
        return logger

    # Create formatters
    # Format: {time}, [{Log Level}], [{python 파일 이름}], {Message}
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s, [%(levelname)s], [%(filename)s], %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s, [%(levelname)s], [%(filename)s], %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler - INFO and above
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Only INFO (20) and above
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handlers
    if file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Main log file - INFO and above
        main_log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        main_handler = logging.FileHandler(main_log_file, mode='w', encoding='utf-8')
        main_handler.setLevel(logging.INFO)  # INFO (20) and above
        main_handler.setFormatter(detailed_formatter)
        logger.addHandler(main_handler)

        # Debug log file - Below INFO (< 20)
        debug_log_file = os.path.join(log_dir, f'train_{timestamp}_debug.log')
        debug_handler = logging.FileHandler(debug_log_file, mode='w', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(LevelFilter(0, logging.INFO - 1))  # Below INFO
        debug_handler.setFormatter(detailed_formatter)
        logger.addHandler(debug_handler)

        # Print log file locations
        if console:
            print(f"Logging to:")
            print(f"  - {main_log_file} (INFO+)")
            print(f"  - {debug_log_file} (DEBUG)")

    return logger


def redirect_warnings_to_logger(logger_instance):
    """
    Redirect Python warnings to logger
    - FutureWarning: Shows summary at WARNING level, details at FUTUREWARN (19) level
    - Other warnings: Logged at WARNING level
    """

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        # Format the warning
        warning_msg = warnings.formatwarning(message, category, filename, lineno, line)

        if category == FutureWarning:
            # Summary for console/main log (WARNING level)
            summary = f"FutureWarning in {filename}:{lineno}"
            logger_instance.warning(summary)

            # Detailed message for debug log (FUTUREWARN level = 19)
            logger_instance.log(FUTURE_WARNING_LEVEL, f"FutureWarning Details:\n{warning_msg.strip()}")
        else:
            # Other warnings go to WARNING level
            logger_instance.warning(warning_msg.strip())

    # Replace warning handler
    warnings.showwarning = warning_handler


# Global logger instance
logger = setup_logger()

# Redirect warnings to logger
redirect_warnings_to_logger(logger)


def get_logger(name=None, **kwargs):
    """
    Get logger instance

    Args:
        name: Logger name (if None, returns global logger)
        **kwargs: Additional arguments for setup_logger
    """
    if name:
        return setup_logger(name, **kwargs)
    return logger