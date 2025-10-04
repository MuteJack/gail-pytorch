# utils/logger.py

""" Import Library """
# Standard library imports
import logging
import sys
import os
import warnings
from datetime import datetime


""" Custom Log Level """
# Add custom log level for FutureWarning details
FUTURE_WARNING_LEVEL = 19
logging.addLevelName(FUTURE_WARNING_LEVEL, "FUTUREWARN")


""" Log Filter """
class LevelFilter(logging.Filter):
    """
    Filter to log only specific level range
    Used to separate debug logs from main logs
    """
    def __init__(self, min_level, max_level):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        """Only pass logs within the level range"""
        return self.min_level <= record.levelno <= self.max_level


""" Logger Setup Function """
def setup_logger(name="GAIL", level=logging.INFO, log_dir="logs", console=True, file=True):
    """
    Setup logger with dual output: console and file

    Log Level Routing:
    - Console: INFO (20) and above
    - log_{timestamp}_info.log: INFO (20) and above
    - log_{timestamp}_debug.log: Below INFO (< 20), typically DEBUG

    Custom levels are supported based on their numeric value:
    - Level < 20 (e.g., TRACE=5, DEBUG=10, VERBOSE=15, FUTUREWARN=19) -> debug.log only
    - Level >= 20 (e.g., INFO=20, WARNING=30, ERROR=40) -> info.log + console

    Args:
        name: Logger name (default: "GAIL")
        level: Logging level for console (default: INFO)
        log_dir: Directory to save log files (default: "logs")
        console: Enable console output (default: True)
        file: Enable file output (default: True)

    Returns:
        Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Avoid adding multiple handlers
    if logger.handlers:
        return logger


    """ Formatters """
    # Format: {time}, [{Log Level}], [{python file name}], {Message}
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s, [%(levelname)s], [%(filename)s], %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s, [%(levelname)s], [%(filename)s], %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    """ Console Handler """
    # Console handler - INFO (20) and above
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)


    """ File Handlers """
    if file:
        # Create log directory if not exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Generate timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Main log file - INFO (20) and above
        main_log_file = os.path.join(log_dir, f'log_{timestamp}_info.log')
        main_handler = logging.FileHandler(main_log_file, mode='w', encoding='utf-8')
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(detailed_formatter)
        logger.addHandler(main_handler)

        # Debug log file - Below INFO (< 20)
        debug_log_file = os.path.join(log_dir, f'log_{timestamp}_debug.log')
        debug_handler = logging.FileHandler(debug_log_file, mode='w', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(LevelFilter(0, logging.INFO - 1))  # Only logs < INFO
        debug_handler.setFormatter(detailed_formatter)
        logger.addHandler(debug_handler)

        # Print log file locations
        if console:
            print(f"Logging to:")
            print(f"  - {main_log_file} (INFO+)")
            print(f"  - {debug_log_file} (DEBUG)")

    return logger



""" Warning Redirection """
def redirect_warnings_to_logger(logger_instance):
    """
    Redirect Python warnings to logger

    Behavior:
    - FutureWarning: Summary at WARNING level, details at FUTUREWARN (19) level
    - Other warnings: Logged at WARNING level

    Args:
        logger_instance: Logger instance to redirect warnings to
    """

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        """Custom warning handler"""
        # Format the warning
        warning_msg = warnings.formatwarning(message, category, filename, lineno, line)

        if category == FutureWarning:
            # Summary for console/main log (WARNING level = 30)
            summary = f"FutureWarning in {filename}:{lineno}"
            logger_instance.warning(summary)

            # Detailed message for debug log (FUTUREWARN level = 19)
            logger_instance.log(FUTURE_WARNING_LEVEL, f"FutureWarning Details:\n{warning_msg.strip()}")
        else:
            # Other warnings go to WARNING level
            logger_instance.warning(warning_msg.strip())

    # Replace default warning handler
    warnings.showwarning = warning_handler


""" Global Logger Instance """
# Create global logger
logger = setup_logger()

# Redirect warnings to logger
redirect_warnings_to_logger(logger)


""" Logger Access Function """
def get_logger(name=None, **kwargs):
    """
    Get logger instance

    Args:
        name: Logger name (if None, returns global logger)
        **kwargs: Additional arguments for setup_logger

    Returns:
        Logger instance
    """
    if name:
        return setup_logger(name, **kwargs)
    return logger

# EOS - End of Script