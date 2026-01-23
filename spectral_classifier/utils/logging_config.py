"""
Logging configuration for spectral transect classification system.

This module provides centralized logging setup with separate console
and file handlers for different verbosity levels.
"""

import logging
from pathlib import Path
from typing import Optional

from ..config import LOG_FORMAT


def setup_logging(
    verbose: bool = True,
    log_file: Optional[Path] = None
) -> None:
    """
    Configure logging for the application.

    Console (terminal): Shows only WARNING and ERROR
    File: Shows INFO and above (or DEBUG if verbose=True)

    Args:
        verbose: Enable debug-level logging in file
        log_file: Optional path to log file. If None, only console logging is used.
    
    Example:
        >>> setup_logging(verbose=True, log_file=Path("output/processing.log"))
    """
    # Set file logging level
    file_level = logging.DEBUG if verbose else logging.INFO

    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler - only WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - INFO/DEBUG and above
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Windows compatibility: Use ASCII encoding with error handling
        file_handler = logging.FileHandler(
            log_file, 
            encoding='ascii', 
            errors='replace'
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)