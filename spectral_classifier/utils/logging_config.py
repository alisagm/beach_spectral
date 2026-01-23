"""
Logging configuration for spectral transect classification system.

This module provides centralized logging setup with separate console
and file handlers for different verbosity levels, plus a ProgressTracker
for clean console progress display.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

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


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

@dataclass
class ProgressTracker:
    """
    Unified progress tracking for batch processing.
    
    Provides:
    - Single-line overwriting console progress (bypasses logging)
    - Elapsed time and ETA calculation
    - Aggregated skip/error statistics (vs per-item logging spam)
    - Summary output at completion
    
    Console output uses direct print() with carriage return to overwrite,
    while detailed logs still go to the log file via standard logging.
    
    Usage:
        tracker = ProgressTracker(total_items=2100, item_noun="transect")
        tracker.set_context("2016")  # Optional: show year/batch context
        
        for i, item in enumerate(items):
            tracker.update(i + 1)
            # ... do work ...
            tracker.add_skipped("points", 5)  # Aggregate skip counts
            
        tracker.finish()
        tracker.print_summary()
    
    Example output:
        [2016] Transect 47/2100 [5m 23s elapsed, ETA: ~2h 10m]
    """
    total_items: int
    item_noun: str = "item"
    
    # Post-init populated fields
    _current: int = field(default=0, init=False)
    _start_time: float = field(default=0.0, init=False)
    _skipped: Dict[str, int] = field(default_factory=dict, init=False)
    _errors: List[str] = field(default_factory=list, init=False)
    _context: str = field(default="", init=False)
    _last_line_len: int = field(default=0, init=False)
    _processed_count: int = field(default=0, init=False)
    
    def __post_init__(self):
        self._start_time = time.time()
        self._skipped = {}
        self._errors = []
        self._current = 0
        self._context = ""
        self._last_line_len = 0
        self._processed_count = 0
    
    def set_context(self, context: str) -> None:
        """
        Set context string displayed in progress (e.g., year being processed).
        
        Args:
            context: Context string like "2016" or "batch 1/5"
        """
        self._context = context
    
    def update(self, current: int, status: str = "") -> None:
        """
        Update progress display (overwrites current line).
        
        Args:
            current: Current item number (1-indexed)
            status: Optional status message (truncated if too long)
        """
        self._current = current
        self._print_progress(status)
    
    def increment(self, status: str = "") -> None:
        """
        Increment progress by 1 and update display.
        
        Args:
            status: Optional status message
        """
        self._current += 1
        self._processed_count += 1
        self._print_progress(status)
    
    def add_skipped(self, category: str, count: int = 1) -> None:
        """
        Record skipped items (aggregated, not logged individually).
        
        Args:
            category: Category of skipped item (e.g., "points", "rasters")
            count: Number of items skipped
        """
        if category not in self._skipped:
            self._skipped[category] = 0
        self._skipped[category] += count
    
    def add_error(self, message: str) -> None:
        """
        Record an error message.
        
        Args:
            message: Error description
        """
        self._errors.append(message)
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self._start_time
    
    @property
    def processed(self) -> int:
        """Get count of successfully processed items."""
        return self._processed_count
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _print_progress(self, status: str = "") -> None:
        """Print single-line progress (overwrites previous line)."""
        elapsed = self.elapsed_seconds
        
        # Calculate ETA
        if self._current > 0:
            rate = self._current / elapsed
            remaining = self.total_items - self._current
            eta = remaining / rate if rate > 0 else 0
            eta_str = f"ETA: ~{self._format_time(eta)}"
        else:
            eta_str = "ETA: --"
        
        # Build progress line
        context_str = f"[{self._context}] " if self._context else ""
        progress_str = f"{self.item_noun.capitalize()} {self._current}/{self.total_items}"
        time_str = f"[{self._format_time(elapsed)} elapsed, {eta_str}]"
        
        # Truncate status if needed
        if status:
            max_status_len = 30
            if len(status) > max_status_len:
                status = status[:max_status_len - 3] + "..."
            status = f" - {status}"
        
        line = f"\r{context_str}{progress_str} {time_str}{status}"
        
        # Pad with spaces to overwrite previous longer line
        padding = max(0, self._last_line_len - len(line))
        line += " " * padding
        
        self._last_line_len = len(line) - padding  # Store actual content length
        
        sys.stdout.write(line)
        sys.stdout.flush()
    
    def finish(self, success_count: int = None) -> None:
        """
        Mark processing complete and print final status line.
        
        Args:
            success_count: Override for successful count (uses _processed_count if None)
        """
        elapsed = self.elapsed_seconds
        count = success_count if success_count is not None else self._processed_count
        
        # Clear the progress line and print completion message
        clear_line = "\r" + " " * self._last_line_len + "\r"
        sys.stdout.write(clear_line)
        
        print(f"Completed {count}/{self.total_items} {self.item_noun}s "
              f"in {self._format_time(elapsed)}")
    
    def print_summary(self) -> None:
        """Print aggregated summary of skipped items and errors."""
        if self._skipped:
            print("\nSkipped during processing:")
            for category, count in sorted(self._skipped.items()):
                print(f"  {category}: {count:,}")
        
        if self._errors:
            print(f"\nErrors ({len(self._errors)}):")
            # Show first few errors
            for err in self._errors[:5]:
                print(f"  - {err}")
            if len(self._errors) > 5:
                print(f"  ... and {len(self._errors) - 5} more")
    
    def reset(self, total_items: int = None) -> None:
        """
        Reset tracker for a new batch (e.g., new year).
        
        Args:
            total_items: New total (keeps current if None)
        """
        if total_items is not None:
            self.total_items = total_items
        self._current = 0
        self._start_time = time.time()
        # Note: Does NOT reset _skipped or _errors - those accumulate across batches
        # Call clear_stats() explicitly if you want to reset those too
    
    def clear_stats(self) -> None:
        """Clear accumulated skip and error statistics."""
        self._skipped = {}
        self._errors = []
        self._processed_count = 0