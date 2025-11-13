"""
Log manager module for handling file logging, rotation, and cleanup.

This module provides functionality to manage log files, including rotation
based on size and time, as well as automatic cleanup of old logs.
"""

import os
import gzip
import shutil
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from apscheduler.schedulers.background import BackgroundScheduler
from logging.handlers import RotatingFileHandler

from .config import config


class CompressingRotatingFileHandler(RotatingFileHandler):
    """
    Custom rotating file handler that compresses old log files.
    
    Extends RotatingFileHandler to compress rotated files with gzip.
    """
    
    def doRollover(self) -> None:
        """
        Do a rollover, as described in __init__().
        """
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]
        
        # Determine the base filename
        base_name = self.baseFilename
        
        # Rotate existing files
        for i in range(self.backupCount - 1, 0, -1):
            old_name = f"{base_name}.{i}.gz"
            new_name = f"{base_name}.{i + 1}.gz"
            if os.path.exists(old_name):
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
        
        # Compress the current log file
        if os.path.exists(base_name):
            compressed_name = f"{base_name}.1.gz"
            with open(base_name, 'rb') as f_in:
                with gzip.open(compressed_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(base_name)
        
        # Open a new log file
        self.stream = self._open()


class LogManager:
    """
    Manages log files, rotation, and cleanup.
    
    This class handles the creation and management of log file handlers,
    as well as periodic cleanup of old log files based on configured policies.
    """
    
    def __init__(self, config_override=None):
        """Initialize the log manager.
        
        Args:
            config_override: Optional config override for testing
        """
        self._handlers: Dict[str, logging.Handler] = {}
        self._scheduler: Optional[BackgroundScheduler] = None
        self._cleanup_lock = threading.Lock()
        self._initialized = False
        
        # Use provided config or global config
        self.config = config_override or config
        
        # Ensure log directory exists
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> None:
        """
        Initialize the log manager.
        
        Sets up file handlers for different log levels and starts the cleanup scheduler.
        """
        if self._initialized:
            return
        
        if not self.config.log_to_file:
            return
        
        # Create file handlers for each log level
        self._setup_file_handlers()
        
        # Start cleanup scheduler
        self._start_cleanup_scheduler()
        
        # Perform initial cleanup
        self._cleanup_logs()
        
        self._initialized = True
    
    def _setup_file_handlers(self) -> None:
        """
        Set up file handlers for different log levels.
        
        Creates separate file handlers for DEBUG, INFO, WARNING, and ERROR levels.
        """
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        
        # Get the JSON formatter from the logging module
        from .logging import JSONFormatter
        formatter = JSONFormatter()
        
        for level_name, level_value in log_levels.items():
            # Create file handler for this level
            filename = self.log_dir / f"app-{datetime.now().strftime('%Y-%m-%d')}-{level_name.lower()}.log"
            
            handler = CompressingRotatingFileHandler(
                filename=str(filename),
                maxBytes=self.config.log_file_max_size * 1024 * 1024,  # Convert MB to bytes
                backupCount=5,  # Keep 5 compressed backups
                encoding='utf-8'
            )
            
            handler.setLevel(level_value)
            handler.setFormatter(formatter)
            
            # Store handler for later use
            self._handlers[level_name] = handler
    
    def _start_cleanup_scheduler(self) -> None:
        """
        Start the background scheduler for periodic log cleanup.
        """
        if self._scheduler is None:
            self._scheduler = BackgroundScheduler()
            self._scheduler.add_job(
                func=self._cleanup_logs,
                trigger="interval",
                hours=self.config.log_cleanup_interval_hours,
                id='log_cleanup'
            )
            self._scheduler.start()
    
    def _cleanup_logs(self) -> None:
        """
        Clean up old log files based on configured policies.
        
        Implements the progressive cleanup strategy:
        1. If total size > max_dir_size_mb, try cleaning with retention_days
        2. If still too large, try with shorter retention periods
        3. Continue until size is acceptable or all options exhausted
        """
        with self._cleanup_lock:
            try:
                # Calculate current log directory size
                total_size = self._calculate_dir_size()
                max_size_bytes = self.config.log_max_dir_size_mb * 1024 * 1024
                target_size_bytes = self.config.log_cleanup_target_size_mb * 1024 * 1024
                
                if total_size <= max_size_bytes:
                    return  # No cleanup needed
                
                # Log cleanup start
                cleanup_logger = self._get_cleanup_logger()
                cleanup_logger.info(
                    f"Starting log cleanup. Current size: {total_size / (1024*1024):.2f}MB, "
                    f"Max allowed: {self.config.log_max_dir_size_mb}MB, "
                    f"Target: {self.config.log_cleanup_target_size_mb}MB"
                )
                
                # Try progressive cleanup
                for retention_days in self.config.get_log_cleanup_retention_days():
                    self._delete_old_logs(retention_days)
                    total_size = self._calculate_dir_size()
                    
                    cleanup_logger.info(
                        f"After cleanup with {retention_days} days retention: "
                        f"{total_size / (1024*1024):.2f}MB"
                    )
                    
                    # Stop if we're under the target size
                    if total_size <= target_size_bytes:
                        break
                
                cleanup_logger.info(
                    f"Log cleanup completed. Final size: {total_size / (1024*1024):.2f}MB"
                )
                
            except Exception as e:
                # Log cleanup errors
                try:
                    cleanup_logger = self._get_cleanup_logger()
                    cleanup_logger.error(f"Error during log cleanup: {e}", exc_info=True)
                except Exception:
                    # If even cleanup logger fails, just print to stderr
                    print(f"Error during log cleanup: {e}")
    
    def _calculate_dir_size(self) -> int:
        """
        Calculate the total size of the log directory in bytes.
        
        Returns:
            int: Total size in bytes
        """
        total_size = 0
        for file_path in self.log_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _delete_old_logs(self, retention_days: int) -> None:
        """
        Delete log files older than the specified retention period.
        
        Args:
            retention_days: Number of days to retain logs
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for file_path in self.log_dir.rglob('*'):
            if file_path.is_file() and file_path.name.endswith(('.log', '.log.gz')):
                # Extract date from filename
                try:
                    # Expected format: app-YYYY-MM-DD-level.log
                    # Handle both .log and .log.gz files
                    stem_name = file_path.stem
                    if stem_name.endswith('.log'):
                        stem_name = stem_name[:-4]  # Remove .log suffix
                    
                    parts = stem_name.split('-')
                    if len(parts) >= 4:  # app-YYYY-MM-DD-level
                        date_part = '-'.join(parts[1:4])  # Get YYYY-MM-DD part
                        file_date = datetime.strptime(date_part, '%Y-%m-%d')
                        
                        if file_date < cutoff_date:
                            file_path.unlink()
                except (ValueError, IndexError):
                    # If we can't parse the date, skip the file
                    continue
    
    def _get_cleanup_logger(self) -> logging.Logger:
        """
        Get a logger for cleanup operations.
        
        Returns:
            logging.Logger: Logger instance for cleanup operations
        """
        # Create a simple file logger for cleanup operations
        cleanup_logger = logging.getLogger('emb_model_provider.cleanup')
        
        # If no handlers exist, create a simple console handler
        if not cleanup_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            cleanup_logger.addHandler(handler)
            cleanup_logger.setLevel(logging.INFO)
        
        return cleanup_logger
    
    def get_handler(self, level_name: str) -> Optional[logging.Handler]:
        """
        Get the file handler for a specific log level.
        
        Args:
            level_name: Name of the log level (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            Optional[logging.Handler]: File handler for the specified level
        """
        return self._handlers.get(level_name)
    
    def shutdown(self) -> None:
        """
        Shutdown the log manager.
        
        Stops the cleanup scheduler and closes all file handlers.
        """
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown()
        
        # Close all handlers
        for handler in self._handlers.values():
            handler.close()
        
        self._handlers.clear()
        self._initialized = False


# Global log manager instance
log_manager = LogManager()