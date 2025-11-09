"""
Unit tests for the log manager module.
"""

import gzip
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from emb_model_provider.core.config import Config
from emb_model_provider.core.log_manager import CompressingRotatingFileHandler, LogManager


class TestCompressingRotatingFileHandler:
    """Test cases for CompressingRotatingFileHandler class."""
    
    def test_rotates_and_compresses_file(self):
        """Test that the handler rotates and compresses files correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test log file
            log_file = Path(temp_dir) / "test.log"
            log_file.write_text("Test log content")
            
            # Create handler with very small max size to force rotation
            handler = CompressingRotatingFileHandler(
                filename=str(log_file),
                maxBytes=10,  # Very small to trigger rotation
                backupCount=2
            )
            
            # Force rotation by calling doRollover directly
            handler.doRollover()
            
            # Close handler to release file locks
            handler.close()
            
            # Check that compressed file exists
            compressed_file = Path(f"{log_file}.1.gz")
            assert compressed_file.exists()
            
            # Check compressed content
            with gzip.open(compressed_file, 'rt') as f:
                content = f.read()
                assert "Test log content" in content


class TestLogManager:
    """Test cases for LogManager class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.log_to_file = True
        self.config.log_dir = self.temp_dir
        self.config.log_file_max_size = 1  # 1MB for testing
        self.config.log_retention_days = 7
        self.config.log_cleanup_interval_hours = 1
        self.config.log_max_dir_size_mb = 50
        self.config.log_cleanup_target_size_mb = 20
        self.config.log_cleanup_retention_days = "7,3,1"
    
    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization_creates_log_directory(self):
        """Test that initialization creates the log directory."""
        log_manager = LogManager(self.config)
        log_manager.initialize()
        
        assert Path(self.temp_dir).exists()
        assert Path(self.temp_dir).is_dir()
    
    def test_initialization_creates_file_handlers(self):
        """Test that initialization creates file handlers for each level."""
        log_manager = LogManager(self.config)
        log_manager.initialize()
        
        # Check that handlers are created for each level
        for level_name in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            handler = log_manager.get_handler(level_name)
            assert handler is not None
            assert isinstance(handler, CompressingRotatingFileHandler)
    
    def test_initialization_skipped_when_log_to_file_is_false(self):
        """Test that initialization is skipped when log_to_file is False."""
        self.config.log_to_file = False
        log_manager = LogManager(self.config)
        log_manager.initialize()
        
        # No handlers should be created
        for level_name in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            handler = log_manager.get_handler(level_name)
            assert handler is None
    
    def test_cleanup_logs_removes_old_files(self):
        """Test that cleanup removes old log files."""
        log_manager = LogManager(self.config)
        
        # Create old log files
        old_date = datetime.now() - timedelta(days=10)
        old_file = Path(self.temp_dir) / f"app-{old_date.strftime('%Y-%m-%d')}-info.log"
        old_file.write_text("Old log content")
        
        # Create recent log files
        recent_date = datetime.now() - timedelta(days=1)
        recent_file = Path(self.temp_dir) / f"app-{recent_date.strftime('%Y-%m-%d')}-info.log"
        recent_file.write_text("Recent log content")
        
        # Run cleanup
        log_manager._delete_old_logs(7)
        
        # Check that old file is removed and recent file remains
        assert not old_file.exists()
        assert recent_file.exists()
    
    def test_cleanup_logs_handles_compressed_files(self):
        """Test that cleanup handles compressed log files."""
        log_manager = LogManager(self.config)
        
        # Create old compressed log files
        old_date = datetime.now() - timedelta(days=10)
        old_file = Path(self.temp_dir) / f"app-{old_date.strftime('%Y-%m-%d')}-info.log.gz"
        with gzip.open(old_file, 'wt') as f:
            f.write("Old compressed log content")
        
        # Run cleanup
        log_manager._delete_old_logs(7)
        
        # Check that old compressed file is removed
        assert not old_file.exists()
    
    def test_progressive_cleanup_strategy(self):
        """Test the progressive cleanup strategy."""
        log_manager = LogManager(self.config)
        
        # Create log files to exceed the size limit
        large_content = "x" * (2 * 1024 * 1024)  # 2MB per file
        
        for i in range(3):  # Create 3 files, total 6MB > 5MB limit
            file_path = Path(self.temp_dir) / f"app-2025-01-0{i+1}-info.log"
            file_path.write_text(large_content)
        
        # Run cleanup
        log_manager._cleanup_logs()
        
        # Check that cleanup was attempted (some files should be removed)
        remaining_files = list(Path(self.temp_dir).glob("*.log"))
        assert len(remaining_files) < 3
    
    def test_calculate_dir_size(self):
        """Test directory size calculation."""
        log_manager = LogManager(self.config)
        
        # Create test files
        (Path(self.temp_dir) / "file1.txt").write_text("x" * 1000)
        (Path(self.temp_dir) / "file2.txt").write_text("x" * 2000)
        
        # Calculate size
        size = log_manager._calculate_dir_size()
        
        # Should be 3000 bytes
        assert size == 3000
    
    def test_shutdown_closes_handlers_and_stops_scheduler(self):
        """Test that shutdown properly closes handlers and stops scheduler."""
        log_manager = LogManager(self.config)
        log_manager.initialize()
        
        # Get a handler to check if it's closed later
        handler = log_manager.get_handler('INFO')
        assert handler is not None
        
        # Shutdown
        log_manager.shutdown()
        
        # Check that handlers are cleared
        assert len(log_manager._handlers) == 0
        assert not log_manager._initialized
    
    @patch('emb_model_provider.core.log_manager.BackgroundScheduler')
    def test_scheduler_start_and_stop(self, mock_scheduler_class):
        """Test that the scheduler is properly started and stopped."""
        mock_scheduler = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler
        
        log_manager = LogManager()
        log_manager.initialize()
        
        # Check that scheduler was started
        mock_scheduler.add_job.assert_called_once()
        mock_scheduler.start.assert_called_once()
        
        # Shutdown
        log_manager.shutdown()
        
        # Check that scheduler was stopped
        mock_scheduler.shutdown.assert_called_once()
    
    def test_cleanup_logger_creation(self):
        """Test cleanup logger creation."""
        log_manager = LogManager()
        logger = log_manager._get_cleanup_logger()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'emb_model_provider.cleanup'
    
    def test_file_handler_rotation(self):
        """Test that file handlers rotate when they reach max size."""
        log_manager = LogManager(self.config)
        log_manager.initialize()
        
        # Get the INFO handler
        handler = log_manager.get_handler('INFO')
        assert handler is not None
        
        # Test that handler has the correct configuration
        assert handler.maxBytes == self.config.log_file_max_size * 1024 * 1024
        assert handler.backupCount == 5
        
        # Close handler
        handler.close()
    
    def test_multiple_initialization_calls(self):
        """Test that multiple initialization calls don't cause issues."""
        log_manager = LogManager(self.config)
        
        # Initialize multiple times
        log_manager.initialize()
        log_manager.initialize()
        log_manager.initialize()
        
        # Should still work correctly
        assert log_manager._initialized
        assert len(log_manager._handlers) == 4  # DEBUG, INFO, WARNING, ERROR
    
    def test_cleanup_with_invalid_filenames(self):
        """Test cleanup handles invalid filenames gracefully."""
        log_manager = LogManager(self.config)
        
        # Create files with invalid date formats
        invalid_file1 = Path(self.temp_dir) / "app-invalid-date-info.log"
        invalid_file1.write_text("content")
        
        invalid_file2 = Path(self.temp_dir) / "not-a-log-file.txt"
        invalid_file2.write_text("content")
        
        # Run cleanup - should not crash
        log_manager._delete_old_logs(7)
        
        # Invalid files should remain
        assert invalid_file1.exists()
        assert invalid_file2.exists()