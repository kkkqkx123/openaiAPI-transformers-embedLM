"""
Unit tests for the logging module.
"""

import json
import logging
import sys
import tempfile
import shutil
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response

from emb_model_provider.core.logging import (
    JSONFormatter,
    RequestIDMiddleware,
    get_logger,
    log_api_error,
    log_model_event,
    log_with_request_id,
    setup_logging,
    shutdown_logging,
)


class TestJSONFormatter:
    """Test cases for JSONFormatter class."""
    
    def test_format_basic_log(self):
        """Test basic log formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data
        assert "request_id" not in log_data
        assert "details" not in log_data
    
    def test_format_log_with_request_id(self):
        """Test log formatting with request ID."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.request_id = "test-request-id"
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["request_id"] == "test-request-id"
    
    def test_format_debug_log_with_details(self):
        """Test DEBUG log formatting with details."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Debug message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.process = 67890
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["level"] == "DEBUG"
        assert "details" in log_data
        assert log_data["details"]["module"] == "test_module"
        assert log_data["details"]["function"] == "test_function"
        assert log_data["details"]["line"] == 1
        assert log_data["details"]["thread"] == 12345
        assert log_data["details"]["process"] == 67890
    
    def test_format_log_with_exception(self):
        """Test log formatting with exception info."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error message",
                args=(),
                exc_info=sys.exc_info()
            )
            
            result = formatter.format(record)
            log_data = json.loads(result)
            
            assert "exception" in log_data
            assert "ValueError" in log_data["exception"]
            assert "Test exception" in log_data["exception"]


class TestRequestIDMiddleware:
    """Test cases for RequestIDMiddleware class."""
    
    @pytest.mark.asyncio
    async def test_request_id_middleware_adds_request_id(self):
        """Test that middleware adds request ID to request state."""
        # Create mock request and response
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.state = MagicMock()
        
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        
        # Create mock call_next
        mock_call_next = AsyncMock(return_value=mock_response)
        
        # Create middleware and process request
        middleware = RequestIDMiddleware(MagicMock())
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        # Verify request ID was added to state
        assert hasattr(mock_request.state, "request_id")
        assert len(mock_request.state.request_id) > 0
        
        # Verify request ID was added to response headers
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == mock_request.state.request_id
        
        # Verify call_next was called
        mock_call_next.assert_called_once_with(mock_request)
    
    @pytest.mark.asyncio
    async def test_request_id_middleware_logs_request_and_response(self):
        """Test that middleware logs request and response."""
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JSONFormatter())
        
        logger = logging.getLogger("emb_model_provider.request")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Create mock request and response
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/embeddings"
        mock_request.state = MagicMock()
        
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        
        # Create mock call_next
        mock_call_next = AsyncMock(return_value=mock_response)
        
        # Create middleware and process request
        middleware = RequestIDMiddleware(MagicMock())
        await middleware.dispatch(mock_request, mock_call_next)
        
        # Get log output
        log_output = log_capture.getvalue()
        log_lines = [line for line in log_output.strip().split('\n') if line]
        
        # Should have two log entries: request start and request completion
        assert len(log_lines) == 2
        
        # Parse log entries
        start_log = json.loads(log_lines[0])
        completion_log = json.loads(log_lines[1])
        
        # Verify request start log
        assert "Request started" in start_log["message"]
        assert "POST" in start_log["message"]
        assert "/embeddings" in start_log["message"]
        assert "request_id" in start_log
        
        # Verify request completion log
        assert "Request completed" in completion_log["message"]
        assert "POST" in completion_log["message"]
        assert "/embeddings" in completion_log["message"]
        assert "Status: 200" in completion_log["message"]
        assert "request_id" in completion_log
        assert completion_log["request_id"] == start_log["request_id"]


class TestLoggingFunctions:
    """Test cases for logging utility functions."""
    
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_log_with_request_id(self):
        """Test logging with request ID."""
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JSONFormatter())
        
        logger = logging.getLogger("test_with_request_id")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log with request ID
        log_with_request_id(
            logger,
            logging.INFO,
            "Test message with request ID",
            request_id="test-request-id"
        )
        
        # Get log output
        log_output = log_capture.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["message"] == "Test message with request ID"
        assert log_data["request_id"] == "test-request-id"
    
    def test_log_model_event(self):
        """Test logging model events."""
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JSONFormatter())
        
        logger = logging.getLogger("emb_model_provider.model")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log model event
        log_model_event(
            "load",
            "all-MiniLM-L12-v2",
            details={"source": "local", "path": "/models/test"},
            request_id="test-request-id"
        )
        
        # Get log output
        log_output = log_capture.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert "Model load" in log_data["message"]
        assert "all-MiniLM-L12-v2" in log_data["message"]
        assert log_data["request_id"] == "test-request-id"
    
    def test_log_api_error(self):
        """Test logging API errors."""
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JSONFormatter())
        
        logger = logging.getLogger("emb_model_provider.api")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        
        # Create mock request - don't use spec as it causes bool(mock_request) to be False
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_url = MagicMock()
        mock_url.path = "/embeddings"
        mock_request.url = mock_url
        
        # Log API error
        log_api_error(
            ValueError("Test error"),
            request=mock_request,
            request_id="test-request-id"
        )
        
        # Get log output
        log_output = log_capture.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert "API error" in log_data["message"]
        assert "ValueError" in log_data["message"]
        assert "Test error" in log_data["message"]
        assert "POST" in log_data["message"]
        assert "/embeddings" in log_data["message"]
        assert log_data["request_id"] == "test-request-id"


class TestSetupLogging:
    """Test cases for setup_logging function."""
    
    def test_setup_logging_configures_root_logger(self):
        """Test that setup_logging properly configures the root logger."""
        # Save original handlers
        original_handlers = logging.getLogger().handlers[:]
        
        try:
            # Setup logging
            setup_logging()
            
            # Verify root logger is configured
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) > 0
            
            # Verify handler uses JSONFormatter
            handler = root_logger.handlers[0]
            assert isinstance(handler.formatter, JSONFormatter)
            
        finally:
            # Restore original handlers
            for handler in logging.getLogger().handlers[:]:
                logging.getLogger().removeHandler(handler)
            for handler in original_handlers:
                logging.getLogger().addHandler(handler)
    
    @patch('emb_model_provider.core.logging.config')
    def test_setup_logging_uses_config_log_level(self, mock_config):
        """Test that setup_logging uses log level from config."""
        mock_config.log_level = "DEBUG"
        mock_config.log_to_file = False
        
        # Setup logging
        setup_logging()
        
        # Verify log level is set correctly
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG


class TestFileLogging:
    """Test cases for file logging functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config to use temp directory
        self.config_patcher = patch('emb_model_provider.core.logging.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.log_level = "INFO"
        self.mock_config.log_to_file = True
        self.mock_config.log_dir = self.temp_dir
        self.mock_config.log_file_max_size = 10
        self.mock_config.log_retention_days = 7
        self.mock_config.log_cleanup_interval_hours = 1
        self.mock_config.log_max_dir_size_mb = 5
        self.mock_config.log_cleanup_retention_days = [7, 3, 1]
    
    def teardown_method(self):
        """Clean up after each test."""
        self.config_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutdown_logging()  # Clean up log manager
    
    def test_setup_logging_creates_file_handlers(self):
        """Test that setup_logging creates file handlers when enabled."""
        # Mock log manager
        with patch('emb_model_provider.core.logging.log_manager') as mock_manager:
            setup_logging()
            
            # Verify log manager was initialized
            mock_manager.initialize.assert_called_once()
            
            # Verify file handlers were added
            root_logger = logging.getLogger()
            handler_count = len(root_logger.handlers)
            assert handler_count >= 2  # Console + at least one file handler
    
    def test_setup_logging_skips_file_handlers_when_disabled(self):
        """Test that setup_logging skips file handlers when disabled."""
        self.mock_config.log_to_file = False
        
        # Mock log manager
        with patch('emb_model_provider.core.logging.log_manager') as mock_manager:
            setup_logging()
            
            # Verify log manager was not initialized
            mock_manager.initialize.assert_not_called()
    
    def test_file_logging_creates_log_files(self):
        """Test that file logging actually creates log files."""
        # Setup logging with file output
        setup_logging()
        
        # Log a message
        logger = logging.getLogger("test_file_logging")
        logger.info("Test message for file logging")
        
        # Check that log files were created
        log_dir = Path(self.temp_dir)
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0
        
        # Check that log file contains our message
        for log_file in log_files:
            content = log_file.read_text()
            if "Test message for file logging" in content:
                # Parse JSON and verify structure
                log_data = json.loads(content.strip())
                assert log_data["level"] == "INFO"
                assert log_data["message"] == "Test message for file logging"
                assert "timestamp" in log_data
                break
        else:
            pytest.fail("Log message not found in any log file")
    
    def test_file_logging_separates_by_level(self):
        """Test that log files are separated by level."""
        # Setup logging with file output
        setup_logging()
        
        # Log messages at different levels
        logger = logging.getLogger("test_level_separation")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check that different log files exist for different levels
        log_dir = Path(self.temp_dir)
        log_files = list(log_dir.glob("*.log"))
        
        # Should have files for different levels
        level_files = {}
        for log_file in log_files:
            if "debug" in log_file.name.lower():
                level_files["DEBUG"] = log_file
            elif "info" in log_file.name.lower():
                level_files["INFO"] = log_file
            elif "warning" in log_file.name.lower():
                level_files["WARNING"] = log_file
            elif "error" in log_file.name.lower():
                level_files["ERROR"] = log_file
        
        # Verify messages are in the correct files
        if "INFO" in level_files:
            content = level_files["INFO"].read_text()
            assert "Info message" in content
        
        if "ERROR" in level_files:
            content = level_files["ERROR"].read_text()
            assert "Error message" in content
    
    def test_shutdown_logging_cleans_up_resources(self):
        """Test that shutdown_logging properly cleans up resources."""
        # Setup logging
        setup_logging()
        
        # Mock log manager
        with patch('emb_model_provider.core.logging.log_manager') as mock_manager:
            shutdown_logging()
            
            # Verify shutdown was called
            mock_manager.shutdown.assert_called_once()
    
    def test_file_logging_with_request_id(self):
        """Test that file logging preserves request ID."""
        # Setup logging with file output
        setup_logging()
        
        # Log a message with request ID
        logger = logging.getLogger("test_request_id")
        logger.info("Test message with request ID", extra={"request_id": "test-123"})
        
        # Check that log file contains request ID
        log_dir = Path(self.temp_dir)
        log_files = list(log_dir.glob("*.log"))
        
        for log_file in log_files:
            content = log_file.read_text()
            if "Test message with request ID" in content:
                log_data = json.loads(content.strip())
                assert log_data["request_id"] == "test-123"
                break
        else:
            pytest.fail("Log message not found in any log file")