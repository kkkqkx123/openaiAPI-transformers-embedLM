"""
Unit tests for the logging module.
"""

import json
import logging
import sys
from io import StringIO
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
        
        # Setup logging
        setup_logging()
        
        # Verify log level is set correctly
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG