"""
Logging module for embedding model provider.

This module provides structured JSON logging functionality and request ID tracking.
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import config
from .log_manager import log_manager


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Formats log records as JSON objects with timestamp, level, message,
    request_id, and optional details.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            str: JSON-formatted log message
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add request ID if available
        if hasattr(record, "request_id"):
            request_id = getattr(record, "request_id", None)
            if request_id is not None:
                log_data["request_id"] = request_id
        
        # Add details for DEBUG level
        if record.levelno == logging.DEBUG:
            log_data["details"] = {
                "module": getattr(record, "module", ""),
                "function": getattr(record, "funcName", ""),
                "line": getattr(record, "lineno", 0),
                "thread": getattr(record, "thread", 0),
                "process": getattr(record, "process", 0),
            }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request ID tracking to all incoming requests.
    
    Generates a unique request ID for each incoming request and adds it
    to the request state for use in logging.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and add request ID tracking.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response: The response from the next middleware or route handler
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Record request start time
        start_time = time.time()
        
        # Log request
        logger = logging.getLogger("emb_model_provider.request")
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"request_id": request_id}
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Time: {process_time:.3f}s",
            extra={"request_id": request_id}
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


def setup_logging() -> None:
    """
    Set up structured JSON logging for the application.
    
    Configures the root logger with JSON formatter and appropriate handlers.
    Includes both console and file output if configured.
    """
    # Get log level from config
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and set JSON formatter
    formatter = JSONFormatter()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Initialize log manager and add file handlers if enabled
    if config.log_to_file:
        log_manager.initialize()
        
        # Add file handlers to root logger for each level
        for level_name in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            handler = log_manager.get_handler(level_name)
            if handler:
                root_logger.addHandler(handler)
    
    # Configure specific loggers
    logging.getLogger("emb_model_provider").setLevel(log_level)
    logging.getLogger("emb_model_provider.request").setLevel(log_level)
    logging.getLogger("emb_model_provider.model").setLevel(log_level)
    logging.getLogger("emb_model_provider.api").setLevel(log_level)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: The name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_with_request_id(
    logger: logging.Logger,
    level: int,
    message: str,
    request_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log a message with optional request ID.
    
    Args:
        logger: The logger to use
        level: The log level
        message: The log message
        request_id: Optional request ID to include
        **kwargs: Additional data to include in the log
    """
    extra = {}
    if request_id:
        extra["request_id"] = request_id
    
    logger.log(level, message, extra=extra, **kwargs)


def log_model_event(
    event_type: str,
    model_name: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log model-related events.
    
    Args:
        event_type: Type of event (e.g., "load", "download", "error")
        model_name: Name of the model
        details: Optional additional details
        request_id: Optional request ID
    """
    logger = logging.getLogger("emb_model_provider.model")
    
    message = f"Model {event_type}: {model_name}"
    if details:
        try:
            details_str = json.dumps(details, ensure_ascii=False)
            message += f" - {details_str}"
        except (TypeError, ValueError):
            # Handle non-serializable objects by converting them to strings
            serializable_details = {}
            for key, value in details.items():
                try:
                    json.dumps(value)  # Test if value is serializable
                    serializable_details[key] = value
                except (TypeError, ValueError):
                    serializable_details[key] = str(value)
            details_str = json.dumps(serializable_details, ensure_ascii=False)
            message += f" - {details_str}"
    
    log_with_request_id(logger, logging.INFO, message, request_id)


def log_api_error(
    error: Exception,
    request: Optional[Request] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log API errors with context.
    
    Args:
        error: The exception that occurred
        request: Optional request object
        request_id: Optional request ID
    """
    logger = logging.getLogger("emb_model_provider.api")
    
    message = f"API error: {type(error).__name__}: {str(error)}"
    if request:
        try:
            message += f" - {request.method} {request.url.path}"
        except AttributeError:
            # In case request or its attributes are None or incomplete
            pass
    
    log_with_request_id(logger, logging.ERROR, message, request_id, exc_info=True)


def shutdown_logging() -> None:
    """
    Shutdown the logging system.
    
    Properly closes file handlers and stops the cleanup scheduler.
    """
    if config.log_to_file:
        log_manager.shutdown()


# Initialize logging when module is imported
setup_logging()