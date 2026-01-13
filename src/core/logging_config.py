"""
NeuroSynth Unified - Structured Logging Configuration
======================================================

Production-ready logging with:
- JSON structured output (for log aggregation)
- Request correlation IDs (for distributed tracing)
- Context binding (user_id, document_id, etc.)
- Standard library integration (captures all loggers)

Usage:
    # At application startup (before any other imports)
    from src.core.logging_config import configure_logging
    configure_logging(json_output=True)

    # In any module
    import structlog
    logger = structlog.get_logger(__name__)

    # Basic logging
    logger.info("Processing document", document_id=doc_id, pages=10)

    # With bound context (persists across calls)
    log = logger.bind(user_id=user_id, request_id=request_id)
    log.info("Starting request")
    log.info("Request complete", duration_ms=150)

Environment Variables:
    LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    LOG_FORMAT: json, console (default: json in production, console in dev)
    LOG_INCLUDE_TIMESTAMP: true/false (default: true)
"""

import logging
import logging.config
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, WrappedLogger


# =============================================================================
# CONTEXT VARIABLES (Thread-safe request context)
# =============================================================================

# Request correlation ID - set per request in middleware
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# User ID - set after authentication
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

# Document ID - set during document operations
document_id_var: ContextVar[Optional[str]] = ContextVar("document_id", default=None)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request ID in context."""
    request_id_var.set(request_id)


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())[:8]  # Short ID for readability


# =============================================================================
# CUSTOM PROCESSORS
# =============================================================================

def add_request_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """Add request context from context variables."""
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id

    user_id = user_id_var.get()
    if user_id:
        event_dict["user_id"] = user_id

    document_id = document_id_var.get()
    if document_id:
        event_dict["document_id"] = document_id

    return event_dict


def add_service_info(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """Add service metadata."""
    event_dict["service"] = "neurosynth"
    event_dict["version"] = os.getenv("APP_VERSION", "3.0.0")
    return event_dict


def add_timestamp_iso(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """Add ISO 8601 timestamp with timezone."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def rename_event_key(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """Rename 'event' to 'message' for consistency with common log formats."""
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def add_log_level(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """Add uppercase log level."""
    event_dict["level"] = method_name.upper()
    return event_dict


def drop_color_codes(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """Remove color codes from message for JSON output."""
    if "message" in event_dict and isinstance(event_dict["message"], str):
        # Remove ANSI escape codes
        import re
        event_dict["message"] = re.sub(r'\x1b\[[0-9;]*m', '', event_dict["message"])
    return event_dict


# =============================================================================
# CONFIGURATION
# =============================================================================

def configure_logging(
    json_output: Optional[bool] = None,
    log_level: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Configure structured logging for the application.

    Args:
        json_output: If True, output JSON. If None, auto-detect from environment.
        log_level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var.
        include_timestamp: Include ISO timestamp in logs.

    Environment Variables:
        LOG_LEVEL: Override log level
        LOG_FORMAT: "json" or "console"
        ENVIRONMENT: "production" enables JSON by default
    """
    # Determine log level
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Determine output format
    if json_output is None:
        log_format = os.getenv("LOG_FORMAT", "").lower()
        if log_format == "json":
            json_output = True
        elif log_format == "console":
            json_output = False
        else:
            # Auto-detect: JSON in production, console in development
            env = os.getenv("ENVIRONMENT", "development").lower()
            json_output = env in ("production", "prod", "staging")

    # Build processor chain
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        add_request_context,
        add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if include_timestamp:
        shared_processors.insert(0, add_timestamp_iso)

    # Add format-specific processors (before final renderer)
    if json_output:
        # JSON output for production
        shared_processors.extend([
            rename_event_key,
            drop_color_codes,
            structlog.processors.format_exc_info,
        ])
        final_processor = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        # Console output for development
        shared_processors.append(structlog.processors.ExceptionPrettyPrinter())
        final_processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Configure structlog for direct usage (bypasses stdlib to avoid double-rendering)
    structlog.configure(
        processors=shared_processors + [final_processor],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging for third-party libraries
    # These logs go through stdlib -> ProcessorFormatter -> same output format
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structlog": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    final_processor,
                ],
                "foreign_pre_chain": shared_processors,
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "structlog",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": True,
            },
            # Reduce noise from verbose libraries
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "asyncio": {"level": "WARNING"},
            "uvicorn.access": {"level": "WARNING"},
            "faiss": {"level": "WARNING"},
        },
    })

    # Log configuration complete
    logger = structlog.get_logger("logging_config")
    logger.info(
        "Logging configured",
        format="json" if json_output else "console",
        level=log_level,
    )


# =============================================================================
# FASTAPI MIDDLEWARE
# =============================================================================

class RequestLoggingMiddleware:
    """
    Middleware to add request correlation and log requests.

    Usage:
        from src.core.logging_config import RequestLoggingMiddleware
        app.add_middleware(RequestLoggingMiddleware)
    """

    def __init__(self, app):
        self.app = app
        self.logger = structlog.get_logger("request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate and set request ID
        request_id = generate_request_id()
        set_request_id(request_id)

        # Extract request info
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        query_string = scope.get("query_string", b"").decode("utf-8")

        # Get client IP
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        # Track timing
        import time
        start_time = time.perf_counter()

        # Log request start
        self.logger.info(
            "Request started",
            method=method,
            path=path,
            query=query_string if query_string else None,
            client_ip=client_ip,
        )

        # Track response status
        status_code = 500  # Default if we can't capture

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.logger.exception(
                "Request failed",
                method=method,
                path=path,
                error=str(e),
            )
            raise
        finally:
            # Log request completion
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_method = self.logger.info if status_code < 400 else self.logger.warning
            log_method(
                "Request completed",
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
            )

            # Clear context
            request_id_var.set(None)
            user_id_var.set(None)
            document_id_var.set(None)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger with context binding support
    """
    return structlog.get_logger(name)


def bind_user(user_id: str) -> None:
    """Bind user ID to current context (persists for request duration)."""
    user_id_var.set(user_id)


def bind_document(document_id: str) -> None:
    """Bind document ID to current context."""
    document_id_var.set(document_id)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Auto-configure if this module is imported and logging isn't already configured
if not structlog.is_configured():
    # Don't auto-configure - let main.py do it explicitly
    pass
