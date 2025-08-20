"""
FastAPI middleware for request logging and monitoring.
Provides async, non-blocking logging of API requests and responses.
"""

import time
import uuid
import asyncio
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from logging_config import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    Features:
    - Async logging that doesn't block request processing
    - Request/response timing
    - Request ID generation for tracing
    - Error logging with stack traces
    - Optional response body logging (disabled by default for performance)
    """

    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.logger = get_logger("middleware.request")
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log details asynchronously."""

        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Extract session ID from request if available
        session_id = None
        if request.method == "POST" and "application/json" in request.headers.get(
            "content-type", ""
        ):
            try:
                # Try to extract session_id from request body without consuming it
                body = await request.body()
                if body:
                    import json

                    data = json.loads(body)
                    session_id = data.get("session_id")

                # Restore the body for the actual request handler
                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive
            except Exception:
                pass  # If we can't parse, just continue without session_id

        # Log request start (async)
        asyncio.create_task(self._log_request_start(request, request_id, session_id))

        response = None
        error = None

        try:
            # Process the request
            response = await call_next(request)

        except Exception as e:
            error = e
            # Create error response
            from fastapi.responses import JSONResponse

            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
            )

        finally:
            # Calculate processing time
            processing_time = time.time() - start_time

            # Log response (async)
            asyncio.create_task(
                self._log_request_end(
                    request, response, request_id, session_id, processing_time, error
                )
            )

        return response

    async def _log_request_start(
        self, request: Request, request_id: str, session_id: Optional[str]
    ):
        """Log request start details asynchronously."""
        try:
            log_data = {
                "event": "request_start",
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params)
                if request.query_params
                else None,
                "user_agent": request.headers.get("user-agent"),
                "client_ip": self._get_client_ip(request),
                "request_id": request_id,
            }

            if session_id:
                log_data["session_id"] = session_id

            # Log request body if enabled (be careful with sensitive data)
            if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        # Only log first 1000 chars to avoid huge logs
                        log_data["request_body"] = body.decode("utf-8")[:1000]
                except Exception as e:
                    log_data["request_body_error"] = str(e)

            self.logger.info("Incoming request", extra=log_data)

        except Exception as e:
            self.logger.error(f"Error logging request start: {e}", exc_info=True)

    async def _log_request_end(
        self,
        request: Request,
        response: Response,
        request_id: str,
        session_id: Optional[str],
        processing_time: float,
        error: Optional[Exception],
    ):
        """Log request completion details asynchronously."""
        try:
            log_data = {
                "event": "request_end",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time": round(processing_time, 4),
                "request_id": request_id,
            }

            if session_id:
                log_data["session_id"] = session_id

            # Log response body if enabled and it's not a streaming response
            if (
                self.log_response_body
                and not isinstance(response, StreamingResponse)
                and hasattr(response, "body")
            ):
                try:
                    # Only log first 1000 chars
                    body_str = response.body.decode("utf-8")[:1000]
                    log_data["response_body"] = body_str
                except Exception as e:
                    log_data["response_body_error"] = str(e)

            # Log error details if there was an exception
            if error:
                log_data["error"] = str(error)
                log_data["error_type"] = type(error).__name__
                self.logger.error("Request failed", extra=log_data, exc_info=error)
            else:
                # Choose log level based on status code
                if response.status_code >= 500:
                    self.logger.error(
                        "Request completed with server error", extra=log_data
                    )
                elif response.status_code >= 400:
                    self.logger.warning(
                        "Request completed with client error", extra=log_data
                    )
                else:
                    self.logger.info("Request completed successfully", extra=log_data)

        except Exception as e:
            self.logger.error(f"Error logging request end: {e}", exc_info=True)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        # Check for forwarded IP headers (common in proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring application performance.
    Tracks slow requests and resource usage.
    """

    def __init__(self, app: ASGIApp, slow_request_threshold: float = 5.0):
        super().__init__(app)
        self.logger = get_logger("middleware.performance")
        self.slow_request_threshold = slow_request_threshold
        self.concurrent_requests = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance."""
        start_time = time.time()
        
        # Track concurrent requests
        self.concurrent_requests += 1
        max_concurrent = getattr(self, '_max_concurrent', 0)
        if self.concurrent_requests > max_concurrent:
            self._max_concurrent = self.concurrent_requests
        
        # Log high concurrency for expensive endpoints
        if self.concurrent_requests > 5 and request.url.path in ["/chat", "/retrieve"]:
            asyncio.create_task(self._log_high_concurrency(request, self.concurrent_requests))

        try:
            response = await call_next(request)
        finally:
            processing_time = time.time() - start_time
            self.concurrent_requests -= 1
            
            # Log slow requests
            if processing_time > self.slow_request_threshold:
                asyncio.create_task(self._log_slow_request(request, processing_time))
            
            # Log performance metrics for expensive endpoints
            if request.url.path in ["/chat", "/retrieve", "/ingest"]:
                asyncio.create_task(self._log_endpoint_metrics(request, processing_time))

        return response

    async def _log_slow_request(self, request: Request, processing_time: float):
        """Log slow request details."""
        try:
            self.logger.warning(
                "Slow request detected",
                extra={
                    "event": "slow_request",
                    "method": request.method,
                    "path": request.url.path,
                    "processing_time": round(processing_time, 4),
                    "threshold": self.slow_request_threshold,
                },
            )
        except Exception as e:
            self.logger.error(f"Error logging slow request: {e}", exc_info=True)
    
    async def _log_high_concurrency(self, request: Request, concurrent_count: int):
        """Log high concurrency events."""
        try:
            self.logger.info(
                "High concurrency detected",
                extra={
                    "event": "high_concurrency",
                    "method": request.method,
                    "path": request.url.path,
                    "concurrent_requests": concurrent_count,
                },
            )
        except Exception as e:
            self.logger.error(f"Error logging high concurrency: {e}", exc_info=True)
    
    async def _log_endpoint_metrics(self, request: Request, processing_time: float):
        """Log performance metrics for expensive endpoints."""
        try:
            self.logger.info(
                "Endpoint performance metrics",
                extra={
                    "event": "endpoint_metrics",
                    "method": request.method,
                    "path": request.url.path,
                    "processing_time": round(processing_time, 4),
                    "max_concurrent_seen": getattr(self, '_max_concurrent', 0),
                },
            )
        except Exception as e:
            self.logger.error(f"Error logging endpoint metrics: {e}", exc_info=True)


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking and logging application errors.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("middleware.errors")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track and log errors."""
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log the error asynchronously
            asyncio.create_task(self._log_error(request, e))

            # Re-raise to let FastAPI's error handlers deal with it
            raise

    async def _log_error(self, request: Request, error: Exception):
        """Log error details."""
        try:
            self.logger.error(
                "Unhandled error in request",
                extra={
                    "event": "unhandled_error",
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                exc_info=error,
            )
        except Exception as log_error:
            # Fallback logging to avoid infinite recursion
            print(f"Error logging error: {log_error}")
            import traceback

            traceback.print_exc()
