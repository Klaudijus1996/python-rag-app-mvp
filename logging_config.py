"""
Logging configuration module for the RAG application.
Provides structured logging with daily file rotation and async capabilities.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import queue
import threading
import json
from datetime import datetime, timezone


class AsyncQueueHandler(logging.Handler):
    """Async-safe logging handler that uses a queue to avoid blocking."""

    def __init__(self, target_handler: logging.Handler):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue()
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self._start_worker()

    def _start_worker(self):
        """Start the background worker thread."""
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Background worker that processes log records."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for records with timeout to allow shutdown check
                record = self.queue.get(timeout=1.0)
                if record is None:  # Sentinel value for shutdown
                    break
                self.target_handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Avoid logging errors in the logging system
                print(f"Error in async logging worker: {e}")

    def emit(self, record):
        """Emit a log record asynchronously."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # If queue is full, just drop the log to avoid blocking
            pass

    def close(self):
        """Clean shutdown of the async handler."""
        self.shutdown_event.set()
        self.queue.put(None)  # Sentinel to stop worker
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "processing_time"):
            log_entry["processing_time"] = record.processing_time

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class LoggingConfig:
    """Centralized logging configuration."""

    def __init__(self, app_env: str = "dev", log_dir: Optional[str] = None):
        self.app_env = app_env
        self.log_dir = Path(log_dir or "storage/logs")
        self.log_level = logging.DEBUG if app_env == "dev" else logging.INFO
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self, use_async: bool = True, use_json: bool = False):
        """
        Setup simplified logging configuration with daily rotation.

        Args:
            use_async: Whether to use async logging handlers
            use_json: Whether to use JSON structured logging
        """
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatters
        if use_json:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        handlers = []

        # Console handler (always synchronous for immediate output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

        # Daily log file handler - single file for everything
        today = datetime.now().strftime("%Y-%m-%d")
        log_file_path = self.log_dir / f"{today}.log"

        # Use regular FileHandler for daily logs (simpler and more reliable)
        daily_log_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        daily_log_handler.setLevel(self.log_level)
        daily_log_handler.setFormatter(formatter)

        if use_async:
            handlers.append(AsyncQueueHandler(daily_log_handler))
        else:
            handlers.append(daily_log_handler)

        # Configure root logger
        root_logger.setLevel(self.log_level)
        for handler in handlers:
            root_logger.addHandler(handler)

        # Configure uvicorn access logger to use our daily handler
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.handlers = [handlers[-1]]  # Use the daily log handler
        access_logger.propagate = False

        return handlers

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance with the given name."""
        return logging.getLogger(name)

    @staticmethod
    def cleanup_async_handlers():
        """Cleanup async handlers on shutdown."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, AsyncQueueHandler):
                handler.close()


# Global logging configuration instance
_logging_config: Optional[LoggingConfig] = None


def setup_logging(
    app_env: str = None,
    log_dir: str = None,
    use_async: bool = True,
    use_json: bool = False,
):
    """
    Setup application logging.

    Args:
        app_env: Application environment (dev/prod)
        log_dir: Directory for log files
        use_async: Whether to use async logging
        use_json: Whether to use JSON structured logging
    """
    global _logging_config

    if app_env is None:
        app_env = os.getenv("APP_ENV", "dev")

    _logging_config = LoggingConfig(app_env=app_env, log_dir=log_dir)
    return _logging_config.setup_logging(use_async=use_async, use_json=use_json)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def cleanup_logging():
    """Cleanup logging resources."""
    if _logging_config:
        _logging_config.cleanup_async_handlers()


# Context manager for request logging
class RequestLoggingContext:
    """Context manager to add request-specific logging context."""

    def __init__(self, request_id: str, session_id: str = None, user_id: str = None):
        self.request_id = request_id
        self.session_id = session_id
        self.user_id = user_id
        self.old_factory = None

    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            # Only set if not already present to avoid overwrite error
            if not hasattr(record, "request_id"):
                record.request_id = self.request_id
            if self.session_id and not hasattr(record, "session_id"):
                record.session_id = self.session_id
            if self.user_id and not hasattr(record, "user_id"):
                record.user_id = self.user_id
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
