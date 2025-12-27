from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class UTCFormatter(logging.Formatter):
    """Log formatter with UTC timestamps."""
    def formatTime(self, record, datefmt=None):
        return _utc_ts()


def setup_logging(
    name: str = "adsb_collector",
    level: Optional[str] = None,
    json_logs: Optional[bool] = None,
) -> logging.Logger:
    """
    Configure a logger that writes to stdout.
    Environment variables:
      LOG_LEVEL: DEBUG/INFO/WARNING/ERROR
      LOG_JSON:  1/true/yes -> emit JSON logs
    """
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    emit_json = json_logs if json_logs is not None else (os.getenv("LOG_JSON", "").lower() in ("1", "true", "yes", "y"))

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, lvl, logging.INFO))
    logger.propagate = False  # don't double-log if root handlers exist

    # Clear existing handlers (important when modules are reloaded)
    logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(getattr(logging, lvl, logging.INFO))

    if emit_json:
        handler.setFormatter(JSONFormatter())
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        formatter = UTCFormatter(fmt=fmt)
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class JSONFormatter(logging.Formatter):
    """Emit structured JSON logs."""
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": _utc_ts(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include exception info if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # Add any extra fields passed as `extra={...}`
        for k, v in record.__dict__.items():
            if k in (
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
                "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName", "process"
            ):
                continue
            # Only include JSON-safe items (fallback to str)
            try:
                json.dumps(v)
                payload[k] = v
            except Exception:
                payload[k] = str(v)

        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
