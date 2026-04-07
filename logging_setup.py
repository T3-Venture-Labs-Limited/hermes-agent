"""
Structured logging setup for the Myah agent container.

Configures Loguru with a JSON sink to stdout (same format as the platform
backend's logger.py), bridges stdlib logging, initializes Sentry, and sets
up OpenTelemetry tracing for observability in SigNoz Cloud.

Call setup_logging() once at container startup, before any other imports
that use the logging module.
"""

import datetime
import json
import logging
import os
import sys
import traceback
from typing import Optional

from loguru import logger


_SERVICE = "myah-agent"

# Module-level OTel state (initialized lazily via setup_otel())
_tracer: Optional["trace.Tracer"] = None
_otel_enabled = False


def _json_sink(message) -> None:
    """Write log records as single-line JSON to stdout."""
    record = message.record
    _utc_time = record["time"].astimezone(datetime.timezone.utc)
    log_entry = {
        "ts": _utc_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": record["level"].name.lower(),
        "service": _SERVICE,
        "user_id": os.environ.get("MYAH_USER_ID", "unknown"),
        "msg": record["message"],
        "caller": f"{record['name']}:{record['function']}:{record['line']}",
    }

    if record["extra"]:
        # Filter out loguru internals
        extras = {k: v for k, v in record["extra"].items() if not k.startswith("_")}
        if extras:
            log_entry["extra"] = extras

    if record["exception"] is not None:
        exc = record["exception"]
        log_entry["error"] = "".join(
            traceback.format_exception(exc.type, exc.value, exc.traceback)
        ).rstrip()

    sys.stdout.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


class _InterceptHandler(logging.Handler):
    """Bridge stdlib logging to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(level: str = "DEBUG") -> None:
    """Configure Loguru JSON logging and intercept stdlib logging.

    Call once at container startup before any other module imports.
    """
    logger.remove()
    logger.add(_json_sink, level=level, colorize=False)

    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)

    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "aiohttp", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info(
        "Agent logging configured",
        extra={
            "service": _SERVICE,
            "user_id": os.environ.get("MYAH_USER_ID", "unknown"),
        },
    )


def setup_sentry() -> None:
    """Initialize Sentry if SENTRY_DSN_AGENT is set.

    Call after setup_logging() at container startup.
    """
    dsn = os.environ.get("SENTRY_DSN_AGENT", "")
    if not dsn:
        return

    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0,
            environment=os.environ.get("ENV", "production"),
        )
        sentry_sdk.set_tag("user_id", os.environ.get("MYAH_USER_ID", "unknown"))
        sentry_sdk.set_tag("service", _SERVICE)
        logger.info("Sentry error tracking enabled for agent container")
    except Exception as e:
        logger.warning(f"Sentry init failed: {e}")


def setup_otel() -> Optional["trace.Tracer"]:
    """Initialize OpenTelemetry tracing if ENABLE_OTEL is set.

    Configures OTLP HTTP exporter for SigNoz Cloud with proper
    kill-switch behavior (app continues if SigNoz is unreachable).

    Returns the tracer instance or None if OTel is disabled/unavailable.
    """
    global _tracer, _otel_enabled

    if _tracer is not None:
        return _tracer

    if os.environ.get("ENABLE_OTEL", "").lower() not in ("true", "1", "yes"):
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")

        if not endpoint:
            logger.warning(
                "ENABLE_OTEL is set but OTEL_EXPORTER_OTLP_ENDPOINT is empty"
            )
            return None

        # Parse headers from key=value,key2=value2 format
        headers = {}
        if headers_str:
            for header in headers_str.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()

        # HTTP exporter requires /v1/traces path
        if not endpoint.endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"

        resource = Resource.create(attributes={SERVICE_NAME: _SERVICE})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer(_SERVICE)
        _otel_enabled = True

        logger.info(
            "OpenTelemetry tracing enabled",
            extra={"endpoint": endpoint.split("?")[0], "service": _SERVICE},
        )
        return _tracer

    except ImportError:
        logger.warning("opentelemetry-sdk not installed, OTel disabled")
        return None
    except Exception as e:
        logger.error(f"OpenTelemetry init failed: {e}")
        return None


def get_tracer() -> Optional["trace.Tracer"]:
    """Get the initialized OTel tracer, or None if not enabled."""
    global _tracer
    if _tracer is None and not _otel_enabled:
        _tracer = setup_otel()
    return _tracer


def is_otel_enabled() -> bool:
    """Check if OpenTelemetry is enabled and initialized."""
    return _otel_enabled
