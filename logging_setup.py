"""
Structured logging setup for the Myah agent container.

Configures Loguru with a JSON sink to stdout (same format as the platform
backend's logger.py) and bridges stdlib logging so all agent log output
is structured and discoverable by Grafana Alloy.

Call setup_logging() once at container startup, before any other imports
that use the logging module.
"""

import datetime
import json
import logging
import os
import sys
import traceback

from loguru import logger


_SERVICE = 'myah-agent'


def _json_sink(message) -> None:
    """Write log records as single-line JSON to stdout."""
    record = message.record
    _utc_time = record['time'].astimezone(datetime.timezone.utc)
    log_entry = {
        'ts': _utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        'level': record['level'].name.lower(),
        'service': _SERVICE,
        'user_id': os.environ.get('MYAH_USER_ID', 'unknown'),
        'msg': record['message'],
        'caller': f'{record["name"]}:{record["function"]}:{record["line"]}',
    }

    if record['extra']:
        # Filter out loguru internals
        extras = {k: v for k, v in record['extra'].items() if not k.startswith('_')}
        if extras:
            log_entry['extra'] = extras

    if record['exception'] is not None:
        exc = record['exception']
        log_entry['error'] = ''.join(
            traceback.format_exception(exc.type, exc.value, exc.traceback)
        ).rstrip()

    sys.stdout.write(json.dumps(log_entry, ensure_ascii=False, default=str) + '\n')
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

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = 'DEBUG') -> None:
    """Configure Loguru JSON logging and intercept stdlib logging.

    Call once at container startup before any other module imports.
    """
    logger.remove()
    logger.add(_json_sink, level=level, colorize=False)

    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)

    # Silence noisy third-party loggers
    for name in ('httpx', 'httpcore', 'aiohttp', 'urllib3'):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info('Agent logging configured', extra={'service': _SERVICE, 'user_id': os.environ.get('MYAH_USER_ID', 'unknown')})


def setup_sentry() -> None:
    """Initialize Sentry if SENTRY_DSN_AGENT is set.

    Call after setup_logging() at container startup.
    """
    dsn = os.environ.get('SENTRY_DSN_AGENT', '')
    if not dsn:
        return

    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0,
            environment=os.environ.get('ENV', 'production'),
        )
        sentry_sdk.set_tag('user_id', os.environ.get('MYAH_USER_ID', 'unknown'))
        sentry_sdk.set_tag('service', _SERVICE)
        logger.info('Sentry error tracking enabled for agent container')
    except Exception as e:
        logger.warning(f'Sentry init failed: {e}')
