"""
Sentry integration for the Myah agent container.

Call setup_sentry() at container startup before other Hermes imports.
Uses stdlib logging exclusively — no Loguru dependency.
"""

import logging
import os

_SERVICE = 'myah-agent'

log = logging.getLogger(__name__)


def setup_sentry() -> None:
    """Initialize Sentry if SENTRY_DSN_AGENT is set.

    Enables error capture, distributed tracing, profiling, and log forwarding.
    Safe to call when Sentry is not configured — returns silently.
    """
    dsn = os.environ.get('SENTRY_DSN_AGENT', '')
    if not dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get('ENV', 'production'),
            release=os.environ.get('SENTRY_RELEASE'),
            traces_sample_rate=1.0,
            profile_session_sample_rate=1.0,
            profile_lifecycle='trace',
            send_default_pii=True,
            enable_logs=True,
            integrations=[
                LoggingIntegration(
                    level=logging.WARNING,
                    event_level=logging.ERROR,
                ),
            ],
        )
        sentry_sdk.set_tag('user_id', os.environ.get('MYAH_USER_ID', 'unknown'))
        sentry_sdk.set_tag('service', _SERVICE)
        log.info('Sentry error tracking, tracing and logging enabled for agent container')
    except Exception as e:
        log.warning('Sentry init failed: %s', e)
