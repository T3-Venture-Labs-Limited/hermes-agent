# ── Myah: TelemetryHook protocol — fork-friendly addition, upstream-ready ──
"""Telemetry hook protocol.

Hermes runtime code uses ``get_telemetry_hook()`` to emit observability
signals (exceptions, breadcrumbs, spans, tags, contexts) without coupling
to any specific SDK.  The default ``_NullTelemetryHook`` is a no-op, so
stock Hermes installations pay zero overhead.  Downstream consumers
(Myah, others) implement the protocol against their own backend
(Sentry, OpenTelemetry, Datadog, …) and call ``register_telemetry_hook``
once at startup.

Design notes:
- The hook is a module-level singleton.  Hermes's agent loop is
  primarily synchronous; threading the hook through every callsite would
  be invasive without buying anything.
- ``start_span`` returns a :class:`TelemetrySpan`-shaped object.  Real
  implementations (e.g. ``sentry_sdk.Span``) already satisfy this shape
  via duck typing.  Callers MUST be able to use it as a context manager
  AND call ``set_data``/``finish`` explicitly — both patterns appear in
  the migrated callsites in :mod:`gateway.run` and
  :mod:`gateway.platforms.api_server`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ContextManager, Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class TelemetrySpan(Protocol):
    """Minimum surface of a span/transaction returned by ``start_span``.

    Implementations MUST be usable as a context manager (``with span:``)
    so a ``finally``-free ``with`` block measures the wrapped work.  They
    SHOULD also expose ``set_data``/``finish`` for callers that want
    fine-grained control (the AI-monitoring path in ``gateway.run`` enters
    the span manually, attaches token-usage data after the agent run, then
    exits).
    """

    def set_data(self, key: str, value: Any) -> None: ...
    def finish(self, **kwargs: Any) -> Any: ...
    def __enter__(self) -> 'TelemetrySpan': ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...


@runtime_checkable
class TelemetryHook(Protocol):
    """Pluggable telemetry surface for Hermes runtime code.

    Implementations may forward to Sentry, OpenTelemetry, Datadog, etc.
    The no-op :class:`_NullTelemetryHook` is the default when nothing is
    registered.
    """

    def capture_exception(self, exc: BaseException, **kwargs: Any) -> None: ...

    def add_breadcrumb(
        self,
        *,
        category: str,
        message: str,
        level: str = 'info',
        data: Optional[dict] = None,
    ) -> None: ...

    def start_span(
        self,
        *,
        op: str,
        description: str = '',
        **kwargs: Any,
    ) -> ContextManager[TelemetrySpan]: ...

    def set_tag(self, key: str, value: Any) -> None: ...

    def set_context(self, name: str, value: dict) -> None: ...


class _NullSpan:
    """No-op span used by :class:`_NullTelemetryHook`.

    Behaves like a sentry ``Span``/``Transaction``: usable as a context
    manager, with stub ``set_data``/``finish`` methods.
    """

    def set_data(self, key: str, value: Any) -> None:  # noqa: ARG002 (interface)
        return None

    def finish(self, **kwargs: Any) -> None:  # noqa: ARG002 (interface)
        return None

    def __enter__(self) -> '_NullSpan':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


class _NullTelemetryHook:
    """Default no-op telemetry hook.

    Every method returns silently; ``start_span`` returns a context
    manager that does nothing.  Stock Hermes installations get this hook
    and pay zero cost.
    """

    def capture_exception(self, exc: BaseException, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    def add_breadcrumb(
        self,
        *,
        category: str,  # noqa: ARG002
        message: str,  # noqa: ARG002
        level: str = 'info',  # noqa: ARG002
        data: Optional[dict] = None,  # noqa: ARG002
    ) -> None:
        return None

    @contextmanager
    def start_span(
        self,
        *,
        op: str,  # noqa: ARG002
        description: str = '',  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Iterator[_NullSpan]:
        yield _NullSpan()

    def set_tag(self, key: str, value: Any) -> None:  # noqa: ARG002
        return None

    def set_context(self, name: str, value: dict) -> None:  # noqa: ARG002
        return None


# Module-level singleton.  Initialized to the no-op hook so calling
# ``get_telemetry_hook()`` is always safe — no caller needs an
# ``if hook is not None`` guard.
_telemetry_hook: TelemetryHook = _NullTelemetryHook()


def register_telemetry_hook(hook: TelemetryHook) -> None:
    """Register the process-wide telemetry hook.

    Call this once at startup (typically from the platform's launcher or
    a plugin's registration entry point).  Subsequent calls overwrite the
    previously registered hook — the last writer wins.
    """
    global _telemetry_hook
    _telemetry_hook = hook


def get_telemetry_hook() -> TelemetryHook:
    """Return the currently registered telemetry hook.

    Always returns a non-None object.  If no hook has been registered,
    the default :class:`_NullTelemetryHook` is returned.
    """
    return _telemetry_hook


def reset_telemetry_hook() -> None:
    """Reset the telemetry hook back to the no-op default.

    Primarily useful for tests that want a clean slate between cases.
    """
    global _telemetry_hook
    _telemetry_hook = _NullTelemetryHook()


# ───────────────────────────────────────────────────────────────────────
