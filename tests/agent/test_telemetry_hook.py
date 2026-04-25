"""Unit tests for the TelemetryHook protocol and registry.

Verifies the no-op default behaves correctly, that hook registration
swaps the singleton, that every protocol method is reachable, and that
spans returned by ``start_span`` are usable as context managers and via
explicit ``set_data``/``finish`` calls (the dual usage pattern that the
gateway's AI-monitoring path depends on).
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from agent.telemetry import (
    _NullSpan,
    _NullTelemetryHook,
    get_telemetry_hook,
    register_telemetry_hook,
    reset_telemetry_hook,
)


@pytest.fixture(autouse=True)
def _reset_hook():
    """Restore the no-op default after every test."""
    yield
    reset_telemetry_hook()


def test_default_hook_is_null_telemetry_hook():
    assert isinstance(get_telemetry_hook(), _NullTelemetryHook)


def test_null_hook_capture_exception_is_silent():
    hook = get_telemetry_hook()
    # Should not raise even when handed a real exception object.
    hook.capture_exception(RuntimeError('boom'))


def test_null_hook_add_breadcrumb_accepts_keyword_args():
    hook = get_telemetry_hook()
    hook.add_breadcrumb(category='cat', message='hello', level='warning', data={'k': 'v'})
    # Default-arg path
    hook.add_breadcrumb(category='cat', message='hello')


def test_null_hook_set_tag_and_set_context_are_silent():
    hook = get_telemetry_hook()
    hook.set_tag('key', 'value')
    hook.set_context('name', {'a': 1})


def test_null_hook_start_span_returns_context_manager():
    hook = get_telemetry_hook()
    with hook.start_span(op='gen_ai.invoke_agent', description='Hermes') as span:
        # Span object exposes set_data and finish.
        span.set_data('gen_ai.usage.input_tokens', 42)
        span.finish()


def test_null_span_is_a_context_manager():
    span = _NullSpan()
    with span as s:
        assert s is span
        s.set_data('k', 'v')
        s.finish()


def test_register_swaps_the_hook():
    class _Spy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple, dict]] = []

        def capture_exception(self, exc: BaseException, **kwargs: Any) -> None:
            self.calls.append(('capture_exception', (exc,), kwargs))

        def add_breadcrumb(
            self,
            *,
            category: str,
            message: str,
            level: str = 'info',
            data: Optional[dict] = None,
        ) -> None:
            self.calls.append(
                ('add_breadcrumb', (), {'category': category, 'message': message, 'level': level, 'data': data})
            )

        def start_span(self, *, op: str, description: str = '', **kwargs: Any):
            self.calls.append(('start_span', (), {'op': op, 'description': description, **kwargs}))
            return _NullSpan()  # Reuse the no-op span — it satisfies the protocol.

        def set_tag(self, key: str, value: Any) -> None:
            self.calls.append(('set_tag', (key, value), {}))

        def set_context(self, name: str, value: dict) -> None:
            self.calls.append(('set_context', (name, value), {}))

    spy = _Spy()
    register_telemetry_hook(spy)
    assert get_telemetry_hook() is spy


def test_spy_hook_records_every_protocol_method():
    class _Spy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple, dict]] = []

        def capture_exception(self, exc: BaseException, **kwargs: Any) -> None:
            self.calls.append(('capture_exception', (exc,), kwargs))

        def add_breadcrumb(
            self,
            *,
            category: str,
            message: str,
            level: str = 'info',
            data: Optional[dict] = None,
        ) -> None:
            self.calls.append(
                ('add_breadcrumb', (), {'category': category, 'message': message, 'level': level, 'data': data})
            )

        def start_span(self, *, op: str, description: str = '', **kwargs: Any):
            self.calls.append(('start_span', (), {'op': op, 'description': description, **kwargs}))
            return _NullSpan()

        def set_tag(self, key: str, value: Any) -> None:
            self.calls.append(('set_tag', (key, value), {}))

        def set_context(self, name: str, value: dict) -> None:
            self.calls.append(('set_context', (name, value), {}))

    spy = _Spy()
    register_telemetry_hook(spy)

    err = RuntimeError('x')
    get_telemetry_hook().capture_exception(err)
    get_telemetry_hook().add_breadcrumb(category='c', message='m', level='warning', data={'a': 1})
    with get_telemetry_hook().start_span(op='op', description='desc') as span:
        span.set_data('k', 'v')
    get_telemetry_hook().set_tag('tk', 'tv')
    get_telemetry_hook().set_context('ctx', {'foo': 'bar'})

    names = [c[0] for c in spy.calls]
    assert names == ['capture_exception', 'add_breadcrumb', 'start_span', 'set_tag', 'set_context']

    capture = next(c for c in spy.calls if c[0] == 'capture_exception')
    assert capture[1] == (err,)

    breadcrumb = next(c for c in spy.calls if c[0] == 'add_breadcrumb')
    assert breadcrumb[2] == {'category': 'c', 'message': 'm', 'level': 'warning', 'data': {'a': 1}}

    span_call = next(c for c in spy.calls if c[0] == 'start_span')
    assert span_call[2] == {'op': 'op', 'description': 'desc'}


def test_reset_telemetry_hook_restores_default():
    class _Anything:
        def capture_exception(self, exc: BaseException, **kwargs: Any) -> None:
            pass

        def add_breadcrumb(self, *, category: str, message: str, level: str = 'info', data=None) -> None:
            pass

        def start_span(self, *, op: str, description: str = '', **kwargs: Any):
            return _NullSpan()

        def set_tag(self, key: str, value: Any) -> None:
            pass

        def set_context(self, name: str, value: dict) -> None:
            pass

    register_telemetry_hook(_Anything())
    assert not isinstance(get_telemetry_hook(), _NullTelemetryHook)
    reset_telemetry_hook()
    assert isinstance(get_telemetry_hook(), _NullTelemetryHook)
