"""Tests for Myah-added disconnect_mcp_server(name) helper in mcp_tool."""
import asyncio
import pytest

from tools import mcp_tool


def test_disconnect_mcp_server_unknown_name_is_noop():
    with mcp_tool._lock:
        mcp_tool._servers.pop('nope', None)

    result = mcp_tool.disconnect_mcp_server('nope')
    assert result is False


def test_disconnect_mcp_server_removes_only_the_named_server(monkeypatch):
    shut_calls = []

    class FakeServer:
        def __init__(self, name):
            self.name = name

        async def shutdown(self):
            shut_calls.append(self.name)

    server_a = FakeServer('alpha')
    server_b = FakeServer('beta')

    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._servers['alpha'] = server_a
        mcp_tool._servers['beta'] = server_b

    def fake_run_on_mcp_loop(coro, timeout=15):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(mcp_tool, '_run_on_mcp_loop', fake_run_on_mcp_loop)

    result = mcp_tool.disconnect_mcp_server('alpha')

    assert result is True
    assert shut_calls == ['alpha']
    with mcp_tool._lock:
        assert 'alpha' not in mcp_tool._servers
        assert 'beta' in mcp_tool._servers


def test_disconnect_mcp_server_swallows_shutdown_exceptions(monkeypatch):
    class BadServer:
        name = 'bad'
        async def shutdown(self):
            raise RuntimeError('shutdown failed')

    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._servers['bad'] = BadServer()

    def fake_run_on_mcp_loop(coro, timeout=15):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        except Exception:
            pass
        finally:
            loop.close()

    monkeypatch.setattr(mcp_tool, '_run_on_mcp_loop', fake_run_on_mcp_loop)

    result = mcp_tool.disconnect_mcp_server('bad')

    assert result is True
    with mcp_tool._lock:
        assert 'bad' not in mcp_tool._servers
