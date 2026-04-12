"""
Tests for the Myah web platform gateway adapter.

Tests cover:
- Adapter lifecycle (init, connect, disconnect)
- Dual session mapping (_session_streams + _chat_id_streams)
- Structured callback event formatting (_format_tool_event)
- Thread-safe queue operations (call_soon_threadsafe)
- Stream management (creation, cleanup, orphan sweep)
- Auth validation (_check_auth returns Optional[web.Response])
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_adapter(auth_key: str = "", **extra_kwargs):
    """Construct a MyahAdapter with register_pre_setup_hook mocked out."""
    extra = dict(extra_kwargs)
    if auth_key:
        extra["auth_key"] = auth_key
    config = PlatformConfig(enabled=True, extra=extra)

    with patch("gateway.platforms.api_server.register_pre_setup_hook"):
        from gateway.platforms.myah import MyahAdapter
        return MyahAdapter(config)


# ── check_myah_requirements ─────────────────────────────────────────────────

class TestCheckRequirements:
    def test_requirements_available(self):
        from gateway.platforms.myah import check_myah_requirements
        assert check_myah_requirements() is True


# ── Init ────────────────────────────────────────────────────────────────────

class TestMyahAdapterInit:
    def test_default_config(self):
        adapter = _make_adapter()
        assert adapter.platform == Platform.MYAH
        assert adapter._auth_key == ""
        assert adapter._streams == {}
        assert adapter._session_streams == {}
        assert adapter._chat_id_streams == {}
        assert adapter._stream_sessions == {}

    def test_auth_key_from_extra(self):
        adapter = _make_adapter(auth_key="test-key-123")
        assert adapter._auth_key == "test-key-123"


# ── Auth ────────────────────────────────────────────────────────────────────
# _check_auth() returns Optional[web.Response]:
#   None       → auth passed
#   web.Response → auth failed (401)


class TestMyahAdapterAuth:
    def test_no_key_configured_allows_all(self):
        """No auth key configured — all requests pass (returns None)."""
        adapter = _make_adapter()
        request = MagicMock()
        request.headers = {}
        assert adapter._check_auth(request) is None

    def test_valid_bearer_token(self):
        """Valid bearer token — returns None (success)."""
        adapter = _make_adapter(auth_key="secret-key")
        request = MagicMock()
        request.headers = {"Authorization": "Bearer secret-key"}
        assert adapter._check_auth(request) is None

    def test_invalid_bearer_token(self):
        """Wrong bearer token — returns 401 response."""
        adapter = _make_adapter(auth_key="secret-key")
        request = MagicMock()
        request.headers = {"Authorization": "Bearer wrong-key"}
        result = adapter._check_auth(request)
        assert result is not None
        assert result.status == 401

    def test_missing_auth_header(self):
        """No Authorization header — returns 401 response."""
        adapter = _make_adapter(auth_key="secret-key")
        request = MagicMock()
        request.headers = {}
        result = adapter._check_auth(request)
        assert result is not None
        assert result.status == 401

    def test_non_bearer_scheme(self):
        """Basic auth scheme instead of Bearer — returns 401."""
        adapter = _make_adapter(auth_key="secret-key")
        request = MagicMock()
        request.headers = {"Authorization": "Basic secret-key"}
        result = adapter._check_auth(request)
        assert result is not None
        assert result.status == 401


# ── Dual session/chat_id mapping ────────────────────────────────────────────

class TestDualMapping:
    def test_independent_mappings(self):
        """session_key and chat_id both resolve to the same stream_id."""
        adapter = _make_adapter()

        stream_id = "stream-001"
        session_key = "agent:main:myah:dm:chat-uuid-1"
        chat_id = "chat-uuid-1"

        adapter._session_streams[session_key] = stream_id
        adapter._chat_id_streams[chat_id] = stream_id
        adapter._streams[stream_id] = asyncio.Queue()

        # Structured callbacks use session_key
        assert adapter._session_streams.get(session_key) == stream_id
        # send() / send_typing() use chat_id
        assert adapter._chat_id_streams.get(chat_id) == stream_id

    def test_multiple_concurrent_streams(self):
        """Multiple chats can have independent active streams."""
        adapter = _make_adapter()

        for i in range(3):
            sid = f"stream-{i}"
            adapter._chat_id_streams[f"chat-{i}"] = sid
            adapter._streams[sid] = asyncio.Queue()

        assert len(adapter._streams) == 3
        assert adapter._chat_id_streams["chat-0"] != adapter._chat_id_streams["chat-1"]


# ── _format_tool_event ──────────────────────────────────────────────────────

class TestFormatToolEvent:
    """Test _format_tool_event with all 4 invocation patterns from run_agent.py."""

    def setup_method(self):
        from gateway.platforms.myah import MyahAdapter
        self.stream_id = "test-stream-42"
        # _format_tool_event is a @staticmethod — no adapter instance needed
        self._fmt = MyahAdapter._format_tool_event

    def test_tool_started(self):
        args = ("tool.started", "web_search", "Searching for...", {"query": "test"})
        result = self._fmt(self.stream_id, args, {})
        assert result["event"] == "tool.started"
        assert result["tool"] == "web_search"
        assert result["preview"] == "Searching for..."
        assert result["args"] == {"query": "test"}
        assert result["run_id"] == self.stream_id
        assert result["stream_id"] == self.stream_id

    def test_tool_completed(self):
        args = ("tool.completed", "web_search", None, None)
        kwargs = {"duration": 1.5, "is_error": False}
        result = self._fmt(self.stream_id, args, kwargs)
        assert result["event"] == "tool.completed"
        assert result["tool"] == "web_search"
        assert result["duration"] == 1.5
        assert result["error"] is False
        assert result["run_id"] == self.stream_id

    def test_thinking(self):
        args = ("_thinking", "Let me analyze this...")
        result = self._fmt(self.stream_id, args, {})
        assert result["event"] == "reasoning.delta"
        assert result["text"] == "Let me analyze this..."
        assert result["run_id"] == self.stream_id

    def test_reasoning_available(self):
        args = ("reasoning.available", "_thinking", "Full reasoning text", None)
        result = self._fmt(self.stream_id, args, {})
        assert result["event"] == "reasoning.available"
        assert result["text"] == "Full reasoning text"
        assert result["run_id"] == self.stream_id

    def test_empty_args_fallback(self):
        result = self._fmt(self.stream_id, (), {})
        assert result["event"] == "status"
        assert result["text"] == "working"
        assert result["run_id"] == self.stream_id

    def test_unknown_event_type_fallback(self):
        args = ("unknown.event.type", "data")
        result = self._fmt(self.stream_id, args, {})
        assert result["event"] == "status"
        assert result["text"] == "unknown.event.type"
        assert result["run_id"] == self.stream_id

    def test_tool_started_non_dict_args(self):
        """Non-dict args[3] should fall back to empty dict."""
        args = ("tool.started", "terminal", "Running command", "not-a-dict")
        result = self._fmt(self.stream_id, args, {})
        assert result["args"] == {}

    def test_tool_started_with_none_preview(self):
        """None preview should become empty string."""
        args = ("tool.started", "file_read", None, {"path": "/tmp"})
        result = self._fmt(self.stream_id, args, {})
        assert result["preview"] == ""


# ── send() and send_typing() ───────────────────────────────────────────────

class TestSendMethods:
    @pytest.mark.asyncio
    async def test_send_pushes_message_delta(self):
        adapter = _make_adapter()
        q = asyncio.Queue()
        stream_id = "stream-send-test"
        adapter._chat_id_streams["chat-1"] = stream_id
        adapter._streams[stream_id] = q

        result = await adapter.send("chat-1", "Hello world")
        assert result.success is True

        event = q.get_nowait()
        assert event["event"] == "message.delta"
        assert event["delta"] == "Hello world"
        assert event["run_id"] == stream_id

    @pytest.mark.asyncio
    async def test_send_no_active_stream(self):
        adapter = _make_adapter()
        result = await adapter.send("nonexistent-chat", "Hello")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_typing_pushes_status(self):
        adapter = _make_adapter()
        q = asyncio.Queue()
        stream_id = "stream-typing-test"
        adapter._chat_id_streams["chat-2"] = stream_id
        adapter._streams[stream_id] = q

        await adapter.send_typing("chat-2")

        event = q.get_nowait()
        assert event["event"] == "status"
        assert event["status"] == "typing"
        assert event["run_id"] == stream_id

    @pytest.mark.asyncio
    async def test_send_typing_no_stream_is_noop(self):
        adapter = _make_adapter()
        # Should not raise
        await adapter.send_typing("nonexistent-chat")


# ── get_chat_info ───────────────────────────────────────────────────────────

class TestGetChatInfo:
    @pytest.mark.asyncio
    async def test_returns_expected_shape(self):
        adapter = _make_adapter()
        info = await adapter.get_chat_info("my-chat-id")
        assert info["chat_id"] == "my-chat-id"
        assert info["platform"] == "myah"
        assert info["type"] == "dm"


# ── get_structured_callbacks ────────────────────────────────────────────────

class TestStructuredCallbacks:
    def test_returns_none_when_no_stream(self):
        adapter = _make_adapter()
        result = adapter.get_structured_callbacks("nonexistent-session-key")
        assert result is None

    def test_returns_dict_with_four_callbacks(self):
        adapter = _make_adapter()
        adapter._loop = asyncio.new_event_loop()
        stream_id = "stream-cb-test"
        session_key = "agent:main:myah:dm:chat-x"
        adapter._session_streams[session_key] = stream_id
        adapter._streams[stream_id] = asyncio.Queue()

        cbs = adapter.get_structured_callbacks(session_key)
        assert cbs is not None
        assert set(cbs.keys()) == {"stream_delta", "tool_progress", "reasoning", "status"}
        adapter._loop.close()

    def test_stream_delta_pushes_event(self):
        adapter = _make_adapter()
        loop = asyncio.new_event_loop()
        adapter._loop = loop
        q = asyncio.Queue()
        stream_id = "stream-delta-test"
        session_key = "agent:main:myah:dm:delta-chat"
        adapter._session_streams[session_key] = stream_id
        adapter._streams[stream_id] = q

        cbs = adapter.get_structured_callbacks(session_key)
        # Simulate call from agent thread — call_soon_threadsafe will
        # schedule on the loop.  We run it manually.
        cbs["stream_delta"]("token text")
        loop.run_until_complete(asyncio.sleep(0))  # Process scheduled callbacks

        event = q.get_nowait()
        assert event["event"] == "message.delta"
        assert event["delta"] == "token text"
        assert event["run_id"] == stream_id
        loop.close()

    def test_stream_delta_ignores_none(self):
        """None text (tool boundary signal) should not push any event."""
        adapter = _make_adapter()
        loop = asyncio.new_event_loop()
        adapter._loop = loop
        q = asyncio.Queue()
        stream_id = "stream-none-test"
        session_key = "agent:main:myah:dm:none-chat"
        adapter._session_streams[session_key] = stream_id
        adapter._streams[stream_id] = q

        cbs = adapter.get_structured_callbacks(session_key)
        cbs["stream_delta"](None)
        loop.run_until_complete(asyncio.sleep(0))

        assert q.empty()
        loop.close()
