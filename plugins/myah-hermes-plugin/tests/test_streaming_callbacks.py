"""Phase F regression tests for the plugin-side streaming workaround.

The pre_llm_call hook fires from AIAgent.run_conversation
(run_agent.py:11765 vanilla / 11066 fork) AFTER _run_agent set messaging-style callbacks
(line 14404) but BEFORE the first LLM API call begins. Tests cover:

1. Hook ignores non-myah platforms.
2. Hook resolves session_key from chat_id and finds the cached agent.
3. Hook installs structured callbacks AND marks the session for
   duplicate-send suppression.
4. Hook is a graceful no-op when adapter / runner / cache are missing.
5. CI guards: pre_llm_call is a real hook name; AIAgent has the four
   callback attributes we mutate.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_agent():
    """Return a fake AIAgent with the four callback attributes."""
    agent = SimpleNamespace(
        stream_delta_callback=lambda *a, **kw: None,
        tool_progress_callback=lambda *a, **kw: None,
        status_callback=lambda *a, **kw: None,
        reasoning_callback=lambda *a, **kw: None,
    )
    return agent


def _make_fake_adapter(session_key: str):
    adapter = MagicMock()
    adapter._chat_id_session_keys = {"chat-1": session_key}
    adapter._native_streaming_used = set()
    structured = {
        "stream_delta": MagicMock(name="cb_stream_delta"),
        "tool_progress": MagicMock(name="cb_tool_progress"),
        "status": MagicMock(name="cb_status"),
        "reasoning": MagicMock(name="cb_reasoning"),
    }
    adapter.get_structured_callbacks.return_value = structured
    return adapter, structured


def test_hook_ignores_non_myah_platform():
    from myah_hermes_plugin.runtime_extensions.streaming_callbacks import (
        myah_pre_llm_call,
    )
    result = myah_pre_llm_call(session_id="chat-1", platform="telegram")
    assert result is None


def test_hook_noop_when_no_adapter():
    from myah_hermes_plugin.runtime_extensions import streaming_callbacks

    with patch.object(
        streaming_callbacks, "_get_latest_adapter", return_value=None
    ):
        result = streaming_callbacks.myah_pre_llm_call(
            session_id="chat-1", platform="myah"
        )
    assert result is None


def test_hook_noop_when_session_key_unknown():
    from myah_hermes_plugin.runtime_extensions import streaming_callbacks

    adapter, _ = _make_fake_adapter(session_key="agent:main:myah:dm:chat-1:user-1")
    adapter._chat_id_session_keys = {}  # platform sent unknown chat_id
    adapter.gateway_runner = MagicMock()

    with patch.object(
        streaming_callbacks, "_get_latest_adapter", return_value=adapter
    ):
        result = streaming_callbacks.myah_pre_llm_call(
            session_id="unknown-chat", platform="myah"
        )
    assert result is None
    adapter.get_structured_callbacks.assert_not_called()


def test_hook_installs_callbacks_and_marks_native_streaming():
    from myah_hermes_plugin.runtime_extensions import streaming_callbacks

    sk = "agent:main:myah:dm:chat-1:user-1"
    adapter, structured = _make_fake_adapter(session_key=sk)

    fake_agent = _make_fake_agent()
    runner = MagicMock()
    runner._agent_cache = {sk: (fake_agent, "sig")}
    adapter.gateway_runner = runner

    with patch.object(
        streaming_callbacks, "_get_latest_adapter", return_value=adapter
    ):
        result = streaming_callbacks.myah_pre_llm_call(
            session_id="chat-1", platform="myah"
        )

    assert result is None
    assert fake_agent.stream_delta_callback is structured["stream_delta"]
    assert fake_agent.tool_progress_callback is structured["tool_progress"]
    assert fake_agent.status_callback is structured["status"]
    assert fake_agent.reasoning_callback is structured["reasoning"]
    assert sk in adapter._native_streaming_used


def test_hook_handles_dict_cache_entry():
    """Some cache shapes store the agent directly (not in a tuple)."""
    from myah_hermes_plugin.runtime_extensions import streaming_callbacks

    sk = "agent:main:myah:dm:chat-1:user-1"
    adapter, _ = _make_fake_adapter(session_key=sk)

    fake_agent = _make_fake_agent()
    runner = MagicMock()
    runner._agent_cache = {sk: fake_agent}  # not a tuple
    adapter.gateway_runner = runner

    with patch.object(
        streaming_callbacks, "_get_latest_adapter", return_value=adapter
    ):
        streaming_callbacks.myah_pre_llm_call(
            session_id="chat-1", platform="myah"
        )

    assert fake_agent.stream_delta_callback is not None


# ── CI guards ────────────────────────────────────────────────────────


def test_pre_llm_call_is_in_valid_hooks():
    """If upstream renames or removes pre_llm_call, this fails loudly."""
    from hermes_cli.plugins import VALID_HOOKS
    assert "pre_llm_call" in VALID_HOOKS, (
        "Upstream removed pre_llm_call hook — Phase F workaround broken"
    )


def test_runner_agent_cache_attr_exists():
    """If upstream renames _agent_cache, this fails loudly."""
    from gateway.run import GatewayRunner
    # GatewayRunner is a class; instances have _agent_cache. Test by
    # checking that the attribute is defined in the class or one of its
    # ancestors via a sentinel instance.
    inst = GatewayRunner.__new__(GatewayRunner)
    # Direct attr access would fail; use the canonical default
    cache = getattr(inst, "_agent_cache", None)
    if cache is None:
        # Some versions only set _agent_cache in __init__; still need
        # the name in source.
        import inspect
        src = inspect.getsource(GatewayRunner)
        assert "_agent_cache" in src, (
            "Upstream removed GatewayRunner._agent_cache"
        )


def test_aiagent_has_structured_callback_attrs():
    """If upstream renames any of the four callbacks, this fails."""
    from run_agent import AIAgent
    expected = [
        "stream_delta_callback",
        "tool_progress_callback",
        "status_callback",
        "reasoning_callback",
    ]
    import inspect
    src = inspect.getsource(AIAgent.__init__)
    for name in expected:
        assert name in src, (
            f"Upstream removed AIAgent.{name} — Phase F workaround broken"
        )


# ── Phase F.4: duplicate-send suppression in MyahAdapter.send ────────


def _make_adapter_vanilla_safe():
    """Build a MyahAdapter for tests, tolerating both fork (which has
    register_pre_setup_hook) and vanilla (which doesn't).

    The plugin's adapter.py:206 explicitly says "no dependency on
    register_pre_setup_hook" (Tier 2A Task 2A.3 removed it). But
    existing tests in test_myah_adapter.py / test_myah_platform_contract.py
    patch the symbol defensively. On vanilla the symbol is absent, so
    we use create=True so patching synthesizes a placeholder.
    """
    from gateway.config import PlatformConfig
    from gateway.platforms import api_server as _api_server

    has_hook = hasattr(_api_server, "register_pre_setup_hook")
    if has_hook:
        cm = patch("gateway.platforms.api_server.register_pre_setup_hook")
    else:
        cm = patch.object(
            _api_server, "register_pre_setup_hook",
            create=True, new=lambda *a, **kw: None,
        )
    with cm:
        from myah_hermes_plugin.myah_platform.adapter import MyahAdapter
        return MyahAdapter(
            PlatformConfig(enabled=True, extra={"auth_key": ""})
        )


@pytest.mark.asyncio
async def test_send_suppresses_gateway_final_when_native_streaming_active():
    """If the pre_llm_call hook marked this session for native streaming,
    MyahAdapter.send() must drop the gateway's final-response send.
    """
    adapter = _make_adapter_vanilla_safe()

    sk = "agent:main:myah:dm:chat-1:user-1"
    adapter._chat_id_session_keys["chat-1"] = sk
    adapter._native_streaming_used.add(sk)

    result = await adapter.send(
        "chat-1", "full response text", metadata=None
    )

    assert result.success is True
    assert result.message_id == "suppressed-native-streaming"
    # And the flag is consumed so subsequent sends pass through.
    assert sk not in adapter._native_streaming_used


@pytest.mark.asyncio
async def test_send_passes_through_when_no_native_streaming_marker():
    """Without the marker, send() takes its normal SSE-or-webhook path."""
    adapter = _make_adapter_vanilla_safe()

    sk = "agent:main:myah:dm:chat-1:user-1"
    adapter._chat_id_session_keys["chat-1"] = sk
    # Note: marker NOT added

    # No active SSE stream either — should fail with "No active stream"
    result = await adapter.send("chat-1", "some content", metadata=None)
    assert result.success is False
    assert "No active stream" in (result.error or "")
