"""Test for Bug A: gateway approval-notify callback must accept BOTH shapes.

Two callers invoke the registered notify callback today:

1. Legacy terminal-command approval — calls ``cb(approval_data: dict)``.
2. Modern action confirmation (``request_action_confirmation``) — calls
   ``cb(session_key: str, payload: dict)`` with payload containing
   ``type="tool.confirmation_required"``, ``confirmation_id``, etc.

The current implementation in ``gateway/run.py`` is single-arity
``def _approval_notify_sync(approval_data: dict)`` which raises a
``TypeError`` when the modern caller invokes it with two positional args,
breaking every approval-bearing tool (notably ``cronjob`` creation).

The fix is a module-level async helper ``_dispatch_approval_notify`` that
routes both shapes to the right adapter method.  The original closure in
``gateway/run.py:_approval_notify_sync`` becomes a thin wrapper that
schedules this helper on the running event loop.
"""

import asyncio
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock


# ── Myah: Bug A regression coverage — variadic callback dispatch ────


class _RecordingAdapter:
    """Fake adapter that records send_* calls without performing IO.

    By default this *does* expose ``send_action_confirmation`` so the
    modern path uses the structured SSE event.  Pass
    ``support_action_confirmation=False`` (or use
    ``_RecordingAdapterTextOnly``) to test the text fallback path.
    """

    support_action_confirmation = True

    def __init__(self):
        self.send_calls: List[Tuple[str, str, Dict[str, Any]]] = []
        self.exec_approval_calls: List[Dict[str, Any]] = []
        self.action_confirmation_calls: List[Dict[str, Any]] = []

    async def send(self, chat_id: str, content: str, metadata: Dict[str, Any] | None = None, **kwargs: Any):
        self.send_calls.append((chat_id, content, metadata or {}))
        return MagicMock(success=True)

    async def send_exec_approval(self, chat_id: str, command: str, session_key: str, description: str, metadata: Dict[str, Any] | None = None):
        self.exec_approval_calls.append({
            "chat_id": chat_id,
            "command": command,
            "session_key": session_key,
            "description": description,
            "metadata": metadata or {},
        })
        return MagicMock(success=True)

    async def send_action_confirmation(self, session_key: str, payload: Dict[str, Any]):
        self.action_confirmation_calls.append({
            "session_key": session_key,
            "payload": payload,
        })
        return MagicMock(success=True)

    def pause_typing_for_chat(self, chat_id: str) -> None:  # pragma: no cover - noop
        return None


class _RecordingAdapterTextOnly:
    """Fake adapter without send_action_confirmation — tests text fallback."""

    def __init__(self):
        self.send_calls: List[Tuple[str, str, Dict[str, Any]]] = []
        self.exec_approval_calls: List[Dict[str, Any]] = []

    async def send(self, chat_id: str, content: str, metadata: Dict[str, Any] | None = None, **kwargs: Any):
        self.send_calls.append((chat_id, content, metadata or {}))
        return MagicMock(success=True)

    async def send_exec_approval(self, chat_id: str, command: str, session_key: str, description: str, metadata: Dict[str, Any] | None = None):
        self.exec_approval_calls.append({
            "chat_id": chat_id,
            "command": command,
            "session_key": session_key,
            "description": description,
            "metadata": metadata or {},
        })
        return MagicMock(success=True)

    def pause_typing_for_chat(self, chat_id: str) -> None:  # pragma: no cover - noop
        return None


class TestApprovalNotifyVariadicDispatch:
    """Verify both legacy and modern callback shapes are routed correctly."""

    def test_modern_two_arg_emits_structured_sse_event(self):
        """When called with (session_key, payload) and the adapter exposes
        ``send_action_confirmation``, dispatch to the structured SSE path
        so the frontend renders the interactive ``ConfirmationCard``.

        Acceptance criteria:
        - The dispatcher must NOT raise.
        - It must call ``adapter.send_action_confirmation`` with the
          session_key from the callback args and the full payload.
        - It must NOT fall through to plain text via ``adapter.send``.
        - It must NOT call ``send_exec_approval`` (legacy command path).
        """
        from gateway.run import _dispatch_approval_notify  # type: ignore[attr-defined]

        adapter = _RecordingAdapter()
        chat_id = "chat-abc"
        bound_session_key = "session-xyz"
        payload = {
            "type": "tool.confirmation_required",
            "confirmation_id": "conf-123",
            "action_type": "cron_create",
            "description": "Create a cron job that runs every 5 minutes",
            "options": ["approve", "approve_session", "deny"],
            "metadata": {"schedule_display": "every 5m"},
        }

        # Modern callback shape: callback(session_key, payload) — 2 positional args
        asyncio.run(_dispatch_approval_notify(
            adapter, chat_id, bound_session_key,
            bound_session_key, payload,  # callback args
        ))

        # Structured path used — exec_approval and text fallback NOT used
        assert adapter.exec_approval_calls == [], (
            "send_exec_approval is the legacy terminal-command path; modern "
            "action confirmations must NOT route through it"
        )
        assert adapter.send_calls == [], (
            "with send_action_confirmation available, the dispatcher must not "
            "fall through to plain-text via adapter.send"
        )
        assert adapter.action_confirmation_calls, (
            "expected exactly one send_action_confirmation call"
        )
        call = adapter.action_confirmation_calls[0]
        assert call["session_key"] == bound_session_key
        assert call["payload"]["confirmation_id"] == "conf-123"
        assert call["payload"]["action_type"] == "cron_create"
        assert call["payload"]["options"] == ["approve", "approve_session", "deny"]
        assert call["payload"]["metadata"]["schedule_display"] == "every 5m"

    def test_modern_falls_back_to_text_when_adapter_lacks_method(self):
        """Adapters that don't expose send_action_confirmation should
        get the plain-text approval card via adapter.send."""
        from gateway.run import _dispatch_approval_notify  # type: ignore[attr-defined]

        adapter = _RecordingAdapterTextOnly()
        bound_session_key = "session-text"
        payload = {
            "type": "tool.confirmation_required",
            "confirmation_id": "conf-text-1",
            "action_type": "cron_create",
            "description": "Test",
            "options": ["approve", "deny"],
        }

        asyncio.run(_dispatch_approval_notify(
            adapter, "chat-text", bound_session_key,
            bound_session_key, payload,
        ))

        assert adapter.send_calls, "text fallback must call adapter.send"
        chat_id_sent, content_sent, _ = adapter.send_calls[0]
        assert chat_id_sent == "chat-text"
        assert "conf-text-1" in content_sent or "approve" in content_sent.lower()

    def test_legacy_one_arg_preserves_button_approval(self):
        """When called with a single dict, dispatch to send_exec_approval
        (legacy rich-button path) with the existing semantics."""
        from gateway.run import _dispatch_approval_notify  # type: ignore[attr-defined]

        adapter = _RecordingAdapter()
        chat_id = "chat-abc"
        bound_session_key = "session-xyz"
        approval_data = {
            "command": "rm -rf /tmp/test",
            "description": "Delete temp files",
        }

        # Legacy callback shape: callback(approval_data) — 1 positional dict
        asyncio.run(_dispatch_approval_notify(
            adapter, chat_id, bound_session_key,
            approval_data,  # callback arg
        ))

        assert adapter.exec_approval_calls, "expected send_exec_approval call for the legacy path"
        call = adapter.exec_approval_calls[0]
        assert call["command"] == "rm -rf /tmp/test"
        assert call["session_key"] == bound_session_key
        assert call["description"] == "Delete temp files"
        assert call["chat_id"] == chat_id

    def test_unknown_shape_does_not_raise(self):
        """Bad input must log + return, not propagate (callers swallow)."""
        from gateway.run import _dispatch_approval_notify  # type: ignore[attr-defined]

        adapter = _RecordingAdapter()
        # No exception when we hand it nothing recognizable
        asyncio.run(_dispatch_approval_notify(
            adapter, "chat-abc", "session-xyz",
            "not-a-dict-or-tuple",  # callback arg
        ))
        # No call should be made
        assert adapter.send_calls == []
        assert adapter.exec_approval_calls == []
# ────────────────────────────────────────────────────────────────────
