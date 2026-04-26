"""Test for Bug B: /myah/v1/confirm/{stream_id} dispatches to the right resolver.

Two queues exist in tools/approval.py:

* ``_gateway_queues`` — legacy terminal-command approvals; resolved by
  ``resolve_gateway_approval(session_key, choice)``.
* ``_action_queues`` — modern action confirmations (cron creation,
  plugin install, etc.); resolved by
  ``resolve_action_confirmation(confirmation_id, choice)``.

The current ``_handle_confirm_endpoint`` only calls
``resolve_gateway_approval`` — Approve/Deny clicks for cron approval
cards arrive but go nowhere because the entry sits in
``_action_queues``.

The fix accepts an optional ``confirmation_id`` in the body:
* present  → ``resolve_action_confirmation(confirmation_id, choice)``
* absent   → ``resolve_gateway_approval(session_key, choice)`` (legacy)
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig


def _make_adapter(auth_key: str = ""):
    extra = dict()
    if auth_key:
        extra["auth_key"] = auth_key
    config = PlatformConfig(enabled=True, extra=extra)
    with patch("gateway.platforms.api_server.register_pre_setup_hook"):
        from gateway.platforms.myah import MyahAdapter
        return MyahAdapter(config)


def _make_app(adapter) -> web.Application:
    """Mount only the /myah/v1/confirm/{stream_id} route for testing."""
    app = web.Application()
    app.router.add_post("/myah/v1/confirm/{stream_id}", adapter._handle_confirm_endpoint)
    return app


# ── Myah: Bug B regression coverage — modern + legacy dispatch ────


class TestConfirmEndpointActionDispatch:
    @pytest.mark.asyncio
    async def test_with_confirmation_id_routes_to_action_resolver(self):
        """When body includes confirmation_id, call resolve_action_confirmation."""
        adapter = _make_adapter()
        stream_id = "stream-1"
        session_key = "sess-1"
        adapter._stream_sessions[stream_id] = session_key

        with patch("tools.approval.resolve_action_confirmation", return_value=True) as mock_action, \
             patch("tools.approval.resolve_gateway_approval", return_value=0) as mock_legacy:
            async with TestClient(TestServer(_make_app(adapter))) as cli:
                resp = await cli.post(
                    f"/myah/v1/confirm/{stream_id}",
                    json={"confirmation_id": "conf-xyz", "choice": "approve"},
                )
                body = await resp.json()

        assert resp.status == 200, body
        assert body.get("ok") is True
        mock_action.assert_called_once_with("conf-xyz", "approve")
        mock_legacy.assert_not_called()

    @pytest.mark.asyncio
    async def test_without_confirmation_id_tries_legacy_then_action_queue(self):
        """No confirmation_id → try resolve_gateway_approval first (legacy
        terminal-command path).  When that returns 0 (nothing pending in the
        legacy queue, the common case for cron approvals) fall through to
        ``resolve_action_confirmation_by_session`` so the frontend's
        ``ConfirmationCard`` POST resolves the cron approval without needing
        to know the confirmation_id explicitly."""
        adapter = _make_adapter()
        stream_id = "stream-2"
        session_key = "sess-2"
        adapter._stream_sessions[stream_id] = session_key

        with patch("tools.approval.resolve_gateway_approval", return_value=0) as mock_legacy, \
             patch("tools.approval.resolve_action_confirmation_by_session", return_value=1) as mock_action_session, \
             patch("tools.approval.resolve_action_confirmation", return_value=False) as mock_action_byid:
            async with TestClient(TestServer(_make_app(adapter))) as cli:
                resp = await cli.post(
                    f"/myah/v1/confirm/{stream_id}",
                    json={"choice": "approve"},
                )
                body = await resp.json()

        assert resp.status == 200, body
        assert body.get("ok") is True
        mock_legacy.assert_called_once_with(session_key, "approve")
        mock_action_session.assert_called_once_with(session_key, "approve")
        mock_action_byid.assert_not_called()  # confirmation_id branch not taken

    @pytest.mark.asyncio
    async def test_without_confirmation_id_legacy_resolves_first(self):
        """When the legacy queue has a pending entry, resolve it without
        consulting the action queue (preserves legacy single-resolution
        semantics for terminal-command approvals)."""
        adapter = _make_adapter()
        stream_id = "stream-2b"
        session_key = "sess-2b"
        adapter._stream_sessions[stream_id] = session_key

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_legacy, \
             patch("tools.approval.resolve_action_confirmation_by_session", return_value=99) as mock_action_session:
            async with TestClient(TestServer(_make_app(adapter))) as cli:
                resp = await cli.post(
                    f"/myah/v1/confirm/{stream_id}",
                    json={"choice": "approve"},
                )
                body = await resp.json()

        assert resp.status == 200, body
        assert body.get("ok") is True
        mock_legacy.assert_called_once_with(session_key, "approve")
        # Action-queue resolver should NOT be called when legacy already resolved
        mock_action_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_confirmation_id_returns_404(self):
        """resolve_action_confirmation returning False (unknown id) → 404."""
        adapter = _make_adapter()
        stream_id = "stream-3"
        adapter._stream_sessions[stream_id] = "sess-3"

        with patch("tools.approval.resolve_action_confirmation", return_value=False):
            async with TestClient(TestServer(_make_app(adapter))) as cli:
                resp = await cli.post(
                    f"/myah/v1/confirm/{stream_id}",
                    json={"confirmation_id": "missing", "choice": "approve"},
                )
                body = await resp.json()

        assert resp.status == 404, body
        # error message should hint which queue was searched (debugging aid)
        assert "action" in body.get("error", "").lower() or "confirmation" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_invalid_choice_rejected(self):
        """Choice must be one of approve/approve_session/deny."""
        adapter = _make_adapter()
        stream_id = "stream-4"
        adapter._stream_sessions[stream_id] = "sess-4"

        async with TestClient(TestServer(_make_app(adapter))) as cli:
            resp = await cli.post(
                f"/myah/v1/confirm/{stream_id}",
                json={"confirmation_id": "anything", "choice": "maybe"},
            )
            body = await resp.json()

        assert resp.status == 400, body

    @pytest.mark.asyncio
    async def test_no_session_for_stream_returns_404(self):
        """Unknown stream_id → 404 before either resolver is called."""
        adapter = _make_adapter()
        # do NOT populate _stream_sessions

        with patch("tools.approval.resolve_action_confirmation") as mock_action, \
             patch("tools.approval.resolve_gateway_approval") as mock_legacy:
            async with TestClient(TestServer(_make_app(adapter))) as cli:
                resp = await cli.post(
                    "/myah/v1/confirm/nonexistent",
                    json={"choice": "approve"},
                )

        assert resp.status == 404
        mock_action.assert_not_called()
        mock_legacy.assert_not_called()
# ─────────────────────────────────────────────────────────────────
