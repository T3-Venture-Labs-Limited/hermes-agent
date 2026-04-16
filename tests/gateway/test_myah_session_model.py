"""Tests for Myah session-scoped model override endpoint.

Verifies:
    - PUT /myah/api/sessions/{session_key}/model writes to
      gateway_runner._session_model_overrides
    - GET returns the current override
    - Bad model id returns 400
    - Cross-session isolation (override on one session doesn't affect another)
"""

import json
from unittest.mock import MagicMock, AsyncMock

import pytest
from aiohttp import web

from gateway.platforms.myah_management import (
    handle_get_session_model,
    handle_put_session_model,
)


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner._session_model_overrides = {}
    runner._evict_cached_agent = MagicMock()
    runner._agent_cache = {}
    return runner


@pytest.fixture
def app_with_runner(mock_runner):
    app = web.Application()
    adapter = MagicMock()
    adapter.gateway_runner = mock_runner
    app["myah_adapter"] = adapter
    return app


@pytest.fixture
def fake_switch_model(monkeypatch):
    """Stub switch_model so tests don't do real network probes."""
    from hermes_cli import model_switch

    def _fake(**kwargs):
        raw = kwargs.get("raw_input", "")
        if raw == "unknown/bogus-model":
            result = MagicMock()
            result.success = False
            result.error_message = "Model 'unknown/bogus-model' not recognized"
            return result
        result = MagicMock()
        result.success = True
        result.new_model = raw or "anthropic/claude-opus-4.6"
        result.target_provider = kwargs.get("explicit_provider") or "anthropic"
        result.provider_label = "Anthropic"
        result.api_key = "sk-test"
        result.base_url = "https://api.anthropic.com"
        result.api_mode = "anthropic"
        result.warning_message = None
        return result

    monkeypatch.setattr(model_switch, "switch_model", _fake)
    return _fake


async def _make_put_request(app, session_key: str, body: dict):
    request = web.Request.__new__(web.Request)
    request._app = app
    request._match_info = {"id": session_key}
    request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
async def test_put_session_model_happy_path(app_with_runner, fake_switch_model, mock_runner):
    request = await _make_put_request(
        app_with_runner,
        "agent:main:myah:dm:chat123",
        {"model": "anthropic/claude-opus-4.6"},
    )
    response = await handle_put_session_model(request)
    assert response.status == 200
    data = json.loads(response.body)
    assert data["model"] == "anthropic/claude-opus-4.6"
    assert data["provider"] == "anthropic"
    assert mock_runner._session_model_overrides["agent:main:myah:dm:chat123"]["model"] == "anthropic/claude-opus-4.6"
    mock_runner._evict_cached_agent.assert_called_once_with("agent:main:myah:dm:chat123")


@pytest.mark.asyncio
async def test_put_session_model_bad_model(app_with_runner, fake_switch_model, mock_runner):
    request = await _make_put_request(
        app_with_runner,
        "agent:main:myah:dm:chat123",
        {"model": "unknown/bogus-model"},
    )
    response = await handle_put_session_model(request)
    assert response.status == 400
    assert mock_runner._session_model_overrides == {}
    mock_runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_put_session_model_missing_model_field(app_with_runner, fake_switch_model, mock_runner):
    request = await _make_put_request(
        app_with_runner,
        "agent:main:myah:dm:chat123",
        {},
    )
    response = await handle_put_session_model(request)
    assert response.status == 400
    assert mock_runner._session_model_overrides == {}


@pytest.mark.asyncio
async def test_put_session_model_cross_session_isolation(app_with_runner, fake_switch_model, mock_runner):
    req_a = await _make_put_request(app_with_runner, "sk_A", {"model": "anthropic/claude-opus-4.6"})
    await handle_put_session_model(req_a)
    req_b = await _make_put_request(app_with_runner, "sk_B", {"model": "xiaomi/mimo-v2-pro"})
    await handle_put_session_model(req_b)
    assert mock_runner._session_model_overrides["sk_A"]["model"] == "anthropic/claude-opus-4.6"
    assert mock_runner._session_model_overrides["sk_B"]["model"] == "xiaomi/mimo-v2-pro"


@pytest.mark.asyncio
async def test_get_session_model_returns_current_override(app_with_runner, mock_runner):
    mock_runner._session_model_overrides["agent:main:myah:dm:chat123"] = {
        "model": "anthropic/claude-opus-4.6",
        "provider": "anthropic",
    }
    request = web.Request.__new__(web.Request)
    request._app = app_with_runner
    request._match_info = {"id": "agent:main:myah:dm:chat123"}
    response = await handle_get_session_model(request)
    assert response.status == 200
    data = json.loads(response.body)
    assert data["model"] == "anthropic/claude-opus-4.6"
    assert data["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_get_session_model_no_override_returns_empty(app_with_runner, mock_runner):
    request = web.Request.__new__(web.Request)
    request._app = app_with_runner
    request._match_info = {"id": "agent:main:myah:dm:unknown"}
    response = await handle_get_session_model(request)
    assert response.status == 200
    data = json.loads(response.body)
    assert data["model"] == ""
    assert data["provider"] == ""
