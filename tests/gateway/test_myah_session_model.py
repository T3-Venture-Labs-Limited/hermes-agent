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
from aiohttp.test_utils import make_mocked_request

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
    # Use make_mocked_request so `request.app` resolves properly via the
    # aiohttp public API — not a manual `_app` assignment which doesn't match
    # real requests in production.
    request = make_mocked_request(
        "PUT",
        f"/myah/api/sessions/{session_key}/model",
        match_info={"id": session_key},
        app=app,
    )
    request.json = AsyncMock(return_value=body)
    return request


def _make_get_request(app, session_key: str):
    return make_mocked_request(
        "GET",
        f"/myah/api/sessions/{session_key}/model",
        match_info={"id": session_key},
        app=app,
    )


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
    request = _make_get_request(app_with_runner, "agent:main:myah:dm:chat123")
    response = await handle_get_session_model(request)
    assert response.status == 200
    data = json.loads(response.body)
    assert data["model"] == "anthropic/claude-opus-4.6"
    assert data["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_get_session_model_no_override_returns_empty(app_with_runner, mock_runner):
    request = _make_get_request(app_with_runner, "agent:main:myah:dm:unknown")
    response = await handle_get_session_model(request)
    assert response.status == 200
    data = json.loads(response.body)
    assert data["model"] == ""
    assert data["provider"] == ""


@pytest.mark.asyncio
async def test_message_body_model_override_sets_session_override(fake_switch_model, monkeypatch):
    """Test that POST /myah/v1/message with 'model' field applies session override."""
    from gateway.platforms.myah import MyahAdapter
    from gateway.config import PlatformConfig

    adapter = MyahAdapter(PlatformConfig(enabled=True, extra={'auth_key': ''}))
    mock_runner = MagicMock()
    mock_runner._session_model_overrides = {}
    mock_runner._evict_cached_agent = MagicMock()
    adapter.gateway_runner = mock_runner

    # Stub handle_message so we don't dispatch for real
    adapter.handle_message = AsyncMock()
    adapter._push_event_sync = MagicMock()

    # Build a request with model in body
    request = MagicMock()
    request.headers = {}
    request.json = AsyncMock(return_value={
        'message': 'hello',
        'session_id': 'chat123',
        'user_id': 'user1',
        'model': 'anthropic/claude-opus-4.6',
    })
    request.path = '/myah/v1/message'

    # Set the adapter's loop
    import asyncio
    adapter._loop = asyncio.get_running_loop()

    response = await adapter._handle_message_endpoint(request)

    # The override should have been applied before handle_message dispatch
    assert any(
        ov.get('model') == 'anthropic/claude-opus-4.6'
        for ov in mock_runner._session_model_overrides.values()
    ), f"override not set: {mock_runner._session_model_overrides}"


@pytest.mark.asyncio
async def test_message_body_provider_pins_explicit_provider(monkeypatch):
    """POST /myah/v1/message with both 'model' and 'provider' must pin the
    provider (skip switch_model's auto-detect).

    Regression for the home/new-chat first-message bug where OAuth-only
    providers like openai-codex fell back to OpenRouter because the
    platform never persisted the session override in time and Hermes saw
    only {model:'gpt-5.4-mini'} with no provider context.
    """
    from gateway.platforms.myah import MyahAdapter
    from gateway.config import PlatformConfig

    # Capture switch_model kwargs so we can assert explicit_provider flows through
    captured_kwargs = {}

    def _capture(**kwargs):
        captured_kwargs.update(kwargs)
        result = MagicMock()
        result.success = True
        result.new_model = kwargs.get('raw_input', '') or 'gpt-5.4-mini'
        result.target_provider = kwargs.get('explicit_provider') or 'openrouter'
        result.provider_label = result.target_provider
        result.api_key = 'sk-test'
        result.base_url = 'https://example.test'
        result.api_mode = ''
        result.error_message = ''
        return result

    from hermes_cli import model_switch
    monkeypatch.setattr(model_switch, 'switch_model', _capture)

    adapter = MyahAdapter(PlatformConfig(enabled=True, extra={'auth_key': ''}))
    mock_runner = MagicMock()
    mock_runner._session_model_overrides = {}
    mock_runner._evict_cached_agent = MagicMock()
    adapter.gateway_runner = mock_runner
    adapter.handle_message = AsyncMock()
    adapter._push_event_sync = MagicMock()

    request = MagicMock()
    request.headers = {}
    request.json = AsyncMock(return_value={
        'message': 'hello',
        'session_id': 'chat123',
        'user_id': 'user1',
        'model': 'gpt-5.4-mini',
        'provider': 'openai-codex',
    })
    request.path = '/myah/v1/message'

    import asyncio
    adapter._loop = asyncio.get_running_loop()

    await adapter._handle_message_endpoint(request)

    assert captured_kwargs.get('explicit_provider') == 'openai-codex', (
        f'provider not forwarded to switch_model: {captured_kwargs}'
    )
    # And the resolved override should carry the pinned provider
    assert any(
        ov.get('provider') == 'openai-codex'
        for ov in mock_runner._session_model_overrides.values()
    ), f'override did not pin provider: {mock_runner._session_model_overrides}'


@pytest.mark.asyncio
async def test_run_completed_emits_model_and_provider(monkeypatch):
    """run.completed event includes model + provider read from cached agent."""
    from gateway.platforms.myah import MyahAdapter
    from gateway.config import PlatformConfig

    adapter = MyahAdapter(PlatformConfig(enabled=True, extra={'auth_key': ''}))

    mock_agent = MagicMock()
    mock_agent.model = 'anthropic/claude-opus-4.6'
    mock_agent.provider = 'anthropic'

    mock_runner = MagicMock()
    mock_runner._session_model_overrides = {}
    adapter.gateway_runner = mock_runner
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter._active_sessions = {}

    emitted = []

    def _capture(stream_id, event):
        emitted.append(event)

    adapter._push_event_sync = _capture
    adapter._streams = {'myah_test': MagicMock()}

    from gateway.platforms.base import MessageEvent, MessageType, SessionSource
    from gateway.config import Platform
    source = SessionSource(
        platform=Platform.MYAH,
        chat_id='chat123',
        chat_type='dm',
        user_id='user1',
    )
    event = MessageEvent(
        text='hello',
        message_type=MessageType.TEXT,
        source=source,
        message_id='myah_test',
    )

    # Pre-populate session_key -> stream mapping as _handle_message_endpoint would
    from gateway.session import build_session_key
    _sk = build_session_key(source, group_sessions_per_user=True, thread_sessions_per_user=False)
    adapter._session_streams[_sk] = 'myah_test'
    adapter._chat_id_streams['chat123'] = 'myah_test'
    adapter._stream_sessions['myah_test'] = _sk

    # Seed agent cache under the correct session_key
    mock_runner._agent_cache = {_sk: (mock_agent, 'sig_hash')}

    await adapter._dispatch_message(event, 'myah_test', 'chat123', _sk)

    run_completed = [e for e in emitted if e.get('event') == 'run.completed']
    assert run_completed, f"no run.completed event; got events: {[e.get('event') for e in emitted]}"
    assert run_completed[0].get('model') == 'anthropic/claude-opus-4.6'
    assert run_completed[0].get('provider') == 'anthropic'
