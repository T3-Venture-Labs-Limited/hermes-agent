"""Regression test for active_provider auto-sync on successful chat.

Production scenario: a user has multiple providers in their credential pool
(e.g., openai-codex via OAuth + openrouter via API key). The entrypoint heal
block at agent/scripts/entrypoint.sh sets active_provider to the first OAuth
provider in a hardcoded tuple — for users with both codex and openrouter,
that's always 'openai-codex'. When the user then selects an OpenRouter model
in the chat UI, the platform sends the right model+provider in the per-message
override (chat works because the per-message override bypasses the auth
chain), but auth.json:active_provider stays stale. Cron's resolve_provider
('auto') reads the stale active_provider and pairs it with the user's chosen
model from config.yaml — producing 400s when the model family doesn't match
the provider (e.g. gemini-2.5-flash-lite via Codex).

POST /myah/v1/active-provider (test_active_provider_endpoint.py) fixes this
on EXPLICIT onboarding actions. This file tests the IMPLICIT auto-heal that
fires on every chat: when the adapter's _handle_message_endpoint applies a
successful per-message model+provider override, it ALSO writes that provider
as auth.json:active_provider (when it differs). Users whose state was already
broken before our fix landed heal automatically on their next chat.
"""

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request


class _FakeRunner:
    """Minimal runner stand-in tracking session-override state."""

    def __init__(self) -> None:
        self._session_model_overrides: dict[str, dict] = {}

    def get_session_override(self, session_key: str) -> dict | None:
        return self._session_model_overrides.get(session_key)

    def set_session_override(self, session_key: str, override: dict) -> None:
        self._session_model_overrides[session_key] = dict(override)


def _make_adapter_with_runner():
    from gateway.config import PlatformConfig

    with patch('gateway.platforms.api_server.register_pre_setup_hook'):
        from myah_hermes_plugin.myah_platform.adapter import MyahAdapter

        adapter = MyahAdapter(PlatformConfig(enabled=True, extra={'auth_key': ''}))
    runner = _FakeRunner()
    adapter.gateway_runner = runner
    return adapter, runner


def _install_fake_model_switch(target_provider: str, new_model: str = 'haiku'):
    fake_mod = types.ModuleType('hermes_cli.model_switch')

    def _switch_model(**kwargs):  # noqa: ARG001
        return SimpleNamespace(
            success=True,
            error_message='',
            new_model=new_model,
            target_provider=target_provider,
            api_key='ak',
            base_url='https://x',
            api_mode='chat_completions',
        )

    fake_mod.switch_model = _switch_model
    sys.modules['hermes_cli.model_switch'] = fake_mod
    return fake_mod


@pytest.fixture
def fake_switch_to_openrouter():
    fake_mod = _install_fake_model_switch(target_provider='openrouter')
    yield fake_mod
    sys.modules.pop('hermes_cli.model_switch', None)


def _make_message_request(body: dict):
    request = make_mocked_request('POST', '/myah/v1/message')
    request.json = AsyncMock(return_value=body)
    return request


def _seed_auth_store(active: str | None, pool_keys: list[str]):
    """Write a starting auth.json that exercises the heal path."""
    from hermes_cli.auth import _save_auth_store

    pool: dict[str, list[dict]] = {
        pid: [{'id': f'{pid}-1', 'access_token': f'fake-{pid}', 'auth_type': 'api_key', 'priority': 0}]
        for pid in pool_keys
    }
    providers: dict[str, dict] = {pid: {} for pid in (pool_keys if active is None else [active])}
    store: dict = {
        'credential_pool': pool,
        'providers': providers,
    }
    if active is not None:
        store['active_provider'] = active
    _save_auth_store(store)


def _read_auth_store():
    from hermes_cli.auth import _load_auth_store

    return _load_auth_store()


@pytest.mark.asyncio
async def test_chat_heals_stale_active_provider(fake_switch_to_openrouter):
    """A successful chat with provider=openrouter must heal active_provider.

    This is the production case: the entrypoint heal locked active_provider
    to openai-codex even though the user has both codex AND openrouter in
    the pool. After our fix lands, the user's NEXT chat with an openrouter
    model will heal the state without any explicit re-onboarding action.
    """
    _seed_auth_store(active='openai-codex', pool_keys=['openai-codex', 'openrouter'])
    adapter, _runner = _make_adapter_with_runner()

    request = _make_message_request({
        'message': 'hello',
        'session_id': 'chat-heal-1',
        'user_id': 'u-heal-1',
        'model': 'haiku',
        'provider': 'openrouter',
    })

    resp = await adapter._handle_message_endpoint(request)

    # The chat itself must accept (202 — adapter starts an async task).
    assert resp.status == 202, f'expected 202, got {resp.status} body={resp.body!r}'

    # auth.json:active_provider must now be openrouter.
    store = _read_auth_store()
    assert store['active_provider'] == 'openrouter', (
        f'expected active_provider=openrouter after chat, got {store["active_provider"]!r}'
    )
    # providers entry must exist for the new active provider.
    assert 'openrouter' in store.get('providers', {}), (
        f'expected providers to contain openrouter, got {list(store.get("providers", {}).keys())}'
    )


@pytest.mark.asyncio
async def test_chat_does_not_change_active_provider_when_already_correct(fake_switch_to_openrouter):
    """If active_provider already matches, the heal is a no-op (no churn)."""
    _seed_auth_store(active='openrouter', pool_keys=['openrouter'])
    adapter, _runner = _make_adapter_with_runner()

    request = _make_message_request({
        'message': 'hello',
        'session_id': 'chat-noop-1',
        'user_id': 'u-noop-1',
        'model': 'haiku',
        'provider': 'openrouter',
    })

    resp = await adapter._handle_message_endpoint(request)

    assert resp.status == 202

    store = _read_auth_store()
    assert store['active_provider'] == 'openrouter'


@pytest.mark.asyncio
async def test_chat_heal_failure_does_not_break_chat():
    """If auth.json write fails, chat must still proceed (best-effort heal)."""
    _seed_auth_store(active='openai-codex', pool_keys=['openai-codex', 'openrouter'])
    fake_mod = _install_fake_model_switch(target_provider='openrouter')
    try:
        adapter, _runner = _make_adapter_with_runner()

        request = _make_message_request({
            'message': 'hello',
            'session_id': 'chat-fail-heal',
            'user_id': 'u-fail-heal',
            'model': 'haiku',
            'provider': 'openrouter',
        })

        # Simulate auth.json write failure by patching _save_auth_store to raise.
        with patch('hermes_cli.auth._save_auth_store', side_effect=OSError('disk full')):
            resp = await adapter._handle_message_endpoint(request)

        # Chat must still succeed (heal is best-effort).
        assert resp.status == 202, (
            f'chat must not fail when heal fails; got {resp.status} body={resp.body!r}'
        )
    finally:
        sys.modules.pop('hermes_cli.model_switch', None)
        # Re-import the real module so other tests don't see our fake.
        sys.modules.pop('hermes_cli.model_switch', None)
