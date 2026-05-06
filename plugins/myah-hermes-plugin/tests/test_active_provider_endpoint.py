"""Tests for POST /myah/v1/active-provider — sync auth.json:active_provider.

Background: Bug B from PR #74 (May 1) — Myah's onboarding handlers add credentials
to ``auth.json:credential_pool`` but never set ``auth.json:active_provider``. Cron
jobs that auto-resolve the provider read a stale ``active_provider`` set by the
entrypoint heal block and pair it with config.yaml's model field, producing
requests to chatgpt.com/backend-api/codex with non-Codex model ids.

This endpoint lets the platform write ``active_provider`` from any onboarding
flow (API-key save, OAuth complete, manual switch) without hermes-core changes.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request


def _make_request(body: dict, headers: dict | None = None):
    """Build a mocked request for /myah/v1/active-provider."""
    request = make_mocked_request(
        'POST',
        '/myah/v1/active-provider',
        headers=headers or {},
    )
    request.json = AsyncMock(return_value=body)
    return request


def _make_adapter(auth_key: str = ''):
    """Construct a MyahAdapter with register_pre_setup_hook mocked out."""
    from gateway.config import PlatformConfig
    with patch('gateway.platforms.api_server.register_pre_setup_hook'):
        from myah_hermes_plugin.myah_platform.adapter import MyahAdapter
        return MyahAdapter(PlatformConfig(enabled=True, extra={'auth_key': auth_key}))


def _seed_auth_store(credential_pool: dict, active_provider: str | None = None,
                     providers: dict | None = None) -> None:
    """Write an auth.json into the isolated HERMES_HOME for the test."""
    from hermes_cli.auth import _save_auth_store
    store: dict = {
        'credential_pool': credential_pool,
        'providers': providers if providers is not None else {},
    }
    if active_provider is not None:
        store['active_provider'] = active_provider
    _save_auth_store(store)


def _read_auth_store() -> dict:
    from hermes_cli.auth import _load_auth_store
    return _load_auth_store()


@pytest.mark.asyncio
async def test_endpoint_requires_bearer_auth():
    adapter = _make_adapter(auth_key='secret-token')
    request = _make_request({'provider': 'openrouter'})  # no Authorization header
    resp = await adapter._handle_active_provider_endpoint(request)
    assert resp.status == 401
    body = json.loads(resp.body)
    assert 'error' in body


@pytest.mark.asyncio
async def test_endpoint_requires_provider_field():
    adapter = _make_adapter()  # no auth_key → auth disabled
    request = _make_request({})
    resp = await adapter._handle_active_provider_endpoint(request)
    assert resp.status == 400
    body = json.loads(resp.body)
    assert 'provider' in body['error'].lower()


@pytest.mark.asyncio
async def test_endpoint_rejects_empty_provider():
    adapter = _make_adapter()
    request = _make_request({'provider': ''})
    resp = await adapter._handle_active_provider_endpoint(request)
    assert resp.status == 400
    body = json.loads(resp.body)
    assert 'provider' in body['error'].lower()


@pytest.mark.asyncio
async def test_endpoint_rejects_unknown_provider():
    _seed_auth_store(credential_pool={'openai-codex': {'api_key': 'sk-x'}})
    adapter = _make_adapter()
    request = _make_request({'provider': 'nonexistent'})
    resp = await adapter._handle_active_provider_endpoint(request)
    assert resp.status == 400
    body = json.loads(resp.body)
    assert 'not in credential pool' in body['error']


@pytest.mark.asyncio
async def test_endpoint_sets_active_provider():
    _seed_auth_store(
        credential_pool={
            'openai-codex': {'api_key': 'sk-c'},
            'openrouter': {'api_key': 'sk-or'},
        },
        active_provider='openai-codex',
        providers={'openai-codex': {}},
    )
    adapter = _make_adapter()
    request = _make_request({'provider': 'openrouter'})
    resp = await adapter._handle_active_provider_endpoint(request)
    assert resp.status == 200
    body = json.loads(resp.body)
    assert body['active_provider'] == 'openrouter'
    assert body['previous'] == 'openai-codex'

    # Verify auth.json was actually written
    store = _read_auth_store()
    assert store['active_provider'] == 'openrouter'


@pytest.mark.asyncio
async def test_endpoint_sets_active_provider_first_time():
    _seed_auth_store(
        credential_pool={'openrouter': {'api_key': 'sk-or'}},
        active_provider=None,
        providers={},
    )
    adapter = _make_adapter()
    request = _make_request({'provider': 'openrouter'})
    resp = await adapter._handle_active_provider_endpoint(request)
    assert resp.status == 200
    body = json.loads(resp.body)
    assert body['active_provider'] == 'openrouter'
    assert body['previous'] is None

    store = _read_auth_store()
    assert store['active_provider'] == 'openrouter'


@pytest.mark.asyncio
async def test_endpoint_creates_providers_entry_if_missing():
    _seed_auth_store(
        credential_pool={'openrouter': {'api_key': 'sk-or'}},
        providers={},  # no 'openrouter' key in providers dict
    )
    adapter = _make_adapter()
    request = _make_request({'provider': 'openrouter'})
    resp = await adapter._handle_active_provider_endpoint(request)
    assert resp.status == 200

    store = _read_auth_store()
    assert isinstance(store.get('providers'), dict)
    assert 'openrouter' in store['providers']
    assert store['providers']['openrouter'] == {}
