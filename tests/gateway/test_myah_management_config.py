"""Tests for /myah/api/config endpoints — subprocess fix, reset, ETag, restart."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Set up a tmp HERMES_HOME with a minimal config.yaml."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        'model: anthropic/claude-opus-4.6\n'
        'auxiliary:\n'
        '  vision:\n'
        '    provider: auto\n'
        "    model: ''\n"
    )
    return tmp_path


# ── Task 3: subprocess fix ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_config_uses_in_process_set_config_value(home, monkeypatch):
    """PATCH must call set_config_value directly, not via subprocess."""
    from gateway.platforms import myah_management as mgmt

    called_with = []

    def fake_set(key, value):
        called_with.append((key, value))

    monkeypatch.setattr(mgmt, 'set_config_value', fake_set)

    request = make_mocked_request('PATCH', '/myah/api/config')
    from unittest.mock import AsyncMock
    request.json = AsyncMock(return_value={
        'auxiliary.vision.model': 'google/gemini-2.5-flash',
    })

    from gateway.platforms.myah_management import handle_patch_config
    resp = await handle_patch_config(request)

    assert resp.status == 200
    import json
    body = json.loads(resp.body)
    assert body['ok'] is True
    assert ('auxiliary.vision.model', 'google/gemini-2.5-flash') in called_with


# ── Task 4: SOUL ETag ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_soul_returns_etag_header(home):
    (home / 'SOUL.md').write_text('You are Myah.\n')

    request = make_mocked_request('GET', '/myah/api/config/soul')

    from gateway.platforms.myah_management import handle_get_soul
    resp = await handle_get_soul(request)

    assert resp.status == 200
    assert 'ETag' in resp.headers
    etag = resp.headers['ETag']
    assert etag.startswith('"sha256-')
    assert resp.text == 'You are Myah.\n'


@pytest.mark.asyncio
async def test_put_soul_without_if_match_returns_428(home):
    (home / 'SOUL.md').write_text('initial\n')

    request = make_mocked_request('PUT', '/myah/api/config/soul')
    from unittest.mock import AsyncMock
    request.text = AsyncMock(return_value='new content\n')

    from gateway.platforms.myah_management import handle_put_soul
    resp = await handle_put_soul(request)
    assert resp.status == 428
    import json
    body = json.loads(resp.body)
    assert 'if-match' in body['error'].lower()


@pytest.mark.asyncio
async def test_put_soul_with_matching_if_match_succeeds(home):
    (home / 'SOUL.md').write_text('initial\n')

    from gateway.platforms.myah_management import handle_get_soul, handle_put_soul, _soul_etag
    etag = _soul_etag('initial\n')

    request = make_mocked_request(
        'PUT', '/myah/api/config/soul',
        headers={'If-Match': etag},
    )
    from unittest.mock import AsyncMock
    request.text = AsyncMock(return_value='new content\n')

    resp = await handle_put_soul(request)
    assert resp.status == 200
    assert 'ETag' in resp.headers
    assert resp.headers['ETag'] != etag
    assert (home / 'SOUL.md').read_text() == 'new content\n'


@pytest.mark.asyncio
async def test_put_soul_with_stale_if_match_returns_412(home):
    (home / 'SOUL.md').write_text('changed by other tab\n')

    stale_etag = '"sha256-abc123"'
    request = make_mocked_request(
        'PUT', '/myah/api/config/soul',
        headers={'If-Match': stale_etag},
    )
    from unittest.mock import AsyncMock
    request.text = AsyncMock(return_value='my edit\n')

    from gateway.platforms.myah_management import handle_put_soul
    resp = await handle_put_soul(request)
    assert resp.status == 412
    import json
    body = json.loads(resp.body)
    assert body['current_body'] == 'changed by other tab\n'
    assert 'ETag' in resp.headers


@pytest.mark.asyncio
async def test_put_soul_rejects_over_hard_cap(home):
    (home / 'SOUL.md').write_text('initial\n')

    from gateway.platforms.myah_management import handle_put_soul, _soul_etag
    etag = _soul_etag('initial\n')

    huge = 'x' * 40_000
    request = make_mocked_request(
        'PUT', '/myah/api/config/soul',
        headers={'If-Match': etag},
    )
    from unittest.mock import AsyncMock
    request.text = AsyncMock(return_value=huge)

    resp = await handle_put_soul(request)
    assert resp.status == 413
    import json
    body = json.loads(resp.body)
    assert body['limit'] == 32_768
    assert body['got'] == 40_000
    assert (home / 'SOUL.md').read_text() == 'initial\n'


@pytest.mark.asyncio
async def test_put_soul_warns_over_soft_limit(home):
    (home / 'SOUL.md').write_text('initial\n')

    from gateway.platforms.myah_management import handle_put_soul, _soul_etag
    etag = _soul_etag('initial\n')

    longish = 'x' * 10_000
    request = make_mocked_request(
        'PUT', '/myah/api/config/soul',
        headers={'If-Match': etag},
    )
    from unittest.mock import AsyncMock
    request.text = AsyncMock(return_value=longish)

    resp = await handle_put_soul(request)
    assert resp.status == 200
    import json
    body = json.loads(resp.body)
    assert 'warning' in body


@pytest.mark.asyncio
async def test_get_soul_exposes_limit_headers(home):
    (home / 'SOUL.md').write_text('hi\n')

    request = make_mocked_request('GET', '/myah/api/config/soul')
    from gateway.platforms.myah_management import handle_get_soul
    resp = await handle_get_soul(request)
    assert resp.headers['X-Soul-Soft-Warn-Chars'] == '8192'
    assert resp.headers['X-Soul-Hard-Cap-Chars'] == '32768'


# ── Task 5: Restart endpoint ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restart_returns_409_when_runs_are_in_flight(home, monkeypatch):
    from gateway.platforms import myah_management as mgmt

    fake_runner = MagicMock()
    fake_runner._running_agents = {
        'session-A': MagicMock(name='resolved-agent'),
        'session-B': object(),
    }
    monkeypatch.setattr(mgmt, '_gateway_runner', fake_runner, raising=False)

    request = make_mocked_request('POST', '/myah/api/gateway/restart')
    from gateway.platforms.myah_management import handle_gateway_restart
    resp = await handle_gateway_restart(request)
    assert resp.status == 409
    import json
    body = json.loads(resp.body)
    assert body['error'] == 'busy'
    assert set(body['busy_sessions']) == {'session-A', 'session-B'}


@pytest.mark.asyncio
async def test_restart_returns_202_when_idle(home, monkeypatch):
    from gateway.platforms import myah_management as mgmt

    fake_runner = MagicMock()
    fake_runner._running_agents = {}
    monkeypatch.setattr(mgmt, '_gateway_runner', fake_runner, raising=False)

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr('subprocess.run', fake_run)

    request = make_mocked_request('POST', '/myah/api/gateway/restart')
    from gateway.platforms.myah_management import handle_gateway_restart
    resp = await handle_gateway_restart(request)
    assert resp.status == 202
    import json
    body = json.loads(resp.body)
    assert body['status'] == 'restarting'
    assert calls == [['supervisorctl', 'restart', 'hermes']]


# ── Task 6: Schema endpoint ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_schema_returns_aux_tasks(home):
    request = make_mocked_request('GET', '/myah/api/config/schema')
    from gateway.platforms.myah_management import handle_get_schema
    resp = await handle_get_schema(request)
    assert resp.status == 200
    import json
    body = json.loads(resp.body)

    aux = body.get('auxiliary', {})
    assert 'title_generation' in aux
    assert 'follow_up_generation' in aux
    assert aux['title_generation']['provider']['default'] == 'openrouter'
    assert aux['title_generation']['model']['default'] == 'google/gemini-2.5-flash'


@pytest.mark.asyncio
async def test_get_schema_includes_main_model_section(home):
    request = make_mocked_request('GET', '/myah/api/config/schema')
    from gateway.platforms.myah_management import handle_get_schema
    resp = await handle_get_schema(request)
    import json
    body = json.loads(resp.body)
    assert 'model' in body


# ── Task 7: Reset endpoint ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_aux_vision_restores_defaults(home, monkeypatch):
    from gateway.platforms import myah_management as mgmt

    called = []
    monkeypatch.setattr(mgmt, 'set_config_value', lambda k, v: called.append((k, v)))

    request = make_mocked_request(
        'POST', '/myah/api/config/reset/aux_vision',
        match_info={'section': 'aux_vision'},
    )
    from gateway.platforms.myah_management import handle_reset_section
    resp = await handle_reset_section(request)
    assert resp.status == 200

    keys_reset = {k for k, _ in called}
    assert 'auxiliary.vision.provider' in keys_reset
    assert 'auxiliary.vision.model' in keys_reset


@pytest.mark.asyncio
async def test_reset_rejects_unknown_section(home):
    request = make_mocked_request(
        'POST', '/myah/api/config/reset/gibberish',
        match_info={'section': 'gibberish'},
    )
    from gateway.platforms.myah_management import handle_reset_section
    resp = await handle_reset_section(request)
    assert resp.status == 400


@pytest.mark.asyncio
async def test_reset_aux_title_generation_restores_defaults(home, monkeypatch):
    from gateway.platforms import myah_management as mgmt

    called = []
    monkeypatch.setattr(mgmt, 'set_config_value', lambda k, v: called.append((k, v)))

    request = make_mocked_request(
        'POST', '/myah/api/config/reset/aux_title_generation',
        match_info={'section': 'aux_title_generation'},
    )
    from gateway.platforms.myah_management import handle_reset_section
    resp = await handle_reset_section(request)
    assert resp.status == 200
    keys_reset = {k for k, _ in called}
    assert 'auxiliary.title_generation.provider' in keys_reset
    model_val = next(v for k, v in called if k == 'auxiliary.title_generation.model')
    assert model_val == 'google/gemini-2.5-flash'


# ── Task 8: MCP registry refresh ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_mcp_calls_register_mcp_servers(home, monkeypatch):
    import sys
    import types

    registered = []

    # Fake mcp_tool module
    fake_mcp_tool = types.ModuleType('tools.mcp_tool')
    fake_mcp_tool.register_mcp_servers = lambda s: registered.append(s) or list(s.keys())
    fake_mcp_tool.disconnect_mcp_server = MagicMock(return_value=True)
    fake_mcp_tool.shutdown_mcp_servers = MagicMock()
    sys.modules['tools.mcp_tool'] = fake_mcp_tool

    evictions = []
    fake_runner = MagicMock()
    fake_runner._agent_cache = {'sess-1': MagicMock(), 'sess-2': MagicMock()}
    fake_runner._evict_cached_agent = lambda k: evictions.append(k)

    from gateway.platforms import myah_management as mgmt
    monkeypatch.setattr(mgmt, '_gateway_runner', fake_runner, raising=False)

    request = make_mocked_request('POST', '/myah/api/mcp')
    from unittest.mock import AsyncMock as AM
    request.json = AM(return_value={
        'name': 'github',
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-github'],
        'env': {},
    })

    from gateway.platforms.myah_management import handle_add_mcp
    resp = await handle_add_mcp(request)

    assert resp.status in (200, 201)
    assert len(registered) == 1
    assert 'github' in registered[0]
    assert set(evictions) == {'sess-1', 'sess-2'}

    sys.modules.pop('tools.mcp_tool', None)


@pytest.mark.asyncio
async def test_remove_mcp_uses_per_server_disconnect(home, monkeypatch):
    import sys
    import types

    (home / 'config.yaml').write_text(
        'mcp_servers:\n'
        '  github:\n'
        '    command: npx\n'
        '    args: []\n'
        '  linear:\n'
        '    command: npx\n'
        '    args: []\n'
    )

    calls = {'disconnect': [], 'shutdown_all': 0}

    fake_mcp_tool = types.ModuleType('tools.mcp_tool')
    fake_mcp_tool.disconnect_mcp_server = lambda n: calls['disconnect'].append(n) or True
    fake_mcp_tool.shutdown_mcp_servers = lambda: None
    fake_mcp_tool.register_mcp_servers = lambda s: list(s.keys())
    sys.modules['tools.mcp_tool'] = fake_mcp_tool

    request = make_mocked_request(
        'DELETE', '/myah/api/mcp/github',
        match_info={'name': 'github'},
    )

    from gateway.platforms.myah_management import handle_remove_mcp
    resp = await handle_remove_mcp(request)
    assert resp.status == 200

    assert calls['disconnect'] == ['github']
    assert calls['shutdown_all'] == 0

    sys.modules.pop('tools.mcp_tool', None)


@pytest.mark.asyncio
async def test_remove_mcp_returns_404_for_missing_name(home, monkeypatch):
    import sys
    import types

    (home / 'config.yaml').write_text('mcp_servers: {}\n')

    fake_mcp_tool = types.ModuleType('tools.mcp_tool')
    fake_mcp_tool.disconnect_mcp_server = MagicMock(return_value=False)
    fake_mcp_tool.shutdown_mcp_servers = MagicMock()
    fake_mcp_tool.register_mcp_servers = MagicMock(return_value=[])
    sys.modules['tools.mcp_tool'] = fake_mcp_tool

    request = make_mocked_request(
        'DELETE', '/myah/api/mcp/nonexistent',
        match_info={'name': 'nonexistent'},
    )
    from gateway.platforms.myah_management import handle_remove_mcp
    resp = await handle_remove_mcp(request)
    assert resp.status == 404

    sys.modules.pop('tools.mcp_tool', None)


# ── Task 9: Memory-provider leak fix ──────────────────────────────────────


@pytest.mark.asyncio
async def test_put_session_model_calls_full_teardown(home, monkeypatch):
    """Session model switch must shutdown_memory_provider and close before eviction."""
    from gateway.platforms import myah_management as mgmt

    teardown_calls = []

    class FakeAgent:
        def shutdown_memory_provider(self):
            teardown_calls.append('shutdown_memory_provider')

        def close(self):
            teardown_calls.append('close')

    fake_runner = MagicMock()
    # _agent_cache stores tuples (agent, metadata) per run.py
    fake_runner._agent_cache = {'test-session': (FakeAgent(), {'created_ts': 0})}
    fake_runner._evict_cached_agent = lambda k: teardown_calls.append(f'evict:{k}')
    fake_runner._session_model_overrides = {}
    monkeypatch.setattr(mgmt, '_gateway_runner', fake_runner, raising=False)

    # Stub switch_model
    from hermes_cli import model_switch

    def _fake_switch(**kwargs):
        result = MagicMock()
        result.success = True
        result.new_model = 'anthropic/claude-haiku-3'
        result.target_provider = 'anthropic'
        result.provider_label = 'Anthropic'
        result.api_key = ''
        result.base_url = ''
        result.api_mode = ''
        result.warning_message = None
        return result

    monkeypatch.setattr(model_switch, 'switch_model', _fake_switch)

    app = web.Application()
    adapter = MagicMock()
    adapter.gateway_runner = fake_runner
    app['myah_adapter'] = adapter

    request = make_mocked_request(
        'PUT', '/myah/api/sessions/test-session/model',
        match_info={'id': 'test-session'},
        app=app,
    )
    from unittest.mock import AsyncMock as AM
    request.json = AM(return_value={'model': 'anthropic/claude-haiku-3'})

    from gateway.platforms.myah_management import handle_put_session_model
    resp = await handle_put_session_model(request)
    assert resp.status == 200

    assert 'shutdown_memory_provider' in teardown_calls
    assert 'close' in teardown_calls
    assert 'evict:test-session' in teardown_calls
    # Teardown must precede eviction
    mp_idx = teardown_calls.index('shutdown_memory_provider')
    evict_idx = teardown_calls.index('evict:test-session')
    assert mp_idx < evict_idx


# ── Task 10: Last-reseed endpoint ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_last_reseed_returns_timestamps(home):
    marker = home / '.myah_last_reseed'
    marker.write_text('config=2026-04-18T00:00:00Z\nsoul=2026-04-17T12:00:00Z\n')

    request = make_mocked_request('GET', '/myah/api/config/last-reseed')
    from gateway.platforms.myah_management import handle_get_last_reseed
    resp = await handle_get_last_reseed(request)
    assert resp.status == 200
    import json
    body = json.loads(resp.body)
    assert body['config'] == '2026-04-18T00:00:00Z'
    assert body['soul'] == '2026-04-17T12:00:00Z'


@pytest.mark.asyncio
async def test_get_last_reseed_returns_empty_when_no_marker(home):
    marker = home / '.myah_last_reseed'
    marker.unlink(missing_ok=True)

    request = make_mocked_request('GET', '/myah/api/config/last-reseed')
    from gateway.platforms.myah_management import handle_get_last_reseed
    resp = await handle_get_last_reseed(request)
    assert resp.status == 200
    import json
    body = json.loads(resp.body)
    assert body == {}
