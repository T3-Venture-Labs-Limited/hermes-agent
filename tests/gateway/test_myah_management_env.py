"""Tests for env var management endpoints in myah_management.py."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.myah_management import (
    _DENIED_ENV_VARS,
    _auth_middleware,
    handle_delete_env,
    handle_list_env,
    handle_set_env,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

_SAMPLE_OPTIONAL_ENV_VARS = {
    'ANTHROPIC_API_KEY': {
        'description': 'Anthropic API key',
        'url': 'https://console.anthropic.com',
        'category': 'provider',
        'password': True,
        'tools': [],
    },
    'OPENAI_API_KEY': {
        'description': 'OpenAI API key',
        'url': 'https://platform.openai.com',
        'category': 'provider',
        'password': True,
        'tools': [],
    },
}


@pytest.fixture
def mock_env_on_disk():
    """A fake .env dict returned by load_env()."""
    return {'ANTHROPIC_API_KEY': 'sk-ant-test-1234567890abcd'}


# ── handle_list_env ──────────────────────────────────────────────────────────


def _list_env_patches(env_on_disk: dict):
    """Context manager: patch the two lazy imports inside handle_list_env."""
    return (
        patch('hermes_cli.config.OPTIONAL_ENV_VARS', _SAMPLE_OPTIONAL_ENV_VARS),
        patch('hermes_cli.config.load_env', return_value=env_on_disk),
    )


class TestListEnv:
    @pytest.mark.asyncio
    async def test_returns_known_vars_only(self, mock_env_on_disk):
        """Only vars in OPTIONAL_ENV_VARS appear in the response."""
        p1, p2 = _list_env_patches(mock_env_on_disk)
        with p1, p2:
            request = MagicMock()
            response = await handle_list_env(request)

        data = json.loads(response.body)
        assert isinstance(data, dict)
        assert set(data.keys()) == {'ANTHROPIC_API_KEY', 'OPENAI_API_KEY'}

    @pytest.mark.asyncio
    async def test_set_key_is_marked_is_set_true(self, mock_env_on_disk):
        """Keys present in .env have is_set=True."""
        p1, p2 = _list_env_patches(mock_env_on_disk)
        with p1, p2:
            request = MagicMock()
            response = await handle_list_env(request)

        data = json.loads(response.body)
        assert data['ANTHROPIC_API_KEY']['is_set'] is True
        assert data['OPENAI_API_KEY']['is_set'] is False

    @pytest.mark.asyncio
    async def test_values_are_redacted(self, mock_env_on_disk):
        """Raw secret values must never appear in the response."""
        p1, p2 = _list_env_patches(mock_env_on_disk)
        with p1, p2:
            request = MagicMock()
            response = await handle_list_env(request)

        data = json.loads(response.body)
        anthropic_info = data['ANTHROPIC_API_KEY']
        raw_value = mock_env_on_disk['ANTHROPIC_API_KEY']

        # Raw value must NOT appear
        assert anthropic_info['redacted_value'] != raw_value
        # Redacted form contains '...' for long values or '***' for short ones
        redacted = anthropic_info['redacted_value']
        assert '...' in redacted or redacted == '***'

    @pytest.mark.asyncio
    async def test_short_value_uses_stars(self):
        """Values shorter than 12 chars use '***' instead of prefix...suffix."""
        p1, p2 = _list_env_patches({'ANTHROPIC_API_KEY': 'short'})
        with p1, p2:
            request = MagicMock()
            response = await handle_list_env(request)

        data = json.loads(response.body)
        assert data['ANTHROPIC_API_KEY']['redacted_value'] == '***'

    @pytest.mark.asyncio
    async def test_metadata_fields_present(self, mock_env_on_disk):
        """Each entry carries description, url, category, is_password, tools."""
        p1, p2 = _list_env_patches(mock_env_on_disk)
        with p1, p2:
            request = MagicMock()
            response = await handle_list_env(request)

        data = json.loads(response.body)
        for key, info in data.items():
            assert 'is_set' in info, f'{key} missing is_set'
            assert 'description' in info, f'{key} missing description'
            assert 'redacted_value' in info, f'{key} missing redacted_value'
            assert 'url' in info, f'{key} missing url'
            assert 'category' in info, f'{key} missing category'
            assert 'is_password' in info, f'{key} missing is_password'
            assert 'tools' in info, f'{key} missing tools'


# ── handle_set_env ───────────────────────────────────────────────────────────


class TestSetEnv:
    @pytest.mark.asyncio
    async def test_set_valid_key(self):
        """Valid key+value saves successfully and returns ok=True."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': 'NEW_API_KEY', 'value': 'sk-test-123'})

        with patch('hermes_cli.config.save_env_value') as mock_save:
            response = await handle_set_env(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data['ok'] is True
        assert data['key'] == 'NEW_API_KEY'
        mock_save.assert_called_once_with('NEW_API_KEY', 'sk-test-123')

    @pytest.mark.asyncio
    async def test_set_rejects_empty_key(self):
        """Empty key returns 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': '', 'value': 'x'})

        response = await handle_set_env(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_set_rejects_whitespace_only_key(self):
        """Whitespace-only key (stripped to empty) returns 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': '   ', 'value': 'x'})

        response = await handle_set_env(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_set_rejects_empty_value(self):
        """Empty value returns 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': 'MY_KEY', 'value': ''})

        response = await handle_set_env(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_set_rejects_oversized_value(self):
        """Values exceeding 4096 chars return 422."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': 'MY_KEY', 'value': 'a' * 4097})

        response = await handle_set_env(request)

        assert response.status == 422

    @pytest.mark.asyncio
    async def test_set_accepts_max_length_value(self):
        """Values exactly 4096 chars are accepted."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': 'MY_KEY', 'value': 'a' * 4096})

        with patch('hermes_cli.config.save_env_value'):
            response = await handle_set_env(request)

        assert response.status == 200

    @pytest.mark.asyncio
    @pytest.mark.parametrize('denied_key', sorted(_DENIED_ENV_VARS)[:5])
    async def test_set_rejects_denied_vars(self, denied_key):
        """Protected system vars return 422 with 'protected' in the error message."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': denied_key, 'value': '/evil'})

        response = await handle_set_env(request)

        assert response.status == 422
        data = json.loads(response.body)
        assert 'protected' in data['error'].lower()

    @pytest.mark.asyncio
    async def test_set_rejects_path_explicitly(self):
        """PATH is the canonical protected var — verify explicitly."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': 'PATH', 'value': '/evil'})

        response = await handle_set_env(request)

        assert response.status == 422
        data = json.loads(response.body)
        assert 'protected' in data['error'].lower()

    @pytest.mark.asyncio
    async def test_set_rejects_invalid_json(self):
        """Malformed request body returns 400."""
        request = MagicMock()
        request.json = AsyncMock(side_effect=Exception('not json'))

        response = await handle_set_env(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_set_propagates_save_exception(self):
        """If save_env_value raises, the endpoint returns 500."""
        request = MagicMock()
        request.json = AsyncMock(return_value={'key': 'MY_KEY', 'value': 'val'})

        with patch('hermes_cli.config.save_env_value', side_effect=OSError('disk full')):
            response = await handle_set_env(request)

        assert response.status == 500


# ── handle_delete_env ────────────────────────────────────────────────────────


class TestDeleteEnv:
    @pytest.mark.asyncio
    async def test_delete_existing_key(self):
        """Successfully deleting a key returns ok=True."""
        request = MagicMock()
        request.match_info = {'key': 'OLD_API_KEY'}

        with patch('hermes_cli.config.remove_env_value', return_value=True) as mock_remove:
            response = await handle_delete_env(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data['ok'] is True
        assert data['key'] == 'OLD_API_KEY'
        mock_remove.assert_called_once_with('OLD_API_KEY')

    @pytest.mark.asyncio
    async def test_delete_missing_key_returns_404(self):
        """Deleting a key not in .env returns 404."""
        request = MagicMock()
        request.match_info = {'key': 'NONEXISTENT_KEY'}

        with patch('hermes_cli.config.remove_env_value', return_value=False):
            response = await handle_delete_env(request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_delete_propagates_exception(self):
        """If remove_env_value raises, the endpoint returns 500."""
        request = MagicMock()
        request.match_info = {'key': 'SOME_KEY'}

        with patch('hermes_cli.config.remove_env_value', side_effect=OSError('disk error')):
            response = await handle_delete_env(request)

        assert response.status == 500

    @pytest.mark.asyncio
    async def test_delete_empty_key_returns_400(self):
        """An empty key path param returns 400."""
        request = MagicMock()
        request.match_info = {'key': ''}

        response = await handle_delete_env(request)

        assert response.status == 400


# ── _auth_middleware ─────────────────────────────────────────────────────────


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_allows_request_when_no_auth_key(self):
        """With auth_key='', all requests pass through."""
        inner = AsyncMock(return_value=MagicMock(status=200))
        wrapped = _auth_middleware(inner, auth_key='')

        request = MagicMock()
        request.headers = {}

        response = await wrapped(request)
        inner.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_allows_valid_bearer_token(self):
        """Correct Bearer token passes through to the inner handler."""
        inner = AsyncMock(return_value=MagicMock(status=200))
        wrapped = _auth_middleware(inner, auth_key='secret-token')

        request = MagicMock()
        request.headers = {'Authorization': 'Bearer secret-token'}

        response = await wrapped(request)
        inner.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_rejects_missing_auth_header(self):
        """Request without Authorization header returns 401."""
        inner = AsyncMock(return_value=MagicMock(status=200))
        wrapped = _auth_middleware(inner, auth_key='secret-token')

        request = MagicMock()
        request.headers = {}

        response = await wrapped(request)
        assert response.status == 401
        inner.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_wrong_token(self):
        """Wrong Bearer token returns 401."""
        inner = AsyncMock(return_value=MagicMock(status=200))
        wrapped = _auth_middleware(inner, auth_key='secret-token')

        request = MagicMock()
        request.headers = {'Authorization': 'Bearer wrong-token'}

        response = await wrapped(request)
        assert response.status == 401
        inner.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_non_bearer_scheme(self):
        """Non-Bearer auth scheme (e.g. Basic) is rejected."""
        inner = AsyncMock(return_value=MagicMock(status=200))
        wrapped = _auth_middleware(inner, auth_key='secret-token')

        request = MagicMock()
        request.headers = {'Authorization': 'Basic secret-token'}

        response = await wrapped(request)
        assert response.status == 401
        inner.assert_not_called()


# ── _DENIED_ENV_VARS coverage ────────────────────────────────────────────────


class TestDeniedEnvVars:
    def test_contains_critical_system_vars(self):
        """Spot-check that key system vars are protected."""
        for var in ('PATH', 'HOME', 'SHELL', 'PYTHONPATH', 'LD_PRELOAD'):
            assert var in _DENIED_ENV_VARS, f'{var} should be in _DENIED_ENV_VARS'

    def test_is_frozenset(self):
        """_DENIED_ENV_VARS must be immutable (frozenset)."""
        assert isinstance(_DENIED_ENV_VARS, frozenset)
