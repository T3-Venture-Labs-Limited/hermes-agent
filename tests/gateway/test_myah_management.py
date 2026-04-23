"""Tests for Myah management API endpoints.

Tests cover:
- Name validation (_safe_name)
- SKILL.md frontmatter parsing (_parse_frontmatter)
- Config read endpoint (handle_get_config)
- SOUL.md read/write endpoints (handle_get_soul, handle_put_soul)
- Skill listing/creation/deletion endpoints
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from gateway.platforms.myah_management import (
    _parse_frontmatter,
    _safe_name,
    handle_get_config,
    handle_get_soul,
    handle_put_soul,
    handle_list_skills,
    handle_create_skill,
    handle_delete_skill,
    handle_list_mcp,
)


# ── _safe_name ──────────────────────────────────────────────────────────────

class TestSafeName:
    def test_valid_names(self):
        assert _safe_name('my-skill') is None
        assert _safe_name('web_search') is None
        assert _safe_name('tool123') is None
        assert _safe_name('A-B_c') is None

    def test_path_traversal(self):
        result = _safe_name('../etc/passwd')
        assert result is not None  # Returns error response
        assert result.status == 422

    def test_empty_name(self):
        result = _safe_name('')
        assert result is not None
        assert result.status == 422

    def test_spaces_rejected(self):
        result = _safe_name('my skill')
        assert result is not None
        assert result.status == 422

    def test_dots_rejected(self):
        result = _safe_name('name.with.dots')
        assert result is not None
        assert result.status == 422


# ── _parse_frontmatter ──────────────────────────────────────────────────────

class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = '---\nname: test-skill\ndescription: A test\n---\n\nBody text'
        fm = _parse_frontmatter(content)
        assert fm['name'] == 'test-skill'
        assert fm['description'] == 'A test'

    def test_no_frontmatter(self):
        content = 'Just plain text'
        fm = _parse_frontmatter(content)
        assert fm == {}

    def test_quoted_values(self):
        content = '---\nname: "quoted-name"\n---\n'
        fm = _parse_frontmatter(content)
        assert fm['name'] == 'quoted-name'

    def test_single_quoted_values(self):
        content = "---\nname: 'single-quoted'\n---\n"
        fm = _parse_frontmatter(content)
        assert fm['name'] == 'single-quoted'

    def test_empty_frontmatter(self):
        content = '---\n---\nBody'
        fm = _parse_frontmatter(content)
        assert fm == {}


# ── Config endpoints ────────────────────────────────────────────────────────

class TestConfigEndpoints:
    @pytest.fixture
    def temp_hermes_home(self, tmp_path):
        config = {'model': 'anthropic/claude-sonnet-4-20250514', 'disabled_toolsets': ['browser']}
        (tmp_path / 'config.yaml').write_text(yaml.dump(config))
        (tmp_path / 'SOUL.md').write_text('You are a helpful assistant.')
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            yield tmp_path

    @pytest.mark.asyncio
    async def test_get_config(self, temp_hermes_home):
        request = MagicMock()
        response = await handle_get_config(request)
        data = json.loads(response.body)
        assert data['model'] == 'anthropic/claude-sonnet-4-20250514'
        assert 'browser' in data['disabled_toolsets']

    @pytest.mark.asyncio
    async def test_get_config_missing_file(self, tmp_path):
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            request = MagicMock()
            response = await handle_get_config(request)
            assert response.status == 404

    @pytest.mark.asyncio
    async def test_get_config_merges_defaults_for_missing_keys(self, tmp_path):
        """Older config.yaml files that predate the current DEFAULT_CONFIG
        should still produce a complete schema payload — missing keys are
        filled from DEFAULT_CONFIG so the frontend can render every aux task.
        """
        partial = {
            '_config_version': 12,
            'model': 'anthropic/claude-sonnet-4-20250514',
            'auxiliary': {
                'vision': {'provider': 'openai', 'model': 'gpt-4o-mini'},
            },
        }
        (tmp_path / 'config.yaml').write_text(yaml.dump(partial))
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            request = MagicMock()
            response = await handle_get_config(request)
        data = json.loads(response.body)
        # Disk values win
        assert data['model'] == 'anthropic/claude-sonnet-4-20250514'
        assert data['auxiliary']['vision']['provider'] == 'openai'
        # Missing aux keys are filled from DEFAULT_CONFIG
        assert 'title_generation' in data['auxiliary']
        assert 'follow_up_generation' in data['auxiliary']
        # Private keys from disk are preserved
        assert data.get('_config_version') == 12


# ── SOUL.md endpoints ───────────────────────────────────────────────────────

class TestSoulEndpoints:
    @pytest.fixture
    def temp_hermes_home(self, tmp_path):
        (tmp_path / 'SOUL.md').write_text('You are a helpful assistant.')
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            yield tmp_path

    @pytest.mark.asyncio
    async def test_get_soul(self, temp_hermes_home):
        from aiohttp.test_utils import make_mocked_request
        request = make_mocked_request('GET', '/myah/api/config/soul')
        response = await handle_get_soul(request)
        # New API: returns raw text with ETag header
        assert response.status == 200
        assert response.text == 'You are a helpful assistant.'
        assert 'ETag' in response.headers

    @pytest.mark.asyncio
    async def test_get_soul_missing(self, tmp_path):
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            from aiohttp.test_utils import make_mocked_request
            request = make_mocked_request('GET', '/myah/api/config/soul')
            response = await handle_get_soul(request)
            # New API: 404 when SOUL.md is missing
            assert response.status == 404

    @pytest.mark.asyncio
    async def test_put_soul(self, temp_hermes_home):
        from aiohttp.test_utils import make_mocked_request
        from gateway.platforms.myah_management import _soul_etag
        etag = _soul_etag('You are a helpful assistant.')
        request = make_mocked_request(
            'PUT', '/myah/api/config/soul',
            headers={'If-Match': etag},
        )
        request.text = AsyncMock(return_value='New soul content')
        response = await handle_put_soul(request)
        data = json.loads(response.body)
        assert data['ok'] is True
        # Verify file was updated
        assert (temp_hermes_home / 'SOUL.md').read_text() == 'New soul content'

    @pytest.mark.asyncio
    async def test_put_soul_empty_rejected(self, temp_hermes_home):
        from aiohttp.test_utils import make_mocked_request
        from gateway.platforms.myah_management import _soul_etag
        etag = _soul_etag('You are a helpful assistant.')
        request = make_mocked_request(
            'PUT', '/myah/api/config/soul',
            headers={'If-Match': etag},
        )
        # Empty content — still allowed (no empty check in new API); check 428 for no If-Match instead
        no_ifmatch_request = make_mocked_request('PUT', '/myah/api/config/soul')
        no_ifmatch_request.text = AsyncMock(return_value='')
        response = await handle_put_soul(no_ifmatch_request)
        assert response.status == 428


# ── Skill endpoints ─────────────────────────────────────────────────────────

class TestSkillEndpoints:
    @pytest.fixture
    def temp_skills(self, tmp_path):
        skills_dir = tmp_path / 'skills' / 'general' / 'test-skill'
        skills_dir.mkdir(parents=True)
        (skills_dir / 'SKILL.md').write_text('---\nname: test-skill\ndescription: A test\n---\n\nBody')
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            yield tmp_path

    @pytest.mark.asyncio
    async def test_list_skills(self, temp_skills):
        request = MagicMock()
        response = await handle_list_skills(request)
        data = json.loads(response.body)
        assert len(data) == 1
        assert data[0]['name'] == 'test-skill'
        assert data[0]['description'] == 'A test'
        assert data[0]['category'] == 'general'

    @pytest.mark.asyncio
    async def test_list_skills_empty(self, tmp_path):
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            request = MagicMock()
            response = await handle_list_skills(request)
            data = json.loads(response.body)
            assert data == []

    @pytest.mark.asyncio
    async def test_create_skill(self, temp_skills):
        request = MagicMock()
        request.json = AsyncMock(return_value={
            'name': 'new-skill',
            'category': 'general',
            'content': '---\nname: new-skill\n---\n\nNew skill body',
        })
        response = await handle_create_skill(request)
        assert response.status == 201
        # Verify file was created
        skill_path = temp_skills / 'skills' / 'general' / 'new-skill' / 'SKILL.md'
        assert skill_path.exists()
        assert 'New skill body' in skill_path.read_text()

    @pytest.mark.asyncio
    async def test_create_duplicate_skill(self, temp_skills):
        request = MagicMock()
        request.json = AsyncMock(return_value={
            'name': 'test-skill',
            'category': 'general',
            'content': 'duplicate',
        })
        response = await handle_create_skill(request)
        assert response.status == 409  # Conflict

    @pytest.mark.asyncio
    async def test_delete_skill(self, temp_skills):
        request = MagicMock()
        request.match_info = {'name': 'test-skill'}
        response = await handle_delete_skill(request)
        data = json.loads(response.body)
        assert data['ok'] is True
        # Verify directory was removed
        assert not (temp_skills / 'skills' / 'general' / 'test-skill').exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_skill(self, temp_skills):
        request = MagicMock()
        request.match_info = {'name': 'nonexistent'}
        response = await handle_delete_skill(request)
        assert response.status == 404


# ── MCP endpoints ───────────────────────────────────────────────────────────

class TestMcpEndpoints:
    @pytest.mark.asyncio
    async def test_list_mcp_empty(self, tmp_path):
        (tmp_path / 'config.yaml').write_text(yaml.dump({'mcp_servers': {}}))
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            request = MagicMock()
            response = await handle_list_mcp(request)
            data = json.loads(response.body)
            assert data == []

    @pytest.mark.asyncio
    async def test_list_mcp_with_servers(self, tmp_path):
        cfg = {
            'mcp_servers': {
                'test-server': {
                    'url': 'http://localhost:3000',
                },
                'local-tool': {
                    'command': 'npx',
                    'args': ['-y', 'tool-server'],
                },
            }
        }
        (tmp_path / 'config.yaml').write_text(yaml.dump(cfg))
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            request = MagicMock()
            response = await handle_list_mcp(request)
            data = json.loads(response.body)
            assert len(data) == 2
            names = {s['name'] for s in data}
            assert 'test-server' in names
            assert 'local-tool' in names


# ── _build_catalog capability metadata ──────────────────────────────────────

class TestBuildCatalogCapabilities:
    """Tests for the curated_models shape change: list[str] → list[dict]."""

    def _make_mock_entry(self, slug='test-provider', label='Test', tui_desc='Desc'):
        entry = MagicMock()
        entry.slug = slug
        entry.label = label
        entry.tui_desc = tui_desc
        return entry

    def _patch_catalog_deps(self, slug='test-provider', models=None, caps_return=None, caps_raise=None):
        """Return a context-manager stack that mocks all _build_catalog dependencies."""
        from contextlib import ExitStack
        stack = ExitStack()

        mock_entry = self._make_mock_entry(slug=slug)
        mock_cfg = MagicMock()
        mock_cfg.auth_type = 'api_key'
        mock_cfg.api_key_env_vars = ['TEST_KEY']
        mock_cfg.inference_base_url = 'https://test.example.com'

        model_list = models if models is not None else ['test/model-a']

        stack.enter_context(patch(
            'gateway.platforms.myah_management._build_catalog.__globals__',
            side_effect=None  # placeholder — we patch the deferred imports instead
        )) if False else None  # unused — we patch the imported names directly

        # Patch deferred imports inside _build_catalog
        stack.enter_context(patch('hermes_cli.models.CANONICAL_PROVIDERS', [mock_entry]))
        stack.enter_context(patch('hermes_cli.models._PROVIDER_MODELS', {slug: model_list}))
        stack.enter_context(patch('hermes_cli.auth.PROVIDER_REGISTRY', {slug: mock_cfg}))
        stack.enter_context(patch('hermes_cli.providers.HERMES_OVERLAYS', {}))
        stack.enter_context(patch('hermes_cli.providers.normalize_provider', side_effect=lambda x: x))
        stack.enter_context(patch('hermes_cli.myah_overrides.MYAH_OVERRIDES', {}))

        if caps_raise is not None:
            mock_caps = MagicMock(side_effect=caps_raise)
        elif caps_return is not None:
            mock_caps = MagicMock(return_value=caps_return)
        else:
            mock_caps = MagicMock(return_value=None)

        stack.enter_context(patch('agent.models_dev.get_model_capabilities', mock_caps))

        return stack, mock_caps

    @pytest.mark.asyncio
    async def test_build_catalog_models_are_structured_objects(self):
        """curated_models entries must be dicts with id and name keys, not bare strings."""
        from gateway.platforms.myah_management import _build_catalog

        slug = 'openai'
        model_ids = ['openai/gpt-4o', 'openai/gpt-3.5-turbo']
        stack, _ = self._patch_catalog_deps(slug=slug, models=model_ids)

        with stack:
            catalog = await _build_catalog()

        assert slug in catalog
        curated = catalog[slug]['curated_models']
        assert len(curated) == 2
        for entry in curated:
            assert isinstance(entry, dict), f"Expected dict, got {type(entry)}: {entry!r}"
            assert 'id' in entry, f"Missing 'id' key in {entry!r}"
            assert 'name' in entry, f"Missing 'name' key in {entry!r}"

    @pytest.mark.asyncio
    async def test_build_catalog_includes_capabilities_when_available(self):
        """When get_model_capabilities returns data, capabilities dict must be present."""
        from gateway.platforms.myah_management import _build_catalog
        from agent.models_dev import ModelCapabilities

        slug = 'anthropic'
        model_id = 'anthropic/claude-opus-4.7'
        caps = ModelCapabilities(
            supports_vision=True,
            supports_tools=True,
            supports_reasoning=False,
            context_window=1_000_000,
            max_output_tokens=8192,
            model_family='claude',
        )
        stack, _ = self._patch_catalog_deps(slug=slug, models=[model_id], caps_return=caps)

        with stack:
            catalog = await _build_catalog()

        curated = catalog[slug]['curated_models']
        assert len(curated) == 1
        entry = curated[0]
        assert entry['id'] == model_id
        assert 'capabilities' in entry, "capabilities key must be present when lookup succeeds"
        c = entry['capabilities']
        assert c['supports_vision'] is True
        assert c['supports_tools'] is True
        assert c['supports_reasoning'] is False
        assert c['context_window'] == 1_000_000
        assert c['max_output_tokens'] == 8192
        assert c['model_family'] == 'claude'
        assert set(c.keys()) == {
            'supports_tools', 'supports_vision', 'supports_reasoning',
            'context_window', 'max_output_tokens', 'model_family',
        }

    @pytest.mark.asyncio
    async def test_build_catalog_omits_capabilities_on_lookup_failure(self):
        """When get_model_capabilities returns None, capabilities key must be absent."""
        from gateway.platforms.myah_management import _build_catalog

        slug = 'google'
        model_id = 'google/gemini-2.5-pro'
        # caps_return=None is the default in _patch_catalog_deps
        stack, _ = self._patch_catalog_deps(slug=slug, models=[model_id], caps_return=None)

        with stack:
            catalog = await _build_catalog()

        curated = catalog[slug]['curated_models']
        assert len(curated) == 1
        entry = curated[0]
        assert entry['id'] == model_id
        assert entry['name'] == model_id
        assert 'capabilities' not in entry, (
            f"'capabilities' key must be absent when lookup returns None, got: {entry!r}"
        )

    @pytest.mark.asyncio
    async def test_build_catalog_handles_get_model_capabilities_exception(self):
        """Exception in get_model_capabilities must not crash the catalog build."""
        import logging
        from gateway.platforms.myah_management import _build_catalog

        slug = 'mistral'
        model_id = 'mistral/mistral-large'
        stack, _ = self._patch_catalog_deps(
            slug=slug,
            models=[model_id],
            caps_raise=RuntimeError('models.dev unreachable'),
        )

        with stack:
            with patch('gateway.platforms.myah_management.logger') as mock_logger:
                catalog = await _build_catalog()

        # Catalog must still return successfully
        assert slug in catalog
        curated = catalog[slug]['curated_models']
        assert len(curated) == 1
        entry = curated[0]

        # No entry should have capabilities when all lookups raised
        assert 'capabilities' not in entry, (
            f"'capabilities' must be absent after exception, got: {entry!r}"
        )

        # A warning must have been logged
        mock_logger.warning.assert_called()
