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


# ── SOUL.md endpoints ───────────────────────────────────────────────────────

class TestSoulEndpoints:
    @pytest.fixture
    def temp_hermes_home(self, tmp_path):
        (tmp_path / 'SOUL.md').write_text('You are a helpful assistant.')
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            yield tmp_path

    @pytest.mark.asyncio
    async def test_get_soul(self, temp_hermes_home):
        request = MagicMock()
        response = await handle_get_soul(request)
        data = json.loads(response.body)
        assert data['content'] == 'You are a helpful assistant.'

    @pytest.mark.asyncio
    async def test_get_soul_missing(self, tmp_path):
        with patch('gateway.platforms.myah_management._hermes_home', return_value=tmp_path):
            request = MagicMock()
            response = await handle_get_soul(request)
            data = json.loads(response.body)
            assert data['content'] == ''

    @pytest.mark.asyncio
    async def test_put_soul(self, temp_hermes_home):
        request = MagicMock()
        request.json = AsyncMock(return_value={'content': 'New soul content'})
        response = await handle_put_soul(request)
        data = json.loads(response.body)
        assert data['content'] == 'New soul content'
        # Verify file was updated
        assert (temp_hermes_home / 'SOUL.md').read_text() == 'New soul content'

    @pytest.mark.asyncio
    async def test_put_soul_empty_rejected(self, temp_hermes_home):
        request = MagicMock()
        request.json = AsyncMock(return_value={'content': '   '})
        response = await handle_put_soul(request)
        assert response.status == 422


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
