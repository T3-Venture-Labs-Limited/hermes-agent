"""Tests for env var capture during skill_manage(action='create')."""

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.skill_manager_tool import skill_manage


@pytest.fixture
def skill_dir(tmp_path, monkeypatch):
    """Redirect SKILLS_DIR to a temp dir and isolate from real ~/.hermes."""
    skills = tmp_path / 'skills'
    skills.mkdir()
    monkeypatch.setattr('tools.skill_manager_tool.SKILLS_DIR', skills)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    return skills


SKILL_WITH_ENV = textwrap.dedent("""\
    ---
    name: test-skill
    description: A test skill
    prerequisites:
      env_vars: [TEST_API_KEY]
    ---
    # Test Skill
    Use TEST_API_KEY to do things.
""")

SKILL_WITHOUT_ENV = textwrap.dedent("""\
    ---
    name: simple-skill
    description: A simple skill
    ---
    # Simple Skill
    No env vars needed.
""")


class TestSkillCreateEnvCapture:
    """Env var capture is triggered when a created skill declares prerequisites."""

    def test_create_with_missing_env_triggers_callback(self, skill_dir, monkeypatch):
        """When an env var is missing, _capture_required_environment_variables is called."""
        mock_capture = MagicMock(return_value={
            'missing_names': [],
            'setup_skipped': False,
            'gateway_setup_hint': None,
        })
        monkeypatch.setattr(
            'tools.skill_manager_tool._capture_required_environment_variables',
            mock_capture,
        )
        monkeypatch.setattr('tools.skill_manager_tool._ENV_CAPTURE_AVAILABLE', True)
        monkeypatch.delenv('TEST_API_KEY', raising=False)

        with patch('agent.skill_utils.get_all_skills_dirs', return_value=[skill_dir]):
            result = json.loads(skill_manage(action='create', name='test-skill', content=SKILL_WITH_ENV))

        assert result['success'] is True
        assert result.get('setup_complete') is True
        mock_capture.assert_called_once()
        call_args = mock_capture.call_args
        assert call_args[0][0] == 'test-skill'  # skill_name
        assert any(e['name'] == 'TEST_API_KEY' for e in call_args[0][1])  # missing_entries

    def test_create_with_present_env_skips_callback(self, skill_dir, monkeypatch):
        """When env vars are already set, capture is not called."""
        mock_capture = MagicMock()
        monkeypatch.setattr(
            'tools.skill_manager_tool._capture_required_environment_variables',
            mock_capture,
        )
        monkeypatch.setattr('tools.skill_manager_tool._ENV_CAPTURE_AVAILABLE', True)
        # Patch _is_env_var_persisted to return True (env var is considered present)
        monkeypatch.setattr(
            'tools.skill_manager_tool._is_env_var_persisted',
            lambda name: True,
        )
        monkeypatch.setenv('TEST_API_KEY', 'sk-test-123')

        with patch('agent.skill_utils.get_all_skills_dirs', return_value=[skill_dir]):
            result = json.loads(skill_manage(action='create', name='test-skill', content=SKILL_WITH_ENV))

        assert result['success'] is True
        mock_capture.assert_not_called()

    def test_create_without_env_vars_skips_capture(self, skill_dir, monkeypatch):
        """Skills with no env var requirements skip the capture entirely."""
        mock_capture = MagicMock()
        monkeypatch.setattr(
            'tools.skill_manager_tool._capture_required_environment_variables',
            mock_capture,
        )
        monkeypatch.setattr('tools.skill_manager_tool._ENV_CAPTURE_AVAILABLE', True)

        with patch('agent.skill_utils.get_all_skills_dirs', return_value=[skill_dir]):
            result = json.loads(skill_manage(action='create', name='simple-skill', content=SKILL_WITHOUT_ENV))

        assert result['success'] is True
        mock_capture.assert_not_called()
        assert 'setup_needed' not in result

    def test_create_with_callback_failure_still_succeeds(self, skill_dir, monkeypatch):
        """If env capture raises, skill creation still succeeds (best-effort)."""
        monkeypatch.setattr(
            'tools.skill_manager_tool._capture_required_environment_variables',
            MagicMock(side_effect=RuntimeError('callback exploded')),
        )
        monkeypatch.setattr('tools.skill_manager_tool._ENV_CAPTURE_AVAILABLE', True)
        monkeypatch.delenv('TEST_API_KEY', raising=False)

        with patch('agent.skill_utils.get_all_skills_dirs', return_value=[skill_dir]):
            result = json.loads(skill_manage(action='create', name='test-skill', content=SKILL_WITH_ENV))

        assert result['success'] is True  # create succeeds despite callback error

    def test_create_reports_still_missing_vars(self, skill_dir, monkeypatch):
        """When user skips secret entry, result includes setup_needed."""
        monkeypatch.setattr(
            'tools.skill_manager_tool._capture_required_environment_variables',
            MagicMock(return_value={
                'missing_names': ['TEST_API_KEY'],
                'setup_skipped': True,
                'gateway_setup_hint': None,
            }),
        )
        monkeypatch.setattr('tools.skill_manager_tool._ENV_CAPTURE_AVAILABLE', True)
        monkeypatch.delenv('TEST_API_KEY', raising=False)

        with patch('agent.skill_utils.get_all_skills_dirs', return_value=[skill_dir]):
            result = json.loads(skill_manage(action='create', name='test-skill', content=SKILL_WITH_ENV))

        assert result['success'] is True
        assert result['setup_needed'] is True
        assert 'TEST_API_KEY' in result['missing_env_vars']
