"""Tests for config v17 → v18 migration that adds title_generation and follow_up_generation."""
import pytest
from hermes_cli.config import DEFAULT_CONFIG, load_config


def test_default_config_declares_title_generation():
    aux = DEFAULT_CONFIG.get('auxiliary', {})
    assert 'title_generation' in aux, 'title_generation not declared in DEFAULT_CONFIG.auxiliary'
    task = aux['title_generation']
    assert task['provider'] == 'openrouter'
    assert task['model'] == 'google/gemini-2.5-flash'
    assert task['timeout'] == 15
    assert task['base_url'] == ''
    assert task['api_key'] == ''


def test_default_config_declares_follow_up_generation():
    aux = DEFAULT_CONFIG.get('auxiliary', {})
    assert 'follow_up_generation' in aux
    task = aux['follow_up_generation']
    assert task['provider'] == 'openrouter'
    assert task['model'] == 'google/gemini-2.5-flash'
    assert task['timeout'] == 30


def test_config_version_is_18():
    assert DEFAULT_CONFIG['_config_version'] == 18, (
        'Bump _config_version to 18 when adding title_generation and follow_up_generation '
        'to ensure existing users get the new defaults via migration.'
    )


def test_load_config_returns_new_aux_tasks_for_existing_user(tmp_path, monkeypatch):
    """Simulate an existing user on v17; load_config should surface v18 defaults via merge."""
    import yaml

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    existing = {
        '_config_version': 17,
        'model': 'anthropic/claude-opus-4.6',
        'auxiliary': {
            'vision': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 120},
        },
    }
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(existing))
    config = load_config()
    # Merged with defaults, so new tasks appear even in an old config file
    assert 'title_generation' in config['auxiliary']
    assert config['auxiliary']['title_generation']['model'] == 'google/gemini-2.5-flash'
    assert 'follow_up_generation' in config['auxiliary']
