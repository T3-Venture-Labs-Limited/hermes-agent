"""Regression coverage for Myah aux-task defaults in ``DEFAULT_CONFIG``.

Historically this file asserted a v17→v18 migration that hard-pinned
``title_generation`` and ``follow_up_generation`` to
``provider=openrouter, model=google/gemini-2.5-flash``. That design
pre-dated multi-provider support in Myah — it silently routed aux tasks
through OpenRouter even for users who brought their own Anthropic / OpenAI
key, effectively cross-billing them.

Upstream Hermes later added ``title_generation`` natively with
``provider=auto`` (use the user's main provider). During the 2026-04-19
upstream merge we adopted the same policy for our Myah-only
``follow_up_generation`` task. This test now guards that both Myah aux
tasks remain declared with the ``auto`` policy so the hard pin cannot
regress.
"""
import pytest
from hermes_cli.config import DEFAULT_CONFIG, load_config


def test_default_config_declares_title_generation_with_auto_provider():
    aux = DEFAULT_CONFIG.get('auxiliary', {})
    assert 'title_generation' in aux, 'title_generation not declared in DEFAULT_CONFIG.auxiliary'
    task = aux['title_generation']
    # Upstream policy: auto-resolve to the user's main provider so aux
    # tasks inherit whichever UserLLMKeys credentials are in effect.
    assert task['provider'] == 'auto'
    assert task['model'] == ''
    assert task['base_url'] == ''
    assert task['api_key'] == ''


def test_default_config_declares_follow_up_generation_with_auto_provider():
    aux = DEFAULT_CONFIG.get('auxiliary', {})
    assert 'follow_up_generation' in aux, (
        'follow_up_generation is a Myah-only aux task; it must remain '
        'declared in DEFAULT_CONFIG.auxiliary so the /myah/v1/aux router '
        'can resolve it.'
    )
    task = aux['follow_up_generation']
    assert task['provider'] == 'auto'
    assert task['model'] == ''


def test_config_version_matches_upstream():
    # Upstream bumps _config_version when DEFAULT_CONFIG grows new fields.
    # Myah tracks upstream exactly — we do not fork the migration chain.
    # If this fails after an upstream merge, bump the expected value and
    # confirm the new migration block in hermes_cli/config.py is compatible
    # with Myah's defaults.
    assert DEFAULT_CONFIG['_config_version'] == 19


def test_load_config_surfaces_aux_tasks_for_existing_user(tmp_path, monkeypatch):
    """An existing user on an older config version gets the new aux tasks
    surfaced automatically via the defaults-merge path in ``load_config``."""
    import yaml

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    existing = {
        '_config_version': 17,
        'model': 'anthropic/claude-opus-4.6',
        'auxiliary': {
            'vision': {
                'provider': 'auto', 'model': '', 'base_url': '',
                'api_key': '', 'timeout': 120,
            },
        },
    }
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(existing))
    config = load_config()
    # Merged with defaults, so new tasks appear even in an old config file
    assert 'title_generation' in config['auxiliary']
    assert config['auxiliary']['title_generation']['provider'] == 'auto'
    assert 'follow_up_generation' in config['auxiliary']
    assert config['auxiliary']['follow_up_generation']['provider'] == 'auto'
