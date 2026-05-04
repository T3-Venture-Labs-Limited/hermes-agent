"""Test fixtures for myah-hermes-plugin.

Sets up an isolated HERMES_HOME per test so secrets_tool's reads/writes
to ~/.hermes/.env are scoped to a tempdir.

The hermes-fork's top-level tests/conftest.py provides the same fixture
(plus credential blanking, locale pinning, etc.) but its scope is the
tests/ subtree. When this plugin is run on its own (e.g. `pytest
plugins/myah-hermes-plugin/tests/`), this conftest takes over.

Also registers the Myah platform with the gateway platform registry at
session start so tests that go through ``GatewayRunner._run_agent`` (and
thus ``_get_platform_tools``) can resolve ``"myah"`` without booting the
full plugin discovery machinery.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a per-test tempdir."""
    fake_hermes_home = tmp_path / "hermes_test"
    fake_hermes_home.mkdir()
    (fake_hermes_home / "sessions").mkdir()
    (fake_hermes_home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_hermes_home))
    # Blank any credential vars that could leak into _is_secret_like_name
    # checks via os.environ inspection.
    for name in list(os.environ.keys()):
        upper = name.upper()
        if upper.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
            monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(autouse=True, scope="session")
def _register_myah_platform_for_tests():
    """Register the Myah platform with the gateway registry once per session.

    Phase 4d (2026-05-04): the platform is registered at runtime by the
    plugin's ``register(ctx)`` callback, but tests don't run the full
    plugin discovery cycle. Registering here makes the registry-aware
    fall-through paths in core (e.g. ``_get_platform_tools``,
    ``_is_user_authorized``) see Myah just like they would in production.
    """
    from gateway.platform_registry import PlatformEntry, platform_registry

    platform_registry.register(
        PlatformEntry(
            name="myah",
            label="🌐 Myah",
            adapter_factory=lambda cfg: None,  # tests construct adapters directly
            check_fn=lambda: True,
            allowed_users_env="MYAH_ALLOWED_USERS",
            allow_all_env="MYAH_ALLOW_ALL_USERS",
            default_toolset="hermes-myah",
            skip_user_authorization=True,
            skip_home_channel_prompt=True,
            connect_last=True,
            source="plugin",
        )
    )
    yield
    platform_registry.unregister("myah")
