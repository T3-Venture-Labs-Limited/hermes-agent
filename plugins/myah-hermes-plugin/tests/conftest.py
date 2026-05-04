"""Test fixtures for myah-hermes-plugin.

Sets up an isolated HERMES_HOME per test so secrets_tool's reads/writes
to ~/.hermes/.env are scoped to a tempdir.

The hermes-fork's top-level tests/conftest.py provides the same fixture
(plus credential blanking, locale pinning, etc.) but its scope is the
tests/ subtree. When this plugin is run on its own (e.g. `pytest
plugins/myah-hermes-plugin/tests/`), this conftest takes over.
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
