"""Regression tests for ``GatewayRunner._is_user_authorized`` on plugin platforms.

PR pair 1 of Phase 4d.5 originally dropped the ``_registry_entry`` lookup
along with the (correctly removed) ``skip_user_authorization`` early-return,
but the same name is still consumed by the upstream-supported
``allow_all_env`` and ``allowed_users_env`` fall-through paths further down
in the function. Without the lookup, every plugin platform (including Myah)
hit ``NameError: name '_registry_entry' is not defined`` for any chat
message that reached the user-auth path.

These tests exercise the plugin path end-to-end so a future regression
surfaces immediately.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    """Strip any auth env vars that might leak from the host."""
    for key in (
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "TESTPLATFORM_ALLOWED_USERS",
        "TESTPLATFORM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_runner(platform: Platform):
    """Build a minimally instantiated GatewayRunner suitable for ``_is_user_authorized``."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})
    runner.adapters = {platform: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    return runner


@pytest.fixture
def registered_test_platform():
    """Register a temporary plugin platform in the registry for the duration of the test."""
    entry = PlatformEntry(
        name="testplatform",
        label="Test Platform",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        allowed_users_env="TESTPLATFORM_ALLOWED_USERS",
        allow_all_env="TESTPLATFORM_ALLOW_ALL_USERS",
        source="plugin",
    )
    platform_registry.register(entry)
    try:
        yield entry
    finally:
        platform_registry.unregister("testplatform")


def _source(user_id: str = "user-1") -> SessionSource:
    return SessionSource(
        platform=Platform("testplatform"),
        user_id=user_id,
        chat_id="chat-1",
        user_name="tester",
        chat_type="dm",
    )


def test_plugin_platform_allow_all_env_authorizes(monkeypatch, registered_test_platform):
    """Setting the plugin's allow_all_env env var to a truthy value authorizes any user.

    This is the path that crashed with NameError on _registry_entry before the
    fix; if this test passes the lookup is in place.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TESTPLATFORM_ALLOW_ALL_USERS", "true")

    runner = _make_runner(Platform("testplatform"))

    assert runner._is_user_authorized(_source()) is True


def test_plugin_platform_no_allowlists_rejects(monkeypatch, registered_test_platform):
    """With no allow_all flag and no allowlists configured, the user is rejected.

    Crucially, the function must reach this rejection path *without* raising
    NameError — the registry lookup defines _registry_entry as ``None`` would
    only be safe if the platform isn't registered; here it IS registered, so
    we exercise the ``_registry_entry is not None`` branch.
    """
    _clear_auth_env(monkeypatch)

    runner = _make_runner(Platform("testplatform"))

    assert runner._is_user_authorized(_source()) is False


def test_plugin_platform_allowed_users_env_authorizes(
    monkeypatch, registered_test_platform
):
    """The plugin's allowed_users_env env var feeds the platform-specific allowlist."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TESTPLATFORM_ALLOWED_USERS", "user-1,user-2")

    runner = _make_runner(Platform("testplatform"))

    assert runner._is_user_authorized(_source(user_id="user-1")) is True
    assert runner._is_user_authorized(_source(user_id="user-3")) is False
