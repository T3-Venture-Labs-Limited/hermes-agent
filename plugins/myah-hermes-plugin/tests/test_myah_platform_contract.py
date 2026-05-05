"""Plugin contract tests for the Myah platform.

Phase 4d (2026-05-04) moved the Myah platform adapter out of upstream
Hermes core and into ``myah-hermes-plugin``. The previous ADDING_A_PLATFORM
contract — which asserted that Myah was hardcoded into the core ``Platform``
enum, ``PLATFORM_HINTS``, ``cron/scheduler.py``, ``_create_adapter`` etc. —
no longer applies; those assertions would *fail by design* in the new world.

These tests instead assert the new contract: when the plugin's
``register(ctx)`` is invoked, it correctly registers a platform named
``"myah"`` with the expected capability fields on its ``PlatformEntry``,
and the adapter is importable from the plugin namespace.
"""

from __future__ import annotations

from typing import Any

import pytest


# ── Plugin entry point + register() invocation ────────────────────────────


class _RecordingContext:
    """Minimal PluginContext stand-in capturing every register_* call."""

    def __init__(self) -> None:
        self.tools: list[dict] = []
        self.platforms: list[dict] = []
        self.hooks: list[tuple[str, Any]] = []

    def register_tool(self, **kwargs: Any) -> None:
        self.tools.append(kwargs)

    def register_platform(self, **kwargs: Any) -> None:
        self.platforms.append(kwargs)

    def register_hook(self, name: str, callback: Any) -> None:
        self.hooks.append((name, callback))


@pytest.fixture
def captured_registration() -> _RecordingContext:
    """Run the plugin's register(ctx) and capture what it registered."""
    from myah_hermes_plugin.myah_platform import register

    ctx = _RecordingContext()
    register(ctx)
    return ctx


# ── Platform registration contract (replaces the old ADDING_A_PLATFORM
# ── checklist for Myah) ────────────────────────────────────────────────


class TestPluginRegistersMyahPlatform:
    """Phase 4d invariants on ``ctx.register_platform`` for Myah."""

    def test_platform_registered(self, captured_registration: _RecordingContext) -> None:
        """register() must call ctx.register_platform exactly once for Myah."""
        myah_entries = [p for p in captured_registration.platforms if p.get("name") == "myah"]
        assert len(myah_entries) == 1, (
            "myah-hermes-plugin's register() must call ctx.register_platform "
            "exactly once for the Myah platform."
        )

    def test_platform_label(self, captured_registration: _RecordingContext) -> None:
        entry = next(p for p in captured_registration.platforms if p.get("name") == "myah")
        assert entry.get("label") == "🌐 Myah"

    def test_allowed_users_env(self, captured_registration: _RecordingContext) -> None:
        """Replaces the hardcoded MYAH_ALLOWED_USERS entry in core's platform_env_map."""
        entry = next(p for p in captured_registration.platforms if p.get("name") == "myah")
        assert entry.get("allowed_users_env") == "MYAH_ALLOWED_USERS"

    def test_allow_all_env(self, captured_registration: _RecordingContext) -> None:
        """Replaces the hardcoded MYAH_ALLOW_ALL_USERS entry in core's platform_allow_all_map."""
        entry = next(p for p in captured_registration.platforms if p.get("name") == "myah")
        assert entry.get("allow_all_env") == "MYAH_ALLOW_ALL_USERS"

    def test_platform_hint_present(self, captured_registration: _RecordingContext) -> None:
        """Replaces the 'myah' entry deleted from PLATFORM_HINTS in prompt_builder."""
        entry = next(p for p in captured_registration.platforms if p.get("name") == "myah")
        hint = entry.get("platform_hint")
        assert isinstance(hint, str) and "Myah" in hint

    def test_check_fn_returns_bool(self, captured_registration: _RecordingContext) -> None:
        """check_fn must be callable and return a bool (aiohttp availability)."""
        entry = next(p for p in captured_registration.platforms if p.get("name") == "myah")
        check_fn = entry.get("check_fn")
        assert callable(check_fn)
        result = check_fn()
        assert isinstance(result, bool)

    def test_adapter_factory_constructible(
        self, captured_registration: _RecordingContext
    ) -> None:
        """adapter_factory must produce a MyahAdapter instance from a PlatformConfig."""
        from gateway.config import PlatformConfig

        entry = next(p for p in captured_registration.platforms if p.get("name") == "myah")
        factory = entry.get("adapter_factory")
        assert callable(factory)

        # The adapter registers a pre-setup hook on api_server during
        # __init__; patch it out to keep the unit test side-effect free.
        from unittest.mock import patch

        with patch("gateway.platforms.api_server.register_pre_setup_hook"):
            adapter = factory(PlatformConfig(enabled=True, extra={}))
        from myah_hermes_plugin.myah_platform.adapter import MyahAdapter

        assert isinstance(adapter, MyahAdapter)


# ── Importability contract ─────────────────────────────────────────────────


class TestImportability:
    """The adapter and runtime_admin must be importable from the plugin namespace."""

    def test_adapter_importable_from_plugin(self) -> None:
        from myah_hermes_plugin.myah_platform.adapter import MyahAdapter

        assert MyahAdapter is not None

    def test_runtime_admin_importable_from_plugin(self) -> None:
        from myah_hermes_plugin.myah_platform.runtime_admin import (
            register_runtime_admin_routes,
        )

        assert callable(register_runtime_admin_routes)

    def test_adapter_NOT_importable_from_old_core_path(self) -> None:
        """gateway.platforms.myah must no longer exist in core."""
        with pytest.raises(ImportError):
            __import__("gateway.platforms.myah", fromlist=["MyahAdapter"])


# ── Toolset preset (Item 7 from old contract — still applies) ─────────────


def test_hermes_myah_toolset_preset_exists() -> None:
    """``toolsets.py::TOOLSETS`` registers the ``hermes-myah`` preset.

    This entry stays in core because the toolset preset is consumed by
    other Hermes machinery (CLI tools menu, gateway tools config) before
    plugins have registered. Phase 4f will revisit whether the preset
    should also move to the plugin.
    """
    from toolsets import TOOLSETS

    assert "hermes-myah" in TOOLSETS, (
        "TOOLSETS must include a 'hermes-myah' preset for the Myah platform"
    )


# ── Cronjob deliver schema mention (description-only, still applies) ──────


def test_myah_in_cronjob_deliver_schema() -> None:
    """``cronjob_tools.py`` deliver-parameter description still mentions myah.

    The schema description teaches the model how to address Myah chats
    (``myah:<chat_id>:<thread_id>``); even with the platform extracted
    to a plugin, this guidance text remains in core because it's part
    of the generic deliver-target syntax.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[3] / "tools" / "cronjob_tools.py").read_text(
        encoding="utf-8"
    )
    assert "myah" in src.lower(), "cronjob deliver schema must mention myah"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
