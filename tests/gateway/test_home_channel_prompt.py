"""Tests for home channel onboarding prompt exemption.

Verifies that the "📬 No home channel is set..." first-message prompt
is correctly skipped for plugin-registered platforms that opt out via
``skip_home_channel_prompt`` (e.g. Myah's web DMs), while remaining
active for messaging platforms (Telegram, Discord, etc.).

Phase 4d (2026-05-04): the original test mirrored the conditional in
``gateway/run.py`` literally and referenced ``Platform.MYAH``. After
Phase 4d the platform was removed from core's ``Platform`` enum, and
the home-channel skip became registry-driven (per
``PlatformEntry.skip_home_channel_prompt``). This rewrite asserts the
NEW invariant: a plugin platform whose registry entry has
``skip_home_channel_prompt=True`` is skipped, regardless of which
identifier it uses, and platforms without that flag still trigger the
prompt.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry


# ── Helpers ────────────────────────────────────────────────────────────────


def _check_should_send_prompt(platform: Platform, history: list) -> bool:
    """Replicate the gateway's home-channel decision logic against the registry.

    Mirrors the check in ``gateway/run.py`` after Phase 4d: built-in
    LOCAL/WEBHOOK platforms always skip; everything else consults the
    registry's ``skip_home_channel_prompt`` flag (False by default).
    """
    if history:
        return False
    if not platform:
        return False
    if platform in (Platform.LOCAL, Platform.WEBHOOK):
        return False
    entry = platform_registry.get(platform.value) if platform else None
    if entry is not None and entry.skip_home_channel_prompt:
        return False
    return True


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def fake_skipping_plugin_platform() -> Iterator[str]:
    """Register a fake plugin platform that opts out of the home-channel prompt."""
    name = "test_skipping_plugin"
    platform_registry.register(
        PlatformEntry(
            name=name,
            label="Fake (Skipping)",
            adapter_factory=lambda cfg: None,  # never instantiated in this test
            check_fn=lambda: True,
            skip_home_channel_prompt=True,
            source="plugin",
        )
    )
    try:
        yield name
    finally:
        platform_registry.unregister(name)


@pytest.fixture
def fake_nonskipping_plugin_platform() -> Iterator[str]:
    """Register a fake plugin platform that does NOT opt out (default)."""
    name = "test_nonskipping_plugin"
    platform_registry.register(
        PlatformEntry(
            name=name,
            label="Fake (Non-skipping)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            skip_home_channel_prompt=False,
            source="plugin",
        )
    )
    try:
        yield name
    finally:
        platform_registry.unregister(name)


# ── Tests ──────────────────────────────────────────────────────────────────


class TestHomeChannelPromptExemption:
    """The home-channel prompt skip is driven by registry capability flags."""

    def test_plugin_with_skip_flag_is_exempt(
        self, fake_skipping_plugin_platform: str
    ) -> None:
        """Plugin platform with skip_home_channel_prompt=True NEVER prompts."""
        platform = Platform(fake_skipping_plugin_platform)
        assert _check_should_send_prompt(platform, history=[]) is False

    def test_plugin_without_skip_flag_does_prompt(
        self, fake_nonskipping_plugin_platform: str
    ) -> None:
        """Plugin platform with skip_home_channel_prompt=False (default) prompts."""
        platform = Platform(fake_nonskipping_plugin_platform)
        assert _check_should_send_prompt(platform, history=[]) is True

    def test_telegram_prompts_when_no_history(self) -> None:
        """Built-in messaging platforms still prompt — registry default."""
        assert _check_should_send_prompt(Platform.TELEGRAM, history=[]) is True

    def test_discord_prompts_when_no_history(self) -> None:
        assert _check_should_send_prompt(Platform.DISCORD, history=[]) is True

    def test_slack_prompts_when_no_history(self) -> None:
        assert _check_should_send_prompt(Platform.SLACK, history=[]) is True

    def test_mattermost_prompts_when_no_history(self) -> None:
        assert _check_should_send_prompt(Platform.MATTERMOST, history=[]) is True

    def test_webhook_never_prompts(self) -> None:
        """Built-in WEBHOOK is hardcoded-skipped (no registry lookup needed)."""
        assert _check_should_send_prompt(Platform.WEBHOOK, history=[]) is False

    def test_local_never_prompts(self) -> None:
        """Built-in LOCAL (CLI) is hardcoded-skipped (no registry lookup needed)."""
        assert _check_should_send_prompt(Platform.LOCAL, history=[]) is False

    def test_subsequent_messages_never_trigger_prompt(self) -> None:
        """Non-empty history short-circuits the check before platform inspection."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert _check_should_send_prompt(Platform.TELEGRAM, history) is False
        # Same for plugin platforms — the history check is first.
        # We don't need to register a fake here; arbitrary unknown platforms
        # also short-circuit on non-empty history.

    def test_runner_uses_registry_skip_flag(
        self, fake_skipping_plugin_platform: str
    ) -> None:
        """Smoke test: gateway/run.py's _hp_entry path resolves through the registry.

        We can't easily construct a full GatewayRunner end-to-end test of
        the message flow without bringing up the full session store; but
        we can verify that the registry lookup the gateway does at the
        prompt site sees our fake plugin entry.
        """
        platform = Platform(fake_skipping_plugin_platform)
        entry = platform_registry.get(platform.value)
        assert entry is not None
        assert entry.skip_home_channel_prompt is True


# ── Backward-compat smoke test for the Myah migration (2026-05-04) ─────────


def test_myah_no_longer_in_core_platform_enum() -> None:
    """Phase 4d invariant: Platform.MYAH was deleted from core."""
    assert not hasattr(Platform, "MYAH"), (
        "Platform.MYAH must NOT exist in the core enum after Phase 4d. "
        "Plugin platforms resolve via Platform('myah') through _missing_."
    )
