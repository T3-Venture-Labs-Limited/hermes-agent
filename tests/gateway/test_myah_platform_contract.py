"""ADDING_A_PLATFORM compliance regression test for the Myah platform.

These tests guard against silent regressions when someone refactors a
Hermes core file and accidentally drops a Myah registration. Each test
maps 1:1 to a checklist item from gateway/platforms/ADDING_A_PLATFORM.md.

If a test fails, the platform integration has lost a registration and
must be restored before merge.
"""

from pathlib import Path

import pytest


HERMES_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    """Read a file under hermes-agent/ relative path as text."""
    return (HERMES_ROOT / rel).read_text(encoding="utf-8")


# ── Item 2: Platform enum ──────────────────────────────────────────────────


def test_platform_myah_in_enum():
    """`Platform.MYAH` exists in `gateway.config.Platform`."""
    from gateway.config import Platform

    assert hasattr(Platform, "MYAH")
    assert Platform.MYAH.value == "myah"


# ── Item 8: cronjob deliver schema description ─────────────────────────────


def test_myah_in_cronjob_deliver_schema():
    """`cronjob_tools.py` deliver-parameter description mentions myah."""
    src = _read("tools/cronjob_tools.py")
    # Description-only field — match case-insensitively to be robust
    # to future re-formatting that capitalises "Myah".
    assert "myah" in src.lower(), "cronjob deliver schema must mention myah"


# ── Item 11 (auto-discovery): channel directory must NOT skip myah ─────────


def test_myah_not_in_channel_directory_skip_set():
    """Verify Myah is auto-included by `for plat in Platform` loop in
    `gateway/channel_directory.py`. The loop skips only entries listed
    in `_SKIP_SESSION_DISCOVERY` ({"local", "api_server", "webhook"});
    Myah must not be in that frozenset.
    """
    src = _read("gateway/channel_directory.py")
    # Locate the literal frozenset definition.
    assert "_SKIP_SESSION_DISCOVERY = frozenset(" in src
    # Extract the literal contents — naive but sufficient for this test.
    skip_block = src.split("_SKIP_SESSION_DISCOVERY = frozenset(", 1)[1]
    skip_block = skip_block.split(")", 1)[0]
    assert '"myah"' not in skip_block, (
        "Myah must NOT be in _SKIP_SESSION_DISCOVERY — it relies on the "
        "auto-discovery loop in channel_directory.py to be enumerated."
    )


# ── Item 13: hermes status --all must show a Myah row ──────────────────────


def test_myah_row_in_hermes_cli_status():
    """`hermes_cli/status.py` Messaging Platforms dict contains a Myah row."""
    src = _read("hermes_cli/status.py")
    # The dict literal uses string keys. Match the canonical "Myah" row.
    assert '"Myah"' in src, "hermes_cli/status.py must include a Myah row"


# ── Item 14: setup wizard must offer Myah ──────────────────────────────────


def test_myah_in_gateway_wizard_platforms():
    """`hermes_cli/gateway.py::_PLATFORMS` contains an entry with key='myah'."""
    src = _read("hermes_cli/gateway.py")
    # Tolerate both single and double quote styles.
    assert ('"key": "myah"' in src) or ("'key': 'myah'" in src), (
        "hermes_cli/gateway.py _PLATFORMS list must contain a Myah entry"
    )


# ── Item 7: dedicated toolset preset ───────────────────────────────────────


def test_hermes_myah_toolset_preset_exists():
    """`toolsets.py::TOOLSETS` registers the `hermes-myah` preset.

    NOTE: the dict is named `TOOLSETS` in current Hermes (not
    `TOOLSET_PRESETS` as referred to in some older docs). Verified
    2026-04-25.
    """
    from toolsets import TOOLSETS

    assert "hermes-myah" in TOOLSETS, (
        "TOOLSETS must include a 'hermes-myah' preset for the Myah platform"
    )


# ── Item 8 (delivery routing): cron scheduler platform map ─────────────────


def test_myah_in_cron_scheduler_platform_map():
    """`cron/scheduler.py` includes Platform.MYAH in its delivery map."""
    src = _read("cron/scheduler.py")
    assert "Platform.MYAH" in src, (
        "cron/scheduler.py must route deliver='myah' to Platform.MYAH"
    )


# ── Item 9: send_message_tool platform map ────────────────────────────────


def test_myah_in_send_message_tool_platform_map():
    """`tools/send_message_tool.py` maps 'myah' → Platform.MYAH."""
    src = _read("tools/send_message_tool.py")
    assert "Platform.MYAH" in src, (
        "send_message_tool.py must map 'myah' to Platform.MYAH"
    )


# ── Item 6: prompt builder platform hint ───────────────────────────────────


def test_myah_in_prompt_builder_platform_hints():
    """`agent/prompt_builder.py::PLATFORM_HINTS` has a 'myah' entry."""
    from agent.prompt_builder import PLATFORM_HINTS

    assert "myah" in PLATFORM_HINTS, (
        "PLATFORM_HINTS must include a 'myah' entry so the agent gets "
        "platform-appropriate guidance when running on Myah."
    )


# ── Item 3: adapter factory dispatches Platform.MYAH ───────────────────────


def test_myah_handled_by_adapter_factory():
    """`gateway/run.py::_create_adapter` (a method on GatewayRunner) handles
    Platform.MYAH and constructs MyahAdapter.
    """
    src = _read("gateway/run.py")
    # The factory is a method, so we can't trivially import + introspect.
    # Use text scan: confirm both the dispatch branch and the import.
    assert "Platform.MYAH" in src
    assert "from gateway.platforms.myah import MyahAdapter" in src, (
        "_create_adapter must import and instantiate MyahAdapter for "
        "Platform.MYAH"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
