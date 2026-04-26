"""Test for Bug E: _origin_from_env() return-None branch must emit a structured warning.

When session contextvars (HERMES_SESSION_PLATFORM, HERMES_SESSION_CHAT_ID) are
unset at cron-creation time, the origin capture silently returns None and the
job is persisted with `origin=null` and `deliver="origin"` — a corrupted shape
that makes the cron run successfully but never deliver. The Bug E fix adds a
structured WARNING log so the next regression takes a minute to diagnose, not
a day.
"""

import logging
from unittest.mock import patch


# ── Myah: Bug E — origin capture failure must emit a warning ────
class TestOriginFromEnvLogsOnFailure:
    def test_returns_none_when_session_env_missing_and_warns(self, caplog):
        """When HERMES_SESSION_PLATFORM/CHAT_ID are both empty, return None and
        emit a WARNING log line containing the unset state for debugging."""
        from tools.cronjob_tools import _origin_from_env

        # Force every session env lookup to return empty
        with patch("gateway.session_context.get_session_env", return_value=""):
            with caplog.at_level(logging.WARNING, logger="tools.cronjob_tools"):
                result = _origin_from_env()

        assert result is None, "expected None when session env is missing"

        # Find a WARNING-level log line that mentions origin capture failure
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "expected at least one WARNING log when origin capture fails"

        joined = " ".join(r.getMessage() for r in warnings)
        # The log must surface enough state to debug a regression
        assert "origin" in joined.lower(), f"warning must mention origin: {joined!r}"
        # Should contain at least one piece of debug context: env-var name, thread, or fallback indicator
        assert any(
            token in joined for token in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "platform=", "chat_id=")
        ), f"warning must surface session env state: {joined!r}"

    def test_returns_origin_when_session_env_set_no_warning(self, caplog):
        """Happy path: when both platform + chat_id are set, return the dict
        and do NOT emit a WARNING (only debug-level)."""
        from tools.cronjob_tools import _origin_from_env

        env_map = {
            "HERMES_SESSION_PLATFORM": "myah",
            "HERMES_SESSION_CHAT_ID": "abc-123",
            "HERMES_SESSION_CHAT_NAME": "Test Chat",
            "HERMES_SESSION_THREAD_ID": "",
        }

        def _fake_get(key):
            return env_map.get(key, "")

        with patch("gateway.session_context.get_session_env", side_effect=_fake_get):
            with caplog.at_level(logging.WARNING, logger="tools.cronjob_tools"):
                result = _origin_from_env()

        assert result == {
            "platform": "myah",
            "chat_id": "abc-123",
            "chat_name": "Test Chat",
            "thread_id": None,
        }

        # No WARNING should be emitted on the happy path
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, f"unexpected warnings on happy path: {[r.getMessage() for r in warnings]}"
# ────────────────────────────────────────────────────────────────
