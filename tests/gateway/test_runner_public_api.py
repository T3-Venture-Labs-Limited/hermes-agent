"""Unit tests for GatewayRunner public API.

These tests verify the public Myah-platform integration surface added in
Workstream B Phase 1.  They cover the contract that external integrations
(Myah platform adapters, future plugins) rely on instead of reaching into
private `_session_model_overrides`, `_agent_cache`, or `_running_agents`
attributes.

The tests construct a minimal runner via ``GatewayRunner.__new__`` (no
``__init__``) and seed only the dicts/locks the public methods touch —
identical pattern to ``test_agent_cache.py`` so they share the same
runtime profile.
"""

import threading
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest


def _make_runner():
    """Construct a minimal runner exposing only the fields the API touches."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


class TestGetSessionOverride:
    def test_returns_none_for_unknown_session(self):
        runner = _make_runner()
        assert runner.get_session_override("telegram:42") is None

    def test_returns_dict_when_set(self):
        runner = _make_runner()
        runner._session_model_overrides["telegram:42"] = {
            "model": "claude-sonnet-4",
            "provider": "anthropic",
        }
        result = runner.get_session_override("telegram:42")
        assert result == {"model": "claude-sonnet-4", "provider": "anthropic"}

    def test_handles_none_overrides_dict(self):
        """Defensive: legacy code paths may set the dict to None."""
        runner = _make_runner()
        runner._session_model_overrides = None
        assert runner.get_session_override("telegram:42") is None


class TestSetSessionOverride:
    def test_writes_value_visible_to_get(self):
        runner = _make_runner()
        runner.set_session_override(
            "telegram:42",
            {"model": "claude-sonnet-4", "provider": "anthropic"},
        )
        assert runner.get_session_override("telegram:42") == {
            "model": "claude-sonnet-4",
            "provider": "anthropic",
        }

    def test_overwrites_existing(self):
        runner = _make_runner()
        runner.set_session_override("telegram:42", {"model": "claude-sonnet-4"})
        runner.set_session_override("telegram:42", {"model": "gpt-5"})
        assert runner.get_session_override("telegram:42") == {"model": "gpt-5"}

    def test_evicts_cached_agent(self):
        runner = _make_runner()
        sentinel_agent = MagicMock(name="cached_agent")
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:42"] = (sentinel_agent, "sig1")

        runner.set_session_override(
            "telegram:42", {"model": "claude-sonnet-4"},
        )

        with runner._agent_cache_lock:
            assert "telegram:42" not in runner._agent_cache

    def test_initialises_overrides_dict_if_none(self):
        runner = _make_runner()
        runner._session_model_overrides = None
        runner.set_session_override("telegram:42", {"model": "claude-sonnet-4"})
        assert runner.get_session_override("telegram:42") == {"model": "claude-sonnet-4"}

    def test_eviction_failure_does_not_undo_override(self):
        """If the eviction path raises, the override write must still stand."""
        runner = _make_runner()
        # Force eviction to raise by replacing the lock with a broken context manager.
        broken_lock = MagicMock()
        broken_lock.__enter__.side_effect = RuntimeError("boom")
        runner._agent_cache_lock = broken_lock

        runner.set_session_override("telegram:42", {"model": "claude-sonnet-4"})
        # Override is still present even though the eviction blew up.
        assert runner.get_session_override("telegram:42") == {"model": "claude-sonnet-4"}


class TestEvictSessionAgent:
    def test_returns_false_for_unknown_session(self):
        runner = _make_runner()
        assert runner.evict_session_agent("telegram:42") is False

    def test_returns_true_after_one_is_set(self):
        runner = _make_runner()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:42"] = (MagicMock(), "sig1")
        assert runner.evict_session_agent("telegram:42") is True

    def test_removes_entry(self):
        runner = _make_runner()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:42"] = (MagicMock(), "sig1")
        runner.evict_session_agent("telegram:42")
        with runner._agent_cache_lock:
            assert "telegram:42" not in runner._agent_cache

    def test_does_not_clear_session_override(self):
        """Eviction is cache-only — overrides are independent state."""
        runner = _make_runner()
        runner._session_model_overrides["telegram:42"] = {"model": "claude-sonnet-4"}
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:42"] = (MagicMock(), "sig1")
        runner.evict_session_agent("telegram:42")
        assert runner.get_session_override("telegram:42") == {"model": "claude-sonnet-4"}


class TestIsSessionRunning:
    def test_false_initially(self):
        runner = _make_runner()
        assert runner.is_session_running("telegram:42") is False

    def test_true_when_agent_present(self):
        runner = _make_runner()
        runner._running_agents["telegram:42"] = MagicMock(name="agent")
        assert runner.is_session_running("telegram:42") is True

    def test_true_for_pending_sentinel(self):
        """A session in the pre-await sentinel state is also considered running."""
        from gateway.run import _AGENT_PENDING_SENTINEL

        runner = _make_runner()
        runner._running_agents["telegram:42"] = _AGENT_PENDING_SENTINEL
        assert runner.is_session_running("telegram:42") is True


class TestIterRunningSessionKeys:
    def test_empty_when_no_runs(self):
        runner = _make_runner()
        assert runner.iter_running_session_keys() == []

    def test_returns_snapshot_of_keys(self):
        runner = _make_runner()
        runner._running_agents["a"] = MagicMock()
        runner._running_agents["b"] = MagicMock()
        keys = runner.iter_running_session_keys()
        assert sorted(keys) == ["a", "b"]

    def test_snapshot_decoupled_from_live_dict(self):
        """The returned list must not change when the dict is mutated."""
        runner = _make_runner()
        runner._running_agents["a"] = MagicMock()
        snapshot = runner.iter_running_session_keys()
        runner._running_agents["b"] = MagicMock()
        assert snapshot == ["a"]


class TestIterCachedSessionKeys:
    def test_empty_when_no_cached_agents(self):
        runner = _make_runner()
        assert runner.iter_cached_session_keys() == []

    def test_returns_keys(self):
        runner = _make_runner()
        with runner._agent_cache_lock:
            runner._agent_cache["a"] = (MagicMock(), "sig")
            runner._agent_cache["b"] = (MagicMock(), "sig")
        assert sorted(runner.iter_cached_session_keys()) == ["a", "b"]


class TestGetCachedAgent:
    def test_none_when_missing(self):
        runner = _make_runner()
        assert runner.get_cached_agent("telegram:42") is None

    def test_returns_agent_from_tuple_entry(self):
        runner = _make_runner()
        agent = MagicMock(name="agent")
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:42"] = (agent, "sig")
        assert runner.get_cached_agent("telegram:42") is agent


class TestGetCachedAgentAttribution:
    def test_none_when_missing(self):
        runner = _make_runner()
        assert runner.get_cached_agent_attribution("telegram:42") is None

    def test_returns_model_and_provider(self):
        runner = _make_runner()
        agent = MagicMock()
        agent.model = "claude-sonnet-4"
        agent.provider = "anthropic"
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:42"] = (agent, "sig")
        result = runner.get_cached_agent_attribution("telegram:42")
        assert result == {"model": "claude-sonnet-4", "provider": "anthropic"}

    def test_returns_empty_strings_when_attrs_missing(self):
        runner = _make_runner()
        agent = MagicMock(spec=[])  # no model/provider attributes
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:42"] = (agent, "sig")
        result = runner.get_cached_agent_attribution("telegram:42")
        assert result == {"model": "", "provider": ""}


class TestPublicApiTransactional:
    """Cross-method invariants that the migration depends on."""

    def test_set_override_then_eviction_then_re_set(self):
        """Round-trip: write → evict → re-write keeps state consistent."""
        runner = _make_runner()

        runner.set_session_override("s", {"model": "m1", "provider": "p1"})
        assert runner.get_session_override("s") == {"model": "m1", "provider": "p1"}

        evicted = runner.evict_session_agent("s")
        assert evicted is False  # nothing was cached
        # Override survives standalone eviction.
        assert runner.get_session_override("s") == {"model": "m1", "provider": "p1"}

        runner.set_session_override("s", {"model": "m2", "provider": "p2"})
        assert runner.get_session_override("s") == {"model": "m2", "provider": "p2"}

    def test_set_override_while_session_running_evicts_for_next_run(self):
        """An override applied during a run takes effect on the NEXT run.

        The current run keeps using the in-flight agent; only the cached
        copy (used for the next message) is invalidated.  This matches
        the existing `_handle_model_command` semantics for `--session`.
        """
        runner = _make_runner()
        running_agent = MagicMock(name="running")
        cached_agent = MagicMock(name="cached")
        runner._running_agents["s"] = running_agent
        with runner._agent_cache_lock:
            runner._agent_cache["s"] = (cached_agent, "sig")

        runner.set_session_override("s", {"model": "m1"})

        # The running agent reference is untouched...
        assert runner._running_agents["s"] is running_agent
        assert runner.is_session_running("s") is True
        # ...but the cached copy is gone, so the next run rebuilds.
        with runner._agent_cache_lock:
            assert "s" not in runner._agent_cache
