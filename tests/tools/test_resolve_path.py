"""Tests for _resolve_path() — TERMINAL_CWD-aware path resolution in file_tools."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def isolated_resolve_state():
    """Clear ``"default"``-keyed pollution from ``_get_live_tracking_cwd``'s
    backing stores so the test falls through to ``TERMINAL_CWD``.

    ``_resolve_path_for_task`` consults ``_get_live_tracking_cwd(task_id)``
    which reads ``tools.file_tools._file_ops_cache`` and
    ``tools.terminal_tool._active_environments``. Other tests in the suite
    may have left a ``"default"`` entry in either store; without this
    isolation, the test sees a stale ``live_cwd`` instead of falling
    through to ``TERMINAL_CWD``. Under pytest-xdist (-n 4) the pollution
    is non-deterministic — depends on which tests share the same worker.

    Mutates the existing dicts in place (``.pop("default", None)``) rather
    than rebinding the module attributes — rebinding would also discard
    entries other tests added under different keys, which can change the
    semantics of subsequent tests. Restores the prior ``"default"`` entry
    after the test so any later test relying on it sees the same value.
    """
    from tools import file_tools

    snapshots: list = []

    with file_tools._file_ops_lock:
        snapshots.append(("file_ops", file_tools._file_ops_cache.pop("default", None)))

    try:
        from tools import terminal_tool

        with terminal_tool._env_lock:
            snapshots.append(
                ("active_env", terminal_tool._active_environments.pop("default", None))
            )
    except (ImportError, AttributeError):  # pragma: no cover - terminal_tool is core
        terminal_tool = None  # type: ignore[assignment]

    try:
        yield
    finally:
        for kind, value in snapshots:
            if value is None:
                continue
            if kind == "file_ops":
                with file_tools._file_ops_lock:
                    file_tools._file_ops_cache.setdefault("default", value)
            elif kind == "active_env" and terminal_tool is not None:
                with terminal_tool._env_lock:
                    terminal_tool._active_environments.setdefault("default", value)


class TestResolvePath:
    """Verify _resolve_path respects TERMINAL_CWD for worktree isolation."""

    def test_relative_path_uses_terminal_cwd(
        self, monkeypatch, tmp_path, isolated_resolve_state
    ):
        """Relative paths resolve against TERMINAL_CWD, not process CWD."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        result = _resolve_path("foo/bar.py")
        assert result == (tmp_path / "foo" / "bar.py")

    def test_absolute_path_ignores_terminal_cwd(self, monkeypatch, tmp_path):
        """Absolute paths are unaffected by TERMINAL_CWD."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        absolute = (tmp_path / "already-absolute.txt").resolve()
        result = _resolve_path(str(absolute))
        assert result == absolute

    def test_falls_back_to_cwd_without_terminal_cwd(self, monkeypatch):
        """Without TERMINAL_CWD, falls back to os.getcwd()."""
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        from tools.file_tools import _resolve_path

        result = _resolve_path("some_file.txt")
        assert result == Path(os.getcwd()) / "some_file.txt"

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        """~ is expanded before TERMINAL_CWD join (already absolute)."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        result = _resolve_path("~/notes.txt")
        # After expanduser, ~/notes.txt becomes absolute → TERMINAL_CWD ignored
        assert result == Path.home() / "notes.txt"

    def test_result_is_resolved(
        self, monkeypatch, tmp_path, isolated_resolve_state
    ):
        """Output path has no '..' components."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        result = _resolve_path("a/../b/file.txt")
        assert ".." not in str(result)
        assert result == (tmp_path / "b" / "file.txt")

    def test_relative_path_prefers_live_file_ops_cwd(self, monkeypatch, tmp_path):
        """Live env.cwd must win after the terminal session changes directory."""
        start_dir = tmp_path / "start"
        live_dir = tmp_path / "worktree"
        start_dir.mkdir()
        live_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(start_dir))

        from tools import file_tools

        task_id = "live-cwd"
        fake_ops = SimpleNamespace(
            env=SimpleNamespace(cwd=str(live_dir)),
            cwd=str(start_dir),
        )

        with file_tools._file_ops_lock:
            previous = file_tools._file_ops_cache.get(task_id)
            file_tools._file_ops_cache[task_id] = fake_ops

        try:
            result = file_tools._resolve_path("nested/file.txt", task_id=task_id)
        finally:
            with file_tools._file_ops_lock:
                if previous is None:
                    file_tools._file_ops_cache.pop(task_id, None)
                else:
                    file_tools._file_ops_cache[task_id] = previous

        assert result == live_dir / "nested" / "file.txt"
