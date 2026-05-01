"""Tests for ``gateway/platforms/myah_event_log.py``.

Sibling-table event log used by the Myah platform adapter. Pure data
layer: schema + append + back-fill + position recovery. The conftest
autouse ``_isolate_hermes_home`` fixture sets ``HERMES_HOME`` to a
per-test tempdir so writes can't touch the real ``~/.hermes/state.db``.

Module-level state persists across tests in the same xdist worker, so
each test calls ``reset_state_for_tests()`` first.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from gateway.platforms import myah_event_log


@pytest.fixture(autouse=True)
def _reset_event_log_state(monkeypatch):
    """Reset module-level singletons before EACH test, and pin
    ``hermes_state.DEFAULT_DB_PATH`` to the current per-test HERMES_HOME
    so SessionDB and ``myah_event_log._db_path()`` agree on the file."""
    myah_event_log.reset_state_for_tests()
    import hermes_state
    monkeypatch.setattr(
        hermes_state,
        "DEFAULT_DB_PATH",
        myah_event_log._db_path(),
    )
    yield
    myah_event_log.reset_state_for_tests()


def _seed_session(session_id: str = "s1") -> None:
    """Create the parent ``sessions`` row so the FK on
    ``myah_session_events`` doesn't reject inserts. Uses the current
    ``DEFAULT_DB_PATH`` (pinned to per-test HERMES_HOME by the fixture)."""
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        db.create_session(session_id=session_id, source="myah", model="x")
    finally:
        db.close()


# ── _ensure_schema ─────────────────────────────────────────────────────


def test_ensure_schema_creates_table():
    """First append() lazy-creates the schema."""
    _seed_session("s1")
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "hi"})

    db_path = myah_event_log._db_path()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='myah_session_events'"
        ).fetchall()
    assert rows == [("myah_session_events",)]


def test_ensure_schema_is_idempotent():
    """Calling _ensure_schema twice doesn't error."""
    myah_event_log._ensure_schema()
    myah_event_log._ensure_schema()  # second call returns immediately
    # Reset and re-run to exercise the path that re-creates after a reset
    myah_event_log.reset_state_for_tests()
    myah_event_log._ensure_schema()

    db_path = myah_event_log._db_path()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='myah_session_events'"
        ).fetchall()
    assert len(rows) == 1


# ── append_event ───────────────────────────────────────────────────────


def test_append_event_writes_row():
    """A persistable event is written with monotonic position + JSON payload."""
    _seed_session("s1")
    e1 = {"event": "message.delta", "delta": "hello"}
    e2 = {"event": "tool.started", "call_id": "c1", "tool": "terminal"}

    myah_event_log.append_event("s1", e1)
    myah_event_log.append_event("s1", e2)

    db_path = myah_event_log._db_path()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT session_id, position, hermes_message_id, event_type, payload "
            "FROM myah_session_events WHERE session_id = ? ORDER BY position",
            ("s1",),
        ).fetchall()

    assert len(rows) == 2
    assert rows[0][0] == "s1"
    assert rows[0][1] == 1  # position monotonic from 1
    assert rows[0][2] is None  # hermes_message_id starts NULL
    assert rows[0][3] == "message.delta"
    assert json.loads(rows[0][4]) == e1
    assert rows[1][1] == 2
    assert rows[1][3] == "tool.started"
    assert json.loads(rows[1][4]) == e2


def test_append_event_filters_non_persistable_types():
    """status / unknown / no-event-key events are skipped without writing."""
    _seed_session("s1")
    myah_event_log.append_event("s1", {"event": "status", "text": "thinking"})
    myah_event_log.append_event("s1", {"event": "noise", "x": 1})
    myah_event_log.append_event("s1", {"delta": "no event key"})
    # One real event after the noise — proves the noise didn't bump position.
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "real"})

    db_path = myah_event_log._db_path()
    with sqlite3.connect(str(db_path)) as conn:
        # The status / noise / missing-event paths skip BEFORE _ensure_schema
        # is called, so the table may not exist yet. Make sure it does
        # before querying so the test fails cleanly if behavior changes.
        try:
            rows = conn.execute(
                "SELECT position, event_type FROM myah_session_events "
                "WHERE session_id = ? ORDER BY position",
                ("s1",),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

    assert rows == [(1, "message.delta")]


def test_append_event_no_op_for_invalid_inputs():
    """Empty session_id / non-dict event / None event must no-op cleanly."""
    # No exception, no schema creation triggered.
    myah_event_log.append_event("", {"event": "message.delta"})
    myah_event_log.append_event("s1", None)  # type: ignore[arg-type]
    myah_event_log.append_event("s1", "not a dict")  # type: ignore[arg-type]
    myah_event_log.append_event("s1", 42)  # type: ignore[arg-type]
    # No row should exist; table may not even exist.
    db_path = myah_event_log._db_path()
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            try:
                rows = conn.execute(
                    "SELECT COUNT(*) FROM myah_session_events"
                ).fetchall()
                assert rows[0][0] == 0
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet — that's fine


# ── backfill_message_id ────────────────────────────────────────────────


def _append_real_message(session_id: str, content: str) -> int:
    """Insert a real ``messages`` row so back-fill FK passes. Returns the id."""
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        return db.append_message(session_id, role="assistant", content=content)
    finally:
        db.close()


def test_backfill_message_id_links_pending_rows():
    """back-fill updates only the rows with NULL hermes_message_id for the session."""
    _seed_session("s1")
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "x"})
    myah_event_log.append_event("s1", {"event": "tool.started", "call_id": "c"})
    myah_event_log.append_event("s1", {"event": "run.completed", "output": []})
    msg_id = _append_real_message("s1", "x")

    myah_event_log.backfill_message_id("s1", msg_id)

    db_path = myah_event_log._db_path()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT position, hermes_message_id FROM myah_session_events "
            "WHERE session_id = ? ORDER BY position",
            ("s1",),
        ).fetchall()
    assert rows == [(1, msg_id), (2, msg_id), (3, msg_id)]


def test_backfill_does_not_re_link_already_linked_rows():
    """A second back-fill (for a later run) only touches the new NULL rows."""
    _seed_session("s1")
    # Run 1
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "a"})
    myah_event_log.append_event("s1", {"event": "run.completed", "output": []})
    msg1 = _append_real_message("s1", "a")
    myah_event_log.backfill_message_id("s1", msg1)
    # Run 2
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "b"})
    myah_event_log.append_event("s1", {"event": "run.completed", "output": []})
    msg2 = _append_real_message("s1", "b")
    myah_event_log.backfill_message_id("s1", msg2)

    db_path = myah_event_log._db_path()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT position, hermes_message_id FROM myah_session_events "
            "WHERE session_id = ? ORDER BY position",
            ("s1",),
        ).fetchall()
    assert rows == [(1, msg1), (2, msg1), (3, msg2), (4, msg2)]


# ── position recovery after restart ────────────────────────────────────


def test_position_recovery_after_process_restart():
    """After reset_state_for_tests(), positions resume from the DB max — not 1."""
    _seed_session("s1")
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "1"})
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "2"})
    # Simulate process restart — wipe in-memory counter
    myah_event_log.reset_state_for_tests()
    myah_event_log.append_event("s1", {"event": "message.delta", "delta": "3"})

    db_path = myah_event_log._db_path()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT position FROM myah_session_events "
            "WHERE session_id = ? ORDER BY position",
            ("s1",),
        ).fetchall()
    assert [r[0] for r in rows] == [1, 2, 3]


# ── get_max_assistant_message_id (race-fix helper) ─────────────────────


def test_max_assistant_message_id_returns_none_for_empty_session():
    """No messages yet → None (not 0, not error)."""
    _seed_session("empty_session")
    assert myah_event_log.get_max_assistant_message_id("empty_session") is None


def test_max_assistant_message_id_returns_max_id():
    """Returns the highest id among assistant messages."""
    _seed_session("s1")
    msg1 = _append_real_message("s1", "first")
    msg2 = _append_real_message("s1", "second")
    msg3 = _append_real_message("s1", "third")
    result = myah_event_log.get_max_assistant_message_id("s1")
    assert result == msg3
    # Sanity: the ids are monotonic increasing
    assert msg1 < msg2 < msg3


def test_max_assistant_message_id_filters_by_after_id():
    """Race-fix path: only returns ids strictly greater than after_id."""
    _seed_session("s1")
    msg1 = _append_real_message("s1", "before run")
    msg2 = _append_real_message("s1", "during run")

    # Pretend the run started after msg1 was already in the DB.
    # Only msg2 should qualify.
    result = myah_event_log.get_max_assistant_message_id("s1", after_id=msg1)
    assert result == msg2

    # If after_id is the most recent message, no qualifying row exists.
    result_none = myah_event_log.get_max_assistant_message_id("s1", after_id=msg2)
    assert result_none is None


def test_max_assistant_message_id_ignores_non_assistant_roles():
    """Tool / user / session_meta rows must not satisfy the query."""
    _seed_session("s1")
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        # Insert several non-assistant messages
        db.append_message("s1", role="user", content="q")
        db.append_message("s1", role="tool", tool_call_id="c1", content="r")
        # Then one assistant
        asst_id = db.append_message("s1", role="assistant", content="a")
        # Then more non-assistants AFTER the assistant — these must NOT be returned.
        db.append_message("s1", role="user", content="q2")
        db.append_message("s1", role="tool", tool_call_id="c2", content="r2")
    finally:
        db.close()

    assert myah_event_log.get_max_assistant_message_id("s1") == asst_id


def test_max_assistant_message_id_tolerates_missing_db():
    """Querying when the DB file doesn't exist returns None (not raise)."""
    # autouse fixture has reset state but not created any DB; this test just
    # passes session_id without seeding to confirm graceful handling.
    assert myah_event_log.get_max_assistant_message_id("never_seeded") is None


def test_max_assistant_message_id_handles_empty_session_id():
    """Defensive: empty session_id returns None without touching the DB."""
    assert myah_event_log.get_max_assistant_message_id("") is None
    assert myah_event_log.get_max_assistant_message_id(None) is None  # type: ignore[arg-type]
