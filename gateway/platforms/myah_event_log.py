"""Sibling-table event log for the Myah platform adapter.

Captures the chronological raw-event stream the Myah adapter emits to its
SSE queue, so the platform's reload path can reconstruct the same output[]
the live path renders. Pure data layer; no business logic.

Why this lives here (and not as a plugin under plugins/observability/):
    - The data captured is intrinsically Myah-shaped (it's the event vocabulary
      the Myah platform emits). Generic observability plugins should be
      reusable across platforms; this is not.
    - post_llm_call (the hook a generic observability plugin would use) only
      receives `assistant_response` as a string for non-Codex providers
      (run_agent.py:13089-13106 + 12662). It cannot see tool.confirmation_required
      or secret.required — both of which the Myah frontend renders.
    - gateway/platforms/ is the documented platform-extension surface. Living
      alongside myah.py keeps the data source and the storage co-located.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

# Events that contribute to output[] rendering. status events are
# transient UI hints and not persisted. Unknown events are skipped.
_PERSIST_EVENT_TYPES = frozenset({
    "message.delta",
    "reasoning.delta",
    "reasoning.available",
    "tool.started",
    "tool.completed",
    "tool.confirmation_required",
    "secret.required",
    "secret.resolved",
    "run.completed",
    "run.failed",
})

_lock = threading.Lock()
_position_per_session: dict[str, int] = {}
_schema_ensured = False


def _db_path():
    """Path to the shared SessionDB file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS myah_session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    hermes_message_id INTEGER,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (hermes_message_id) REFERENCES messages(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_mse_session_pos ON myah_session_events(session_id, position);
CREATE INDEX IF NOT EXISTS idx_mse_session_msg ON myah_session_events(session_id, hermes_message_id);
"""


def _ensure_schema() -> None:
    """Create the events table on first use. Idempotent + thread-safe."""
    global _schema_ensured
    if _schema_ensured:
        return
    with _lock:
        if _schema_ensured:
            return
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(path)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA_DDL)
            conn.commit()
        _schema_ensured = True


def _next_position(session_id: str) -> int:
    """Get the next monotonic position for a session. Lazy-loads from DB
    on first access (handles process restart)."""
    with _lock:
        pos = _position_per_session.get(session_id)
        if pos is None:
            try:
                with sqlite3.connect(str(_db_path())) as conn:
                    row = conn.execute(
                        "SELECT MAX(position) FROM myah_session_events WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                pos = (row[0] or 0) if row else 0
            except sqlite3.Error:
                pos = 0
        pos += 1
        _position_per_session[session_id] = pos
        return pos


def append_event(session_id: str, event: dict) -> None:
    """Persist one raw event. No-op for empty session_id / non-dict event /
    filtered event types. Failures log but never raise — the live SSE
    stream must not break because of a persistence error."""
    if not session_id or not isinstance(event, dict):
        return
    event_type = event.get("event")
    if event_type not in _PERSIST_EVENT_TYPES:
        return
    _ensure_schema()
    position = _next_position(session_id)
    try:
        payload = json.dumps(event, default=str)
    except (TypeError, ValueError) as exc:
        logger.debug("[myah_event_log] payload serialization failed: %s", exc)
        payload = json.dumps({"event": event_type, "_serialization_error": True})
    try:
        with sqlite3.connect(str(_db_path())) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "INSERT INTO myah_session_events "
                "(session_id, position, hermes_message_id, event_type, payload, created_at) "
                "VALUES (?, ?, NULL, ?, ?, ?)",
                (session_id, position, event_type, payload, time.time()),
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning("[myah_event_log] append failed for %s: %s", session_id, exc)


def backfill_message_id(session_id: str, hermes_message_id: int) -> None:
    """At run.completed, link this run's events (those with NULL
    hermes_message_id) to the just-written assistant message row."""
    if not session_id or hermes_message_id is None:
        return
    try:
        with sqlite3.connect(str(_db_path())) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "UPDATE myah_session_events SET hermes_message_id = ? "
                "WHERE session_id = ? AND hermes_message_id IS NULL",
                (hermes_message_id, session_id),
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning("[myah_event_log] back-fill failed for %s: %s", session_id, exc)


def get_max_assistant_message_id(session_id: str, after_id: int = 0) -> int | None:
    """Return ``MAX(messages.id)`` where ``session_id=?`` AND ``role='assistant'``
    AND ``id > after_id``.

    Used by the Myah adapter to decide which assistant message a just-completed
    run's events should be linked to. Passing ``after_id`` (the max assistant
    message id captured at run START) eliminates a race with concurrent cron
    appends: without it, a cron append landing in the microseconds between the
    gateway writing the agent's assistant message and the adapter's
    ``run.completed`` handler firing would cause the adapter to back-fill the
    cron's id instead of the agent's.

    Uses a separate sqlite3 connection rather than ``SessionDB`` private
    attributes (``_lock``, ``_conn``) so upstream Hermes refactors of
    SessionDB internals don't silently break the Myah adapter.

    Returns ``None`` if no qualifying message exists or on any sqlite error
    (callers must tolerate ``None``).
    """
    if not session_id:
        return None
    try:
        with sqlite3.connect(str(_db_path())) as conn:
            row = conn.execute(
                "SELECT MAX(id) FROM messages "
                "WHERE session_id = ? AND role = 'assistant' AND id > ?",
                (session_id, after_id),
            ).fetchone()
        return row[0] if row and row[0] is not None else None
    except sqlite3.Error as exc:
        logger.debug("[myah_event_log] max-assistant lookup failed: %s", exc)
        return None


def reset_state_for_tests() -> None:
    """Reset module-level state. For test isolation only."""
    global _schema_ensured
    with _lock:
        _position_per_session.clear()
        _schema_ensured = False
