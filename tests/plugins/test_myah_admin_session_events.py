"""Tests for the myah-admin sessions/events read endpoint.

Covers ``GET /sessions/{id}/events`` in
``plugins/myah-admin/dashboard/_sessions_and_lifecycle.py``. The endpoint
reads from the ``myah_session_events`` sibling table populated by the
Myah platform adapter's event log.

Conftest's autouse ``_isolate_hermes_home`` fixture sets ``HERMES_HOME``
to a per-test tempdir, so writes can't touch the real ``~/.hermes``.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Module loader ───────────────────────────────────────────────────────────


def _load_lifecycle_module():
    """Import ``_sessions_and_lifecycle`` from ``plugins/myah-admin/dashboard``.

    Same pattern as ``test_myah_admin_sessions_lifecycle.py`` — see that
    file for the explanation of why a hyphenated package name forces
    path-based loading.
    """
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "myah-admin" / "dashboard"

    pkg_name = "_myah_admin_dashboard_under_test_events"
    if pkg_name not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(pkg_name, loader=None)
        )
        pkg.__path__ = [str(plugin_dir)]
        sys.modules[pkg_name] = pkg

    common_path = plugin_dir / "_common.py"
    common_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}._common", common_path
    )
    common_mod = importlib.util.module_from_spec(common_spec)
    sys.modules[f"{pkg_name}._common"] = common_mod
    common_spec.loader.exec_module(common_mod)

    target_path = plugin_dir / "_sessions_and_lifecycle.py"
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}._sessions_and_lifecycle", target_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}._sessions_and_lifecycle"] = mod
    spec.loader.exec_module(mod)
    return mod, common_mod


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def lifecycle_app(monkeypatch):
    """FastAPI app with the lifecycle router mounted, auth bypassed."""
    monkeypatch.delenv("HERMES_WEB_SESSION_TOKEN", raising=False)
    mod, common_mod = _load_lifecycle_module()
    # Bypass require_session_token: the router has it as a global Depends,
    # so we override at the FastAPI dependency-overrides level rather than
    # patching the symbol (which is captured by the router's Depends).
    app = FastAPI()
    app.include_router(mod.router)
    # Override the auth dependency
    app.dependency_overrides[common_mod.require_session_token] = lambda: None
    app.dependency_overrides[mod.require_session_token] = lambda: None
    return app, mod


@pytest.fixture
def client(lifecycle_app):
    app, _mod = lifecycle_app
    return TestClient(app)


def _ensure_events_schema(db_path: Path) -> None:
    """Create the events table in the test DB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS myah_session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                hermes_message_id INTEGER,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mse_session_pos
                ON myah_session_events(session_id, position);
        """)
        conn.commit()


def _insert_event(
    db_path: Path,
    session_id: str,
    position: int,
    event_type: str,
    payload: dict,
    msg_id: int | None = None,
) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO myah_session_events "
            "(session_id, position, hermes_message_id, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, position, msg_id, event_type, json.dumps(payload), time.time()),
        )
        conn.commit()


# ── Tests ──────────────────────────────────────────────────────────────────


def test_get_session_events_returns_chronological_payload(client):
    """Pre-populated events surface in position order with decoded payloads."""
    from hermes_constants import get_hermes_home
    db_path = get_hermes_home() / "state.db"
    _ensure_events_schema(db_path)

    e1 = {"event": "message.delta", "delta": "hello "}
    e2 = {"event": "message.delta", "delta": "world"}
    e3 = {"event": "tool.started", "call_id": "c1", "tool": "terminal"}
    _insert_event(db_path, "s1", 1, "message.delta", e1)
    _insert_event(db_path, "s1", 2, "message.delta", e2, msg_id=42)
    _insert_event(db_path, "s1", 3, "tool.started", e3, msg_id=42)

    resp = client.get("/sessions/s1/events")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == "s1"
    assert len(body["events"]) == 3
    assert [ev["position"] for ev in body["events"]] == [1, 2, 3]
    assert body["events"][0]["event_type"] == "message.delta"
    assert body["events"][0]["payload"] == e1
    assert body["events"][0]["hermes_message_id"] is None
    assert body["events"][1]["payload"] == e2
    assert body["events"][1]["hermes_message_id"] == 42
    assert body["events"][2]["payload"] == e3
    assert all("created_at" in ev for ev in body["events"])


def test_get_session_events_empty_for_unknown_session(client):
    """Unknown session id returns empty list, not 404."""
    from hermes_constants import get_hermes_home
    db_path = get_hermes_home() / "state.db"
    _ensure_events_schema(db_path)
    # Insert events for a different session
    _insert_event(db_path, "other", 1, "message.delta", {"event": "message.delta"})

    resp = client.get("/sessions/does-not-exist/events")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"session_id": "does-not-exist", "events": []}


def test_get_session_events_tolerates_missing_table(client):
    """When myah_session_events table doesn't exist, return empty list (not 500)."""
    # Do NOT call _ensure_events_schema. The DB file may or may not exist;
    # either way the endpoint should return cleanly.
    from hermes_constants import get_hermes_home
    db_path = get_hermes_home() / "state.db"
    if db_path.exists():
        # Create a DB without the events table (simulate fresh SessionDB)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
            conn.commit()

    resp = client.get("/sessions/s1/events")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"session_id": "s1", "events": []}
