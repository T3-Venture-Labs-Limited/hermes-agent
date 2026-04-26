"""Myah-admin plugin — backend API routes.

Mounted at ``/api/plugins/myah-admin/`` by the dashboard plugin loader inside
each per-user Hermes container. The platform backend reaches these routes
through the loopback ``hermes dashboard`` server (port 9119) using the
per-container session token (see ``platform/backend/open_webui/utils/hermes_web.py``).

What this plugin owns:
    Every Myah-specific admin operation that does NOT require live
    ``GatewayRunner`` state:
      * SOUL CRUD
      * Skill discovery + CRUD
      * Plugin CRUD
      * MCP server CRUD
      * Provider catalog + credentials
      * Session title / append-message ops
      * Config aux-resolved + last-reseed + reset
      * Slash-command discovery

What this plugin does NOT own:
    Runner-coupled admin (session model overrides, cache eviction, MCP
    refresh, gateway restart busy-check). Those live on the gateway under
    ``/myah/v1/admin/*`` (see ``gateway/platforms/myah_runtime_admin.py``).
    The plugin reaches the gateway via the localhost HTTP client in
    ``_common.gateway_client``.

Sub-router layout:
    ``_soul_and_config.py``      — SOUL, aux-resolved, commands, reset, last-reseed
    ``_skills_plugins_mcp.py``   — Skills CRUD, Plugins CRUD, MCP CRUD, toolset toggle
    ``_providers.py``            — Provider catalog, credentials, models
    ``_sessions_and_lifecycle.py`` — Session ops, global model, session model overrides

Telemetry registration:
    On import, register Myah's ``SentryHook`` with the ``agent.telemetry``
    registry so Hermes runtime instrumentation routes through Sentry.
    Idempotent + silent on import failure.

Import strategy:
    The dashboard plugin loader (``hermes_cli/web_server.py:_mount_plugin_api_routes``)
    builds the spec via ``importlib.util.spec_from_file_location`` with a
    synthetic module name (``hermes_dashboard_plugin_myah-admin``) and does
    NOT set ``submodule_search_locations``. As a result:
      * Package-relative imports (``from . import _common``) fail with
        "attempted relative import with no known parent package".
      * The hyphen in ``myah-admin`` makes
        ``import plugins.myah_admin.dashboard._common`` invalid.
    Therefore we load each sibling module by absolute path, the same pattern
    the SentryHook registration block below uses.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from fastapi import APIRouter

# ── Telemetry: register SentryHook (best-effort, idempotent) ────────────────
try:
    from plugins.myah_admin.myah_hook import register_sentry_hook
    register_sentry_hook()
except ImportError:
    try:
        _hook_path = os.path.join(os.path.dirname(__file__), '..', 'myah_hook.py')
        _spec = importlib.util.spec_from_file_location('_myah_admin_hook', _hook_path)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.register_sentry_hook()
    except Exception:
        # Telemetry is best-effort — never block plugin load.
        pass


# ── Sibling module loader ────────────────────────────────────────────────────


def _load_sibling(module_name: str, file_name: str):
    """Load a sibling .py file as a synthetic module, registering it in
    sys.modules under ``module_name`` so that any internal cross-imports
    (e.g. between sub-routers and ``_common``) resolve.

    Returns the loaded module. Raises ImportError on failure (we don't
    want silent route loss — if a sub-router fails to load, plugin
    initialisation should surface it loudly).
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, file_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"could not build spec for {file_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load _common FIRST so sub-routers can import it. Each sub-router has its
# own fallback shim (see implementer notes), but pre-registering _common
# under the canonical module name avoids the duplicate-load that would
# otherwise produce two distinct ``require_session_token`` instances.
_common = _load_sibling('myah_admin_dashboard_common', '_common.py')

# Load each sub-router in order. Each module exports a top-level ``router``.
_soul_mod = _load_sibling(
    'myah_admin_dashboard_soul_and_config', '_soul_and_config.py'
)
_skills_mod = _load_sibling(
    'myah_admin_dashboard_skills_plugins_mcp', '_skills_plugins_mcp.py'
)
_providers_mod = _load_sibling(
    'myah_admin_dashboard_providers', '_providers.py'
)
_sessions_mod = _load_sibling(
    'myah_admin_dashboard_sessions_and_lifecycle', '_sessions_and_lifecycle.py'
)


# ── Top-level router (mount point for the dashboard plugin loader) ──────────

router = APIRouter()

# Liveness probe — the platform's `hermes_web.py::web_call` against this path
# is the canonical reachability check used by the platform's
# `/api/v1/containers/{user_id}/web-health` endpoint.
@router.get("/health")
async def health() -> dict:
    """Liveness probe for the platform's hermes_web client."""
    return {"status": "ok", "plugin": "myah-admin"}


# Mount sub-routers. Order is informational; FastAPI dispatches on path,
# not registration order, so collisions would be a bug at design time.
router.include_router(_soul_mod.router)
router.include_router(_skills_mod.router)
router.include_router(_providers_mod.router)
router.include_router(_sessions_mod.router)
