"""Runtime-control admin surface for the Myah platform adapter.

This module exposes the small set of admin operations that **must** run in the
gateway process because they touch the live ``GatewayRunner`` (session model
overrides, agent cache eviction, busy-check). Everything else (file-system
admin: SOUL, skills, plugins, MCP CRUD, providers, reset) lives in the
``plugins/myah-admin/`` dashboard plugin which runs in the ``hermes dashboard``
process.

Mounting:
    Routes are added under ``/myah/v1/admin/*`` via
    ``register_runtime_admin_routes(app, *, runner, auth_key)`` from
    ``gateway/platforms/myah.py::_register_routes_on_app``.

Auth:
    Same Bearer-token model as the rest of the Myah adapter. The platform
    backend forwards every request with ``Authorization: Bearer <MYAH_ADAPTER_AUTH_KEY>``.
    The ``myah-admin`` dashboard plugin reaches this surface via
    ``http://localhost:8642`` (same container) using the same key — read from
    ``MYAH_ADAPTER_AUTH_KEY`` env var inside the container.

Why a separate module:
    Keeps ``myah.py`` focused on chat I/O. The whole ``myah_management.py`` (-2,035
    LOC) was deleted in this PR; this module replaces only the runner-coupled
    sliver (~150 LOC). All file-system admin moved to the plugin.
"""

from __future__ import annotations

import hmac
import logging
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aiohttp import web

    from gateway.run import GatewayRunner

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore

logger = logging.getLogger(__name__)


# ── Auth helper ─────────────────────────────────────────────────────────────


def _check_auth(request: "web.Request", auth_key: Optional[str]) -> Optional["web.Response"]:
    """Same Bearer-token model as MyahAdapter._check_auth."""
    if not auth_key:
        return None
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        token = header[7:].strip()
        if hmac.compare_digest(token, auth_key):
            return None
    return web.json_response({"error": "Invalid or missing auth token"}, status=401)


# ── Handlers ────────────────────────────────────────────────────────────────


def _make_handlers(runner: "GatewayRunner", auth_key: Optional[str]):
    """Build closure-captured handlers bound to a specific runner + auth key.

    The runner is captured at registration time so handlers don't depend on a
    module-level global (which made testing ``myah_management`` ugly).
    """

    async def get_session_override(request: "web.Request") -> "web.Response":
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        session_key = request.match_info["session_key"]
        override = runner.get_session_override(session_key)
        return web.json_response({"override": override})

    async def put_session_override(request: "web.Request") -> "web.Response":
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        session_key = request.match_info["session_key"]
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "Body must be an object"}, status=400)
        # The override dict shape is whatever ``GatewayRunner.SessionOverride``
        # accepts — typically {model, provider, base_url?}. Pass through.
        try:
            runner.set_session_override(session_key, body)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("[myah-admin] set_session_override failed")
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response({"ok": True, "session_key": session_key})

    async def delete_session_override(request: "web.Request") -> "web.Response":
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        session_key = request.match_info["session_key"]
        # No public API for "remove"; setting to empty dict is the closest we
        # can do without exposing internals. The convention is that an empty
        # override means "no override active".
        runner.set_session_override(session_key, {})  # type: ignore[arg-type]
        return web.json_response({"ok": True})

    async def get_active_sessions(request: "web.Request") -> "web.Response":
        """List the keys of sessions with an in-flight run.

        Used by the dashboard plugin's gateway-restart endpoint to enforce a
        busy-check before issuing ``supervisorctl restart hermes``.
        """
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        keys = list(runner.iter_running_session_keys())
        return web.json_response({"active_session_keys": keys, "count": len(keys)})

    async def evict_all_caches(request: "web.Request") -> "web.Response":
        """Evict every cached agent.

        Called by the plugin after writes that change agent assembly (global
        model change, MCP add/remove, toolset toggle). Idempotent.
        """
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        evicted = 0
        for key in list(runner.iter_cached_session_keys()):
            if runner.evict_session_agent(key):
                evicted += 1
        return web.json_response({"ok": True, "evicted": evicted})

    async def evict_session_cache(request: "web.Request") -> "web.Response":
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        session_key = request.match_info["session_key"]
        evicted = runner.evict_session_agent(session_key)
        return web.json_response({"ok": True, "evicted": evicted})

    async def reload_mcp(request: "web.Request") -> "web.Response":
        """Re-read MCP servers from config.yaml and re-register them.

        Called by the plugin after writing to ``mcp_servers`` in config.yaml.
        """
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        try:
            from agent.mcp_registry import register_mcp_servers
        except Exception:  # pragma: no cover — module path may shift upstream
            logger.exception("[myah-admin] failed to import register_mcp_servers")
            return web.json_response(
                {"error": "MCP registry module not available"}, status=500
            )
        try:
            register_mcp_servers()
        except Exception as exc:
            logger.exception("[myah-admin] register_mcp_servers failed")
            return web.json_response({"error": str(exc)}, status=500)
        # Evict caches so next message picks up the new toolset.
        evicted = 0
        for key in list(runner.iter_cached_session_keys()):
            if runner.evict_session_agent(key):
                evicted += 1
        return web.json_response({"ok": True, "evicted": evicted})

    async def disconnect_mcp(request: "web.Request") -> "web.Response":
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        name = request.match_info["name"]
        try:
            from agent.mcp_registry import disconnect_mcp_server
        except Exception:  # pragma: no cover
            logger.exception("[myah-admin] failed to import disconnect_mcp_server")
            return web.json_response(
                {"error": "MCP registry module not available"}, status=500
            )
        try:
            disconnect_mcp_server(name)
        except Exception as exc:
            logger.exception("[myah-admin] disconnect_mcp_server failed")
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response({"ok": True, "name": name})

    async def gateway_restart(request: "web.Request") -> "web.Response":
        """Busy-check + ``supervisorctl restart hermes``.

        Returns 409 if any session has an in-flight run. The frontend can
        retry after the run completes. Equivalent to the legacy
        ``handle_gateway_restart`` endpoint, kept on the gateway because the
        busy check requires runner state.
        """
        if (resp := _check_auth(request, auth_key)) is not None:
            return resp
        active = list(runner.iter_running_session_keys())
        if active:
            return web.json_response(
                {
                    "error": "Cannot restart while runs are in flight",
                    "active_session_keys": active,
                },
                status=409,
            )
        supervisorctl = shutil.which("supervisorctl")
        if not supervisorctl:
            return web.json_response(
                {"error": "supervisorctl not available in container"}, status=503
            )
        try:
            result = subprocess.run(
                [supervisorctl, "restart", "hermes"],
                capture_output=True,
                timeout=30,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return web.json_response({"error": "Restart timed out"}, status=504)
        if result.returncode != 0:
            return web.json_response(
                {
                    "error": "supervisorctl restart failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                status=500,
            )
        return web.json_response({"ok": True, "stdout": result.stdout})

    return {
        "get_session_override": get_session_override,
        "put_session_override": put_session_override,
        "delete_session_override": delete_session_override,
        "get_active_sessions": get_active_sessions,
        "evict_all_caches": evict_all_caches,
        "evict_session_cache": evict_session_cache,
        "reload_mcp": reload_mcp,
        "disconnect_mcp": disconnect_mcp,
        "gateway_restart": gateway_restart,
    }


# ── Public registrar ────────────────────────────────────────────────────────


def register_runtime_admin_routes(
    app: "web.Application",
    *,
    runner: "GatewayRunner",
    auth_key: Optional[str],
) -> None:
    """Add ``/myah/v1/admin/*`` routes to the shared aiohttp app.

    Called from ``MyahAdapter._register_routes_on_app`` as part of the
    pre-setup hook (i.e. before the router is frozen).
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for runtime admin routes")

    handlers = _make_handlers(runner, auth_key)

    app.router.add_get(
        "/myah/v1/admin/sessions/{session_key}/override",
        handlers["get_session_override"],
    )
    app.router.add_put(
        "/myah/v1/admin/sessions/{session_key}/override",
        handlers["put_session_override"],
    )
    app.router.add_delete(
        "/myah/v1/admin/sessions/{session_key}/override",
        handlers["delete_session_override"],
    )
    app.router.add_get(
        "/myah/v1/admin/sessions/active",
        handlers["get_active_sessions"],
    )
    app.router.add_post(
        "/myah/v1/admin/cache/evict-all",
        handlers["evict_all_caches"],
    )
    app.router.add_post(
        "/myah/v1/admin/cache/evict/{session_key}",
        handlers["evict_session_cache"],
    )
    app.router.add_post(
        "/myah/v1/admin/mcp/refresh",
        handlers["reload_mcp"],
    )
    app.router.add_post(
        "/myah/v1/admin/mcp/disconnect/{name}",
        handlers["disconnect_mcp"],
    )
    app.router.add_post(
        "/myah/v1/admin/gateway/restart",
        handlers["gateway_restart"],
    )

    logger.info(
        "[myah-admin] runtime-control routes registered (9 endpoints under /myah/v1/admin/)"
    )
