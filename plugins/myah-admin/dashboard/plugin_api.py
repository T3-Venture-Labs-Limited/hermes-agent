"""Myah-admin plugin — backend API routes.

Mounted at /api/plugins/myah-admin/ by the dashboard plugin system inside
the per-user Hermes container. The platform backend reaches these routes
through the loopback `hermes web` server using the per-container session
token (see platform/backend/open_webui/utils/hermes_web.py).

Phase 0 only ships a /health endpoint so the platform can verify the
plugin is reachable end-to-end. Real admin endpoints land in later phases.

On module import, also register Myah's :class:`SentryHook` against the
``agent.telemetry`` hook registry so Hermes runtime telemetry routes
through the Sentry SDK that the agent container's
``logging_setup.setup_sentry()`` initialized.  Registration is
idempotent and silent when ``sentry_sdk`` is unavailable.
"""

from fastapi import APIRouter

# Import-time side effect: register the SentryHook as the process-wide
# telemetry hook.  Safe even if the agent container's logging_setup has
# already registered its own hook — the last writer wins, and both
# adapters delegate to the same sentry_sdk module.
try:
    from plugins.myah_admin.myah_hook import register_sentry_hook
    register_sentry_hook()
except ImportError:
    # The plugin loader may import this module before the
    # ``plugins.myah_admin`` package is on sys.path (e.g. when the
    # plugin lives at ``plugins/myah-admin/`` and Python sees the
    # hyphen).  Fall back to a relative-style import of the sibling
    # module so registration still happens.
    try:
        import importlib
        import os
        _hook_path = os.path.join(os.path.dirname(__file__), '..', 'myah_hook.py')
        _spec = importlib.util.spec_from_file_location('_myah_admin_hook', _hook_path)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.register_sentry_hook()
    except Exception:
        # Telemetry registration is best-effort — never block the
        # plugin from loading.
        pass

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Liveness probe for the platform's hermes_web client."""
    return {"status": "ok", "plugin": "myah-admin"}
