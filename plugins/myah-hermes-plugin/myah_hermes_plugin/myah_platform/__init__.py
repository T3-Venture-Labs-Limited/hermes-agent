"""Myah platform plugin entry point.

Registers Myah-specific tools, hooks, and platform adapter.

This is the canonical Hermes plugin entry point — declared in
``pyproject.toml`` as ``hermes_agent.plugins -> myah-platform``. Hermes'
``PluginManager`` calls :func:`register` once at startup with a
``PluginContext`` instance.

Phase 4d (2026-05-04) added the ``ctx.register_platform(...)`` wiring that
moves the Myah platform adapter out of upstream Hermes core into this
plugin. Earlier phases (4b/4c) bootstrapped the package skeleton and the
secrets tool. Phase 4f will follow with cron/status_hint/boot_md hooks.
"""

from typing import Any

from .. import sentry_init
from ..myah_tools import secrets_tool
from .pre_dispatch_hook import myah_pre_gateway_dispatch

# Platform-hint string injected by ``agent.prompt_builder.get_platform_hint``
# when the agent is running on the Myah platform. This text used to live
# inside core's ``PLATFORM_HINTS`` dict; Phase 4d moved it onto the
# ``PlatformEntry`` so the plugin owns its own prompt copy.
_MYAH_PLATFORM_HINT = (
    "User is interacting via the Myah web platform. "
    "Full markdown rendering is supported including code blocks, tables, "
    "images (via URL), and links. The user can see tool call progress "
    "and reasoning in real time."
)


def _validate_myah_config(config: Any) -> bool:
    """Reject empty/invalid Myah configurations early in adapter creation.

    Currently a permissive check — the adapter handles missing auth keys
    by serving requests unauthenticated (single-tenant local dev path).
    Returns True so the registry always proceeds to instantiation; the
    adapter itself surfaces specific errors if it cannot start.
    """
    return True


def register(ctx: Any) -> None:
    """Register Myah platform extensions with the Hermes runtime.

    Wires three things:

    1. **Secrets tool** (Phase 4c): the ``secrets`` tool under the
       ``hermes-myah`` toolset.
    2. **Platform adapter** (Phase 4d): the Myah web platform via
       ``ctx.register_platform(...)`` with capability fields supported by
       upstream's ``PlatformEntry`` (``allowed_users_env``,
       ``allow_all_env``, ``platform_hint``).
    3. **Phase 4f (TODO)**: cron status_hint plumbing, boot_md hook,
       offline cron delivery — all currently no-ops in upstream Hermes.

    Phase 4d (TODO follow-up): the secrets tool's ``request`` action
    needs the platform adapter to wire a session-keyed
    ``secrets_request`` callback. The adapter already provides the
    HTTP transport (``POST /myah/v1/secret/{stream_id}``). The
    callback registration belongs here and will land in a follow-up
    PR alongside the cron hooks.
    """
    # ── Sentry / TelemetryHook initialization (Tier 2A Task 2A.7) ──────
    # Idempotent — silently no-ops when SENTRY_DSN_AGENT is unset (the
    # OSS-user case). For hosted Myah agent containers SENTRY_DSN_AGENT
    # is injected at spawn time and this call wires up Sentry SDK +
    # registers the SentryHook adapter so Hermes runtime telemetry calls
    # (which only see agent.telemetry.TelemetryHook) route through it.
    sentry_init.setup_sentry()

    # ── pre_gateway_dispatch hook (Tier 2A Task 2A.4) ──────────────────
    # Replaces skip_user_authorization semantics that PR #20 removed.
    # Currently a no-op-allow for Myah-platform messages; reserved as an
    # extension point for future Myah-specific routing logic that must
    # NOT live in upstream gateway/run.py.
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_gateway_dispatch", myah_pre_gateway_dispatch)

    # ── Secrets tool registration (Phase 4c) ───────────────────────────
    ctx.register_tool(
        name="secrets",
        toolset="secrets",
        schema=secrets_tool.SCHEMA,
        handler=secrets_tool.handle,
        emoji="🔐",
        description=(
            "Securely manage API keys and other credentials without exposing "
            "values to the model."
        ),
    )

    # ── Platform adapter registration (Phase 4d) ───────────────────────
    # Local imports avoid pulling aiohttp at module-import time so the
    # plugin still loads cleanly when aiohttp is missing (the adapter's
    # check_fn handles that case below).
    from .adapter import MyahAdapter, check_myah_requirements

    ctx.register_platform(
        name="myah",
        label="🌐 Myah",
        adapter_factory=lambda cfg: MyahAdapter(cfg),
        check_fn=check_myah_requirements,
        validate_config=_validate_myah_config,
        required_env=["MYAH_ADAPTER_AUTH_KEY"],
        install_hint="pip install aiohttp",
        # Capability fields supported by upstream's PlatformEntry. Auth
        # bypass for the platform's authenticated upstream users flows
        # through ``allow_all_env`` (set via ``MYAH_ALLOW_ALL_USERS=true``
        # in the agent container's entrypoint).
        allowed_users_env="MYAH_ALLOWED_USERS",
        allow_all_env="MYAH_ALLOW_ALL_USERS",
        platform_hint=_MYAH_PLATFORM_HINT,
    )

    # ── PLATFORMS bridge (Tier 2C Issue 2 — workaround) ──────────────────
    # Phase 4d moved the Myah platform out of upstream's static
    # hermes_cli/platforms.py registry. Code paths in upstream's
    # hermes_cli/tools_config.py do `PLATFORMS["myah"]` direct lookup or
    # iterate `PLATFORMS.values()` — those don't see plugin-registered
    # platforms.
    #
    # WORKAROUND: mutate tools_config.PLATFORMS at register time so direct
    # lookups succeed. This is fragile (relies on tools_config.PLATFORMS
    # being a plain dict; if upstream changes it to a derived view, this
    # silently stops working). Long-term fix is U-PLAT upstream PR
    # (deferred per spec §5). Sentinel test in
    # plugins/myah-hermes-plugin/tests/test_myah_platform_bridge.py
    # catches the dict-type change.
    try:
        import hermes_cli.tools_config as _tc
        _tc.PLATFORMS["myah"] = {
            "label": "Myah",
            "default_toolset": "hermes-myah",
        }
    except ImportError:
        # tools_config is CLI-only; in pure-gateway runtime, skip silently.
        pass
    # ────────────────────────────────────────────────────────────────────
