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

from ..myah_tools import secrets_tool

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
       ``ctx.register_platform(...)`` with capability fields previously
       hardcoded in core (``allowed_users_env``, ``platform_hint``,
       ``connect_last``, ``skip_user_authorization``,
       ``skip_home_channel_prompt``, ``default_toolset``).
    3. **Phase 4f (TODO)**: cron status_hint plumbing, boot_md hook,
       offline cron delivery — all currently no-ops in upstream Hermes.

    Phase 4d (TODO follow-up): the secrets tool's ``request`` action
    needs the platform adapter to wire a session-keyed
    ``secrets_request`` callback. The adapter already provides the
    HTTP transport (``POST /myah/v1/secret/{stream_id}``). The
    callback registration belongs here and will land in a follow-up
    PR alongside the cron hooks.
    """
    # ── Secrets tool registration (Phase 4c) ───────────────────────────
    ctx.register_tool(
        name="secrets",
        toolset="hermes-myah",
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
        # Capability fields (Phase 4d): replace previously hardcoded
        # Myah-specific behaviour in core.
        allowed_users_env="MYAH_ALLOWED_USERS",
        allow_all_env="MYAH_ALLOW_ALL_USERS",
        platform_hint=_MYAH_PLATFORM_HINT,
        default_toolset="hermes-myah",
        skip_user_authorization=True,  # Myah handles auth via Open WebUI / single-tenant
        skip_home_channel_prompt=True,  # Web DMs don't use home-channel semantics
        connect_last=True,  # Connect after API_SERVER so its pre-setup hook fires
    )
