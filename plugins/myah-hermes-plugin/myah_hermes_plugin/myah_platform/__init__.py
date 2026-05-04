"""Myah platform plugin entry point.

Registers Myah-specific tools and hooks. Phase 4d will add the platform
adapter via ctx.register_platform().
"""

from typing import Any

from ..myah_tools import secrets_tool


def register(ctx: Any) -> None:
    """Register Myah platform extensions with the Hermes runtime.

    Phase 4c: registers secrets_tool under the ``hermes-myah`` toolset.
    The session-key contextvar wiring + secrets-request callback registration
    are deferred to Phase 4d (platform adapter), where they can be installed
    against the right session at the right point in the agent dispatch path.
    Until Phase 4d lands, the secrets tool's ``request`` action will silently
    no-op (the tool reports "Secure secret entry is not available in this
    surface."). ``list`` / ``check`` / ``delete`` / ``inject`` work normally.

    Phase 4d (TODO): adds ctx.register_platform(...) for the Myah adapter.
    Phase 4f (TODO): adds ctx.register_hook(...) for status_hint, boot_md, cron.
    """
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
