"""Myah platform plugin entry point.

This module is the target of the `hermes_agent.plugins` entry-point.
Hermes calls `register(ctx)` at startup to wire up Myah-specific
tools/hooks/platform adapters.

Filled in during Phase 4d (platform adapter), 4c (tools), 4e (admin).
"""

from typing import Any


def register(ctx: Any) -> None:
    """Register Myah platform extensions with the Hermes runtime.

    Phase 4d will call ctx.register_platform(...).
    Phase 4c will call ctx.register_tool(...) for secrets_tool.
    Phase 4f will call ctx.register_hook(...) for status_hint, boot_md, cron.
    """
    pass  # Skeleton — populated by Phases 4c-4f
