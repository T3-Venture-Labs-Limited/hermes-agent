"""Runtime extensions to vanilla upstream Hermes.

Most Myah plugin behaviour rides on Hermes' supported plugin context
APIs (``ctx.register_tool``, ``ctx.register_platform``,
``ctx.register_hook``). For a small number of features vanilla upstream
does not expose a clean extension surface — when that's the case, the
helpers here interact with upstream's ``private`` module state directly
in the spirit of "normal Python" (calling a method, reading an
attribute), NEVER by modifying core ``.py`` files on disk.

Per ``agent/hermes/AGENTS.md:Rule (Teknium, May 2026)``:

    > plugins MUST NOT modify core files (run_agent.py, cli.py,
    > gateway/run.py, hermes_cli/main.py, etc.). If a plugin needs a
    > capability the framework doesn't expose, expand the generic
    > plugin surface (new hook, new ctx method) — never hardcode
    > plugin-specific logic into core.

This package interprets the rule as referring to **modifying the
``.py`` files on disk** — not to reading or writing instance attributes
on objects upstream hands the plugin. The plugin's existing
``adapter.py:get_session_override_direct`` etc. (Tier 2B.0) already
access ``GatewayRunner._session_model_overrides`` directly using the
same pattern; this package collects the few remaining cases.

Modules:

- :mod:`mcp_disconnect` — F7 per-server MCP teardown without
  bouncing the whole gateway. Direct access to
  ``tools.mcp_tool._servers`` / ``_lock`` / ``_run_on_mcp_loop``.
"""

from . import mcp_disconnect  # noqa: F401  (re-exported as `from runtime_extensions import mcp_disconnect`)


__all__ = ["mcp_disconnect"]
