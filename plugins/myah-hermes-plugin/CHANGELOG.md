# Changelog

All notable changes to `myah-hermes-plugin` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-08

First OSS-launch-eligible release. Tier 2C of the Myah OSS Completion epic.

### Compatibility

- **hermes-agent**: SHA-pinned to upstream commit
  `faa13e49f81480771ceeb55991bb0c27edf1a5fb` (Hermes-Agent v0.11-track,
  fetched 2026-05-08 from `NousResearch/Hermes-Agent@main`).
- **Verification:** Mode D litmus test (Tier 2A Task 2A.8) — 9/9 passing
  on stock upstream + plugin (F5/BOOT.md is deferred per spec §3.1 and
  excluded from the Mode D matrix).
- **Python:** ≥ 3.11.
- **aiohttp:** ≥ 3.9, < 4.0.

When `hermes-agent` ships to PyPI, the SHA pin becomes a semver pin
(`hermes-agent>=0.11,<0.12`) — see `pyproject.toml` for the canonical
declaration.

### Vendored upstream features

The plugin vendors the following Myah-platform-specific features that
do not yet exist upstream. Each will be removed as the corresponding
upstream PR (designed in spec §5) merges:

- **F1 — Cron approval card UI flow** (~322 LOC):
  `myah_hermes_plugin.cron_approval` (vendored from upstream
  `tools/approval.py:request_action_confirmation` + dispatcher);
  `myah_hermes_plugin.myah_tools.cron_tool` (shadows upstream's
  `tools/cronjob_tools.py` to import the vendored confirmation
  primitive). Removed when upstream PR U5 lands.
- **F2 — Provider catalog** (Myah V1 picker):
  `myah_hermes_plugin.myah_admin.myah_overrides`. No upstream PR planned
  (data-only, no generic value).
- **F3 — Telemetry hook protocol** (Sentry breadcrumbs, AI monitoring):
  `myah_hermes_plugin.myah_platform.adapter`'s telemetry wiring + plugin
  `register()` Sentry init. Removed when upstream PR U1 lands.
- **F4 — Session-keyed secret capture**:
  `myah_hermes_plugin.myah_tools.secrets_tool`.
  No upstream PR planned for v1.0.0 (revisit if Nous expresses interest).
- **F6 — Cron→Myah delivery metadata enrichment**:
  `MyahAdapter.build_delivery_metadata` (override of polymorphic
  `BasePlatformAdapter.build_delivery_metadata` shipped to fork in
  Tier 2B Task 2B.4 — same diff queued as upstream PR U-CRON).
- **F7 — MCP per-server disconnect**:
  `tools.mcp_tool.disconnect_mcp_server` (fork-side; same diff queued
  as upstream PR U-MCP).

### Deferred from this release

- **F5 — BOOT.md startup hook**: requires upstream
  `register_gateway_event_hook` (PR U-HOOK in spec §5). OSS users on
  stock+plugin do **not** get BOOT.md until U-HOOK merges. Mode D test
  matrix excludes the F5 row per spec §3.1.

### Architectural notes

- Plugin runs in **standalone-mode adapter** on `MYAH_GATEWAY_PORT`
  (default `8643`). One-way door per spec Tier 2A Task 2A.3 — hosted
  Myah keeps standalone mode permanently even if upstream PR U2
  (`register_pre_setup_hook`) merges later.
- Plugin uses **direct attribute access** against upstream-native private
  dicts (`_session_model_overrides`, `_agent_cache`, etc.) per spec
  §3.2.1's 2026-05-07-evening discovery, NOT the v2 plan's plugin-local
  vendored dicts. This unblocked Tier 2B without depending on upstream
  PRs U4 / U-OVERRIDE (both downgraded to "optional future robustness").
- A CI guard test (`tests/test_upstream_runner_attrs_present.py`) asserts
  the upstream private attrs exist; if a future upstream rename breaks
  the plugin, CI flags it loudly before deploy.

### Distribution

- The plugin is shipped to OSS users via `pip install` (from PyPI when
  available, or `pip install <local-source>` from the fork's
  `plugins/myah-hermes-plugin/` directory).
- Hosted Myah's stock+plugin agent image (`agent/Dockerfile.stock`)
  installs the plugin from local source — see `myah` parent repo
  `agent/Dockerfile.stock` for the canonical image build.
- The dashboard plugin (`myah_admin/`) is materialized at image build
  time via `myah-hermes-plugin install --dashboard-only --target
  /opt/myah/plugins/`. Hermes' filesystem-discovery loader picks it up
  on container start. Image SHA = plugin version; atomic rollback.

## [0.3.0] — 2026-05-07 (internal-only, Tier 2B)

- Tier 2B Task 2B.3: migrated `agent/hermes/plugins/myah-admin/` into
  `myah_hermes_plugin.myah_admin/` (Phase 4e) with a new
  `myah-hermes-plugin install --dashboard-only` console script.
- Tier 2B Task 2B.4: shipped polymorphic
  `BasePlatformAdapter.build_delivery_metadata` to the fork +
  `MyahAdapter.build_delivery_metadata` override (Phase 4f); deletes
  `cron/scheduler.py`'s hardcoded `if platform_name == "myah"` branch.
- Tier 2B Task 2B.0: replaced 19 plugin callsites of fork-only
  `GatewayRunner` methods with direct attribute access against
  upstream-native private dicts; deletes 8 fork-only methods +
  `SessionOverride` TypedDict.

## [0.2.0] — 2026-04-28 (internal-only, Phase 4d)

- Phase 4d: moved `gateway/platforms/myah.py` adapter from the fork
  into the plugin via `ctx.register_platform()`.
- Phase 4c: moved `tools/secrets_tool.py` into the plugin.

## [0.1.0] — 2026-04-21 (internal-only, Phase 4b)

- Phase 4b: empty skeleton, pip-installable, `hermes_agent.plugins`
  entry point registered.
