"""Skills, plugins, MCP servers, and toolset write-side handlers.

Ported from the legacy ``gateway/platforms/myah_management.py`` (aiohttp,
in-process inside the gateway) to FastAPI inside the ``myah-admin``
dashboard plugin (port 9119, separate process from the gateway).

Why a separate file from ``plugin_api.py``:
    The legacy admin surface is being decomposed into themed sub-modules
    so each one stays small, testable, and reviewable. ``plugin_api.py``
    keeps responsibility for plugin lifecycle (Sentry hook registration,
    health probe). This module owns the file-system CRUD for skills,
    plugins, and MCP servers, plus the toolset toggle that ``hermes
    tools enable/disable`` writes to ``config.yaml``.

Process boundary:
    The dashboard plugin runs in the ``hermes dashboard`` process. The
    handlers below NEVER call ``GatewayRunner`` methods directly — that
    would cross processes. Instead, after writes that mutate gateway
    state (MCP add/remove, toolset toggle), they delegate to the
    runtime-control admin surface at
    ``http://localhost:{API_SERVER_PORT}/myah/v1/admin/*`` via
    :class:`GatewayClient` from ``_common``. The gateway-side handlers
    take care of MCP registry refresh and agent cache eviction.

Cache-eviction semantics (verified against
``gateway/platforms/myah_runtime_admin.py``):
    - ``POST /mcp/refresh``      — already evicts all cached agents.
                                   Caller does NOT need a separate
                                   ``/cache/evict-all`` call.
    - ``POST /mcp/disconnect/{name}`` — does NOT evict caches. Caller
                                   MUST follow up with
                                   ``/cache/evict-all``.
    - ``hermes tools enable/disable`` — does NOT touch the runner. We
                                   evict via ``/cache/evict-all`` so the
                                   next message rebuilds the agent with
                                   the new toolset selection.

Import strategy:
    The dashboard plugin loader (``hermes_cli/web_server.py::
    _mount_plugin_api_routes``) imports plugin api files via
    ``importlib.util.spec_from_file_location``, which strips the parent
    package context — so relative imports (``from ._common import ...``)
    cannot work, and the package-qualified name
    ``plugins.myah-admin.dashboard._common`` is not importable because
    of the hyphen in ``myah-admin``.

    We therefore load ``_common.py`` from disk via
    ``spec_from_file_location`` ourselves. This mirrors how
    ``plugin_api.py`` loads ``../myah_hook.py``. The cost is one extra
    import-time stat call; the benefit is that the module is loadable
    in three contexts: (1) the live dashboard process, (2) the test
    suite when imported via the same file-based shim, and (3) directly
    via ``importlib.util.spec_from_file_location`` for ad-hoc scripts.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Import the sibling _common module without a package context ─────────────


def _load_common():
    """Load the sibling ``_common.py`` via file path.

    The dashboard plugin loader registers our parent file as
    ``hermes_dashboard_plugin_myah-admin`` in ``sys.modules`` but does
    not establish a package, so ``from ._common import ...`` fails with
    ``ImportError: attempted relative import with no known parent
    package``. We resolve the file directly.
    """
    here = Path(__file__).resolve().parent
    common_path = here / "_common.py"
    spec = importlib.util.spec_from_file_location(
        "myah_admin_dashboard_common", common_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {common_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_common()
require_session_token = _common.require_session_token
gateway_client = _common.gateway_client
hermes_home_str = _common.hermes_home


def _hermes_home() -> Path:
    """Return the active ``HERMES_HOME`` as a ``Path``.

    ``_common.hermes_home`` returns a ``str`` to keep the type consistent
    with ``hermes_constants.get_hermes_home`` callers that print it; we
    want a ``Path`` for filesystem ops.
    """
    return Path(hermes_home_str())


# ── Constants & helpers ─────────────────────────────────────────────────────


_NAME_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _validate_name(name: str, *, field: str = 'name') -> None:
    """Raise ``HTTPException`` 422 when ``name`` is invalid.

    Mirrors the legacy ``_safe_name`` regex (alphanumeric + ``-`` + ``_``).
    Rejects path-traversal sequences (``..``, ``/``, ``\\``) implicitly
    because they fail the regex.
    """
    if not name or not _NAME_RE.match(name):
        # 422 — Unprocessable Content. Numeric literal avoids starlette's
        # ``HTTP_422_UNPROCESSABLE_ENTITY``/``_CONTENT`` deprecation churn.
        raise HTTPException(
            status_code=422,
            detail=f'{field} must be alphanumeric with hyphens/underscores only',
        )


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from a SKILL.md file.

    Copied verbatim from
    ``gateway/platforms/myah_management.py::_parse_frontmatter`` (line 1471).
    Kept inline rather than imported because that module is being
    deleted as part of the hermes-first cleanup; the dashboard plugin
    must not depend on its continued existence.
    """
    fm: dict = {}
    m = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ': ' in line:
                k, v = line.split(': ', 1)
                fm[k.strip()] = v.strip().strip('"\'')
    return fm


async def _async_subprocess(*cmd: str, timeout: float = 10) -> tuple[int, str, str]:
    """Run a subprocess without blocking the event loop.

    Mirrors ``gateway/platforms/myah_management.py::_async_subprocess``
    (line 77). Returns ``(returncode, stdout, stderr)``.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode or 0, stdout.decode(), stderr.decode()


def _schedule_restart() -> None:
    """Schedule a deferred ``supervisorctl restart hermes``.

    Copied from
    ``gateway/platforms/myah_management.py::_schedule_restart`` (line 1483).
    The 2 s delay lets the HTTP response reach the client before the
    process dies; without it the client sees a connection reset.
    """
    def _do_restart() -> None:
        try:
            subprocess.run(
                ['supervisorctl', 'restart', 'hermes'],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.error(f'Failed to restart hermes: {e}')

    try:
        loop = asyncio.get_event_loop()
        loop.call_later(
            2.0,
            lambda: asyncio.ensure_future(asyncio.to_thread(_do_restart)),
        )
    except RuntimeError:
        # No running event loop — run synchronously as fallback.
        _do_restart()


# ── Pydantic request bodies ─────────────────────────────────────────────────


class CreateSkillBody(BaseModel):
    name: str
    category: str = 'general'
    content: str


class UpdateSkillBody(BaseModel):
    content: str


class CreatePluginBody(BaseModel):
    name: str
    content: str


class UpdatePluginBody(BaseModel):
    content: str


class AddMCPBody(BaseModel):
    name: str
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    api_key: str | None = None


class ToggleToolsetBody(BaseModel):
    enabled: bool = True


# ── Router ──────────────────────────────────────────────────────────────────


router = APIRouter(dependencies=[Depends(require_session_token)])


# ── Skills ──────────────────────────────────────────────────────────────────
#
# The legacy GET /skills (list) is intentionally NOT ported — Hermes ships
# its own ``GET /api/skills`` endpoint that lists skills uniformly across
# CLI / dashboard / gateway. The legacy single-skill GET has no Hermes
# equivalent and is ported below.


@router.get('/skills/{name}')
async def get_skill(name: str) -> dict[str, Any]:
    """Return the full SKILL.md content for ``name``."""
    _validate_name(name)
    skills_dir = _hermes_home() / 'skills'
    if not skills_dir.exists():
        raise HTTPException(status_code=404, detail='Skill not found')
    for skill_md in skills_dir.rglob('SKILL.md'):
        fm = _parse_frontmatter(skill_md.read_text())
        if fm.get('name', skill_md.parent.name) == name:
            category = (
                skill_md.parent.parent.name
                if skill_md.parent.parent != skills_dir
                else 'general'
            )
            return {
                'name': name,
                'category': category,
                'content': skill_md.read_text(),
            }
    raise HTTPException(status_code=404, detail='Skill not found')


@router.post('/skills', status_code=201)
async def create_skill(body: CreateSkillBody) -> dict[str, Any]:
    """Create a new skill at ``$HERMES_HOME/skills/{category}/{name}/SKILL.md``."""
    _validate_name(body.name, field='name')
    _validate_name(body.category, field='category')
    if not body.content.strip():
        raise HTTPException(status_code=400, detail='content is required')

    skill_dir = _hermes_home() / 'skills' / body.category / body.name
    skill_path = skill_dir / 'SKILL.md'
    if skill_path.exists():
        raise HTTPException(status_code=409, detail='Skill already exists')

    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(body.content)

    return {
        'name': body.name,
        'category': body.category,
        'content': body.content,
    }


@router.put('/skills/{name}')
async def update_skill(name: str, body: UpdateSkillBody) -> dict[str, Any]:
    """Update an existing skill's SKILL.md content."""
    _validate_name(name)
    if not body.content.strip():
        raise HTTPException(status_code=400, detail='content is required')

    skills_dir = _hermes_home() / 'skills'
    if not skills_dir.exists():
        raise HTTPException(status_code=404, detail='Skill not found')
    for skill_md in skills_dir.rglob('SKILL.md'):
        fm = _parse_frontmatter(skill_md.read_text())
        if fm.get('name', skill_md.parent.name) == name:
            skill_md.write_text(body.content)
            return {'name': name, 'content': body.content}
    raise HTTPException(status_code=404, detail='Skill not found')


@router.delete('/skills/{name}')
async def delete_skill(name: str) -> dict[str, Any]:
    """Delete a skill's directory tree."""
    _validate_name(name)
    skills_dir = _hermes_home() / 'skills'
    if not skills_dir.exists():
        raise HTTPException(status_code=404, detail='Skill not found')
    for skill_md in skills_dir.rglob('SKILL.md'):
        fm = _parse_frontmatter(skill_md.read_text())
        if fm.get('name', skill_md.parent.name) == name:
            shutil.rmtree(skill_md.parent, ignore_errors=True)
            return {'ok': True}
    raise HTTPException(status_code=404, detail='Skill not found')


# ── Plugins ─────────────────────────────────────────────────────────────────


@router.get('/plugins')
async def list_plugins() -> list[dict[str, Any]]:
    """List user plugins from ``$HERMES_HOME/plugins/*.py``."""
    plugins_dir = _hermes_home() / 'plugins'
    if not plugins_dir.exists():
        return []

    result: list[dict[str, Any]] = []
    for f in sorted(plugins_dir.glob('*.py')):
        if f.name.startswith('_'):
            continue
        text = f.read_text()
        description = ''
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                description = stripped.lstrip('# ').rstrip()
                break
        result.append({
            'filename': f.name,
            'name': f.stem,
            'description': description[:200],
            'content': text,
        })
    return result


@router.post('/plugins', status_code=201)
async def create_plugin(body: CreatePluginBody) -> dict[str, Any]:
    """Create a new plugin file. Schedules a gateway restart."""
    _validate_name(body.name)
    if not body.content.strip():
        raise HTTPException(status_code=400, detail='content is required')

    try:
        compile(body.content, f'{body.name}.py', 'exec')
    except SyntaxError as e:
        # Legacy returned 422; preserve that contract for the frontend.
        raise HTTPException(
            status_code=422, detail=f'Python syntax error: {e}',
        ) from e

    plugins_dir = _hermes_home() / 'plugins'
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_path = plugins_dir / f'{body.name}.py'
    if plugin_path.exists():
        raise HTTPException(status_code=409, detail='Plugin already exists')

    plugin_path.write_text(body.content)
    _schedule_restart()

    return {
        'filename': f'{body.name}.py',
        'name': body.name,
        'content': body.content,
    }


@router.put('/plugins/{name}')
async def update_plugin(name: str, body: UpdatePluginBody) -> dict[str, Any]:
    """Update an existing plugin file. Schedules a gateway restart."""
    _validate_name(name)
    if not body.content.strip():
        raise HTTPException(status_code=400, detail='content is required')

    try:
        compile(body.content, f'{name}.py', 'exec')
    except SyntaxError as e:
        raise HTTPException(
            status_code=422, detail=f'Python syntax error: {e}',
        ) from e

    plugin_path = _hermes_home() / 'plugins' / f'{name}.py'
    if not plugin_path.exists():
        raise HTTPException(status_code=404, detail='Plugin not found')

    plugin_path.write_text(body.content)
    _schedule_restart()

    return {'name': name, 'content': body.content}


@router.delete('/plugins/{name}')
async def delete_plugin(name: str) -> dict[str, Any]:
    """Delete a plugin file. Schedules a gateway restart."""
    _validate_name(name)
    plugin_path = _hermes_home() / 'plugins' / f'{name}.py'
    if not plugin_path.exists():
        raise HTTPException(status_code=404, detail='Plugin not found')

    plugin_path.unlink()
    _schedule_restart()

    return {'ok': True}


# ── MCP Servers ─────────────────────────────────────────────────────────────


@router.get('/mcp')
async def list_mcp() -> list[dict[str, Any]]:
    """List MCP servers from ``$HERMES_HOME/config.yaml``."""
    config_path = _hermes_home() / 'config.yaml'
    if not config_path.exists():
        return []
    cfg = yaml.safe_load(config_path.read_text()) or {}
    servers = cfg.get('mcp_servers', {}) or {}
    return [
        {
            'name': k,
            'url': v.get('url') if isinstance(v, dict) else None,
            'command': v.get('command') if isinstance(v, dict) else None,
            'args': v.get('args', []) if isinstance(v, dict) else [],
            'status': 'unknown',
        }
        for k, v in servers.items()
    ]


@router.post('/mcp')
async def add_mcp(body: AddMCPBody) -> dict[str, Any]:
    """Add an MCP server.

    Persists to ``config.yaml``, optionally injects an API key into
    ``$HERMES_HOME/.env`` as ``MCP_<NAME>_API_KEY``, then asks the
    gateway to refresh its in-process MCP registry. The gateway-side
    ``/mcp/refresh`` handler also evicts agent caches so the next
    message picks up the new toolset (verified at
    ``gateway/platforms/myah_runtime_admin.py:144-168``).
    """
    _validate_name(body.name)
    if not body.url and not body.command:
        raise HTTPException(
            status_code=422, detail='Either url or command is required',
        )

    # Optional API key injection.
    if body.api_key:
        env_path = _hermes_home() / '.env'
        env_key = f'MCP_{body.name.upper()}_API_KEY'
        existing = env_path.read_text() if env_path.exists() else ''
        lines = [
            line for line in existing.splitlines()
            if not line.startswith(f'{env_key}=')
        ]
        lines.append(f'{env_key}={body.api_key}')
        env_path.write_text('\n'.join(lines) + '\n')

    # Build server config dict.
    server_cfg: dict[str, Any] = {}
    if body.url:
        server_cfg['url'] = body.url
    elif body.command:
        server_cfg['command'] = body.command
        server_cfg['args'] = body.args
        if body.env:
            server_cfg['env'] = body.env

    config_path = _hermes_home() / 'config.yaml'
    cfg = yaml.safe_load(config_path.read_text()) or {} if config_path.exists() else {}
    if 'mcp_servers' not in cfg or cfg['mcp_servers'] is None:
        cfg['mcp_servers'] = {}
    cfg['mcp_servers'][body.name] = server_cfg
    config_path.write_text(yaml.safe_dump(cfg, default_flow_style=False))

    # Delegate refresh + cache eviction to the gateway. Best-effort: log
    # and continue if the gateway's runtime-control surface is down,
    # because the on-disk write has already succeeded and the next
    # gateway restart will pick it up.
    try:
        await gateway_client.request_or_raise('POST', '/mcp/refresh')
    except HTTPException as exc:
        logger.warning(f'[myah-admin] /mcp/refresh failed for {body.name}: {exc.detail}')

    return {
        'name': body.name,
        'url': body.url,
        'command': body.command,
        'args': body.args,
        'status': 'unknown',
    }


@router.delete('/mcp/{name}')
async def remove_mcp(name: str) -> dict[str, Any]:
    """Remove an MCP server.

    Updates ``config.yaml`` first, then asks the gateway to disconnect
    the in-process client and to evict agent caches. The disconnect
    endpoint does NOT evict caches itself (verified at
    ``gateway/platforms/myah_runtime_admin.py:170-186``), so we follow
    up with ``/cache/evict-all``.
    """
    _validate_name(name)

    config_path = _hermes_home() / 'config.yaml'
    cfg = yaml.safe_load(config_path.read_text()) or {} if config_path.exists() else {}
    servers = cfg.get('mcp_servers') or {}
    if name not in servers:
        raise HTTPException(
            status_code=404, detail=f'MCP server {name!r} not found',
        )

    del servers[name]
    cfg['mcp_servers'] = servers
    config_path.write_text(yaml.safe_dump(cfg, default_flow_style=False))

    # Best-effort gateway-side cleanup. Log and continue on failure.
    try:
        await gateway_client.request_or_raise('POST', f'/mcp/disconnect/{name}')
    except HTTPException as exc:
        logger.warning(
            f'[myah-admin] /mcp/disconnect/{name} failed: {exc.detail}',
        )
    try:
        await gateway_client.request_or_raise('POST', '/cache/evict-all')
    except HTTPException as exc:
        logger.warning(
            f'[myah-admin] /cache/evict-all after removing {name} failed: {exc.detail}',
        )

    return {'ok': True}


# ── Toolsets (write side only) ──────────────────────────────────────────────
#
# The legacy GET /toolsets is intentionally NOT ported — Hermes ships
# ``GET /api/tools/toolsets`` natively. Only the toggle write handler
# moves here.


@router.patch('/toolsets/{name}')
async def toggle_toolset(name: str, body: ToggleToolsetBody) -> dict[str, Any]:
    """Enable or disable a toolset by name.

    Shells out to ``hermes tools enable|disable <name>``, which is the
    canonical writer for ``config.yaml``'s ``disabled_toolsets`` list.
    Then asks the gateway to evict all cached agents so the next
    message rebuilds with the new toolset selection.
    """
    _validate_name(name)
    action = 'enable' if body.enabled else 'disable'

    returncode, _, stderr = await _async_subprocess(
        'hermes', 'tools', action, name, timeout=10,
    )
    if returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f'hermes tools {action} failed: {stderr.strip()}',
        )

    # Force a rebuild on the next message. Best-effort.
    try:
        await gateway_client.request_or_raise('POST', '/cache/evict-all')
    except HTTPException as exc:
        logger.warning(
            f'[myah-admin] /cache/evict-all after toolset toggle failed: {exc.detail}',
        )

    return {'name': name, 'enabled': body.enabled}
