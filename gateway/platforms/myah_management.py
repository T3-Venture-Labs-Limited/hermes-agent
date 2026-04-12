"""
Myah Management API Handlers

In-process REST handlers for managing Hermes agent configuration, skills,
plugins, MCP servers, and toolsets. These run inside the Hermes gateway
process, giving direct access to config files, the skills directory,
the SessionDB, and the toolset registry — without docker exec.

All handlers receive an aiohttp web.Request and return web.Response.
Auth is checked by the caller (via _auth_middleware wrapper).
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from aiohttp import web

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _hermes_home() -> Path:
    """Return the Hermes home directory (profile-aware)."""
    return get_hermes_home()


async def _async_subprocess(*cmd: str, timeout: float = 10) -> tuple:
    """Run a subprocess without blocking the event loop.

    Returns (returncode, stdout, stderr).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode, stdout.decode(), stderr.decode()


def _safe_name(name: str) -> Optional[web.Response]:
    """Validate a resource name. Returns an error response if invalid, None if OK."""
    if not name or not _NAME_RE.match(name):
        return web.json_response(
            {'error': 'Name must be alphanumeric with hyphens/underscores only'},
            status=422,
        )
    return None


# ── Config endpoints ──────────────────────────────────────────────────────────


async def handle_get_config(request: web.Request) -> web.Response:
    """GET /myah/api/config — Read config.yaml as JSON."""
    config_path = _hermes_home() / 'config.yaml'
    if not config_path.exists():
        return web.json_response({'error': 'config.yaml not found'}, status=404)
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
        return web.json_response(cfg)
    except Exception as e:
        logger.error('Failed to read config.yaml: %s', e)
        return web.json_response({'error': str(e)}, status=500)


async def handle_patch_config(request: web.Request) -> web.Response:
    """PATCH /myah/api/config — Update config keys via hermes config set.

    Body: {"key1": "value1", "key2": "value2"}
    Uses the official `hermes config set` CLI command, which handles type
    coercion (bool, int, float), nested dotted keys, and .env synchronization.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    if not isinstance(body, dict):
        return web.json_response({'error': 'Body must be a JSON object'}, status=400)

    errors = []
    for key, value in body.items():
        try:
            returncode, _, stderr = await _async_subprocess(
                'hermes', 'config', 'set', str(key), str(value), timeout=10,
            )
            if returncode != 0:
                errors.append({'key': key, 'error': stderr.strip()})
        except asyncio.TimeoutError:
            errors.append({'key': key, 'error': 'timeout'})
        except Exception as e:
            errors.append({'key': key, 'error': str(e)})

    if errors:
        return web.json_response({'ok': False, 'errors': errors}, status=207)

    # Re-read to return the updated config
    config_path = _hermes_home() / 'config.yaml'
    cfg = yaml.safe_load(config_path.read_text()) or {}
    return web.json_response({'ok': True, 'config': cfg})


async def handle_get_model(request: web.Request) -> web.Response:
    """GET /myah/api/config/model — Read the current model from config."""
    config_path = _hermes_home() / 'config.yaml'
    if not config_path.exists():
        return web.json_response({'model': ''})
    cfg = yaml.safe_load(config_path.read_text()) or {}
    return web.json_response({'model': cfg.get('model', '')})


async def handle_put_model(request: web.Request) -> web.Response:
    """PUT /myah/api/config/model — Update the model via hermes config set.

    The gateway re-reads config.yaml on every incoming message, so the
    new model takes effect immediately on the next message (no restart).
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    model = body.get('model', '')
    if not model:
        return web.json_response({'error': 'model is required'}, status=400)

    returncode, _, stderr = await _async_subprocess(
        'hermes', 'config', 'set', 'model', model, timeout=10,
    )
    if returncode != 0:
        return web.json_response(
            {'error': f'hermes config set failed: {stderr.strip()}'}, status=500
        )
    return web.json_response({'model': model})


# ── SOUL.md endpoints ─────────────────────────────────────────────────────────


async def handle_get_soul(request: web.Request) -> web.Response:
    """GET /myah/api/config/soul — Read SOUL.md content."""
    soul_path = _hermes_home() / 'SOUL.md'
    if not soul_path.exists():
        return web.json_response({'content': ''})
    return web.json_response({'content': soul_path.read_text()})


async def handle_put_soul(request: web.Request) -> web.Response:
    """PUT /myah/api/config/soul — Update SOUL.md content.

    SOUL.md changes take effect on next message because prompt_builder.py
    re-reads SOUL.md on every prompt assembly. No restart or cache
    invalidation needed.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    content = body.get('content', '')
    if not content.strip():
        return web.json_response({'error': 'SOUL content cannot be empty'}, status=422)

    soul_path = _hermes_home() / 'SOUL.md'
    try:
        soul_path.write_text(content)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

    return web.json_response({'content': content})


# ── Toolset endpoints ─────────────────────────────────────────────────────────


async def handle_list_toolsets(request: web.Request) -> web.Response:
    """GET /myah/api/toolsets — List all toolsets with enabled state.

    Reads disabled_toolsets from config.yaml, then inspects the
    toolsets module and tool registry for metadata.
    """
    try:
        import toolsets as ts
        from tools.registry import registry

        config_path = _hermes_home() / 'config.yaml'
        cfg = yaml.safe_load(config_path.read_text()) or {} if config_path.exists() else {}
        disabled = cfg.get('disabled_toolsets', []) or []

        tool_map = registry.get_tool_to_toolset_map()
        result = []
        for ts_name, ts_info in sorted(ts.TOOLSETS.items()):
            # Skip platform-specific and internal toolsets
            if ts_name.startswith('hermes-') or ts_name in ('safe', 'messaging', 'search', 'honcho'):
                continue
            tools = [
                {
                    'name': t,
                    'description': registry._tools[t].description if t in registry._tools else '',
                    'toolset': ts_name,
                    'emoji': getattr(registry._tools.get(t), 'emoji', None),
                }
                for t, mapped_ts in tool_map.items()
                if mapped_ts == ts_name and t in registry._tools
            ]
            result.append({
                'name': ts_name,
                'description': ts_info.get('description', ts_name),
                'enabled': ts_name not in disabled,
                'tools': tools,
            })
        return web.json_response(result)
    except Exception as e:
        logger.error('Failed to list toolsets: %s', e)
        return web.json_response({'error': str(e)}, status=500)


async def handle_toggle_toolset(request: web.Request) -> web.Response:
    """PATCH /myah/api/toolsets/{name} — Enable or disable a toolset.

    Uses `hermes tools enable/disable` CLI command which writes to
    config.yaml's disabled_toolsets list. Changes take effect on next
    message due to config signature invalidation.
    """
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    enabled = body.get('enabled', True)
    action = 'enable' if enabled else 'disable'

    returncode, _, stderr = await _async_subprocess(
        'hermes', 'tools', action, name, timeout=10,
    )
    if returncode != 0:
        return web.json_response(
            {'error': f'hermes tools {action} failed: {stderr.strip()}'}, status=500
        )
    return web.json_response({'name': name, 'enabled': enabled})


# ── Skill endpoints ───────────────────────────────────────────────────────────


async def handle_list_skills(request: web.Request) -> web.Response:
    """GET /myah/api/skills — List all skills from the skills directory."""
    skills_dir = _hermes_home() / 'skills'
    if not skills_dir.exists():
        return web.json_response([])

    result = []
    for skill_md in sorted(skills_dir.rglob('SKILL.md')):
        content = skill_md.read_text()
        fm = _parse_frontmatter(content)
        category = skill_md.parent.parent.name if skill_md.parent.parent != skills_dir else 'general'
        if category.startswith('.'):
            continue
        result.append({
            'name': fm.get('name', skill_md.parent.name),
            'category': category,
            'description': fm.get('description', '')[:200],
            'source': 'local',
            'trust': 'local',
        })
    return web.json_response(result)


async def handle_get_skill(request: web.Request) -> web.Response:
    """GET /myah/api/skills/{name} — Get full skill content."""
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    skills_dir = _hermes_home() / 'skills'
    if not skills_dir.exists():
        return web.json_response({'error': 'Skill not found'}, status=404)
    for skill_md in skills_dir.rglob('SKILL.md'):
        fm = _parse_frontmatter(skill_md.read_text())
        if fm.get('name', skill_md.parent.name) == name:
            return web.json_response({
                'name': name,
                'category': skill_md.parent.parent.name if skill_md.parent.parent != skills_dir else 'general',
                'content': skill_md.read_text(),
            })
    return web.json_response({'error': 'Skill not found'}, status=404)


async def handle_create_skill(request: web.Request) -> web.Response:
    """POST /myah/api/skills — Create a new skill."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    name = body.get('name', '')
    category = body.get('category', 'general')
    content = body.get('content', '')

    for n in (name, category):
        err = _safe_name(n)
        if err:
            return err

    if not content.strip():
        return web.json_response({'error': 'content is required'}, status=400)

    skill_dir = _hermes_home() / 'skills' / category / name
    skill_path = skill_dir / 'SKILL.md'

    if skill_path.exists():
        return web.json_response({'error': 'Skill already exists'}, status=409)

    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(content)

    return web.json_response({
        'name': name,
        'category': category,
        'content': content,
    }, status=201)


async def handle_update_skill(request: web.Request) -> web.Response:
    """PUT /myah/api/skills/{name} — Update an existing skill."""
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    content = body.get('content', '')
    if not content.strip():
        return web.json_response({'error': 'content is required'}, status=400)

    skills_dir = _hermes_home() / 'skills'
    if not skills_dir.exists():
        return web.json_response({'error': 'Skill not found'}, status=404)
    for skill_md in skills_dir.rglob('SKILL.md'):
        fm = _parse_frontmatter(skill_md.read_text())
        if fm.get('name', skill_md.parent.name) == name:
            skill_md.write_text(content)
            return web.json_response({'name': name, 'content': content})

    return web.json_response({'error': 'Skill not found'}, status=404)


async def handle_delete_skill(request: web.Request) -> web.Response:
    """DELETE /myah/api/skills/{name} — Delete a skill directory."""
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    skills_dir = _hermes_home() / 'skills'
    if not skills_dir.exists():
        return web.json_response({'error': 'Skill not found'}, status=404)
    for skill_md in skills_dir.rglob('SKILL.md'):
        fm = _parse_frontmatter(skill_md.read_text())
        if fm.get('name', skill_md.parent.name) == name:
            shutil.rmtree(skill_md.parent, ignore_errors=True)
            return web.json_response({'ok': True})

    return web.json_response({'error': 'Skill not found'}, status=404)


# ── Plugin endpoints ──────────────────────────────────────────────────────────


async def handle_list_plugins(request: web.Request) -> web.Response:
    """GET /myah/api/plugins — List all plugins."""
    plugins_dir = _hermes_home() / 'plugins'
    if not plugins_dir.exists():
        return web.json_response([])

    result = []
    for f in sorted(plugins_dir.glob('*.py')):
        if f.name.startswith('_'):
            continue
        text = f.read_text()
        # Extract first non-empty line as description
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
    return web.json_response(result)


async def handle_create_plugin(request: web.Request) -> web.Response:
    """POST /myah/api/plugins — Create a new plugin. Requires restart."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    name = body.get('name', '')
    content = body.get('content', '')
    err = _safe_name(name)
    if err:
        return err

    if not content.strip():
        return web.json_response({'error': 'content is required'}, status=400)

    # Validate Python syntax before writing
    try:
        compile(content, f'{name}.py', 'exec')
    except SyntaxError as e:
        return web.json_response({'error': f'Python syntax error: {e}'}, status=422)

    plugins_dir = _hermes_home() / 'plugins'
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_path = plugins_dir / f'{name}.py'

    if plugin_path.exists():
        return web.json_response({'error': 'Plugin already exists'}, status=409)

    plugin_path.write_text(content)

    # Plugins require restart to be loaded (Python imports are cached)
    _schedule_restart()

    return web.json_response({
        'filename': f'{name}.py',
        'name': name,
        'content': content,
    }, status=201)


async def handle_update_plugin(request: web.Request) -> web.Response:
    """PUT /myah/api/plugins/{name} — Update a plugin. Requires restart."""
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    content = body.get('content', '')
    if not content.strip():
        return web.json_response({'error': 'content is required'}, status=400)

    try:
        compile(content, f'{name}.py', 'exec')
    except SyntaxError as e:
        return web.json_response({'error': f'Python syntax error: {e}'}, status=422)

    plugin_path = _hermes_home() / 'plugins' / f'{name}.py'
    if not plugin_path.exists():
        return web.json_response({'error': 'Plugin not found'}, status=404)

    plugin_path.write_text(content)
    _schedule_restart()

    return web.json_response({'name': name, 'content': content})


async def handle_delete_plugin(request: web.Request) -> web.Response:
    """DELETE /myah/api/plugins/{name} — Delete a plugin. Requires restart."""
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    plugin_path = _hermes_home() / 'plugins' / f'{name}.py'
    if not plugin_path.exists():
        return web.json_response({'error': 'Plugin not found'}, status=404)

    plugin_path.unlink()
    _schedule_restart()

    return web.json_response({'ok': True})


# ── MCP Server endpoints ─────────────────────────────────────────────────────


async def handle_list_mcp(request: web.Request) -> web.Response:
    """GET /myah/api/mcp — List MCP servers from config."""
    config_path = _hermes_home() / 'config.yaml'
    if not config_path.exists():
        return web.json_response([])
    cfg = yaml.safe_load(config_path.read_text()) or {}
    servers = cfg.get('mcp_servers', {}) or {}
    result = [
        {
            'name': k,
            'url': v.get('url') if isinstance(v, dict) else None,
            'command': v.get('command') if isinstance(v, dict) else None,
            'args': v.get('args', []) if isinstance(v, dict) else [],
            'status': 'unknown',
        }
        for k, v in servers.items()
    ]
    return web.json_response(result)


async def handle_add_mcp(request: web.Request) -> web.Response:
    """POST /myah/api/mcp — Add an MCP server via hermes mcp add."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    name = body.get('name', '')
    err = _safe_name(name)
    if err:
        return err

    url = body.get('url')
    command = body.get('command')

    if not url and not command:
        return web.json_response({'error': 'Either url or command is required'}, status=422)

    cmd = ['hermes', 'mcp', 'add', name]
    if url:
        cmd += ['--url', url]
    elif command:
        cmd += ['--command', command]
        for arg in body.get('args', []):
            cmd += ['--args', str(arg)]

    # Handle API key injection into .env
    api_key = body.get('api_key')
    if api_key:
        env_path = _hermes_home() / '.env'
        env_key = f'MCP_{name.upper()}_API_KEY'
        existing = env_path.read_text() if env_path.exists() else ''
        lines = [l for l in existing.splitlines() if not l.startswith(f'{env_key}=')]
        lines.append(f'{env_key}={api_key}')
        env_path.write_text('\n'.join(lines) + '\n')

    returncode, _, stderr = await _async_subprocess(*cmd, timeout=30)
    if returncode != 0:
        return web.json_response(
            {'error': f'hermes mcp add failed: {stderr.strip()}'}, status=500
        )

    return web.json_response({
        'name': name,
        'url': url,
        'command': command,
        'args': body.get('args', []),
        'status': 'unknown',
    }, status=201)


async def handle_remove_mcp(request: web.Request) -> web.Response:
    """DELETE /myah/api/mcp/{name} — Remove an MCP server via hermes mcp remove."""
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    returncode, _, stderr = await _async_subprocess(
        'hermes', 'mcp', 'remove', name, timeout=30,
    )
    if returncode != 0:
        return web.json_response(
            {'error': f'hermes mcp remove failed: {stderr.strip()}'}, status=500
        )

    return web.json_response({'ok': True})


# ── Session endpoints ─────────────────────────────────────────────────────────


async def handle_list_sessions(request: web.Request) -> web.Response:
    """GET /myah/api/sessions — List sessions from SessionDB."""
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        limit = int(request.query.get('limit', '50'))
        offset = int(request.query.get('offset', '0'))
        source = request.query.get('source')
        sessions = db.list_sessions_rich(
            source=source, limit=limit, offset=offset, include_children=False,
        )
        return web.json_response(sessions)
    except Exception as e:
        logger.error('Failed to list sessions: %s', e)
        return web.json_response({'error': str(e)}, status=500)


async def handle_get_session_messages(request: web.Request) -> web.Response:
    """GET /myah/api/sessions/{id}/messages — Get session conversation history."""
    session_id = request.match_info['id']
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        messages = db.get_messages_as_conversation(session_id)
        session = db.get_session(session_id)
        return web.json_response({
            'session_id': session_id,
            'session': session,
            'messages': messages,
        })
    except Exception as e:
        logger.error('Failed to get session messages: %s', e)
        return web.json_response({'error': str(e)}, status=500)


async def handle_set_session_title(request: web.Request) -> web.Response:
    """POST /myah/api/sessions/{id}/title — Set session title."""
    session_id = request.match_info['id']
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    title = body.get('title', '')
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        found = db.set_session_title(session_id, title)
        if not found:
            return web.json_response({'error': 'Session not found'}, status=404)
        return web.json_response({'session_id': session_id, 'title': title})
    except ValueError as e:
        return web.json_response({'error': str(e)}, status=422)
    except Exception as e:
        logger.error('Failed to set session title: %s', e)
        return web.json_response({'error': str(e)}, status=500)


async def handle_append_session_message(request: web.Request) -> web.Response:
    """POST /myah/api/sessions/{id}/append — Append a message to session transcript.

    Used by cron delivery to write output directly to SessionDB without
    triggering an agent run. The message is stored as-is in the session's
    message history.

    Body: {role: "assistant", content: "...", metadata?: {...}}
    """
    session_id = request.match_info['id']
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    role = body.get('role', 'assistant')
    content = body.get('content', '')
    if not content:
        return web.json_response({'error': 'content is required'}, status=400)

    try:
        from hermes_state import SessionDB
        db = SessionDB()
        # Ensure session exists (create if not — handles edge case where
        # the cron job references a session_id that hasn't been used yet)
        db.ensure_session(session_id, source='myah')
        msg_id = db.append_message(session_id, role=role, content=content)
        return web.json_response({
            'session_id': session_id,
            'message_id': msg_id,
            'role': role,
        })
    except Exception as e:
        logger.error('Failed to append message to session %s: %s', session_id, e)
        return web.json_response({'error': str(e)}, status=500)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from a SKILL.md file."""
    fm = {}
    m = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ': ' in line:
                k, v = line.split(': ', 1)
                fm[k.strip()] = v.strip().strip('"\'')
    return fm


def _schedule_restart() -> None:
    """Schedule a deferred restart of the Hermes gateway via supervisorctl.

    Used for plugin changes only. The restart is deferred by 2 seconds so
    the HTTP response reaches the client before the process dies. Without
    this delay, the client gets a connection reset because supervisorctl
    kills our process.
    """
    def _do_restart():
        try:
            subprocess.run(
                ['supervisorctl', 'restart', 'hermes'],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as e:
            logger.error('Failed to restart hermes: %s', e)

    try:
        loop = asyncio.get_event_loop()
        loop.call_later(2.0, lambda: asyncio.ensure_future(
            asyncio.to_thread(_do_restart)
        ))
    except RuntimeError:
        # No running event loop — run synchronously as fallback
        _do_restart()


def _auth_middleware(handler, auth_key: str):
    """Wrap a handler with bearer token auth checking."""
    async def wrapped(request: web.Request) -> web.Response:
        if auth_key:
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer ') or auth_header[7:] != auth_key:
                return web.json_response({'error': 'Unauthorized'}, status=401)
        return await handler(request)
    return wrapped


def register_management_routes(app: web.Application, auth_key: str = "") -> None:
    """Register all management API routes on the aiohttp app.

    Called by MyahAdapter._register_routes_on_app() during connect().
    All routes are wrapped with bearer token auth middleware.
    """
    def _a(handler):
        """Apply auth middleware to a handler."""
        return _auth_middleware(handler, auth_key)

    # Config
    app.router.add_get('/myah/api/config', _a(handle_get_config))
    app.router.add_patch('/myah/api/config', _a(handle_patch_config))
    app.router.add_get('/myah/api/config/model', _a(handle_get_model))
    app.router.add_put('/myah/api/config/model', _a(handle_put_model))
    app.router.add_get('/myah/api/config/soul', _a(handle_get_soul))
    app.router.add_put('/myah/api/config/soul', _a(handle_put_soul))

    # Toolsets
    app.router.add_get('/myah/api/toolsets', _a(handle_list_toolsets))
    app.router.add_patch('/myah/api/toolsets/{name}', _a(handle_toggle_toolset))

    # Skills
    app.router.add_get('/myah/api/skills', _a(handle_list_skills))
    app.router.add_get('/myah/api/skills/{name}', _a(handle_get_skill))
    app.router.add_post('/myah/api/skills', _a(handle_create_skill))
    app.router.add_put('/myah/api/skills/{name}', _a(handle_update_skill))
    app.router.add_delete('/myah/api/skills/{name}', _a(handle_delete_skill))

    # Plugins
    app.router.add_get('/myah/api/plugins', _a(handle_list_plugins))
    app.router.add_post('/myah/api/plugins', _a(handle_create_plugin))
    app.router.add_put('/myah/api/plugins/{name}', _a(handle_update_plugin))
    app.router.add_delete('/myah/api/plugins/{name}', _a(handle_delete_plugin))

    # MCP
    app.router.add_get('/myah/api/mcp', _a(handle_list_mcp))
    app.router.add_post('/myah/api/mcp', _a(handle_add_mcp))
    app.router.add_delete('/myah/api/mcp/{name}', _a(handle_remove_mcp))

    # Sessions
    app.router.add_get('/myah/api/sessions', _a(handle_list_sessions))
    app.router.add_get('/myah/api/sessions/{id}/messages', _a(handle_get_session_messages))
    app.router.add_post('/myah/api/sessions/{id}/title', _a(handle_set_session_title))
    app.router.add_post('/myah/api/sessions/{id}/append', _a(handle_append_session_message))

    logger.info('Myah management routes registered (%d endpoints)', 24)
