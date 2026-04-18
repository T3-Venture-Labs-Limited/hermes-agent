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
import hashlib
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

# ── Myah: in-process config setter (fixes subprocess os.environ staleness) ──
from hermes_cli.config import set_config_value
# ──────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')

# ── Myah: module-level runner reference for restart handler ──────────────
_gateway_runner = None


def set_gateway_runner(runner):
    """Called by MyahAdapter._register_routes_on_app to expose runner to handlers."""
    global _gateway_runner
    _gateway_runner = runner
# ─────────────────────────────────────────────────────────────────────────

# ── Myah: SOUL size limits ──────────────────────────────────────────────
SOUL_SOFT_WARN_CHARS = 8_192
SOUL_HARD_CAP_CHARS = 32_768
# ──────────────────────────────────────────────────────────────────────────


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


def _deep_merge_defaults(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides over defaults.

    Keys present in overrides win; nested dicts are merged key-by-key. Lists
    and scalars from overrides replace defaults entirely.
    """
    result = dict(defaults)
    for key, override_val in overrides.items():
        default_val = result.get(key)
        if isinstance(default_val, dict) and isinstance(override_val, dict):
            result[key] = _deep_merge_defaults(default_val, override_val)
        else:
            result[key] = override_val
    return result


async def handle_get_config(request: web.Request) -> web.Response:
    """GET /myah/api/config — Read config.yaml merged over DEFAULT_CONFIG.

    Older containers may have a config.yaml whose schema predates the current
    ``DEFAULT_CONFIG`` (e.g. missing newly-added auxiliary task keys). Rather
    than relying on ``migrate_config()`` running at gateway startup, we merge
    the on-disk values on top of the current DEFAULT_CONFIG every time we
    serve this endpoint so the frontend always gets a complete schema shape.
    """
    config_path = _hermes_home() / 'config.yaml'
    if not config_path.exists():
        return web.json_response({'error': 'config.yaml not found'}, status=404)
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception as e:
        logger.error('Failed to read config.yaml: %s', e)
        return web.json_response({'error': str(e)}, status=500)

    try:
        from hermes_cli.config import DEFAULT_CONFIG

        # Strip private keys (``_config_version`` etc.) from the defaults view —
        # the on-disk ``_config_version`` wins, but the DEFAULT_CONFIG private
        # keys have no user-facing meaning.
        public_defaults = {k: v for k, v in DEFAULT_CONFIG.items() if not k.startswith('_')}
        merged = _deep_merge_defaults(public_defaults, cfg)
        # Preserve any private keys that were on disk (e.g. _config_version)
        for k, v in cfg.items():
            if k.startswith('_'):
                merged[k] = v
        return web.json_response(merged)
    except Exception as e:  # pragma: no cover — defensive
        logger.error('Failed to merge DEFAULT_CONFIG with config.yaml: %s', e)
        return web.json_response(cfg)


async def handle_patch_config(request: web.Request) -> web.Response:
    """PATCH /myah/api/config — Update config keys via in-process set_config_value.

    Body: {"key1": "value1", "key2": "value2"}
    Uses set_config_value() directly — the subprocess approach succeeded at
    writing config.yaml but left the running gateway process's os.environ stale.

    Composite values (dict / list) are merged into the user config directly
    instead of being stringified via ``set_config_value`` — Python's ``str()``
    renders dicts in repr notation (single-quoted) which YAML cannot parse on
    the next read. See e2e-output/report.md ISSUE-004.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    if not isinstance(body, dict):
        return web.json_response({'error': 'Body must be a JSON object'}, status=400)

    errors = []
    composite_writes: Dict[str, Any] = {}

    for key, value in body.items():
        if isinstance(value, (dict, list)):
            # Defer composite values to a single YAML merge below.
            composite_writes[str(key)] = value
            continue
        try:
            set_config_value(str(key), str(value))
        except Exception as e:
            errors.append({'key': key, 'error': str(e)})

    if composite_writes:
        config_path = _hermes_home() / 'config.yaml'
        try:
            cfg = yaml.safe_load(config_path.read_text()) or {} if config_path.exists() else {}
            # Deep-merge dict writes; replace list writes wholesale.
            cfg = _deep_merge_defaults(cfg, composite_writes)
            config_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
        except Exception as e:
            errors.append({'key': list(composite_writes.keys()), 'error': str(e)})

    if errors:
        return web.json_response({'ok': False, 'errors': errors}, status=207)

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


# ── Myah: Session-scoped model override (T3-932) ─────────────────────────────
# Per-session model overrides via HTTP. Writes directly to the gateway
# runner's _session_model_overrides dict — the same primitive that the
# /model slash command uses in gateway/run.py:_handle_model_command.
# Does NOT persist to config.yaml; override lives for the session lifetime.


async def handle_get_session_model(request: web.Request) -> web.Response:
    """GET /myah/api/sessions/{id}/model — Read current session override."""
    session_key = request.match_info.get('id', '')
    adapter = request.app.get('myah_adapter')
    if adapter is None or adapter.gateway_runner is None:
        return web.json_response({'error': 'Gateway runner not available'}, status=503)

    overrides = getattr(adapter.gateway_runner, '_session_model_overrides', {}) or {}
    override = overrides.get(session_key, {})
    return web.json_response({
        'model': override.get('model', ''),
        'provider': override.get('provider', ''),
    })


async def handle_put_session_model(request: web.Request) -> web.Response:
    """PUT /myah/api/sessions/{id}/model — Set per-session model override.

    Body: {"model": "<model-id>", "provider": "<optional-provider>"}

    Writes to gateway_runner._session_model_overrides[session_key] and
    evicts the cached agent so the next turn rebuilds with the new model.
    Mirrors the --session (non-global) branch of the /model slash command
    in gateway/run.py:_handle_model_command.
    """
    session_key = request.match_info.get('id', '')
    if not session_key:
        return web.json_response({'error': 'session_key is required'}, status=400)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    raw_input = (body.get('model') or '').strip()
    if not raw_input:
        return web.json_response({'error': 'model is required'}, status=400)

    explicit_provider = (body.get('provider') or '').strip()

    adapter = request.app.get('myah_adapter')
    if adapter is None or adapter.gateway_runner is None:
        return web.json_response({'error': 'Gateway runner not available'}, status=503)
    runner = adapter.gateway_runner

    # Load current config to build switch_model() context
    config_path = _hermes_home() / 'config.yaml'
    cfg: Dict[str, Any] = {}
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text()) or {}
        except Exception:
            cfg = {}

    model_cfg = cfg.get('model', {}) if isinstance(cfg.get('model'), dict) else {}
    current_model = model_cfg.get('default') or (cfg.get('model') if isinstance(cfg.get('model'), str) else '')
    current_provider = model_cfg.get('provider', '') or 'openrouter'
    current_base_url = model_cfg.get('base_url', '') or ''
    user_providers = cfg.get('providers')
    custom_providers = cfg.get('custom_providers')

    # Layer current session override on top if present (so we preserve
    # unset fields when the user switches providers).
    existing = (runner._session_model_overrides or {}).get(session_key, {})
    if existing:
        current_model = existing.get('model', current_model)
        current_provider = existing.get('provider', current_provider)
        current_base_url = existing.get('base_url', current_base_url)

    # switch_model() performs network validation calls — run in executor
    # to avoid blocking the event loop.
    from hermes_cli.model_switch import switch_model

    def _run_switch_model():
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key='',
            is_global=False,
            explicit_provider=explicit_provider,
            user_providers=user_providers,
            custom_providers=custom_providers,
        )

    try:
        result = await asyncio.get_running_loop().run_in_executor(None, _run_switch_model)
    except Exception as exc:
        logger.exception('[myah] switch_model failed for session %s', session_key)
        return web.json_response({'error': f'switch_model failed: {exc}'}, status=500)

    if not getattr(result, 'success', False):
        error_msg = getattr(result, 'error_message', '') or 'Model not recognized'
        return web.json_response({'error': error_msg}, status=400)

    # Write the override and evict cached agent
    if runner._session_model_overrides is None:
        runner._session_model_overrides = {}
    runner._session_model_overrides[session_key] = {
        'model': result.new_model,
        'provider': result.target_provider,
        'api_key': getattr(result, 'api_key', '') or '',
        'base_url': getattr(result, 'base_url', '') or '',
        'api_mode': getattr(result, 'api_mode', '') or '',
    }
    # ── Myah: full agent teardown before eviction (fixes memory-provider leak) ──
    cache_entry = (runner._agent_cache or {}).get(session_key)
    if cache_entry is not None:
        old_agent = cache_entry[0] if isinstance(cache_entry, tuple) else cache_entry
        try:
            old_agent.shutdown_memory_provider()
        except Exception as e:
            logger.warning('shutdown_memory_provider failed for %s: %s', session_key, e)
        try:
            old_agent.close()
        except Exception as e:
            logger.warning('agent.close() failed for %s: %s', session_key, e)
    # ─────────────────────────────────────────────────────────────────────
    try:
        runner._evict_cached_agent(session_key)
    except Exception:
        logger.warning('[myah] _evict_cached_agent failed for %s', session_key, exc_info=True)

    return web.json_response({
        'model': result.new_model,
        'provider': result.target_provider,
        'provider_label': getattr(result, 'provider_label', '') or '',
        'warning': getattr(result, 'warning_message', None) or None,
    })


# ─────────────────────────────────────────────────────────────────────────────


# ── SOUL.md endpoints ─────────────────────────────────────────────────────────


def _soul_etag(body: str) -> str:
    """Compute sha256-based ETag for SOUL content."""
    digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
    return f'"sha256-{digest}"'


async def handle_get_soul(request: web.Request) -> web.Response:
    """GET /myah/api/config/soul — Read SOUL.md with ETag."""
    soul_path = _hermes_home() / 'SOUL.md'
    if not soul_path.exists():
        return web.Response(status=404, text='SOUL.md not found')
    body = soul_path.read_text(encoding='utf-8')
    etag = _soul_etag(body)
    response = web.Response(
        text=body,
        content_type='text/markdown',
        headers={'ETag': etag},
    )
    response.headers['X-Soul-Soft-Warn-Chars'] = str(SOUL_SOFT_WARN_CHARS)
    response.headers['X-Soul-Hard-Cap-Chars'] = str(SOUL_HARD_CAP_CHARS)
    return response


async def handle_put_soul(request: web.Request) -> web.Response:
    """PUT /myah/api/config/soul — Write SOUL.md with If-Match concurrency control.

    428 if If-Match missing, 412 if stale (returns current body),
    413 if over 32768-char hard cap, 200 with warning at 8192 chars.
    """
    if_match = request.headers.get('If-Match')
    if not if_match:
        return web.json_response(
            {'error': 'If-Match header required for SOUL writes'},
            status=428,
        )

    new_body = await request.text()

    if len(new_body) > SOUL_HARD_CAP_CHARS:
        return web.json_response(
            {
                'error': (
                    f'SOUL content exceeds {SOUL_HARD_CAP_CHARS} character limit '
                    f'(got {len(new_body)}). SOUL is injected into every turn; '
                    f'keep it focused.'
                ),
                'limit': SOUL_HARD_CAP_CHARS,
                'got': len(new_body),
            },
            status=413,
        )

    soul_path = _hermes_home() / 'SOUL.md'
    current_body = soul_path.read_text(encoding='utf-8') if soul_path.exists() else ''
    current_etag = _soul_etag(current_body)

    if if_match != current_etag:
        response = web.json_response(
            {
                'error': 'precondition failed — SOUL was modified since you read it',
                'current_body': current_body,
            },
            status=412,
        )
        response.headers['ETag'] = current_etag
        return response

    soul_path.write_text(new_body, encoding='utf-8')
    new_etag = _soul_etag(new_body)

    response_body: Dict[str, Any] = {'ok': True}
    if len(new_body) > SOUL_SOFT_WARN_CHARS:
        response_body['warning'] = (
            f'SOUL is {len(new_body)} chars; recommended soft limit is '
            f'{SOUL_SOFT_WARN_CHARS}. This adds to every turn.'
        )
    return web.json_response(
        response_body,
        headers={'ETag': new_etag},
    )


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
    """POST /myah/api/mcp — Add an MCP server and register it in-process."""
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
    args = body.get('args', [])
    env = body.get('env') or {}

    if not url and not command:
        return web.json_response({'error': 'Either url or command is required'}, status=422)

    # Handle API key injection into .env
    api_key = body.get('api_key')
    if api_key:
        env_path = _hermes_home() / '.env'
        env_key = f'MCP_{name.upper()}_API_KEY'
        existing = env_path.read_text() if env_path.exists() else ''
        lines = [line for line in existing.splitlines() if not line.startswith(f'{env_key}=')]
        lines.append(f'{env_key}={api_key}')
        env_path.write_text('\n'.join(lines) + '\n')

    # Build server config dict and write to config.yaml
    server_cfg: Dict[str, Any] = {}
    if url:
        server_cfg['url'] = url
    elif command:
        server_cfg['command'] = command
        server_cfg['args'] = args
        if env:
            server_cfg['env'] = env

    config_path = _hermes_home() / 'config.yaml'
    cfg = yaml.safe_load(config_path.read_text()) or {} if config_path.exists() else {}
    if 'mcp_servers' not in cfg or cfg['mcp_servers'] is None:
        cfg['mcp_servers'] = {}
    cfg['mcp_servers'][name] = server_cfg
    config_path.write_text(yaml.safe_dump(cfg, default_flow_style=False))

    # ── Myah: in-process MCP registry refresh + agent cache eviction ─────
    try:
        from tools.mcp_tool import register_mcp_servers
        register_mcp_servers({name: server_cfg})
    except Exception as e:
        logger.warning('[myah] register_mcp_servers failed for %s: %s', name, e)

    runner = _gateway_runner
    if runner is not None:
        cache = getattr(runner, '_agent_cache', {}) or {}
        for session_key in list(cache.keys()):
            try:
                runner._evict_cached_agent(session_key)
            except Exception as e:
                logger.warning('[myah] evict failed for %s: %s', session_key, e)
    # ─────────────────────────────────────────────────────────────────────

    return web.json_response({
        'name': name,
        'url': url,
        'command': command,
        'args': args,
        'status': 'unknown',
    }, status=200)


async def handle_remove_mcp(request: web.Request) -> web.Response:
    """DELETE /myah/api/mcp/{name} — Remove an MCP server and disconnect in-process."""
    name = request.match_info['name']
    err = _safe_name(name)
    if err:
        return err

    config_path = _hermes_home() / 'config.yaml'
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
    else:
        cfg = {}
    servers = cfg.get('mcp_servers') or {}
    if name not in servers:
        return web.json_response({'error': f'MCP server {name!r} not found'}, status=404)

    del servers[name]
    cfg['mcp_servers'] = servers
    config_path.write_text(yaml.safe_dump(cfg, default_flow_style=False))

    # ── Myah: per-server in-process disconnect ────────────────────────────
    try:
        from tools.mcp_tool import disconnect_mcp_server
        disconnect_mcp_server(name)
    except Exception as e:
        logger.warning('[myah] disconnect_mcp_server failed for %s: %s', name, e)

    runner = _gateway_runner
    if runner is not None:
        cache = getattr(runner, '_agent_cache', {}) or {}
        for session_key in list(cache.keys()):
            try:
                runner._evict_cached_agent(session_key)
            except Exception as e:
                logger.warning('[myah] evict failed for %s: %s', session_key, e)
    # ─────────────────────────────────────────────────────────────────────

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


# ── Environment variable (secret) management ─────────────────────────────────

_DENIED_ENV_VARS = frozenset({
    'PATH', 'HOME', 'SHELL', 'USER', 'LOGNAME',
    'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP',
    'LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH',
    'HERMES_HOME', 'HERMES_PROFILE',
    'NODE_PATH', 'NODE_OPTIONS',
})

# ── Myah: internal env var prefixes hidden from settings UI ───────────────
# Infrastructure vars set by the platform — not user-configurable secrets.
_INTERNAL_PREFIXES = ('MYAH_', 'API_SERVER_', 'HONCHO_', 'HERMES_', 'SENTRY_')
# ──────────────────────────────────────────────────────────────────────────


async def handle_list_env(request: web.Request) -> web.Response:
    """List all known env vars with redacted values and metadata.

    Returns vars defined in OPTIONAL_ENV_VARS plus any user-added secrets
    found in .env that are not internal infrastructure vars.
    Values are always redacted (first 4 + last 4 chars).
    """
    from hermes_cli.config import OPTIONAL_ENV_VARS, load_env

    env_on_disk = load_env()
    result = {}

    def _redact(value: str) -> str | None:
        if not value:
            return None
        return value[:4] + '...' + value[-4:] if len(value) >= 12 else '***'

    # Known vars with rich metadata from OPTIONAL_ENV_VARS
    for var_name, info in OPTIONAL_ENV_VARS.items():
        value = env_on_disk.get(var_name)
        result[var_name] = {
            'is_set': bool(value),
            'redacted_value': _redact(value),
            'description': info.get('description', ''),
            'url': info.get('url'),
            'category': info.get('category', ''),
            'is_password': info.get('password', False),
            'tools': info.get('tools', []),
        }

    # ── Myah: include user-added secrets not in OPTIONAL_ENV_VARS ─────────
    # Custom secrets the user added via the settings UI or secrets tool.
    # Exclude system vars, denied vars, and internal infrastructure vars.
    for var_name, value in env_on_disk.items():
        if var_name in result:
            continue
        if var_name.upper() in _DENIED_ENV_VARS:
            continue
        if any(var_name.startswith(p) for p in _INTERNAL_PREFIXES):
            continue
        result[var_name] = {
            'is_set': bool(value),
            'redacted_value': _redact(value),
            'description': 'User-configured secret',
            'url': None,
            'category': 'custom',
            'is_password': True,
            'tools': [],
        }
    # ──────────────────────────────────────────────────────────────────────

    return web.json_response(result)


async def handle_set_env(request: web.Request) -> web.Response:
    """Set an environment variable in the agent's .env file.

    Body: {"key": "VAR_NAME", "value": "secret-value"}
    The value is written atomically and picked up immediately by os.environ.
    """
    from hermes_cli.config import save_env_value

    try:
        body = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON'}, status=400)

    key = body.get('key', '').strip()
    value = body.get('value', '')

    if not key:
        return web.json_response({'error': 'key is required'}, status=400)
    if key.upper() in _DENIED_ENV_VARS:
        return web.json_response(
            {'error': f'{key} is a protected system variable and cannot be set'},
            status=422,
        )
    if not value:
        return web.json_response({'error': 'value is required'}, status=400)
    if len(value) > 4096:
        return web.json_response({'error': 'value exceeds 4096 char limit'}, status=422)

    try:
        save_env_value(key, value)
    except Exception as exc:
        logger.exception('PUT /myah/api/config/env failed')
        return web.json_response({'error': str(exc)}, status=500)

    return web.json_response({'ok': True, 'key': key})


async def handle_delete_env(request: web.Request) -> web.Response:
    """Remove an environment variable from the agent's .env file.

    URL: DELETE /myah/api/config/env/{key}
    """
    from hermes_cli.config import remove_env_value

    key = request.match_info.get('key', '').strip()
    if not key:
        return web.json_response({'error': 'key is required'}, status=400)

    try:
        removed = remove_env_value(key)
    except Exception as exc:
        logger.exception('DELETE /myah/api/config/env failed')
        return web.json_response({'error': str(exc)}, status=500)

    if not removed:
        return web.json_response({'error': f'{key} not found in .env'}, status=404)

    return web.json_response({'ok': True, 'key': key})


# ── Gateway restart endpoint ─────────────────────────────────────────────────


async def handle_gateway_restart(request: web.Request) -> web.Response:
    """POST /myah/api/gateway/restart — restart gateway via supervisorctl.

    Refuses with 409 if any session has an in-flight turn.
    """
    runner = _gateway_runner
    if runner is not None:
        running = getattr(runner, '_running_agents', {}) or {}
        if running:
            return web.json_response(
                {
                    'error': 'busy',
                    'busy_sessions': list(running.keys()),
                    'message': 'A turn is currently running. Wait for it to finish and retry.',
                },
                status=409,
            )

    subprocess.run(['supervisorctl', 'restart', 'hermes'], check=True)
    return web.json_response(
        {'status': 'restarting', 'estimated_ready_ms': 3000},
        status=202,
    )


# ── Config schema endpoint ────────────────────────────────────────────────────


async def handle_get_schema(request: web.Request) -> web.Response:
    """GET /myah/api/config/schema — return a filtered view of DEFAULT_CONFIG."""
    from hermes_cli.config import DEFAULT_CONFIG

    def describe(value):
        if isinstance(value, dict):
            return {k: describe(v) for k, v in value.items()}
        if isinstance(value, bool):
            return {'type': 'boolean', 'default': value}
        if isinstance(value, int):
            return {'type': 'integer', 'default': value}
        if isinstance(value, float):
            return {'type': 'number', 'default': value}
        if isinstance(value, list):
            return {'type': 'array', 'default': value}
        return {'type': 'string', 'default': value or ''}

    schema = {
        k: describe(v)
        for k, v in DEFAULT_CONFIG.items()
        if not k.startswith('_')
    }
    return web.json_response(schema)


# ── Config reset endpoint ─────────────────────────────────────────────────────

# ── Myah: reset section taxonomy ─────────────────────────────────────────
_RESET_SECTION_KEYS: Dict[str, List[str]] = {
    'model': ['model'],
    'aux_vision': [
        'auxiliary.vision.provider', 'auxiliary.vision.model',
        'auxiliary.vision.base_url', 'auxiliary.vision.api_key',
        'auxiliary.vision.timeout',
    ],
    'aux_web_extract': [
        'auxiliary.web_extract.provider', 'auxiliary.web_extract.model',
        'auxiliary.web_extract.base_url', 'auxiliary.web_extract.api_key',
        'auxiliary.web_extract.timeout',
    ],
    'aux_compression': [
        'auxiliary.compression.provider', 'auxiliary.compression.model',
        'auxiliary.compression.base_url', 'auxiliary.compression.api_key',
        'auxiliary.compression.timeout',
    ],
    'aux_session_search': [
        'auxiliary.session_search.provider', 'auxiliary.session_search.model',
        'auxiliary.session_search.base_url', 'auxiliary.session_search.api_key',
        'auxiliary.session_search.timeout',
    ],
    'aux_skills_hub': [
        'auxiliary.skills_hub.provider', 'auxiliary.skills_hub.model',
        'auxiliary.skills_hub.base_url', 'auxiliary.skills_hub.api_key',
        'auxiliary.skills_hub.timeout',
    ],
    'aux_approval': [
        'auxiliary.approval.provider', 'auxiliary.approval.model',
        'auxiliary.approval.base_url', 'auxiliary.approval.api_key',
        'auxiliary.approval.timeout',
    ],
    'aux_mcp': [
        'auxiliary.mcp.provider', 'auxiliary.mcp.model',
        'auxiliary.mcp.base_url', 'auxiliary.mcp.api_key',
        'auxiliary.mcp.timeout',
    ],
    'aux_flush_memories': [
        'auxiliary.flush_memories.provider', 'auxiliary.flush_memories.model',
        'auxiliary.flush_memories.base_url', 'auxiliary.flush_memories.api_key',
        'auxiliary.flush_memories.timeout',
    ],
    'aux_title_generation': [
        'auxiliary.title_generation.provider', 'auxiliary.title_generation.model',
        'auxiliary.title_generation.base_url', 'auxiliary.title_generation.api_key',
        'auxiliary.title_generation.timeout',
    ],
    'aux_follow_up_generation': [
        'auxiliary.follow_up_generation.provider',
        'auxiliary.follow_up_generation.model',
        'auxiliary.follow_up_generation.base_url',
        'auxiliary.follow_up_generation.api_key',
        'auxiliary.follow_up_generation.timeout',
    ],
    'behavior': [
        'agent.reasoning_effort', 'approvals.mode', 'display.personality',
    ],
    'toolsets': [
        'disabled_toolsets',
    ],
    'advanced': [
        'terminal.backend', 'timezone',
    ],
}
# ─────────────────────────────────────────────────────────────────────────


def _read_default_value(dotted_key: str):
    """Read a value from DEFAULT_CONFIG using a dotted key path."""
    from hermes_cli.config import DEFAULT_CONFIG
    node = DEFAULT_CONFIG
    for part in dotted_key.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


async def handle_reset_section(request: web.Request) -> web.Response:
    """POST /myah/api/config/reset/{section} — revert a section to image defaults."""
    section = request.match_info.get('section', '')

    if section == 'soul':
        src = Path('/opt/myah/defaults/SOUL.md')
        dst = _hermes_home() / 'SOUL.md'
        if not src.exists():
            return web.json_response(
                {'error': 'image defaults not present (are you in a dev container?)'},
                status=503,
            )
        dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
        return web.json_response({'ok': True, 'section': 'soul'})

    if section == 'mcp_servers':
        config_path = _hermes_home() / 'config.yaml'
        cfg = yaml.safe_load(config_path.read_text()) or {}
        previous_names = list((cfg.get('mcp_servers') or {}).keys())

        try:
            set_config_value('mcp_servers', '{}')
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

        try:
            from tools.mcp_tool import disconnect_mcp_server
            for name in previous_names:
                try:
                    disconnect_mcp_server(name)
                except Exception as e:
                    logger.warning('reset mcp_servers: disconnect %s failed: %s', name, e)
        except Exception as e:
            logger.warning('mcp_servers reset: registry refresh failed: %s', e)
        return web.json_response({'ok': True, 'section': 'mcp_servers', 'removed': previous_names})

    if section not in _RESET_SECTION_KEYS:
        return web.json_response({'error': f'unknown section: {section}'}, status=400)

    keys = _RESET_SECTION_KEYS[section]
    errors = []
    composite_resets: Dict[str, Any] = {}
    for key in keys:
        default_val = _read_default_value(key)
        if default_val is None:
            logger.warning('reset: key %s not found in DEFAULT_CONFIG', key)
            continue
        if isinstance(default_val, (dict, list)):
            # Avoid str(dict) Python-repr corruption — apply via YAML merge below.
            composite_resets[key] = default_val
            continue
        try:
            set_config_value(key, str(default_val))
        except Exception as e:
            errors.append({'key': key, 'error': str(e)})

    if composite_resets:
        try:
            config_path = _hermes_home() / 'config.yaml'
            cfg = yaml.safe_load(config_path.read_text()) or {} if config_path.exists() else {}
            for dotted_key, val in composite_resets.items():
                parts = dotted_key.split('.')
                node = cfg
                for part in parts[:-1]:
                    if not isinstance(node.get(part), dict):
                        node[part] = {}
                    node = node[part]
                node[parts[-1]] = val
            config_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
        except Exception as e:
            errors.append({'key': list(composite_resets.keys()), 'error': str(e)})

    if errors:
        return web.json_response({'ok': False, 'errors': errors}, status=207)
    return web.json_response({'ok': True, 'section': section})


# ── Last-reseed endpoint ──────────────────────────────────────────────────────


async def handle_get_last_reseed(request: web.Request) -> web.Response:
    """GET /myah/api/config/last-reseed — when did entrypoint last seed each file.

    The breadcrumb file written by ``agent/scripts/seed_config_files.sh`` lists
    files as a space-separated string (``files=config soul``). The platform
    expects a proper JSON array so the toast can render
    ``files.join(' and ')`` without crashing on a string.
    See e2e-output/report.md ISSUE-009.
    """
    marker = _hermes_home() / '.myah_last_reseed'
    if not marker.exists():
        return web.json_response({})

    result: Dict[str, Any] = {}
    for line in marker.read_text().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            key = k.strip()
            value = v.strip()
            if key == 'files':
                # Normalise to a JSON array regardless of how the breadcrumb
                # encoded it (space-separated string today, possibly JSON
                # tomorrow).
                result[key] = [part for part in value.split() if part]
            else:
                result[key] = value
    return web.json_response(result)


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
    # ── Myah: schema, reset, last-reseed ─────────────────────────────────
    app.router.add_get('/myah/api/config/schema', _a(handle_get_schema))
    app.router.add_post('/myah/api/config/reset/{section}', _a(handle_reset_section))
    app.router.add_get('/myah/api/config/last-reseed', _a(handle_get_last_reseed))
    # ─────────────────────────────────────────────────────────────────────

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

    # ── Myah: gateway restart ─────────────────────────────────────────────
    app.router.add_post('/myah/api/gateway/restart', _a(handle_gateway_restart))
    # ─────────────────────────────────────────────────────────────────────

    # Sessions
    app.router.add_get('/myah/api/sessions', _a(handle_list_sessions))
    app.router.add_get('/myah/api/sessions/{id}/messages', _a(handle_get_session_messages))
    app.router.add_post('/myah/api/sessions/{id}/title', _a(handle_set_session_title))
    app.router.add_post('/myah/api/sessions/{id}/append', _a(handle_append_session_message))

    # ── Myah: Session-scoped model override (T3-932) ──────────────────────────
    app.router.add_get('/myah/api/sessions/{id}/model', _a(handle_get_session_model))
    app.router.add_put('/myah/api/sessions/{id}/model', _a(handle_put_session_model))
    # ─────────────────────────────────────────────────────────────────────────

    # Env var (secrets) management
    app.router.add_get('/myah/api/config/env', _a(handle_list_env))
    app.router.add_put('/myah/api/config/env', _a(handle_set_env))
    app.router.add_delete('/myah/api/config/env/{key}', _a(handle_delete_env))

    logger.info('Myah management routes registered (%d endpoints)', 29)
