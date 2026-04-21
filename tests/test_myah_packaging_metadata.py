"""Myah-specific packaging metadata tests.

Mirrors the upstream `test_packaging_metadata.py` pattern (which asserts
on pyproject.toml structure to keep heavy deps out of the base install)
but focused on the extras Myah's agent/Dockerfile depends on.

If one of these assertions fails, the image build will still succeed but
the runtime will crash when a handler reaches for the module. Catching
it here fails PR CI in ~5 seconds instead of discovering it during a
6-minute image rebuild or, worse, in prod.

Part of T3-XXX post-incident hardening (Tier-1, Component C -- pytest layer).
See docs/gotchas/optional-extra-drift.md for context.
"""

from pathlib import Path

try:
    import tomllib  # stdlib 3.11+
except ImportError:
    import tomli as tomllib  # 3.10 fallback

# This test lives INSIDE the hermes submodule at agent/hermes/tests/.
# From here: tests/ -> hermes/ -> agent/ -> myah-repo-root/
THIS_FILE = Path(__file__).resolve()
HERMES_ROOT = THIS_FILE.parent.parent  # agent/hermes/
MYAH_ROOT = HERMES_ROOT.parent.parent  # myah repo root
AGENT_DOCKERFILE = MYAH_ROOT / 'agent' / 'Dockerfile'


# Extras Myah's Dockerfile currently installs. If you add a new extra to
# the Dockerfile's pip install line, add it here. If you add a new handler
# that uses an optional extra NOT in this list, add it to BOTH places.
MYAH_REQUIRED_EXTRAS = {'messaging', 'cron', 'honcho', 'mcp', 'voice', 'pty', 'web'}


def test_all_myah_required_extras_are_declared_in_pyproject():
    """Every extra Myah's Dockerfile installs must exist in hermes pyproject.toml."""
    data = tomllib.loads((HERMES_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    declared = set(data['project']['optional-dependencies'].keys())
    missing = MYAH_REQUIRED_EXTRAS - declared
    assert not missing, (
        f'Myah Dockerfile installs extras that are not declared in '
        f"hermes pyproject.toml: {missing}. Either add them to pyproject.toml's "
        f'[project.optional-dependencies] or remove them from agent/Dockerfile.'
    )


def test_agent_dockerfile_installs_all_myah_required_extras():
    """The extras list in agent/Dockerfile must include every required extra."""
    if not AGENT_DOCKERFILE.exists():
        # Submodule may be tested in isolation (without the myah repo root).
        # Skip rather than fail — the test is most useful in the full repo context.
        import pytest
        pytest.skip(f'agent/Dockerfile not found at {AGENT_DOCKERFILE}')

    dockerfile_text = AGENT_DOCKERFILE.read_text(encoding='utf-8')
    # Find the line that does `pip install ... -e "/opt/hermes-source[...]"`.
    # The bracketed list must contain each required extra.
    # This is a string-match sanity check, not a full parser — intentional;
    # keeps the test self-contained and easy to update.
    for extra in MYAH_REQUIRED_EXTRAS:
        assert extra in dockerfile_text, (
            f"agent/Dockerfile must install the '{extra}' extra. "
            f"Grep for the pip install line and confirm the extras list includes '{extra}'. "
            f'See docs/gotchas/optional-extra-drift.md'
        )


def test_web_extra_contains_fastapi_and_uvicorn():
    """Sanity: the [web] extra actually provides fastapi/uvicorn (the 2026-04-20 cause)."""
    data = tomllib.loads((HERMES_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    web_extra = data['project']['optional-dependencies'].get('web', [])
    web_names = {
        dep.split('[')[0].split('=')[0].split('>')[0].split('<')[0].strip().lower()
        for dep in web_extra
    }
    # fastapi and uvicorn are the two modules hermes_cli/web_server.py imports.
    # If the [web] extra ever stops providing them, the SystemExit probe in
    # myah_management.py fires and the gateway can't serve OAuth.
    assert 'fastapi' in web_names, f'[web] extra must include fastapi; got {web_names}'
    assert 'uvicorn' in web_names, f'[web] extra must include uvicorn; got {web_names}'
