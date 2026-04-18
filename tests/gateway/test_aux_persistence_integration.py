"""Integration: PATCH aux.vision.model survive simulated restart and env reflects it.

Skipped unless HERMES_INTEGRATION_TESTS=1. Requires docker + myah-agent:local image.
"""
import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('HERMES_INTEGRATION_TESTS') != '1',
    reason='Set HERMES_INTEGRATION_TESTS=1 to run',
)


@pytest.fixture(scope='module')
def agent_container():
    name = 'myah-agent-test-persist'
    subprocess.run(['docker', 'rm', '-f', name], check=False, capture_output=True)

    subprocess.run([
        'docker', 'run', '-d', '--name', name,
        '-p', '8091:8080',
        '-e', 'MYAH_AUTH_KEY=test-key',
        '-e', 'HERMES_MODEL=anthropic/claude-opus-4.6',
        'myah-agent:local',
    ], check=True)

    try:
        import requests
    except ImportError:
        pytest.skip('requests not installed')

    for _ in range(30):
        try:
            r = requests.get('http://localhost:8091/health', timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        subprocess.run(['docker', 'logs', name])
        pytest.fail('agent did not become healthy')

    yield 'http://localhost:8091'

    subprocess.run(['docker', 'rm', '-f', name], check=False, capture_output=True)


def test_patch_aux_vision_model_persists_across_restart(agent_container):
    """PATCH vision model survives supervisorctl restart hermes within container."""
    import requests

    base = agent_container
    headers = {'Authorization': 'Bearer test-key', 'Content-Type': 'application/json'}

    # Patch the model
    r = requests.patch(
        f'{base}/myah/api/config',
        json={'auxiliary.vision.model': 'google/gemini-2.5-flash'},
        headers=headers,
        timeout=10,
    )
    assert r.status_code == 200, f'PATCH failed: {r.text}'

    # Restart hermes inside the container
    subprocess.run(
        ['docker', 'exec', 'myah-agent-test-persist', 'supervisorctl', 'restart', 'hermes'],
        check=True,
        timeout=30,
    )
    time.sleep(5)

    # Wait for health
    for _ in range(20):
        try:
            r2 = requests.get(f'{base}/health', timeout=1)
            if r2.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        pytest.fail('agent did not recover after restart')

    # Read config back
    r3 = requests.get(f'{base}/myah/api/config', headers=headers, timeout=10)
    assert r3.status_code == 200
    cfg = r3.json()
    assert cfg.get('auxiliary', {}).get('vision', {}).get('model') == 'google/gemini-2.5-flash', \
        f"Expected google/gemini-2.5-flash, got: {cfg.get('auxiliary', {}).get('vision', {})}"
