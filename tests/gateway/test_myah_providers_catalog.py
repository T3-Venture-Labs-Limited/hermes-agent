"""Tests for Myah provider gateway endpoints.

Uses direct handler calls with mocked aiohttp requests — same pattern as
test_myah_management.py (no make_app_for_tests needed).
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_request(method="GET", path="/", match_info=None, json_body=None, query=None):
    """Build a mocked aiohttp Request for testing handlers.

    Uses make_mocked_request's native match_info kwarg instead of monkey-
    patching the Request class — class-level patching leaks to every other
    test (including test_myah_session_model.py) running in the same
    interpreter / xdist worker.
    """
    from aiohttp.test_utils import make_mocked_request
    headers = {"Authorization": "Bearer test-token"}
    kwargs = {"headers": headers}
    if match_info is not None:
        kwargs["match_info"] = match_info
    req = make_mocked_request(method, path, **kwargs)
    if query:
        req._rel_url = req._rel_url.with_query(query)
    if json_body is not None:
        async def _json():
            return json_body
        req.json = _json
    return req


@pytest.mark.asyncio
async def test_catalog_includes_all_v1_providers():
    """GET /myah/api/providers?visible=v1 returns every MYAH_OVERRIDES entry
    with v1_visible=True. The tier-1 seven must be present; tier-2 entries
    (zai, kimi, minimax, etc.) round out the catalog."""
    from gateway.platforms.myah_management import handle_list_providers
    from hermes_cli.myah_overrides import MYAH_OVERRIDES

    req = _make_request(query={"visible": "v1"})
    response = await handle_list_providers(req)
    assert response.status == 200
    body = json.loads(response.body)

    # Tier-1: the seven providers users see first
    tier_one = {"openrouter", "openai", "openai-codex", "anthropic",
                "gemini", "xai", "deepseek"}
    assert tier_one.issubset(set(body.keys()))

    # All v1_visible entries in MYAH_OVERRIDES must surface in the catalog
    expected = {slug for slug, override in MYAH_OVERRIDES.items()
                if override.get("v1_visible")}
    assert set(body.keys()) == expected


@pytest.mark.asyncio
async def test_catalog_openai_has_custom_provider_block():
    """The openai entry must have write_type=custom_provider."""
    from gateway.platforms.myah_management import handle_list_providers

    req = _make_request(query={"visible": "v1"})
    response = await handle_list_providers(req)
    body = json.loads(response.body)
    assert body["openai"]["write_type"] == "custom_provider"
    assert body["openai"]["custom_provider"]["model_provider_value"] == "custom:openai-direct"


@pytest.mark.asyncio
async def test_catalog_all_returns_complete_catalog():
    """GET /myah/api/providers?visible=all returns every MYAH_OVERRIDES
    entry plus any unclaimed CANONICAL_PROVIDERS slugs. With every
    override flagged v1_visible=True, the v1 filter and the all filter
    return the same set today — the test guarantees the all-view is a
    superset of the v1-view regardless of future gating."""
    from gateway.platforms.myah_management import handle_list_providers

    req_all = _make_request(query={"visible": "all"})
    all_resp = await handle_list_providers(req_all)
    all_body = json.loads(all_resp.body)

    req_v1 = _make_request(query={"visible": "v1"})
    v1_resp = await handle_list_providers(req_v1)
    v1_body = json.loads(v1_resp.body)

    assert set(v1_body.keys()).issubset(set(all_body.keys()))
    assert all_body, "catalog must never be empty"


@pytest.mark.asyncio
async def test_models_endpoint_unknown_provider_returns_404():
    """GET /myah/api/providers/not-a-real-provider/models returns 404."""
    from gateway.platforms.myah_management import handle_provider_models

    req = _make_request(match_info={"provider_id": "not-a-real-provider"})
    response = await handle_provider_models(req)
    assert response.status == 404


@pytest.mark.asyncio
async def test_connect_credential_missing_api_key_returns_400():
    """POST without api_key returns 400."""
    from gateway.platforms.myah_management import handle_connect_credential

    req = _make_request(method="POST", json_body={}, match_info={"provider_id": "openrouter"})
    response = await handle_connect_credential(req)
    assert response.status == 400


@pytest.mark.asyncio
async def test_connect_credential_unknown_provider_returns_404():
    """POST to unknown provider returns 404."""
    from gateway.platforms.myah_management import handle_connect_credential

    req = _make_request(
        method="POST",
        json_body={"api_key": "sk-test"},
        match_info={"provider_id": "does-not-exist"},
    )
    response = await handle_connect_credential(req)
    assert response.status == 404


@pytest.mark.asyncio
async def test_connect_credential_validation_failure_returns_400():
    """Validation failure returns 400."""
    from gateway.platforms.myah_management import handle_connect_credential

    req = _make_request(
        method="POST",
        json_body={"api_key": "bad-key"},
        match_info={"provider_id": "openrouter"},
    )
    with patch(
        "gateway.platforms.myah_management._validate_api_key",
        AsyncMock(return_value=(False, "auth denied by provider (HTTP 401)")),
    ):
        response = await handle_connect_credential(req)
    assert response.status == 400


@pytest.mark.asyncio
async def test_delete_all_credentials():
    """DELETE /myah/api/providers/{id} clears provider auth."""
    from gateway.platforms.myah_management import handle_delete_all_credentials

    req = _make_request(method="DELETE", match_info={"provider_id": "openrouter"})
    with patch("gateway.platforms.myah_management._build_catalog", AsyncMock(return_value={})):
        with patch("hermes_cli.auth.clear_provider_auth") as mock_clear:
            response = await handle_delete_all_credentials(req)
    assert response.status == 200
    mock_clear.assert_called_once_with("openrouter")


@pytest.mark.asyncio
async def test_oauth_start_unknown_provider_returns_404():
    """POST oauth/start for an unknown provider returns 404."""
    from gateway.platforms.myah_management import handle_oauth_start

    req = _make_request(method="POST", match_info={"provider_id": "no-such-provider"})
    response = await handle_oauth_start(req)
    assert response.status == 404


@pytest.mark.asyncio
async def test_oauth_poll_unknown_session_returns_404():
    """GET oauth/poll for a non-existent session returns 404."""
    from gateway.platforms.myah_management import handle_oauth_poll

    req = _make_request(
        match_info={"provider_id": "openai-codex", "session_id": "nonexistent-sid"}
    )
    response = await handle_oauth_poll(req)
    assert response.status == 404


@pytest.mark.asyncio
async def test_oauth_cancel_unknown_session_returns_ok_false():
    """DELETE oauth/sessions for a non-existent session returns ok=False."""
    from gateway.platforms.myah_management import handle_oauth_cancel

    req = _make_request(method="DELETE", match_info={"session_id": "nonexistent-sid"})
    response = await handle_oauth_cancel(req)
    body = json.loads(response.body)
    assert body["ok"] is False


# ── Myah: catalog contract tests ───────────────────────────────────────────
# These guard the "add a provider = one MYAH_OVERRIDES entry" promise.
# Every v1_visible entry must carry the fields the platform and Hermes rely
# on so that untested providers behave identically to tested ones
# (zai, opencode-zen) without per-provider code changes downstream.


_VALID_WRITE_TYPES = frozenset({
    "env_var",
    "custom_provider",
    "oauth_device_code",
    "oauth_codex",
    "oauth_external",     # coming-soon tile
    "external_process",   # coming-soon tile
})

# Providers whose write flow is a tile-only placeholder. They appear in the
# picker but have no working auth flow yet, so we don't require validation /
# env_var on them.
_COMING_SOON_WRITE_TYPES = frozenset({"oauth_external", "external_process"})


def _v1_visible_entries():
    from hermes_cli.myah_overrides import MYAH_OVERRIDES
    return {slug: entry for slug, entry in MYAH_OVERRIDES.items()
            if entry.get("v1_visible")}


def test_every_v1_visible_entry_has_default_model():
    """Default model drives _resolve_user_model in the platform and the
    initial model pin on first chat. Missing it = blank model selector."""
    missing = [slug for slug, entry in _v1_visible_entries().items()
               if not entry.get("default_model")]
    assert not missing, (
        f"MYAH_OVERRIDES entries missing 'default_model': {missing}. "
        "Every V1 provider must declare a sensible default so the first "
        "chat after connect doesn't hit a blank model."
    )


def test_every_v1_visible_entry_has_valid_write_type():
    """write_type drives handle_connect_credential routing. An invalid or
    missing value means the connect flow can't persist the credential."""
    invalid = {slug: entry.get("write_type")
               for slug, entry in _v1_visible_entries().items()
               if entry.get("write_type") not in _VALID_WRITE_TYPES}
    assert not invalid, (
        f"MYAH_OVERRIDES entries with invalid 'write_type': {invalid}. "
        f"Must be one of {sorted(_VALID_WRITE_TYPES)}."
    )


def test_api_key_providers_declare_env_var_or_custom_provider():
    """For write_type=env_var, handle_connect_credential needs entry.env_var
    to save_env_value(). For write_type=custom_provider, it needs the
    custom_provider block. Coming-soon tiles are exempt."""
    broken = []
    for slug, entry in _v1_visible_entries().items():
        wt = entry.get("write_type")
        if wt in _COMING_SOON_WRITE_TYPES:
            continue
        if wt == "env_var" and not entry.get("env_var"):
            # env_var may be inherited from PROVIDER_REGISTRY.api_key_env_vars
            # in hermes_cli.auth — we only fail if BOTH override and canonical
            # lack it, because _build_catalog at myah_management.py:1472-1474
            # falls back to api_key_env_vars[0] when override.env_var is absent.
            from hermes_cli.auth import PROVIDER_REGISTRY
            canonical = PROVIDER_REGISTRY.get(slug)
            has_canonical = bool(
                canonical and getattr(canonical, "api_key_env_vars", ())
            )
            if not has_canonical:
                broken.append(f"{slug} (env_var missing and no canonical)")
        elif wt == "custom_provider" and not entry.get("custom_provider"):
            broken.append(f"{slug} (custom_provider block missing)")
    assert not broken, (
        "MYAH_OVERRIDES entries can't persist credentials: "
        f"{broken}. Fix by adding env_var/custom_provider block, or mark "
        "as oauth_external/external_process if the flow isn't wired yet."
    )


def test_api_key_providers_with_validation_have_required_fields():
    """If an entry declares validation, the shape must match what
    _validate_api_key expects: url + method + auth."""
    bad = []
    for slug, entry in _v1_visible_entries().items():
        validation = entry.get("validation")
        if not validation:
            continue
        missing = [k for k in ("url", "method", "auth") if k not in validation]
        if missing:
            bad.append(f"{slug} missing {missing}")
        if validation.get("auth") not in (None, "bearer", "x-api-key", "query"):
            bad.append(f"{slug} has unknown auth style {validation.get('auth')!r}")
    assert not bad, (
        f"MYAH_OVERRIDES validation blocks malformed: {bad}. "
        "_validate_api_key reads url/method/auth — mismatched shape means "
        "the connect flow silently accepts every key."
    )


@pytest.mark.asyncio
async def test_catalog_exposes_env_var_for_every_api_key_provider():
    """End-to-end: the merged catalog (CANONICAL + HERMES_OVERLAYS +
    MYAH_OVERRIDES) must expose env_var on every v1_visible entry whose
    write_type is env_var, because handle_connect_credential reads it from
    the merged entry — not from MYAH_OVERRIDES directly."""
    from gateway.platforms.myah_management import handle_list_providers

    req = _make_request(query={"visible": "v1"})
    response = await handle_list_providers(req)
    body = json.loads(response.body)

    missing = []
    for slug, entry in body.items():
        wt = entry.get("write_type")
        if wt == "env_var" and not entry.get("env_var"):
            missing.append(slug)
    assert not missing, (
        f"Merged catalog entries missing env_var: {missing}. "
        "Platform can't inject credentials for these providers — chat "
        "will fail with 'API key missing' on first message."
    )


# ── Appendix Task E: _validate_api_key transient-failure tests ───────────────
# Pins the transient-failure handling added in the Myah marker block above.
# Only 401/403 should reject; all other conditions accept optimistically.
# ─────────────────────────────────────────────────────────────────────────────

_CATALOG_ENTRY_WITH_VALIDATION = {
    "validation": {
        "url": "https://api.example.com/validate",
        "method": "GET",
        "auth": "bearer",
    }
}


@pytest.mark.asyncio
async def test_validate_api_key_accepts_2xx():
    """200 response accepts the key."""
    from gateway.platforms.myah_management import _validate_api_key
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.request.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("gateway.platforms.myah_management._aiohttp.ClientSession",
               return_value=mock_session):
        accepted, reason = await _validate_api_key(_CATALOG_ENTRY_WITH_VALIDATION, "test-key")
    assert accepted is True
    assert "validated" in reason


@pytest.mark.asyncio
async def test_validate_api_key_rejects_401():
    """HTTP 401 rejects the key as a genuine auth failure."""
    from gateway.platforms.myah_management import _validate_api_key
    mock_resp = MagicMock()
    mock_resp.status = 401
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.request.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("gateway.platforms.myah_management._aiohttp.ClientSession",
               return_value=mock_session):
        accepted, reason = await _validate_api_key(_CATALOG_ENTRY_WITH_VALIDATION, "bad-key")
    assert accepted is False
    assert "auth denied" in reason


@pytest.mark.asyncio
async def test_validate_api_key_rejects_403():
    """HTTP 403 rejects the key as a genuine auth failure."""
    from gateway.platforms.myah_management import _validate_api_key
    mock_resp = MagicMock()
    mock_resp.status = 403
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.request.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("gateway.platforms.myah_management._aiohttp.ClientSession",
               return_value=mock_session):
        accepted, reason = await _validate_api_key(_CATALOG_ENTRY_WITH_VALIDATION, "bad-key")
    assert accepted is False
    assert "auth denied" in reason


@pytest.mark.asyncio
async def test_validate_api_key_optimistic_on_429():
    """HTTP 429 (rate-limit) accepts optimistically — not an auth failure."""
    from gateway.platforms.myah_management import _validate_api_key
    mock_resp = MagicMock()
    mock_resp.status = 429
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.request.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("gateway.platforms.myah_management._aiohttp.ClientSession",
               return_value=mock_session):
        accepted, reason = await _validate_api_key(_CATALOG_ENTRY_WITH_VALIDATION, "valid-key")
    assert accepted is True
    assert "optimistic" in reason


@pytest.mark.asyncio
async def test_validate_api_key_optimistic_on_500():
    """HTTP 500 (server error) accepts optimistically."""
    from gateway.platforms.myah_management import _validate_api_key
    mock_resp = MagicMock()
    mock_resp.status = 500
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.request.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("gateway.platforms.myah_management._aiohttp.ClientSession",
               return_value=mock_session):
        accepted, reason = await _validate_api_key(_CATALOG_ENTRY_WITH_VALIDATION, "valid-key")
    assert accepted is True
    assert "optimistic" in reason


@pytest.mark.asyncio
async def test_validate_api_key_optimistic_on_timeout():
    """asyncio.TimeoutError accepts optimistically."""
    import asyncio
    from gateway.platforms.myah_management import _validate_api_key
    mock_session = MagicMock()
    mock_session.request.side_effect = asyncio.TimeoutError()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("gateway.platforms.myah_management._aiohttp.ClientSession",
               return_value=mock_session):
        accepted, reason = await _validate_api_key(_CATALOG_ENTRY_WITH_VALIDATION, "valid-key")
    assert accepted is True
    assert "timeout" in reason


@pytest.mark.asyncio
async def test_validate_api_key_optimistic_on_network_error():
    """Generic network exception accepts optimistically."""
    from gateway.platforms.myah_management import _validate_api_key
    mock_session = MagicMock()
    mock_session.request.side_effect = Exception("DNS resolution failed")
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("gateway.platforms.myah_management._aiohttp.ClientSession",
               return_value=mock_session):
        accepted, reason = await _validate_api_key(_CATALOG_ENTRY_WITH_VALIDATION, "valid-key")
    assert accepted is True
    assert "optimistic" in reason
    assert "error" in reason


@pytest.mark.asyncio
async def test_connect_credential_returns_400_on_auth_denied():
    """Full handler: mock validation returning auth denied yields 400 with reason."""
    from gateway.platforms.myah_management import handle_connect_credential
    req = _make_request(
        method="POST",
        json_body={"api_key": "invalid-key"},
        match_info={"provider_id": "openrouter"},
    )
    with patch(
        "gateway.platforms.myah_management._validate_api_key",
        AsyncMock(return_value=(False, "auth denied by provider (HTTP 401)")),
    ):
        response = await handle_connect_credential(req)
    assert response.status == 400
    body = json.loads(response.body)
    assert "auth denied" in body["error"]


@pytest.mark.asyncio
async def test_connect_credential_succeeds_on_optimistic_accept(tmp_path, monkeypatch):
    """Full handler: optimistic accept still persists the credential."""
    from gateway.platforms.myah_management import handle_connect_credential
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    req = _make_request(
        method="POST",
        json_body={"api_key": "valid-key"},
        match_info={"provider_id": "openrouter"},
    )
    with patch(
        "gateway.platforms.myah_management._validate_api_key",
        AsyncMock(return_value=(True, "optimistic accept (validation timeout)")),
    ), patch(
        "hermes_cli.config.save_env_value",
    ) as mock_save, patch(
        "agent.credential_pool.load_pool",
        return_value=MagicMock(entries=[], save=MagicMock()),
    ):
        response = await handle_connect_credential(req)
    # Should not return 400 — key was persisted despite optimistic accept
    assert response.status != 400
