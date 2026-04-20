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
        AsyncMock(return_value=False),
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
