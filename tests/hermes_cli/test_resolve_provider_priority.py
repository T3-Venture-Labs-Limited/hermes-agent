"""Regression tests for ``hermes_cli.auth.resolve_provider("auto")``.

These tests pin the priority chain that ``resolve_provider`` follows when
the caller asks for ``"auto"`` (or omits ``requested`` entirely).  The
priority is documented in the function's own docstring at
``hermes_cli/auth.py:1112``:

    1. ``auth.json:active_provider`` — when ``get_auth_status`` reports
       the provider as logged in.
    2. Explicit one-off CLI ``api_key``/``base_url`` arguments — always
       map to ``openrouter``/``custom`` regardless of any other state.
    3. ``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY`` env-var heuristic
       (returns ``openrouter``).
    4. Per-provider API-key env-var fallback (returns the matching
       provider id, e.g. ``GLM_API_KEY`` → ``zai``).
    5. AWS Bedrock when ``boto3`` reports usable credentials.
    6. Last-resort: raise ``AuthError(code="no_provider_configured")``.

These tests are pure additions — they assert the *current* behavior of
``hermes_cli.auth.resolve_provider`` and the contract that the Myah
platform plugin's ``POST /myah/v1/active-provider`` endpoint relies on.
They do not exercise any code that was changed in
``fix/active-provider-endpoint``.

Test patterns follow ``tests/hermes_cli/test_runtime_provider_resolution.py``
(monkeypatching helpers on ``hermes_cli.auth``) and
``tests/hermes_cli/test_profiles.py`` (using ``HERMES_HOME`` from the
autouse ``_hermetic_environment`` fixture instead of mocking
``Path.home``).
"""

import json
from pathlib import Path

import pytest

from hermes_cli import auth as auth_mod
from hermes_cli.auth import AuthError, resolve_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_auth_store(active_provider, providers=None, credential_pool=None):
    """Persist an ``auth.json`` with the given fields via the public save path.

    The autouse ``_hermetic_environment`` fixture in ``tests/conftest.py``
    points ``HERMES_HOME`` at a per-test tempdir, so this writes inside
    that sandbox.
    """
    store = {
        "version": auth_mod.AUTH_STORE_VERSION,
        "providers": providers or {},
    }
    if active_provider is not None:
        store["active_provider"] = active_provider
    if credential_pool is not None:
        store["credential_pool"] = credential_pool
    return auth_mod._save_auth_store(store)


def _clear_provider_env(monkeypatch):
    """Remove every API-key env var the auto-detection chain consults.

    The autouse hermetic fixture already strips credential-shaped vars,
    but we redundantly clear the specific ones ``resolve_provider``
    examines so a stray local export can never contaminate a test.
    """
    for name in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GLM_API_KEY",
        "ZAI_API_KEY",
        "Z_AI_API_KEY",
        "KIMI_API_KEY",
        "KIMI_CODING_API_KEY",
        "KIMI_CN_API_KEY",
        "STEPFUN_API_KEY",
        "ARCEEAI_API_KEY",
        "GMI_API_KEY",
        "MINIMAX_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "AI_GATEWAY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def no_bedrock(monkeypatch):
    """Disable Bedrock auto-detection so it never short-circuits a test.

    ``has_aws_credentials`` is imported lazily inside ``resolve_provider``,
    so we patch the symbol on its source module.
    """
    from agent import bedrock_adapter

    monkeypatch.setattr(bedrock_adapter, "has_aws_credentials", lambda: False)
    return monkeypatch


# ===========================================================================
# Branch 1: active_provider in auth.json wins when get_auth_status says
# logged_in.
# ===========================================================================


class TestActiveProviderBranch:
    def test_active_provider_logged_in_wins(self, monkeypatch, no_bedrock):
        """When auth.json says active_provider=X and X is logged_in,
        resolve_provider('auto') returns X."""
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider="openai-codex")

        # Force get_auth_status to report logged_in regardless of provider
        # internals — we're testing the resolve_provider branch, not the
        # status dispatcher.
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        assert resolve_provider("auto") == "openai-codex"

    def test_active_provider_overrides_env_var_heuristic(
        self, monkeypatch, no_bedrock
    ):
        """active_provider takes priority over OPENAI_API_KEY / OPENROUTER_API_KEY.

        This is the critical contract: an env-var key in the user's
        environment must NOT override what the user explicitly chose
        via ``hermes auth login``.
        """
        _clear_provider_env(monkeypatch)
        # Both env vars set — the heuristic at branch 3 would fire if
        # branch 1 didn't pre-empt it.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-should-lose")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-lose")

        _seed_auth_store(active_provider="openai-codex")
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        assert resolve_provider("auto") == "openai-codex"

    def test_active_provider_not_logged_in_falls_through(
        self, monkeypatch, no_bedrock
    ):
        """A stale active_provider whose credentials have been revoked
        must NOT be returned — we fall through to the env-var heuristic.

        Without this behavior, a user whose OAuth refresh token was
        revoked would be locked into the broken provider with no way
        for the auto-resolver to recover.
        """
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fallback")

        _seed_auth_store(active_provider="openai-codex")
        # Simulate revoked / expired creds for the active provider.
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": False, "provider": provider_id},
        )

        # Falls through to branch 3 (OPENROUTER_API_KEY → openrouter).
        assert resolve_provider("auto") == "openrouter"

    def test_active_provider_unknown_id_falls_through(
        self, monkeypatch, no_bedrock
    ):
        """An ``active_provider`` value that isn't in ``PROVIDER_REGISTRY``
        is ignored — the resolver doesn't trust arbitrary strings."""
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fallback")

        _seed_auth_store(active_provider="not-a-real-provider")
        # If get_auth_status were called we'd notice — guard with a
        # tripwire so we know branch 1 short-circuited correctly.
        called = []
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: (called.append(provider_id),
                                      {"logged_in": True})[1],
        )

        assert resolve_provider("auto") == "openrouter"
        # The unknown id is filtered before get_auth_status is invoked.
        assert "not-a-real-provider" not in called


# ===========================================================================
# Branch 2: explicit api_key / base_url short-circuit.
# ===========================================================================


class TestExplicitCredentialBranch:
    def test_explicit_api_key_returns_openrouter_even_with_active_provider(
        self, monkeypatch, no_bedrock
    ):
        """An explicit one-off ``api_key`` argument forces openrouter,
        regardless of any persisted ``active_provider``.

        This is documented behavior — a user passing ``--api-key`` on
        the CLI is overriding everything else for that single invocation.
        """
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider="openai-codex")
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        assert (
            resolve_provider("auto", explicit_api_key="sk-explicit")
            == "openrouter"
        )
        assert (
            resolve_provider("auto", explicit_base_url="https://example.com/v1")
            == "openrouter"
        )


# ===========================================================================
# Branch 3: OPENAI_API_KEY / OPENROUTER_API_KEY env-var heuristic.
# ===========================================================================


class TestOpenRouterEnvVarBranch:
    def test_openrouter_api_key_picked_when_no_active_provider(
        self, monkeypatch, no_bedrock
    ):
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider=None)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        assert resolve_provider("auto") == "openrouter"

    def test_openai_api_key_picked_when_no_active_provider(
        self, monkeypatch, no_bedrock
    ):
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider=None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        assert resolve_provider("auto") == "openrouter"


# ===========================================================================
# Branch 4: per-provider API-key env-var auto-detection.
# ===========================================================================


class TestPerProviderEnvVarBranch:
    def test_provider_specific_env_var_detected(self, monkeypatch, no_bedrock):
        """A provider-specific API-key env var (e.g. ``GLM_API_KEY``)
        causes auto-detection to return that provider when no
        OpenAI/OpenRouter key and no ``active_provider`` are present.

        We deliberately do not pin the provider id to a specific name —
        we assert *that* a provider was detected and that it matches
        the provider whose ``api_key_env_vars`` we set.
        """
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider=None)

        # Pick any non-copilot api_key provider from the registry; copilot
        # is explicitly skipped at line 1206 of auth.py.
        target_id = None
        target_env_var = None
        for pid, pconfig in auth_mod.PROVIDER_REGISTRY.items():
            if pconfig.auth_type != "api_key" or pid == "copilot":
                continue
            if pconfig.api_key_env_vars:
                target_id = pid
                target_env_var = pconfig.api_key_env_vars[0]
                break
        assert target_id is not None, (
            "PROVIDER_REGISTRY has no non-copilot api_key provider — "
            "test setup precondition broken"
        )

        monkeypatch.setenv(target_env_var, "test-key-value")

        assert resolve_provider("auto") == target_id

    def test_copilot_token_does_not_hijack_auto_detect(
        self, monkeypatch, no_bedrock
    ):
        """``GH_TOKEN`` / ``GITHUB_TOKEN`` are commonly set for repo
        access — they must NOT cause ``resolve_provider("auto")`` to
        return ``copilot``. This is enforced at auth.py:1206-1207.

        With no other credentials available, the resolver should fall
        through past branch 4 and ultimately raise
        ``no_provider_configured``.
        """
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider=None)
        monkeypatch.setenv("GH_TOKEN", "ghp_some-github-token")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_another-token")

        with pytest.raises(AuthError) as excinfo:
            resolve_provider("auto")
        assert excinfo.value.code == "no_provider_configured"


# ===========================================================================
# Branch 5: AWS Bedrock fallback.
# ===========================================================================


class TestBedrockBranch:
    def test_bedrock_returned_when_aws_credentials_present(self, monkeypatch):
        """When no API-key env var matches but boto3 reports AWS
        credentials, the resolver returns ``bedrock``."""
        from agent import bedrock_adapter

        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider=None)
        monkeypatch.setattr(bedrock_adapter, "has_aws_credentials", lambda: True)

        assert resolve_provider("auto") == "bedrock"

    def test_bedrock_runs_after_api_key_providers(self, monkeypatch):
        """If both an API-key env var AND AWS credentials are present,
        the API-key provider wins (Bedrock is the lower-priority
        fallback, not a peer)."""
        from agent import bedrock_adapter

        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider=None)
        monkeypatch.setattr(bedrock_adapter, "has_aws_credentials", lambda: True)

        # Find a non-copilot api_key provider with at least one env var
        # to set; this exercises the "API-key beats bedrock" rule.
        target_id = None
        target_env_var = None
        for pid, pconfig in auth_mod.PROVIDER_REGISTRY.items():
            if pconfig.auth_type != "api_key" or pid == "copilot":
                continue
            if pconfig.api_key_env_vars:
                target_id = pid
                target_env_var = pconfig.api_key_env_vars[0]
                break
        assert target_id is not None
        monkeypatch.setenv(target_env_var, "test-key-value")

        assert resolve_provider("auto") == target_id


# ===========================================================================
# Branch 6: last-resort error.
# ===========================================================================


class TestNoProviderConfiguredBranch:
    def test_raises_auth_error_when_nothing_available(
        self, monkeypatch, no_bedrock
    ):
        """With no active_provider, no env-var keys, no AWS creds, the
        resolver raises ``AuthError`` with code ``no_provider_configured``."""
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider=None)

        with pytest.raises(AuthError) as excinfo:
            resolve_provider("auto")
        assert excinfo.value.code == "no_provider_configured"


# ===========================================================================
# Explicit (non-"auto") requested provider.
# ===========================================================================


class TestExplicitRequestedProvider:
    def test_explicit_provider_returned_directly(self, monkeypatch, no_bedrock):
        """When ``requested`` is a real provider id, it is returned
        without consulting auth.json or env vars."""
        _clear_provider_env(monkeypatch)
        # Seed a different active_provider to prove it's ignored.
        _seed_auth_store(active_provider="zai")
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        # Pick a known, registered provider id that's clearly distinct.
        assert resolve_provider("openai-codex") == "openai-codex"

    def test_explicit_provider_does_not_consult_auth_store(
        self, monkeypatch, no_bedrock
    ):
        """Tripwire: if ``requested`` is explicit, ``_load_auth_store``
        must not be called at all."""
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider="zai")

        called = []
        original = auth_mod._load_auth_store
        monkeypatch.setattr(
            auth_mod, "_load_auth_store",
            lambda *a, **kw: (called.append(True), original(*a, **kw))[1],
        )

        resolve_provider("openai-codex")
        assert called == [], (
            "resolve_provider with an explicit provider id consulted "
            "the auth store — the active_provider branch should be "
            "skipped entirely"
        )

    def test_explicit_alias_resolves(self, monkeypatch, no_bedrock):
        """Aliases like 'claude' → 'anthropic' resolve directly without
        consulting auth.json. Picks an alias that's stable across
        provider renames (``claude`` has been the alias for anthropic
        since the provider was added)."""
        _clear_provider_env(monkeypatch)
        _seed_auth_store(active_provider="zai")
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        # We deliberately don't snapshot the alias map — we just confirm
        # that the alias maps to its target provider, not to the
        # active_provider in auth.json.
        result = resolve_provider("claude")
        assert result != "zai"
        assert result in auth_mod.PROVIDER_REGISTRY


# ===========================================================================
# Bug B regression: Myah platform contract.
# ===========================================================================


class TestBugBRegression:
    """Regression for the May 6 cron-and-chat active-provider sync bug.

    Bug B was: when ``auth.json:credential_pool`` contained credentials
    for multiple providers (e.g. both ``openai-codex`` and
    ``openrouter``), ``resolve_provider("auto")`` was supposed to honor
    ``active_provider`` rather than auto-detecting from the pool.

    The Myah platform plugin's ``POST /myah/v1/active-provider``
    endpoint (added in PR ``fix/active-provider-endpoint``) writes
    ``active_provider`` so that subsequent ``resolve_provider("auto")``
    calls — including those made by cron jobs — return the user's
    chosen provider, not whatever happens to be discoverable in the
    credential pool.

    These tests pin that contract.
    """

    def test_active_provider_wins_with_multi_provider_pool(
        self, monkeypatch, no_bedrock
    ):
        """With both ``openai-codex`` and ``openrouter`` credentials in
        the pool, ``resolve_provider("auto")`` returns whatever
        ``active_provider`` says — never auto-detects from the pool."""
        _clear_provider_env(monkeypatch)
        # Simulate the pool layout the Myah platform produces: an entry
        # per credential, keyed by provider id.
        _seed_auth_store(
            active_provider="openai-codex",
            credential_pool={
                "openai-codex": {
                    "entries": [
                        {"access_token": "codex-token", "source": "manual"}
                    ],
                },
                "openrouter": {
                    "entries": [
                        {"access_token": "or-token", "source": "manual"}
                    ],
                },
            },
            providers={
                "openai-codex": {"logged_in": True},
                "openrouter": {"logged_in": True},
            },
        )
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        assert resolve_provider("auto") == "openai-codex"

    def test_active_provider_switches_change_resolution(
        self, monkeypatch, no_bedrock
    ):
        """Flipping ``active_provider`` flips the resolved provider —
        the pool composition is irrelevant when ``active_provider`` is
        set and logged_in.

        This is the exact behavior the Myah ``POST /myah/v1/active-provider``
        endpoint relies on: the platform's onboarding handler writes
        ``active_provider``, and the next ``resolve_provider("auto")``
        from a cron job picks up the change.

        Note: ``openrouter`` is not a registered provider id (it's
        a magic string in the resolver), so this test uses two
        registered OAuth providers — ``openai-codex`` and ``nous`` —
        which is what the Myah platform actually writes.
        """
        _clear_provider_env(monkeypatch)
        pool = {
            "openai-codex": {
                "entries": [{"access_token": "codex-token", "source": "manual"}],
            },
            "nous": {
                "entries": [{"access_token": "nous-token", "source": "manual"}],
            },
        }
        providers = {
            "openai-codex": {"logged_in": True},
            "nous": {"logged_in": True},
        }
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        _seed_auth_store(
            active_provider="nous",
            credential_pool=pool,
            providers=providers,
        )
        assert resolve_provider("auto") == "nous"

        _seed_auth_store(
            active_provider="openai-codex",
            credential_pool=pool,
            providers=providers,
        )
        assert resolve_provider("auto") == "openai-codex"

    def test_active_provider_persisted_to_disk_is_honored(
        self, monkeypatch, no_bedrock
    ):
        """The contract isn't ``in-memory state`` — it's the JSON file
        on disk. Read-back through ``_load_auth_store`` must return the
        ``active_provider`` we wrote, and ``resolve_provider`` must
        honor it.

        This is what cron jobs see: a fresh process loads ``auth.json``
        from disk; whatever the platform's onboarding endpoint last
        wrote is what wins.
        """
        _clear_provider_env(monkeypatch)
        path = _seed_auth_store(active_provider="openai-codex")
        monkeypatch.setattr(
            auth_mod, "get_auth_status",
            lambda provider_id=None: {"logged_in": True, "provider": provider_id},
        )

        # Sanity-check the on-disk file matches what we expect.
        on_disk = json.loads(Path(path).read_text())
        assert on_disk["active_provider"] == "openai-codex"

        assert resolve_provider("auto") == "openai-codex"
