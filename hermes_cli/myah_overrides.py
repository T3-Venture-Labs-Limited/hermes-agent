# ── Myah: provider catalog overrides (not present upstream) ─────────────────
"""MYAH_OVERRIDES — Myah-specific catalog augmentation.

Single-source-of-truth for V1-visible providers and Myah-specific hints
(validation URLs, custom-provider config for synthetic entries like plain
OpenAI, v1 gating). Adding a provider to the Myah picker: add one entry here.
Removing: delete the entry.
"""

MYAH_OVERRIDES: dict = {
    # OpenRouter: NOT in PROVIDER_REGISTRY upstream (treated as the
    # default fallback inside resolve_provider at auth.py:892). We
    # declare the full shape here.
    "openrouter": {
        "display_name": "OpenRouter",
        "description": "200+ models via one key",
        "auth_type": "api_key",
        "env_var": "OPENROUTER_API_KEY",
        "validation": {"url": "https://openrouter.ai/api/v1/auth/key",
                       "method": "GET", "auth": "bearer"},
        "inference_base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "v1_visible": True,
        "write_type": "env_var",
    },

    # OpenAI API: the bare "openai" slug aliases to "openrouter" in
    # providers.ALIASES:170-172, so we CANNOT use provider="openai" in
    # config.yaml — that would route through OpenRouter using an OpenAI
    # API key (rejected). Instead we write a providers: block and set
    # model.provider to "custom:openai-direct".
    "openai": {
        "display_name": "OpenAI API",
        "description": "Use your OpenAI developer API key (sk-...)",
        "auth_type": "api_key",
        "env_var": "OPENAI_API_KEY",
        "validation": {"url": "https://api.openai.com/v1/models",
                       "method": "GET", "auth": "bearer"},
        "inference_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "v1_visible": True,
        "write_type": "custom_provider",
        "custom_provider": {
            "slug": "openai-direct",
            "base_url": "https://api.openai.com/v1",
            "api_mode": "codex_responses",
            "model_provider_value": "custom:openai-direct",
        },
    },

    "openai-codex":  {"default_model": "gpt-5",             "v1_visible": True,  "write_type": "oauth_codex"},
    "anthropic":     {"default_model": "claude-sonnet-4.6", "v1_visible": True,  "write_type": "env_var"},
    "gemini":        {"default_model": "gemini-2.5-flash",  "v1_visible": True,  "write_type": "env_var"},
    "xai":           {"default_model": "grok-4",            "v1_visible": True,  "write_type": "env_var"},
    "deepseek":      {"default_model": "deepseek-chat",     "v1_visible": True,  "write_type": "env_var"},

    # All other Hermes providers (CANONICAL_PROVIDERS entries not listed
    # above) get auto-generated catalog entries with v1_visible=False.
}
# ─────────────────────────────────────────────────────────────────────────────
