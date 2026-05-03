# myah-hermes-plugin

Myah platform connector for [Hermes Agent](https://github.com/NousResearch/Hermes-Agent).

This pip package is the upstream-supported integration point for the
[Myah](https://app.myah.dev) self-hosted web platform. It registers a
Hermes platform adapter, Myah-specific tools, and an admin dashboard
plugin via Hermes' standard plugin extension model
(`hermes_agent.plugins` entry point).

## Status

**Skeleton only.** Phase 4 of the Myah OSS readiness initiative populates
this package incrementally:

- Phase 4b — *this PR* — empty skeleton, pip-installable, entry-point registered.
- Phase 4c — moves `tools/secrets_tool.py` from the hermes fork into this plugin.
- Phase 4d — moves `gateway/platforms/myah.py` (the platform adapter) here, calling
  `ctx.register_platform()`.
- Phase 4e — moves `plugins/myah-admin/` (the admin dashboard plugin) here.
- Phase 4f — moves cron/status_hint/boot_md hooks here.

After Phase 4f, the hermes fork has zero Myah-specific code — `hermes update`
runs cleanly against upstream.

## Install

```bash
# From the hermes fork checkout. Use --no-deps until hermes-agent is
# published to PyPI (the runtime that loads this plugin already has
# hermes-agent importable, so the dep declaration is documentary).
pip install -e plugins/myah-hermes-plugin --no-deps
```

After install, verify the entry point is registered:

```bash
python -c "import importlib.metadata as m; \
  eps = m.entry_points(group='hermes_agent.plugins'); \
  print([e.name for e in eps])"
```

Should include `myah-platform` in the output.

## License

MIT — see `LICENSE`.
