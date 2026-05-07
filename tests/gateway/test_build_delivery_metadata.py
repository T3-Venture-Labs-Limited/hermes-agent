"""BasePlatformAdapter.build_delivery_metadata — optional polymorphic hook
for cron-delivery metadata enrichment.

Subclasses override; the default returns base_metadata unchanged.
Mirrors the on_processing_start/on_processing_complete optional-method
pattern at gateway/platforms/base.py:1871-1885.

Tier 2B Task 2B.4 / Phase 4f: this hook replaces the hardcoded
``if platform_name.lower() == "myah":`` enrichment branch in
cron/scheduler.py:_deliver_result. The same diff is queued as upstream
PR U-CRON for eventual upstream landing.
"""
from __future__ import annotations

import inspect

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter that does NOT override build_delivery_metadata.

    Used to exercise the default implementation. The override-side tests
    live in the plugin's test suite (test_build_delivery_metadata.py)
    where MyahAdapter is exercised.
    """

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):  # pragma: no cover — abstract impl
        return None


def test_default_returns_base_metadata_unchanged():
    """The default implementation is a no-op — returns base_metadata as-is."""
    adapter = _StubAdapter()
    base = {"thread_id": "t-1"}
    result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="ok", base_metadata=base
    )
    assert result == base


def test_default_handles_missing_base_metadata():
    """base_metadata=None returns an empty dict, not None."""
    adapter = _StubAdapter()
    result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="ok", base_metadata=None
    )
    assert result == {}


def test_default_returns_copy_not_reference():
    """The default returns a copy — caller mutations don't affect base_metadata."""
    adapter = _StubAdapter()
    base = {"thread_id": "t-1"}
    result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="ok", base_metadata=base
    )
    result["thread_id"] = "mutated"
    assert base["thread_id"] == "t-1"


def test_signature_is_stable():
    """Method signature: (self, job, status_hint='ok', base_metadata=None)."""
    sig = inspect.signature(BasePlatformAdapter.build_delivery_metadata)
    params = list(sig.parameters)
    assert params == ["self", "job", "status_hint", "base_metadata"]
    assert sig.parameters["status_hint"].default == "ok"
    assert sig.parameters["base_metadata"].default is None
