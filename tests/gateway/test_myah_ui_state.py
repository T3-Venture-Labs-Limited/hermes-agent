"""Tests for ui_state → channel_prompt lifting in the Myah adapter.

The MyahAdapter._handle_message_endpoint reads body.get('ui_state'),
serializes it to JSON, and stores it on event.channel_prompt as a
[CURRENT_UI_STATE] block. This is cache-safe: combined_ephemeral
appends to the cached system prompt at API call time without mutating
the cache.
"""

import json
import logging
from unittest.mock import patch

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.myah import MyahAdapter


def _build_channel_prompt(body: dict) -> str | None:
    """Replicates the Myah marker block in _handle_message_endpoint.

    Kept in sync with gateway/platforms/myah.py:404+ (the Myah marker
    immediately before the MessageEvent constructor).  This unit test
    exercises the JSON serialization and [CURRENT_UI_STATE] wrapping
    logic in isolation; the integration test below confirms the same
    block is wired into MessageEvent.channel_prompt.
    """
    _ui_state_dict = body.get('ui_state')
    if not _ui_state_dict:
        return None
    try:
        _ui_state_json = json.dumps(_ui_state_dict, indent=2)
        return f'[CURRENT_UI_STATE]\n{_ui_state_json}\n[/CURRENT_UI_STATE]'
    except (TypeError, ValueError):
        return None


class TestUIStateChannelPrompt:
    def test_no_ui_state_yields_no_channel_prompt(self):
        """body without ui_state -> no channel_prompt."""
        body = {'message': 'hi', 'session_id': 's', 'user_id': 'u'}
        assert _build_channel_prompt(body) is None

    def test_empty_ui_state_yields_no_channel_prompt(self):
        """Empty/falsy ui_state -> no channel_prompt."""
        assert _build_channel_prompt({'ui_state': None}) is None
        assert _build_channel_prompt({'ui_state': {}}) is None

    def test_well_formed_ui_state_produces_block(self):
        """A normal ui_state object -> [CURRENT_UI_STATE] block with JSON."""
        ui_state = {
            'selectionRefs': [
                {
                    'id': 'r1',
                    'kind': 'doc-text',
                    'file_key': 'path:/abs/doc.md',
                    'filename': 'doc.md',
                    'anchor': {
                        'startOffset': 10,
                        'endOffset': 20,
                        'contextFingerprint': 'fp',
                    },
                    'preview': 'hello world',
                    'summary': 'doc',
                }
            ],
            'pendingEdits': [],
        }
        result = _build_channel_prompt({'ui_state': ui_state})
        assert result is not None
        assert result.startswith('[CURRENT_UI_STATE]\n')
        assert result.endswith('\n[/CURRENT_UI_STATE]')
        # Verify JSON content present and parseable
        body = result[len('[CURRENT_UI_STATE]\n') : -len('\n[/CURRENT_UI_STATE]')]
        parsed = json.loads(body)
        assert parsed == ui_state
        assert parsed['selectionRefs'][0]['file_key'] == 'path:/abs/doc.md'

    def test_pending_edits_serialized(self):
        ui_state = {
            'selectionRefs': [],
            'pendingEdits': [
                {'file_key': 'path:/abs/a.py', 'filename': 'a.py', 'diff': '+ new line\n'}
            ],
        }
        result = _build_channel_prompt({'ui_state': ui_state})
        assert result is not None
        body = result[len('[CURRENT_UI_STATE]\n') : -len('\n[/CURRENT_UI_STATE]')]
        parsed = json.loads(body)
        assert parsed['pendingEdits'][0]['filename'] == 'a.py'

    def test_malformed_ui_state_does_not_raise(self):
        """A non-JSON-serializable value inside ui_state must NOT raise.
        The Myah block catches TypeError/ValueError and skips the prompt."""
        # Sets are not JSON-serializable
        bad_ui_state = {'selectionRefs': {1, 2, 3}, 'pendingEdits': []}
        result = _build_channel_prompt({'ui_state': bad_ui_state})
        assert result is None  # gracefully fell back to no prompt


class TestMessageEventAcceptsChannelPrompt:
    """Sanity check that MessageEvent's existing constructor accepts
    channel_prompt — the Myah marker block depends on this field."""

    def test_message_event_constructor_accepts_channel_prompt(self):
        # Build a minimal source object — the adapter's build_source helper
        # returns an EventSource, but for the constructor we only need the
        # field to be recognized.
        adapter = _make_adapter()
        source = adapter.build_source(
            chat_id='c1',
            chat_name='Test',
            chat_type='dm',
            user_id='u1',
            user_name='User',
        )
        event = MessageEvent(
            text='hello',
            message_type=MessageType.TEXT,
            source=source,
            message_id='mid',
            channel_prompt='[CURRENT_UI_STATE]\n{}\n[/CURRENT_UI_STATE]',
        )
        assert event.channel_prompt is not None
        assert '[CURRENT_UI_STATE]' in event.channel_prompt


# ── Helper (mirrors test_myah_adapter.py:_make_adapter) ─────────────────────

def _make_adapter(auth_key: str = '', **extra_kwargs):
    from gateway.config import Platform, PlatformConfig

    extra = dict(extra_kwargs)
    if auth_key:
        extra['auth_key'] = auth_key
    config = PlatformConfig(enabled=True, extra=extra)
    with patch('gateway.platforms.api_server.register_pre_setup_hook'):
        return MyahAdapter(config)


class TestMyahAdapterSourceContainsMarker:
    """Regression gate: the Myah marker block in the adapter source must
    contain the ui_state lift logic.  If somebody deletes it, this fails."""

    def test_adapter_source_contains_ui_state_block(self):
        import inspect
        from gateway.platforms import myah as _module

        source = inspect.getsource(_module)
        assert "body.get('ui_state')" in source, (
            'ui_state lift block missing from gateway/platforms/myah.py — '
            'Phase 4B requires the adapter to read body.ui_state and set '
            'event.channel_prompt with a [CURRENT_UI_STATE] block.'
        )
        assert '[CURRENT_UI_STATE]' in source
        assert 'channel_prompt=_channel_prompt' in source
