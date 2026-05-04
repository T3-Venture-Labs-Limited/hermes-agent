"""Invariant tests for the Myah platform hint in PLATFORM_HINTS.

The Myah marker block in agent/prompt_builder.py:454+ tells the agent how
to interpret the [CURRENT_UI_STATE] block lifted from body.ui_state by the
adapter (gateway/platforms/myah.py). These tests assert that:

  1. The hint contains a header for editor context.
  2. The hint mentions the no-clobber rule for pendingEdits.
  3. The hint describes what to do when [CURRENT_UI_STATE] is absent.

These are invariant tests — they assert the contract the hint communicates
to the agent, not the exact wording. Wording can be refined; the contract
must hold.
"""

from agent.prompt_builder import PLATFORM_HINTS


class TestMyahPlatformHint:
    def test_myah_hint_present(self):
        assert 'myah' in PLATFORM_HINTS
        assert isinstance(PLATFORM_HINTS['myah'], str)
        assert len(PLATFORM_HINTS['myah']) > 100

    def test_myah_hint_has_editor_context_section(self):
        """The hint must contain a section header that signals editor context.
        The agent uses this to identify when [CURRENT_UI_STATE] semantics apply.
        """
        hint = PLATFORM_HINTS['myah']
        assert '[CURRENT_UI_STATE]' in hint
        # Section header — could be markdown ## or any explicit header.
        assert 'Editor' in hint or 'editor' in hint
        # Must mention selectionRefs and pendingEdits by name so the agent
        # can map JSON fields to behavior.
        assert 'selectionRefs' in hint
        assert 'pendingEdits' in hint

    def test_myah_hint_describes_no_clobber_rule(self):
        """The hint must tell the agent NOT to overwrite the user's
        unsaved edits — the no-clobber rule for pendingEdits."""
        hint = PLATFORM_HINTS['myah']
        # The contract is: read dirty state before suggesting edits;
        # avoid clobbering.  Match either 'clobber' or the canonical
        # 'unsaved'/'dirty state' phrasing.
        assert 'clobber' in hint.lower() or 'unsaved' in hint.lower()

    def test_myah_hint_describes_absent_ui_state(self):
        """The hint must tell the agent how to behave when no
        [CURRENT_UI_STATE] block is shipped (i.e., the common case)."""
        hint = PLATFORM_HINTS['myah']
        # Must explicitly handle the absent case so the agent doesn't
        # hallucinate a missing UI state.
        assert 'absent' in hint.lower() or 'no ' in hint.lower()


class TestPromptBuilderSourceMarkers:
    """Regression gate: the Myah marker comments must remain so upstream
    merge tooling can find this block."""

    def test_marker_comments_present(self):
        import inspect
        from agent import prompt_builder as _module

        source = inspect.getsource(_module)
        assert '# ── Myah: platform hint' in source, (
            'Opening Myah marker comment removed from prompt_builder.py — '
            'upstream-merge tooling needs this to detect Myah additions.'
        )
