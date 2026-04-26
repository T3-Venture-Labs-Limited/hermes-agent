"""Test for Bug D v3 (scheduler half): _build_myah_send_metadata enrichment.

The MyahAdapter's offline-delivery fallback (Bug D v3) expects ``metadata``
to carry ``job_id``, ``job_name``, ``status``, ``ran_at``, and (optionally)
``origin`` so it can build the platform's ``/webhook/run-complete``
payload.  These fields aren't in the upstream ``send_metadata`` that
``_deliver_result`` builds today (only ``thread_id``).

The fix wraps the minimal enrichment in a small, Myah-marked block in
``_deliver_result`` that calls ``_build_myah_send_metadata(job)`` only
when the target platform is ``myah``.  Other platforms see ZERO change.
"""


# ── Myah: Bug D v3 — scheduler metadata enrichment ──────────────


class TestBuildMyahSendMetadata:
    def test_includes_required_fields(self):
        from cron.scheduler import _build_myah_send_metadata

        job = {
            "id": "deadbeef12",
            "name": "test-cron",
            "deliver": "origin",
            "origin": {"platform": "myah", "chat_id": "chat-xyz"},
        }
        meta = _build_myah_send_metadata(job, status_hint="ok")

        # Required by platform/backend/.../routers/processes.py:824-831
        assert meta["job_id"] == "deadbeef12"
        assert meta["job_name"] == "test-cron"
        assert meta["status"] == "ok"
        # ran_at must be ISO-ish; we don't pin the exact value but it
        # must be non-empty so the platform's logger can show it.
        assert isinstance(meta["ran_at"], str)
        assert len(meta["ran_at"]) > 0

    def test_includes_origin_for_chat_id_fallback(self):
        """Adapter falls back to metadata.origin.chat_id when chat_id arg is empty."""
        from cron.scheduler import _build_myah_send_metadata

        job = {
            "id": "abc123",
            "name": "test",
            "deliver": "origin",
            "origin": {"platform": "myah", "chat_id": "origin-chat"},
        }
        meta = _build_myah_send_metadata(job)
        assert isinstance(meta["origin"], dict)
        assert meta["origin"]["chat_id"] == "origin-chat"
        assert meta["origin"]["platform"] == "myah"

    def test_origin_none_when_job_has_none(self):
        """Job without origin → metadata.origin is None or absent."""
        from cron.scheduler import _build_myah_send_metadata

        job = {
            "id": "abc",
            "name": "no-origin-job",
            "deliver": "myah:explicit-chat",
            # no origin
        }
        meta = _build_myah_send_metadata(job)
        assert meta.get("origin") is None

    def test_falls_back_to_id_for_job_name_when_name_missing(self):
        from cron.scheduler import _build_myah_send_metadata

        job = {"id": "deadbeef", "deliver": "origin"}
        meta = _build_myah_send_metadata(job)
        assert meta["job_name"] == "deadbeef"

    def test_status_hint_propagates(self):
        from cron.scheduler import _build_myah_send_metadata

        job = {"id": "x", "name": "y"}
        ok = _build_myah_send_metadata(job, status_hint="ok")
        err = _build_myah_send_metadata(job, status_hint="error")
        assert ok["status"] == "ok"
        assert err["status"] == "error"
# ────────────────────────────────────────────────────────────────
