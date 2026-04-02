# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for _StageHandle and _current_trace ContextVar."""

from __future__ import annotations

import datetime as dt

from pgns_agent._context import _current_trace
from pgns_agent._trace import _StageHandle


class TestStageHandleInit:
    def test_defaults(self) -> None:
        h = _StageHandle(agent_name="my-agent")
        assert h.agent_name == "my-agent"
        assert h.status == "working"
        assert h.completed_at is None
        assert h.duration_ms is None
        assert h.error is None
        assert h.input_summary is None
        assert h.output_summary is None
        assert h.metadata == {}

    def test_started_at_is_utc(self) -> None:
        h = _StageHandle(agent_name="a")
        assert h.started_at.tzinfo is dt.UTC

    def test_server_fields_are_read_only(self) -> None:
        """Handler code cannot overwrite server-owned lifecycle fields."""
        h = _StageHandle(agent_name="a")
        for attr in ("agent_name", "started_at", "completed_at", "duration_ms", "error", "status"):
            try:
                setattr(h, attr, "bogus")
            except AttributeError:
                pass  # expected — property has no setter
            else:
                raise AssertionError(f"Expected AttributeError for {attr}")


class TestStageHandlePublicAPI:
    def test_set_input_summary(self) -> None:
        h = _StageHandle(agent_name="a")
        h.set_input_summary("user query about weather")
        assert h.input_summary == "user query about weather"

    def test_set_output_summary(self) -> None:
        h = _StageHandle(agent_name="a")
        h.set_output_summary("returned forecast data")
        assert h.output_summary == "returned forecast data"

    def test_set_metadata_merge_semantics(self) -> None:
        h = _StageHandle(agent_name="a")
        h.set_metadata({"model": "gpt-4o"})
        h.set_metadata({"tokens": 512})
        assert h.metadata == {"model": "gpt-4o", "tokens": 512}

    def test_set_metadata_overwrites_existing_key(self) -> None:
        h = _StageHandle(agent_name="a")
        h.set_metadata({"model": "gpt-4o"})
        h.set_metadata({"model": "claude-3"})
        assert h.metadata == {"model": "claude-3"}

    def test_metadata_returns_copy(self) -> None:
        h = _StageHandle(agent_name="a")
        h.set_metadata({"key": "val"})
        copy = h.metadata
        copy["extra"] = "nope"
        assert "extra" not in h.metadata


class TestStageHandleFinalize:
    def test_finalize_completed(self) -> None:
        h = _StageHandle(agent_name="a")
        h._finalize(status="completed")  # noqa: SLF001
        assert h.status == "completed"
        assert h.completed_at is not None
        assert h.duration_ms is not None
        assert h.duration_ms >= 0
        assert h.error is None

    def test_finalize_failed(self) -> None:
        h = _StageHandle(agent_name="a")
        h._finalize(status="failed", error="something broke")  # noqa: SLF001
        assert h.status == "failed"
        assert h.error == "something broke"
        assert h.completed_at is not None
        assert h.duration_ms is not None

    def test_finalize_defaults_to_completed(self) -> None:
        h = _StageHandle(agent_name="a")
        h._finalize()  # noqa: SLF001
        assert h.status == "completed"

    def test_finalize_is_idempotent(self) -> None:
        """Second _finalize call is silently ignored — timing and error are preserved."""
        h = _StageHandle(agent_name="a")
        h._finalize(status="failed", error="first error")  # noqa: SLF001
        first_completed_at = h.completed_at
        first_duration_ms = h.duration_ms

        h._finalize(status="completed", error=None)  # noqa: SLF001

        assert h.status == "failed"
        assert h.error == "first error"
        assert h.completed_at is first_completed_at
        assert h.duration_ms is first_duration_ms


class TestCurrentTraceContextVar:
    def test_default_is_none(self) -> None:
        assert _current_trace.get() is None

    def test_set_and_get(self) -> None:
        h = _StageHandle(agent_name="test-agent")
        token = _current_trace.set(h)
        try:
            assert _current_trace.get() is h
        finally:
            _current_trace.reset(token)

    def test_reset_restores_none(self) -> None:
        h = _StageHandle(agent_name="test-agent")
        token = _current_trace.set(h)
        _current_trace.reset(token)
        assert _current_trace.get() is None


class TestStageHandleToWire:
    """Verify _to_wire() serialization for trace propagation."""

    def test_minimal_wire_format_before_finalize(self) -> None:
        """Unfinalized stage includes only required fields."""
        h = _StageHandle(agent_name="my-agent")
        wire = h._to_wire()  # noqa: SLF001
        assert wire["v"] == 1
        assert wire["agent_name"] == "my-agent"
        assert wire["status"] == "working"
        assert "started_at" in wire
        # Not finalized — optional fields omitted
        assert "completed_at" not in wire
        assert "duration_ms" not in wire
        assert "error" not in wire
        assert "input_summary" not in wire
        assert "output_summary" not in wire
        assert "metadata" not in wire

    def test_full_wire_format_after_finalize(self) -> None:
        """Finalized stage with all annotations includes every field."""
        h = _StageHandle(agent_name="annotated")
        h.set_input_summary("query about weather")
        h.set_output_summary("forecast returned")
        h.set_metadata({"model": "gpt-4o", "tokens": 42})
        h._finalize(status="completed")  # noqa: SLF001
        wire = h._to_wire()  # noqa: SLF001

        assert wire["v"] == 1
        assert wire["agent_name"] == "annotated"
        assert wire["status"] == "completed"
        assert wire["started_at"]
        assert wire["completed_at"]
        assert wire["duration_ms"] >= 0
        assert wire["input_summary"] == "query about weather"
        assert wire["output_summary"] == "forecast returned"
        assert wire["metadata"] == {"model": "gpt-4o", "tokens": 42}
        assert "error" not in wire

    def test_failed_wire_format_includes_error(self) -> None:
        """Failed stage includes error string in wire format."""
        h = _StageHandle(agent_name="a")
        h._finalize(status="failed", error="something broke")  # noqa: SLF001
        wire = h._to_wire()  # noqa: SLF001
        assert wire["status"] == "failed"
        assert wire["error"] == "something broke"

    def test_error_truncated_at_512_chars(self) -> None:
        long_error = "x" * 600
        h = _StageHandle(agent_name="a")
        h._finalize(status="failed", error=long_error)  # noqa: SLF001
        wire = h._to_wire()  # noqa: SLF001
        assert len(wire["error"]) < 600
        assert wire["error"] == "x" * 512 + "…[truncated]"

    def test_error_at_exactly_512_chars_not_truncated(self) -> None:
        exact_error = "y" * 512
        h = _StageHandle(agent_name="a")
        h._finalize(status="failed", error=exact_error)  # noqa: SLF001
        wire = h._to_wire()  # noqa: SLF001
        assert wire["error"] == exact_error

    def test_metadata_in_wire_is_a_copy(self) -> None:
        """Mutating the wire dict's metadata must not affect the stage."""
        h = _StageHandle(agent_name="a")
        h.set_metadata({"key": "val"})
        wire = h._to_wire()  # noqa: SLF001
        wire["metadata"]["injected"] = True
        assert "injected" not in h.metadata

    def test_started_at_is_isoformat_string(self) -> None:
        h = _StageHandle(agent_name="a")
        wire = h._to_wire()  # noqa: SLF001
        # Should be parseable as ISO 8601
        parsed = dt.datetime.fromisoformat(wire["started_at"])
        assert parsed.tzinfo is dt.UTC
