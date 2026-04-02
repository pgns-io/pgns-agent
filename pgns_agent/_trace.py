# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal trace handle for automatic pipeline tracing.

``_StageHandle`` captures handler lifecycle data (timing, status, errors) and
exposes public methods for handler authors to annotate traces with summaries
and metadata.  Instances are managed by :class:`AgentServer` and made
available to handlers via the ``task.trace`` property.

This module is internal (``_`` prefix).  The *methods* on ``_StageHandle`` are
public API; the class itself should not be imported by user code.
"""

from __future__ import annotations

__all__: list[str] = []

import datetime as _dt
from typing import Literal

# Valid terminal status values for _StageHandle, mirroring TaskStatus members.
_TraceStatus = Literal["working", "completed", "failed"]


class _StageHandle:
    """Mutable trace context for a single handler invocation.

    Created by :class:`AgentServer` when tracing is enabled, set into the
    ``_current_trace`` context var, and finalized after the handler returns
    or raises.

    Public methods (called by handler code):

    * :meth:`set_input_summary` -- one-line description of the input
    * :meth:`set_output_summary` -- one-line description of the output
    * :meth:`set_metadata` -- merge arbitrary key/value pairs

    Internal methods (called by AgentServer):

    * :meth:`_finalize` -- stamp timing and terminal status
    """

    __slots__ = (
        "_agent_name",
        "_started_at",
        "_completed_at",
        "_duration_ms",
        "_error",
        "_status",
        "_input_summary",
        "_output_summary",
        "_metadata",
    )

    def __init__(self, *, agent_name: str) -> None:
        self._agent_name: str = agent_name
        self._started_at: _dt.datetime = _dt.datetime.now(_dt.UTC)
        self._completed_at: _dt.datetime | None = None
        self._duration_ms: float | None = None
        self._error: str | None = None
        self._status: _TraceStatus = "working"
        self._input_summary: str | None = None
        self._output_summary: str | None = None
        self._metadata: dict[str, object] = {}

    # -- read-only accessors for server-owned fields -----------------------

    @property
    def agent_name(self) -> str:
        """The agent name this trace belongs to."""
        return self._agent_name

    @property
    def started_at(self) -> _dt.datetime:
        """UTC timestamp when the handler started."""
        return self._started_at

    @property
    def completed_at(self) -> _dt.datetime | None:
        """UTC timestamp when the handler completed, or ``None``."""
        return self._completed_at

    @property
    def duration_ms(self) -> float | None:
        """Handler duration in milliseconds, or ``None`` if not finalized."""
        return self._duration_ms

    @property
    def error(self) -> str | None:
        """Error message if the handler failed, or ``None``."""
        return self._error

    @property
    def status(self) -> _TraceStatus:
        """Current status: ``"working"``, ``"completed"``, or ``"failed"``."""
        return self._status

    # -- public API (called by handler authors) ---------------------------

    def set_input_summary(self, summary: str) -> None:
        """Set a one-line summary describing the input to this stage."""
        self._input_summary = summary

    def set_output_summary(self, summary: str) -> None:
        """Set a one-line summary describing the output of this stage."""
        self._output_summary = summary

    def set_metadata(self, data: dict[str, object]) -> None:
        """Merge *data* into the stage metadata.

        Subsequent calls merge keys (``dict.update`` semantics) rather than
        replacing the entire metadata dict::

            task.trace.set_metadata({"model": "gpt-4o"})
            task.trace.set_metadata({"tokens": 512})
            # metadata is now {"model": "gpt-4o", "tokens": 512}
        """
        self._metadata.update(data)

    # -- read-only accessors for handler-set fields ------------------------

    @property
    def input_summary(self) -> str | None:
        """The input summary, or ``None`` if not set."""
        return self._input_summary

    @property
    def output_summary(self) -> str | None:
        """The output summary, or ``None`` if not set."""
        return self._output_summary

    @property
    def metadata(self) -> dict[str, object]:
        """A shallow copy of the current metadata dict."""
        return dict(self._metadata)

    # -- internal API (called by AgentServer) ------------------------------

    def _finalize(
        self,
        *,
        status: _TraceStatus = "completed",
        error: str | None = None,
    ) -> None:
        """Stamp completion time and terminal status.

        Called by :class:`AgentServer` after the handler returns or raises.
        Idempotent: subsequent calls are silently ignored to prevent
        accidental overwrites of timing and error data.
        """
        if self._completed_at is not None:
            return
        now = _dt.datetime.now(_dt.UTC)
        self._completed_at = now
        self._duration_ms = (now - self._started_at).total_seconds() * 1000.0
        self._status = status
        self._error = error

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the current trace state.

        This captures whatever has been recorded so far — timing, status,
        and handler-set annotations.  If the handler is still running (i.e.
        :meth:`_finalize` hasn't been called yet), ``status`` will be
        ``"working"`` and ``completed_at`` / ``duration_ms`` will be absent.

        .. versionadded:: 0.12.0
        """
        return self._to_wire()

    def _to_wire(self) -> dict[str, object]:
        """Serialize this stage to the ``_trace`` wire format.

        Returns a JSON-serializable dict containing timing, status, and
        handler-set annotations.  Used by :meth:`AgentServer.send` to
        propagate trace data to downstream agents.
        """
        stage: dict[str, object] = {
            "v": 1,
            "agent_name": self._agent_name,
            "started_at": self._started_at.isoformat(),
            "status": self._status,
        }
        if self._completed_at is not None:
            stage["completed_at"] = self._completed_at.isoformat()
        if self._duration_ms is not None:
            stage["duration_ms"] = self._duration_ms
        if self._error is not None:
            err = self._error
            if len(err) > 512:
                err = err[:512] + "…[truncated]"
            stage["error"] = err
        if self._input_summary is not None:
            stage["input_summary"] = self._input_summary
        if self._output_summary is not None:
            stage["output_summary"] = self._output_summary
        if self._metadata:
            stage["metadata"] = dict(self._metadata)
        return stage
