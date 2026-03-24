# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Task model exposed to handler functions."""

from __future__ import annotations

__all__ = ["Task", "TaskMetadata", "TaskStatus"]

import dataclasses
import enum
from collections.abc import Awaitable, Callable
from typing import Any


class TaskStatus(enum.StrEnum):
    """A2A task lifecycle states.

    Transitions::

        submitted → working → completed
                            → failed
                  → input-required → working → ...
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"


class _TaskControl:
    """Internal control interface injected by the server.

    Bridges ``Task.update_status`` / ``Task.request_input`` to the server's
    publishing and suspension machinery.  Not part of the public API.
    """

    __slots__ = ("_update_status", "_request_input")

    def __init__(
        self,
        update_status: Callable[[str], Awaitable[None]],
        request_input: Callable[[str], Awaitable[Any]],
    ) -> None:
        self._update_status = update_status
        self._request_input = request_input


@dataclasses.dataclass(frozen=True, slots=True)
class TaskMetadata:
    """Contextual metadata attached to a task.

    Attributes:
        correlation_id: Opaque ID linking request/response pigeons in a
            conversation. ``None`` for the first message in a chain.
        source_agent: Identifier (name or URL) of the agent that sent the
            task, or ``None`` when the task originates from a non-agent caller.
    """

    correlation_id: str | None = None
    source_agent: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    """Immutable snapshot of an inbound task delivered to a handler.

    Attributes:
        id: Unique task identifier assigned by the platform.
        input: The deserialized JSON payload sent by the caller.
        metadata: Correlation and provenance information.
    """

    id: str
    input: Any
    metadata: TaskMetadata = dataclasses.field(default_factory=TaskMetadata)
    _ctrl: _TaskControl | None = dataclasses.field(
        default=None, repr=False, compare=False, hash=False
    )

    async def update_status(self, message: str) -> None:
        """Publish a progress-update pigeon while staying in *working* state.

        Use this to report incremental progress from within a handler::

            @agent.on_task
            async def handle(task):
                await task.update_status("processing step 1 of 3")
                ...
                await task.update_status("processing step 2 of 3")

        Raises:
            RuntimeError: If called outside a task handler.
        """
        if self._ctrl is None:
            raise RuntimeError("update_status() can only be called inside a task handler.")
        await self._ctrl._update_status(message)

    async def request_input(self, prompt: str) -> Any:
        """Suspend the handler and request additional input from the caller.

        Transitions the task to ``input-required``, publishes a status pigeon
        with *prompt* as the message and the task's correlation ID, then
        suspends the handler until the caller sends a follow-up request with
        the same ``task.id``.  When the response arrives the handler resumes
        and this method returns the caller's input payload::

            @agent.on_task
            async def handle(task):
                answer = await task.request_input("What is your name?")
                return {"greeting": f"Hello, {answer}!"}

        Raises:
            RuntimeError: If called outside a task handler.
        """
        if self._ctrl is None:
            raise RuntimeError("request_input() can only be called inside a task handler.")
        return await self._ctrl._request_input(prompt)
