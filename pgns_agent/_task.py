# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Task model exposed to handler functions."""

from __future__ import annotations

__all__ = ["Task", "TaskMetadata", "TaskStatus"]

import asyncio
import dataclasses
import enum
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pgns_agent._artifact import ArtifactRef


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

    __slots__ = ("_update_status", "_request_input", "_store_artifact", "_get_artifact")

    def __init__(
        self,
        update_status: Callable[[str], Awaitable[None]],
        request_input: Callable[[str], Awaitable[Any]],
        store_artifact: Callable[..., Awaitable[ArtifactRef]],
        get_artifact: Callable[..., Awaitable[Any]],
    ) -> None:
        self._update_status = update_status
        self._request_input = request_input
        self._store_artifact = store_artifact
        self._get_artifact = get_artifact


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

    async def store_artifact(
        self,
        data: Any,
        *,
        media_type: str = "application/json",
        ttl_seconds: int | None = None,
        auto_delete: bool = False,
    ) -> ArtifactRef:
        """Store data as an artifact and return a reference.

        The reference contains a URL and access_token that can be included
        in pigeon payloads for other agents to retrieve the artifact.

        Raises:
            RuntimeError: If called outside a task handler.
            ArtifactTooLargeError: If the data exceeds the plan's size limit.
        """
        if self._ctrl is None:
            raise RuntimeError("store_artifact() can only be called inside a task handler.")
        return await self._ctrl._store_artifact(
            data, media_type=media_type, ttl_seconds=ttl_seconds, auto_delete=auto_delete
        )

    async def get_artifact(self, url: str, *, token: str | None = None) -> Any:
        """Download and deserialize an artifact by URL.

        Content is automatically deserialized based on the artifact's media type:
        - application/json -> dict/list
        - text/plain -> str
        - all others -> bytes

        Raises:
            RuntimeError: If called outside a task handler.
            ArtifactNotFoundError: If the artifact is expired, consumed, or missing.
            ArtifactAccessError: If the token is invalid.
        """
        if self._ctrl is None:
            raise RuntimeError("get_artifact() can only be called inside a task handler.")
        return await self._ctrl._get_artifact(url, token=token)

    async def get_artifacts(self, urls: list[str], *, tokens: list[str] | None = None) -> list[Any]:
        """Concurrently download and deserialize multiple artifacts.

        Convenience method for scatter/gather patterns. Uses asyncio.gather()
        for concurrent fetch.

        Raises:
            RuntimeError: If called outside a task handler.
            ValueError: If tokens is provided but length doesn't match urls.
        """
        if self._ctrl is None:
            raise RuntimeError("get_artifacts() can only be called inside a task handler.")
        if tokens is not None and len(tokens) != len(urls):
            raise ValueError(f"Expected {len(urls)} tokens, got {len(tokens)}.")
        coros = [
            self._ctrl._get_artifact(url, token=tokens[i] if tokens else None)
            for i, url in enumerate(urls)
        ]
        return list(await asyncio.gather(*coros))
