# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Test client for unit-testing pgns-agent handlers.

Provides a synchronous :class:`TestClient` that wraps the Starlette test
client with agent-specific convenience methods::

    client = agent.test_client()
    resp = client.send_task({"text": "hello"})
    assert resp.status == "completed"
"""

from __future__ import annotations

__all__ = ["TaskResponse", "TestClient"]

import dataclasses
import uuid
from typing import TYPE_CHECKING, Any

from starlette.testclient import TestClient as StarletteTestClient

if TYPE_CHECKING:
    from pgns_agent._agent_card import AgentCard
    from pgns_agent._server import AgentServer
    from pgns_agent._task import TaskMetadata


@dataclasses.dataclass(frozen=True, slots=True)
class TaskResponse:
    """Typed response from :meth:`TestClient.send_task`.

    Attributes:
        id: The task ID echoed back by the server.
        status: ``"completed"`` when the handler succeeded, ``"failed"``
            otherwise.
        result: The handler's return value (``None`` on failure).
        error: Error message from the server (``None`` on success).
        status_code: The raw HTTP status code for advanced assertions.
    """

    id: str
    status: str
    result: Any = None
    error: str | None = None
    status_code: int = 200


class TestClient:
    """Synchronous test client for unit-testing agent handlers.

    Wraps :class:`starlette.testclient.TestClient` with ergonomic helpers
    that speak the pgns-agent task protocol.  Use via
    :meth:`~pgns_agent.AgentServer.test_client`::

        agent = AgentServer("echo", "echoes input")

        @agent.on_task
        async def handle(task):
            return {"echo": task.input}

        client = agent.test_client()
        resp = client.send_task({"text": "hello"})
        assert resp.status == "completed"
        assert resp.result == {"echo": {"text": "hello"}}
    """

    __test__ = False  # prevent pytest collection

    def __init__(
        self,
        agent: AgentServer,
        *,
        raise_server_exceptions: bool = False,
    ) -> None:
        self._agent = agent
        self._http = StarletteTestClient(
            agent.app(),
            raise_server_exceptions=raise_server_exceptions,
        )

    @property
    def agent(self) -> AgentServer:
        """The :class:`AgentServer` under test."""
        return self._agent

    def send_task(
        self,
        input: Any = None,  # noqa: A002
        *,
        id: str | None = None,  # noqa: A002
        skill: str | None = None,
        metadata: dict[str, Any] | TaskMetadata | None = None,
        prefer_async: bool = False,
    ) -> TaskResponse:
        """Send a task to the agent and return a typed :class:`TaskResponse`.

        Args:
            input: JSON-serializable payload delivered to the handler as
                ``task.input``.
            id: Task ID.  Auto-generated (UUID4) when omitted.
            skill: Skill name to dispatch to.  ``None`` routes to the
                default handler.
            metadata: Correlation / provenance metadata.  Accepts a plain
                dict or a :class:`~pgns_agent.TaskMetadata` instance.
            prefer_async: When ``True``, send the ``Prefer: respond-async``
                header (RFC 7240) to request asynchronous processing.  The
                server returns HTTP 202 with a ``"submitted"`` status
                instead of blocking until the handler completes.
        """
        task_id = id if id is not None else uuid.uuid4().hex

        body: dict[str, Any] = {"id": task_id, "input": input}
        if skill is not None:
            body["skill"] = skill
        if metadata is not None:
            body["metadata"] = _metadata_to_dict(metadata)

        headers: dict[str, str] = {}
        if prefer_async:
            headers["Prefer"] = "respond-async"

        resp = self._http.post("/", json=body, headers=headers)
        data = resp.json()

        if resp.status_code == 200:
            return TaskResponse(
                id=data.get("id", task_id),
                status="completed",
                result=data.get("result"),
                status_code=200,
            )

        if resp.status_code == 202:
            return TaskResponse(
                id=data.get("id", task_id),
                status=data.get("status", "submitted"),
                status_code=202,
            )

        return TaskResponse(
            id=task_id,
            status="failed",
            error=data.get("error"),
            status_code=resp.status_code,
        )

    def build_agent_card(self) -> AgentCard:
        """Build the Agent Card from the server's current state.

        This calls :meth:`~pgns_agent.AgentServer.build_agent_card` directly
        and does **not** exercise the ``/.well-known/agent.json`` HTTP route.

        Returns:
            An :class:`~pgns_agent.AgentCard` built from the server's
            current state.
        """
        return self._agent.build_agent_card()


def _metadata_to_dict(metadata: dict[str, Any] | TaskMetadata) -> dict[str, Any]:
    """Normalise metadata into a plain dict for the JSON body."""
    if isinstance(metadata, dict):
        return metadata
    return {k: v for k, v in dataclasses.asdict(metadata).items() if v is not None}
