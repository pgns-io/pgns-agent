# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for pgns_agent.testing — TestClient and TaskResponse."""

from __future__ import annotations

from typing import Any

import pytest

from pgns_agent import AgentServer, Task, TaskMetadata
from pgns_agent.testing import TaskResponse, TestClient

# ---------------------------------------------------------------------------
# TestClient construction
# ---------------------------------------------------------------------------


class TestTestClientConstruction:
    def test_from_agent_server(self) -> None:
        agent = AgentServer("echo", "echoes input")
        client = TestClient(agent)
        assert client.agent is agent

    def test_from_test_client_method(self) -> None:
        agent = AgentServer("echo", "echoes input")
        client = agent.test_client()
        assert isinstance(client, TestClient)
        assert client.agent is agent


# ---------------------------------------------------------------------------
# send_task — successful dispatch
# ---------------------------------------------------------------------------


class TestSendTaskSuccess:
    def test_default_handler(self) -> None:
        agent = AgentServer("echo", "echoes input")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"echo": task.input}

        client = agent.test_client()
        resp = client.send_task({"text": "hello"})

        assert resp.status == "completed"
        assert resp.result == {"echo": {"text": "hello"}}
        assert resp.status_code == 200
        assert resp.error is None

    def test_named_skill(self) -> None:
        agent = AgentServer("multi", "multi-skill")

        @agent.on_task("greet")
        async def greet(task: Task) -> dict[str, str]:
            return {"greeting": f"Hello {task.input}"}

        client = agent.test_client()
        resp = client.send_task("World", skill="greet")

        assert resp.status == "completed"
        assert resp.result == {"greeting": "Hello World"}

    def test_custom_task_id(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {"got_id": task.id}

        client = agent.test_client()
        resp = client.send_task("x", id="my-task-123")

        assert resp.id == "my-task-123"
        assert resp.result == {"got_id": "my-task-123"}

    def test_auto_generated_task_id(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = agent.test_client()
        resp = client.send_task("x")

        assert resp.id  # non-empty
        assert len(resp.id) == 32  # uuid4 hex

    def test_none_input(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"input_was": task.input}

        client = agent.test_client()
        resp = client.send_task()

        assert resp.status == "completed"
        assert resp.result == {"input_was": None}

    def test_metadata_as_dict(self) -> None:
        agent = AgentServer("a", "b")
        captured: dict[str, str | None] = {}

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            captured["correlation_id"] = task.metadata.correlation_id
            captured["source_agent"] = task.metadata.source_agent
            return {}

        client = agent.test_client()
        client.send_task(
            "x",
            metadata={
                "correlation_id": "corr-abc",
                "source_agent": "other-agent",
            },
        )

        assert captured["correlation_id"] == "corr-abc"
        assert captured["source_agent"] == "other-agent"

    def test_metadata_as_task_metadata(self) -> None:
        agent = AgentServer("a", "b")
        captured: dict[str, str | None] = {}

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            captured["correlation_id"] = task.metadata.correlation_id
            return {}

        client = agent.test_client()
        client.send_task(
            "x",
            metadata=TaskMetadata(correlation_id="corr-xyz"),
        )

        assert captured["correlation_id"] == "corr-xyz"


# ---------------------------------------------------------------------------
# send_task — failure cases
# ---------------------------------------------------------------------------


class TestSendTaskFailure:
    def test_no_handler_returns_failed(self) -> None:
        agent = AgentServer("a", "b")
        client = agent.test_client()
        resp = client.send_task("x")

        assert resp.status == "failed"
        assert resp.status_code == 404
        assert resp.error is not None
        assert resp.result is None

    def test_unknown_skill_no_default_returns_failed(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("only-this")
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = agent.test_client()
        resp = client.send_task("x", skill="other")

        assert resp.status == "failed"
        assert resp.status_code == 404

    def test_handler_exception_returns_failed(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            raise RuntimeError("boom")

        client = agent.test_client()
        resp = client.send_task("x")

        assert resp.status == "failed"
        assert resp.status_code == 500
        assert resp.error is not None

    def test_unknown_skill_falls_back_to_default(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, bool]:
            return {"fallback": True}

        client = agent.test_client()
        resp = client.send_task("x", skill="unknown")

        assert resp.status == "completed"
        assert resp.result == {"fallback": True}


# ---------------------------------------------------------------------------
# build_agent_card
# ---------------------------------------------------------------------------


class TestBuildAgentCard:
    def test_returns_agent_card(self) -> None:
        agent = AgentServer("card-test", "tests cards", version="1.2.3")

        @agent.on_task("summarize")
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        client = agent.test_client()
        card = client.build_agent_card()

        assert card.name == "card-test"
        assert card.description == "tests cards"
        assert card.version == "1.2.3"
        skill_ids = {s.id for s in card.skills}
        assert skill_ids == {"summarize"}

    def test_excludes_default_handler_from_skills(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = agent.test_client()
        card = client.build_agent_card()
        assert card.skills == ()


# ---------------------------------------------------------------------------
# send_task — async mode (prefer_async)
# ---------------------------------------------------------------------------


class TestSendTaskAsync:
    def test_prefer_async_returns_submitted(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"done": True}

        client = agent.test_client()
        resp = client.send_task("x", id="t-async", prefer_async=True)

        assert resp.status == "submitted"
        assert resp.status_code == 202
        assert resp.id == "t-async"
        assert resp.result is None
        assert resp.error is None

    def test_prefer_async_false_is_sync(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"sync": True}

        client = agent.test_client()
        resp = client.send_task("x", prefer_async=False)

        assert resp.status == "completed"
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TaskResponse
# ---------------------------------------------------------------------------


class TestTaskResponse:
    def test_completed_response(self) -> None:
        resp = TaskResponse(id="t1", status="completed", result={"ok": True}, status_code=200)
        assert resp.status == "completed"
        assert resp.result == {"ok": True}
        assert resp.error is None

    def test_failed_response(self) -> None:
        resp = TaskResponse(id="t1", status="failed", error="not found", status_code=404)
        assert resp.status == "failed"
        assert resp.result is None
        assert resp.error == "not found"

    def test_is_frozen(self) -> None:
        resp = TaskResponse(id="t1", status="completed")
        with pytest.raises(AttributeError):
            resp.status = "failed"  # type: ignore[misc]
