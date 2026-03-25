# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""AgentServer tests: constructor, @on_task, entry points, state machine, async, SSE, webhooks."""

from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import hmac as hmac_mod
import json
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from pgns_agent import AgentServer, Task, TaskStatus
from pgns_agent._context import current_task
from pgns_agent._server import DEFAULT_HANDLER_NAME
from pgns_agent._state import TaskState

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_stores_name_and_description(self) -> None:
        agent = AgentServer("my-agent", "Does things")
        assert agent.name == "my-agent"
        assert agent.description == "Does things"

    def test_no_key_local_dev_mode(self) -> None:
        agent = AgentServer("local", "dev mode")
        assert agent.client is None

    def test_with_key_initialises_client(self) -> None:
        # Patch at source module — works because _server.py imports inside __init__.
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            agent = AgentServer("prod", "production", pgns_key="pk_test_abc")
            mock_cls.assert_called_once_with("https://api.pgns.io", api_key="pk_test_abc")
            assert agent.client is mock_cls.return_value

    def test_custom_pgns_url(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            AgentServer("a", "b", pgns_key="pk_test_x", pgns_url="https://custom.pgns.dev")
            mock_cls.assert_called_once_with("https://custom.pgns.dev", api_key="pk_test_x")

    def test_handlers_initially_empty(self) -> None:
        agent = AgentServer("a", "b")
        assert agent.handlers == {}

    def test_handlers_returns_copy(self) -> None:
        agent = AgentServer("a", "b")
        h = agent.handlers
        h["injected"] = lambda t: None  # type: ignore[assignment,return-value]
        assert "injected" not in agent.handlers

    def test_initial_provisioning_state(self) -> None:
        agent = AgentServer("a", "b")
        assert agent.provisioned is False
        assert agent.agent_card is None
        assert agent.roost is None

    def test_pgns_url_property(self) -> None:
        agent = AgentServer("a", "b", pgns_url="https://custom.pgns.dev")
        assert agent.pgns_url == "https://custom.pgns.dev"


# ---------------------------------------------------------------------------
# @on_task decorator
# ---------------------------------------------------------------------------


class TestOnTask:
    def test_bare_decorator_registers_default(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        assert DEFAULT_HANDLER_NAME in agent.handlers
        assert agent.handlers[DEFAULT_HANDLER_NAME] is handle

    def test_named_decorator(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("summarize")
        async def summarize(task: Task) -> dict[str, str]:
            return {"summary": "..."}

        assert "summarize" in agent.handlers
        assert agent.handlers["summarize"] is summarize

    def test_multiple_named_handlers(self) -> None:
        agent = AgentServer("multi", "multi-skill")

        @agent.on_task("translate")
        async def translate(task: Task) -> dict[str, str]:
            return {}

        @agent.on_task("summarize")
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        assert set(agent.handlers.keys()) == {"translate", "summarize"}

    def test_default_plus_named(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def fallback(task: Task) -> dict[str, str]:
            return {}

        @agent.on_task("special")
        async def special(task: Task) -> dict[str, str]:
            return {}

        assert set(agent.handlers.keys()) == {DEFAULT_HANDLER_NAME, "special"}

    def test_duplicate_default_raises(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def first(task: Task) -> dict[str, str]:
            return {}

        with pytest.raises(ValueError, match="already registered"):

            @agent.on_task
            async def second(task: Task) -> dict[str, str]:
                return {}

    def test_duplicate_named_raises(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("dup")
        async def first(task: Task) -> dict[str, str]:
            return {}

        with pytest.raises(ValueError, match="already registered"):

            @agent.on_task("dup")
            async def second(task: Task) -> dict[str, str]:
                return {}

    def test_decorator_returns_original_function(self) -> None:
        agent = AgentServer("a", "b")

        async def my_handler(task: Task) -> None:
            pass

        result = agent.on_task(my_handler)
        assert result is my_handler

    def test_named_decorator_returns_original_function(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("skill")
        async def my_handler(task: Task) -> None:
            pass

        assert agent.handlers["skill"] is my_handler

    def test_explicit_default_name_raises(self) -> None:
        agent = AgentServer("a", "b")
        with pytest.raises(ValueError, match="reserved"):
            agent.on_task(DEFAULT_HANDLER_NAME)


# ---------------------------------------------------------------------------
# app()
# ---------------------------------------------------------------------------


class TestApp:
    def test_returns_starlette_instance(self) -> None:
        from starlette.applications import Starlette

        agent = AgentServer("a", "b")
        assert isinstance(agent.app(), Starlette)

    def test_returns_same_instance(self) -> None:
        agent = AgentServer("a", "b")
        assert agent.app() is agent.app()


# ---------------------------------------------------------------------------
# handler()
# ---------------------------------------------------------------------------


class TestHandler:
    def test_returns_same_as_app(self) -> None:
        agent = AgentServer("a", "b")
        assert agent.handler() is agent.app()


# ---------------------------------------------------------------------------
# listen()
# ---------------------------------------------------------------------------


class TestListen:
    def test_calls_uvicorn_run(self) -> None:
        mock_uvicorn = MagicMock()
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent.listen(3000)
            mock_uvicorn.run.assert_called_once_with(agent.app(), host="127.0.0.1", port=3000)

    def test_custom_host(self) -> None:
        mock_uvicorn = MagicMock()
        agent = AgentServer("a", "b")

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent.listen(8080, host="127.0.0.1")
            mock_uvicorn.run.assert_called_once_with(agent.app(), host="127.0.0.1", port=8080)

    def test_raises_without_uvicorn(self) -> None:
        agent = AgentServer("a", "b")
        with patch.dict(sys.modules, {"uvicorn": None}):
            with pytest.raises(RuntimeError, match="uvicorn"):
                agent.listen(3000)


# ---------------------------------------------------------------------------
# Agent Card endpoint (GET /.well-known/agent.json)
# ---------------------------------------------------------------------------


class TestAgentCardEndpoint:
    def test_returns_card(self) -> None:
        agent = AgentServer("test-agent", "A test agent")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-agent"
        assert data["description"] == "A test agent"
        assert "version" in data

    def test_includes_named_skills(self) -> None:
        agent = AgentServer("multi", "multi-skill")

        @agent.on_task("summarize")
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        @agent.on_task("translate")
        async def translate(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.get("/.well-known/agent.json")
        data = resp.json()
        skill_names = {s["id"] for s in data["skills"]}
        assert skill_names == {"summarize", "translate"}

    def test_excludes_default_from_skills(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.get("/.well-known/agent.json")
        data = resp.json()
        assert data["skills"] == []


# ---------------------------------------------------------------------------
# Task endpoint (POST /)
# ---------------------------------------------------------------------------


class TestTaskEndpoint:
    def test_default_handler(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {"echo": task.input}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "hello"})
        assert resp.status_code == 200
        assert resp.json() == {"id": "t1", "status": "completed", "result": {"echo": "hello"}}

    def test_named_handler(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("greet")
        async def greet(task: Task) -> dict[str, str]:
            return {"greeting": f"Hello {task.input}"}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "World", "skill": "greet"})
        assert resp.status_code == 200
        assert resp.json()["result"] == {"greeting": "Hello World"}

    def test_unknown_skill_falls_back_to_default(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, bool]:
            return {"fallback": True}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "x", "skill": "unknown"})
        assert resp.status_code == 200
        assert resp.json()["result"]["fallback"] is True

    def test_unknown_skill_no_default_returns_404(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("only-this")
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "x", "skill": "other"})
        assert resp.status_code == 404

    def test_invalid_json_returns_400(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.post("/", content=b"not json", headers={"content-type": "application/json"})
        assert resp.status_code == 400

    def test_missing_id_returns_400(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.post("/", json={"input": "x"})
        assert resp.status_code == 400

    def test_handler_exception_returns_500(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            raise RuntimeError("boom")

        client = TestClient(agent.app(), raise_server_exceptions=False)
        resp = client.post("/", json={"id": "t1", "input": "x"})
        assert resp.status_code == 500

    def test_metadata_passed_to_handler(self) -> None:
        agent = AgentServer("a", "b")
        captured: dict[str, str | None] = {}

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            captured["correlation_id"] = task.metadata.correlation_id
            captured["source_agent"] = task.metadata.source_agent
            return {}

        client = TestClient(agent.app())
        resp = client.post(
            "/",
            json={
                "id": "t1",
                "input": "x",
                "metadata": {
                    "correlation_id": "corr-123",
                    "source_agent": "other-agent",
                },
            },
        )
        assert resp.status_code == 200
        assert captured["correlation_id"] == "corr-123"
        assert captured["source_agent"] == "other-agent"

    def test_no_handlers_returns_404(self) -> None:
        agent = AgentServer("a", "b")
        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "x"})
        assert resp.status_code == 404

    def test_non_string_id_returns_400(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": 123, "input": "x"})
        assert resp.status_code == 400
        assert "string" in resp.json()["error"]

    def test_non_dict_metadata_is_ignored(self) -> None:
        agent = AgentServer("a", "b")
        captured: dict[str, str | None] = {}

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            captured["correlation_id"] = task.metadata.correlation_id
            return {}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "x", "metadata": "bad"})
        assert resp.status_code == 200
        assert captured["correlation_id"] is None

    def test_null_skill_uses_default_handler(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, bool]:
            return {"default": True}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "x", "skill": None})
        assert resp.status_code == 200
        assert resp.json()["result"]["default"] is True


# ---------------------------------------------------------------------------
# Task state machine (auto-transitions)
# ---------------------------------------------------------------------------


class TestTaskStateMachine:
    def test_success_response_includes_status_completed(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"ok": True}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "x"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] == {"ok": True}

    def test_failure_response_includes_status_failed(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            raise RuntimeError("boom")

        client = TestClient(agent.app(), raise_server_exceptions=False)
        resp = client.post("/", json={"id": "t1", "input": "x"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["status"] == "failed"
        assert data["id"] == "t1"

    def test_terminal_task_evicted_on_success(self) -> None:
        agent = AgentServer("a", "b")
        captured: list[dict[str, TaskState]] = []

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            # Task should be visible while handler is running (WORKING).
            captured.append(agent.tasks)
            return {}

        client = TestClient(agent.app())
        client.post("/", json={"id": "t-stored", "input": "x"})

        # During handler execution the task was present and WORKING.
        assert "t-stored" in captured[0]
        assert captured[0]["t-stored"].status is TaskStatus.WORKING
        # After completion, terminal entries are evicted.
        assert "t-stored" not in agent.tasks

    def test_terminal_task_evicted_on_failure(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            raise ValueError("nope")

        client = TestClient(agent.app(), raise_server_exceptions=False)
        client.post("/", json={"id": "t-fail", "input": "x"})

        assert "t-fail" not in agent.tasks

    def test_tasks_property_returns_deep_copy(self) -> None:
        agent = AgentServer("a", "b")
        captured: list[dict[str, TaskState]] = []

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            captured.append(agent.tasks)
            return {}

        client = TestClient(agent.app())
        client.post("/", json={"id": "t-copy", "input": "x"})

        # Mutating the snapshot must not affect the internal dict.
        snapshot = captured[0]
        snapshot.pop("t-copy")
        # A fresh snapshot taken during handler should still have the key.
        assert "t-copy" not in snapshot

    def test_context_variable_set_during_handler(self) -> None:
        """The current_task context variable is set while the handler runs."""
        agent = AgentServer("a", "b")
        captured_task_id: list[str | None] = []

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            from pgns_agent import get_current_task

            ct = get_current_task()
            captured_task_id.append(ct.id if ct else None)
            return {}

        client = TestClient(agent.app())
        client.post("/", json={"id": "t-ctx", "input": "x"})

        assert captured_task_id == ["t-ctx"]

    def test_context_variable_reset_after_handler(self) -> None:
        """current_task is None after the handler returns."""
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        client.post("/", json={"id": "t-reset", "input": "x"})

        assert current_task.get() is None

    def test_context_variable_reset_after_handler_exception(self) -> None:
        """current_task is reset even when the handler throws."""
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            raise RuntimeError("boom")

        client = TestClient(agent.app(), raise_server_exceptions=False)
        client.post("/", json={"id": "t-exc-ctx", "input": "x"})

        assert current_task.get() is None


class TestPublishStatusUpdate:
    """Verify status pigeons are published at each auto-transition."""

    def test_publishes_three_pigeons_on_success(self) -> None:
        """submitted, working, completed → 3 publish calls (order non-deterministic
        because submitted/working are fire-and-forget)."""
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            # Simulate provisioned state
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, Any]:
                return {"done": True}

            client = TestClient(agent.app())
            client.post("/", json={"id": "t-pub", "input": "x"})

            assert mock_client.send.await_count == 3
            statuses = sorted(
                call.kwargs["payload"]["status"] for call in mock_client.send.await_args_list
            )
            assert statuses == ["completed", "submitted", "working"]

    def test_publishes_three_pigeons_on_failure(self) -> None:
        """submitted, working, failed → 3 publish calls (order non-deterministic
        because submitted/working are fire-and-forget)."""
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, str]:
                raise RuntimeError("boom")

            client = TestClient(agent.app(), raise_server_exceptions=False)
            client.post("/", json={"id": "t-fail-pub", "input": "x"})

            assert mock_client.send.await_count == 3
            statuses = sorted(
                call.kwargs["payload"]["status"] for call in mock_client.send.await_args_list
            )
            assert statuses == ["failed", "submitted", "working"]

    def test_pigeons_include_task_id_and_timestamp(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, str]:
                return {}

            client = TestClient(agent.app())
            client.post("/", json={"id": "t-fields", "input": "x"})

            for call in mock_client.send.await_args_list:
                payload = call.kwargs["payload"]
                assert payload["task_id"] == "t-fields"
                assert "timestamp" in payload

    def test_pigeons_use_agent_task_status_event_type(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, str]:
                return {}

            client = TestClient(agent.app())
            client.post("/", json={"id": "t-evt", "input": "x"})

            for call in mock_client.send.await_args_list:
                assert call.kwargs["event_type"] == "agent.task.status"

    def test_pigeons_sent_to_own_roost(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-self"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, str]:
                return {}

            client = TestClient(agent.app())
            client.post("/", json={"id": "t-roost", "input": "x"})

            for call in mock_client.send.await_args_list:
                assert call.args[0] == "roost-self"

    def test_correlation_id_propagated_in_status_pigeons(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, str]:
                return {}

            client = TestClient(agent.app())
            client.post(
                "/",
                json={
                    "id": "t-cid",
                    "input": "x",
                    "metadata": {"correlation_id": "corr-test-123"},
                },
            )

            for call in mock_client.send.await_args_list:
                assert call.kwargs["extra_headers"] == {"X-Pgns-CorrelationId": "corr-test-123"}

    def test_local_dev_mode_skips_publish(self) -> None:
        """In local dev mode, no client calls — transitions are just logged."""
        agent = AgentServer("local", "dev mode")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"ok": True}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t-local", "input": "x"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_publish_failure_does_not_affect_response(self) -> None:
        """If pigeon publishing fails, the task still succeeds."""
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(side_effect=RuntimeError("network error"))

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, Any]:
                return {"survived": True}

            client = TestClient(agent.app())
            resp = client.post("/", json={"id": "t-pubfail", "input": "x"})
            assert resp.status_code == 200
            assert resp.json()["result"]["survived"] is True

    def test_no_roost_skips_publish(self) -> None:
        """If roost isn't provisioned, publishing is silently skipped."""
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            agent = AgentServer("a", "b", pgns_key="pk_test")
            # Deliberately leave _roost as None (not provisioned)

            @agent.on_task
            async def handle(task: Task) -> dict[str, Any]:
                return {"ok": True}

            client = TestClient(agent.app())
            resp = client.post("/", json={"id": "t-noroost", "input": "x"})
            assert resp.status_code == 200
            mock_client.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# update_status()
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    def test_publishes_working_pigeon_with_message(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, Any]:
                await task.update_status("step 1 of 3")
                await task.update_status("step 2 of 3")
                return {"done": True}

            client = TestClient(agent.app())
            resp = client.post("/", json={"id": "t-upd", "input": "x"})
            assert resp.status_code == 200

            # submitted, working, 2x update_status (working+message), completed = 5
            assert mock_client.send.await_count == 5
            payloads = [call.kwargs["payload"] for call in mock_client.send.await_args_list]
            working_with_msg = [
                p for p in payloads if p["status"] == "working" and p.get("message")
            ]
            assert len(working_with_msg) == 2
            messages = [p["message"] for p in working_with_msg]
            assert "step 1 of 3" in messages
            assert "step 2 of 3" in messages

    def test_update_status_in_local_dev_mode(self) -> None:
        agent = AgentServer("local", "dev")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            await task.update_status("progress")
            return {"ok": True}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t-local-upd", "input": "x"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_update_status_updates_task_state_message(self) -> None:
        agent = AgentServer("a", "b")
        captured_msg: list[str | None] = []

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            await task.update_status("halfway done")
            state = agent.tasks.get(task.id)
            captured_msg.append(state.message if state else None)
            return {}

        client = TestClient(agent.app())
        client.post("/", json={"id": "t-state-msg", "input": "x"})
        assert captured_msg == ["halfway done"]


# ---------------------------------------------------------------------------
# request_input()
# ---------------------------------------------------------------------------


class TestRequestInput:
    @pytest.mark.asyncio
    async def test_suspends_and_resumes_with_input(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            answer = await task.request_input("What color?")
            return {"answer": answer}

        import httpx

        transport = httpx.ASGITransport(app=agent.app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def resume_after_delay() -> httpx.Response:
                await asyncio.sleep(0.05)
                return await client.post("/", json={"id": "t-input", "input": "blue"})

            initial_resp, resume_resp = await asyncio.gather(
                client.post("/", json={"id": "t-input", "input": "hello"}),
                resume_after_delay(),
            )

            assert initial_resp.status_code == 200
            assert initial_resp.json()["status"] == "completed"
            assert initial_resp.json()["result"]["answer"] == "blue"

            assert resume_resp.status_code == 200
            assert resume_resp.json()["status"] == "input-received"

    @pytest.mark.asyncio
    async def test_multiple_request_input_calls(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            name = await task.request_input("Name?")
            color = await task.request_input("Favorite color?")
            return {"name": name, "color": color}

        import httpx

        transport = httpx.ASGITransport(app=agent.app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def resume_sequence() -> None:
                # Wait for first request_input
                await asyncio.sleep(0.05)
                r1 = await client.post("/", json={"id": "t-multi", "input": "Alice"})
                assert r1.json()["status"] == "input-received"

                # Wait for second request_input
                await asyncio.sleep(0.05)
                r2 = await client.post("/", json={"id": "t-multi", "input": "green"})
                assert r2.json()["status"] == "input-received"

            initial_task = asyncio.create_task(
                client.post("/", json={"id": "t-multi", "input": "start"})
            )
            await resume_sequence()
            initial_resp = await initial_task

            assert initial_resp.status_code == 200
            result = initial_resp.json()["result"]
            assert result == {"name": "Alice", "color": "green"}

    @pytest.mark.asyncio
    async def test_publishes_input_required_and_working_pigeons(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            @agent.on_task
            async def handle(task: Task) -> dict[str, Any]:
                answer = await task.request_input("Question?")
                return {"answer": answer}

            import httpx

            transport = httpx.ASGITransport(app=agent.app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

                async def resume() -> None:
                    await asyncio.sleep(0.05)
                    await client.post("/", json={"id": "t-pig", "input": "answer"})

                await asyncio.gather(
                    client.post("/", json={"id": "t-pig", "input": "x"}),
                    resume(),
                )

            # Expected pigeons: submitted, working, input-required,
            #   working (resume), completed = 5
            assert mock_client.send.await_count == 5
            statuses = [
                call.kwargs["payload"]["status"] for call in mock_client.send.await_args_list
            ]
            assert "input-required" in statuses
            # input-required pigeon should have the prompt as message
            ir_payloads = [
                call.kwargs["payload"]
                for call in mock_client.send.await_args_list
                if call.kwargs["payload"]["status"] == "input-required"
            ]
            assert len(ir_payloads) == 1
            assert ir_payloads[0]["message"] == "Question?"

    @pytest.mark.asyncio
    async def test_in_flight_non_input_required_returns_409(self) -> None:
        """Sending same task_id while WORKING (not input-required) → 409."""
        agent = AgentServer("a", "b")
        hold = asyncio.Event()

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            await hold.wait()
            return {"ok": True}

        import httpx

        transport = httpx.ASGITransport(app=agent.app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def send_duplicate() -> httpx.Response:
                await asyncio.sleep(0.05)
                return await client.post("/", json={"id": "t-dup", "input": "y"})

            initial_task = asyncio.create_task(client.post("/", json={"id": "t-dup", "input": "x"}))
            dup_resp = await send_duplicate()
            assert dup_resp.status_code == 409

            hold.set()
            initial_resp = await initial_task
            assert initial_resp.status_code == 200

    def test_input_required_without_pending_future_returns_500(self) -> None:
        """INPUT_REQUIRED state with no live Future → 500 ghost-state error."""
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"ok": True}

        # Manually inject a task state in INPUT_REQUIRED without a pending future.
        import datetime

        state = TaskState(
            task_id="t-ghost",
            status=TaskStatus.INPUT_REQUIRED,
            created_at=datetime.datetime.now(datetime.UTC),
            updated_at=datetime.datetime.now(datetime.UTC),
        )
        agent._tasks["t-ghost"] = state  # noqa: SLF001

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t-ghost", "input": "x"})
        assert resp.status_code == 500
        assert "no pending future" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Helpers for provisioning tests
# ---------------------------------------------------------------------------


def _mock_agent_card(
    *,
    name: str = "test-agent",
    card_id: str = "card_123",
    description: str = "test",
) -> Any:
    """Create a lightweight mock matching the AgentCard shape."""
    from unittest.mock import MagicMock

    card = MagicMock()
    card.name = name
    card.id = card_id
    card.description = description
    return card


def _mock_roost(
    *,
    name: str = "test-agent-inbox",
    roost_id: str = "roost_456",
    agent_card_id: str = "card_123",
    secret: str | None = "whsec_dGVzdC1zZWNyZXQ=",
) -> Any:
    from unittest.mock import MagicMock

    roost = MagicMock()
    roost.name = name
    roost.id = roost_id
    roost.agent_card_id = agent_card_id
    roost.secret = secret
    return roost


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


class TestProvision:
    @pytest.mark.asyncio
    async def test_local_dev_mode_is_noop(self) -> None:
        agent = AgentServer("local", "dev mode")
        await agent.provision()

        assert agent.provisioned is True
        assert agent.agent_card is None
        assert agent.roost is None

    @pytest.mark.asyncio
    async def test_idempotent_second_call(self) -> None:
        agent = AgentServer("local", "dev mode")
        await agent.provision()
        await agent.provision()  # should be a no-op
        assert agent.provisioned is True

    @pytest.mark.asyncio
    async def test_creates_agent_card_and_roost(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            card = _mock_agent_card(name="my-agent")
            roost = _mock_roost(agent_card_id=card.id)

            mock_client.list_agents = AsyncMock(return_value=[])
            mock_client.create_agent = AsyncMock(return_value=card)
            mock_client.list_roosts = AsyncMock(return_value=[])
            mock_client.create_roost = AsyncMock(return_value=roost)

            agent = AgentServer("my-agent", "Does things", pgns_key="pk_test_abc")
            await agent.provision()

            assert agent.provisioned is True
            assert agent.agent_card is card
            assert agent.roost is roost
            mock_client.create_agent.assert_awaited_once()
            mock_client.create_roost.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finds_existing_agent_card_and_roost(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            card = _mock_agent_card(name="my-agent")
            roost = _mock_roost(agent_card_id=card.id)

            mock_client.list_agents = AsyncMock(return_value=[card])
            mock_client.list_roosts = AsyncMock(return_value=[roost])

            agent = AgentServer("my-agent", "Does things", pgns_key="pk_test_abc")
            await agent.provision()

            assert agent.provisioned is True
            assert agent.agent_card is card
            assert agent.roost is roost
            mock_client.create_agent.assert_not_awaited()
            mock_client.create_roost.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_finds_card_creates_roost(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            card = _mock_agent_card(name="my-agent")
            roost = _mock_roost(agent_card_id=card.id)

            mock_client.list_agents = AsyncMock(return_value=[card])
            mock_client.list_roosts = AsyncMock(return_value=[])
            mock_client.create_roost = AsyncMock(return_value=roost)

            agent = AgentServer("my-agent", "Does things", pgns_key="pk_test_abc")
            await agent.provision()

            assert agent.agent_card is card
            assert agent.roost is roost
            mock_client.create_agent.assert_not_awaited()
            mock_client.create_roost.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_roost_without_secret_skips_webhook(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            card = _mock_agent_card(name="my-agent")
            roost = _mock_roost(agent_card_id=card.id, secret=None)

            mock_client.list_agents = AsyncMock(return_value=[card])
            mock_client.list_roosts = AsyncMock(return_value=[roost])

            agent = AgentServer("my-agent", "Does things", pgns_key="pk_test_abc")
            await agent.provision()

            assert agent.provisioned is True
            # _webhook should be None when no secret
            assert agent._webhook is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_provision_idempotent_with_key(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            card = _mock_agent_card(name="my-agent")
            roost = _mock_roost(agent_card_id=card.id)

            mock_client.list_agents = AsyncMock(return_value=[card])
            mock_client.list_roosts = AsyncMock(return_value=[roost])

            agent = AgentServer("my-agent", "Does things", pgns_key="pk_test_abc")
            await agent.provision()
            await agent.provision()

            # Only called once despite two provision() calls
            assert mock_client.list_agents.await_count == 1


# ---------------------------------------------------------------------------
# Webhook verification
# ---------------------------------------------------------------------------


class TestVerifyWebhook:
    def test_local_dev_mode_parses_without_verification(self) -> None:
        agent = AgentServer("local", "dev mode")
        payload = {"hello": "world"}
        body = json.dumps(payload)

        result = agent.verify_webhook(body, {})
        assert result == payload

    def test_local_dev_mode_accepts_bytes(self) -> None:
        agent = AgentServer("local", "dev mode")
        payload = {"hello": "world"}
        body = json.dumps(payload).encode()

        result = agent.verify_webhook(body, {})
        assert result == payload

    def test_valid_standard_webhook_signature(self) -> None:
        """End-to-end test: sign a payload and verify it."""
        secret = "whsec_" + base64.b64encode(b"test-secret-key!").decode()

        with patch("pgns.sdk.async_client.AsyncPigeonsClient"):
            agent = AgentServer("prod", "production", pgns_key="pk_test_abc")

        # Manually wire up a Webhook verifier (simulating post-provision state)
        from pgns.webhook import Webhook

        agent._webhook = Webhook(secret)  # noqa: SLF001

        # Build a signed request
        payload = {"task": "summarize", "input": "hello"}
        body = json.dumps(payload, separators=(",", ":"))
        timestamp = str(int(time.time()))
        msg_id = "msg_test123"

        key_bytes = base64.b64decode(secret[6:])
        signed_content = f"{msg_id}.{timestamp}.{body}".encode()
        sig = base64.b64encode(
            hmac_mod.new(key_bytes, signed_content, hashlib.sha256).digest()
        ).decode()

        headers = {
            "webhook-id": msg_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": f"v1,{sig}",
        }

        result = agent.verify_webhook(body, headers)
        assert result == payload

    def test_invalid_signature_raises(self) -> None:
        from pgns.errors import WebhookVerificationError

        secret = "whsec_" + base64.b64encode(b"test-secret-key!").decode()

        with patch("pgns.sdk.async_client.AsyncPigeonsClient"):
            agent = AgentServer("prod", "production", pgns_key="pk_test_abc")

        from pgns.webhook import Webhook

        agent._webhook = Webhook(secret)  # noqa: SLF001

        body = json.dumps({"bad": "payload"})
        headers = {
            "webhook-id": "msg_bad",
            "webhook-timestamp": str(int(time.time())),
            "webhook-signature": "v1,aW52YWxpZA==",
        }

        with pytest.raises(WebhookVerificationError):
            agent.verify_webhook(body, headers)

    def test_no_secret_raises_without_verification(self) -> None:
        """When provisioned but roost has no webhook verifier, raise an error."""
        from pgns.errors import WebhookVerificationError

        with patch("pgns.sdk.async_client.AsyncPigeonsClient"):
            agent = AgentServer("prod", "production", pgns_key="pk_test_abc")

        # Simulate provisioned state with no webhook verifier
        agent._webhook = None  # noqa: SLF001

        payload = {"hello": "world"}
        body = json.dumps(payload)
        with pytest.raises(WebhookVerificationError):
            agent.verify_webhook(body, {})


# ---------------------------------------------------------------------------
# Async mode (Prefer: respond-async, RFC 7240)
# ---------------------------------------------------------------------------


class TestAsyncMode:
    def test_returns_202_with_prefer_header(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"ok": True}

        client = TestClient(agent.app())
        resp = client.post(
            "/",
            json={"id": "t-async", "input": "x"},
            headers={"Prefer": "respond-async"},
        )
        assert resp.status_code == 202

    def test_preference_applied_header(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {}

        client = TestClient(agent.app())
        resp = client.post(
            "/",
            json={"id": "t-pref", "input": "x"},
            headers={"Prefer": "respond-async"},
        )
        assert resp.headers["preference-applied"] == "respond-async"

    def test_response_body_has_submitted_status(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {}

        client = TestClient(agent.app())
        resp = client.post(
            "/",
            json={"id": "t-body", "input": "x"},
            headers={"Prefer": "respond-async"},
        )
        data = resp.json()
        assert data["id"] == "t-body"
        assert data["status"] == "submitted"

    def test_sync_mode_still_returns_200(self) -> None:
        """Without Prefer header, behavior is unchanged."""
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {"sync": True}

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t-sync", "input": "x"})
        assert resp.status_code == 200
        assert resp.json()["result"] == {"sync": True}

    def test_duplicate_inflight_task_returns_409(self) -> None:
        """An async task still in-flight rejects duplicates (sync or async)."""
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, Any]:
            return {}

        # Manually place a non-terminal task in the store.
        now = datetime.datetime.now(datetime.UTC)
        agent._tasks["t-dup"] = TaskState(  # noqa: SLF001
            task_id="t-dup",
            status=TaskStatus.WORKING,
            created_at=now,
            updated_at=now,
        )

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t-dup", "input": "y"})
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# _run_async_handler
# ---------------------------------------------------------------------------


class TestRunAsyncHandler:
    @pytest.mark.asyncio
    async def test_publishes_working_and_completed_pigeons(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            async def handler(task: Task) -> dict[str, Any]:
                return {"done": True}

            task = Task(id="t-async-pub", input="x")
            now = datetime.datetime.now(datetime.UTC)
            state = TaskState(
                task_id="t-async-pub",
                status=TaskStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
            )
            agent._tasks["t-async-pub"] = state  # noqa: SLF001

            await agent._run_async_handler(task, handler, state, None, "default")  # noqa: SLF001
            # Allow fire-and-forget tasks to complete.
            await asyncio.sleep(0)

            # working (fire-and-forget) + completed (awaited) = 2 pigeons
            assert mock_client.send.await_count == 2
            statuses = sorted(
                call.kwargs["payload"]["status"] for call in mock_client.send.await_args_list
            )
            assert statuses == ["completed", "working"]

    @pytest.mark.asyncio
    async def test_completed_pigeon_includes_result_artifact(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            async def handler(task: Task) -> dict[str, Any]:
                return {"answer": 42}

            task = Task(id="t-artifact", input="x")
            now = datetime.datetime.now(datetime.UTC)
            state = TaskState(
                task_id="t-artifact",
                status=TaskStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
            )
            agent._tasks["t-artifact"] = state  # noqa: SLF001

            await agent._run_async_handler(task, handler, state, None, "default")  # noqa: SLF001
            await asyncio.sleep(0)

            # Find the completed pigeon call
            completed_call = next(
                c
                for c in mock_client.send.await_args_list
                if c.kwargs["payload"]["status"] == "completed"
            )
            assert completed_call.kwargs["payload"]["artifact"] == {"answer": 42}

    @pytest.mark.asyncio
    async def test_publishes_failed_on_exception(self) -> None:
        with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.send = AsyncMock(
                return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
            )

            agent = AgentServer("a", "b", pgns_key="pk_test")
            roost = MagicMock()
            roost.id = "roost-1"
            roost.secret = "whsec_dGVzdA=="
            agent._roost = roost  # noqa: SLF001
            agent._provisioned = True  # noqa: SLF001

            async def handler(task: Task) -> dict[str, str]:
                raise RuntimeError("boom")

            task = Task(id="t-async-fail", input="x")
            now = datetime.datetime.now(datetime.UTC)
            state = TaskState(
                task_id="t-async-fail",
                status=TaskStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
            )
            agent._tasks["t-async-fail"] = state  # noqa: SLF001

            await agent._run_async_handler(task, handler, state, None, "default")  # noqa: SLF001
            await asyncio.sleep(0)

            assert mock_client.send.await_count == 2
            statuses = sorted(
                call.kwargs["payload"]["status"] for call in mock_client.send.await_args_list
            )
            assert statuses == ["failed", "working"]

    @pytest.mark.asyncio
    async def test_evicts_task_on_completion(self) -> None:
        agent = AgentServer("a", "b")

        async def handler(task: Task) -> dict[str, Any]:
            return {"done": True}

        task = Task(id="t-evict", input="x")
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-evict",
            status=TaskStatus.SUBMITTED,
            created_at=now,
            updated_at=now,
        )
        agent._tasks["t-evict"] = state  # noqa: SLF001

        await agent._run_async_handler(task, handler, state, None, "default")  # noqa: SLF001

        assert "t-evict" not in agent.tasks

    @pytest.mark.asyncio
    async def test_evicts_task_on_failure(self) -> None:
        agent = AgentServer("a", "b")

        async def handler(task: Task) -> dict[str, str]:
            raise RuntimeError("boom")

        task = Task(id="t-evict-fail", input="x")
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-evict-fail",
            status=TaskStatus.SUBMITTED,
            created_at=now,
            updated_at=now,
        )
        agent._tasks["t-evict-fail"] = state  # noqa: SLF001

        await agent._run_async_handler(task, handler, state, None, "default")  # noqa: SLF001

        assert "t-evict-fail" not in agent.tasks

    @pytest.mark.asyncio
    async def test_context_variable_set_during_handler(self) -> None:
        agent = AgentServer("a", "b")
        captured: list[str | None] = []

        async def handler(task: Task) -> dict[str, str]:
            from pgns_agent import get_current_task

            ct = get_current_task()
            captured.append(ct.id if ct else None)
            return {}

        task = Task(id="t-ctx-async", input="x")
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-ctx-async",
            status=TaskStatus.SUBMITTED,
            created_at=now,
            updated_at=now,
        )
        agent._tasks["t-ctx-async"] = state  # noqa: SLF001

        await agent._run_async_handler(task, handler, state, None, "default")  # noqa: SLF001

        assert captured == ["t-ctx-async"]


# ---------------------------------------------------------------------------
# SSE endpoint (GET /tasks/{task_id}/events)
# ---------------------------------------------------------------------------


class TestSSEEndpoint:
    def test_unknown_task_returns_404(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        client = TestClient(agent.app())
        resp = client.get("/tasks/nonexistent/events")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# Task event broadcast
# ---------------------------------------------------------------------------


class TestTaskEventBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_sends_to_subscriber(self) -> None:
        agent = AgentServer("a", "b")
        queue = agent._subscribe_task("t-bcast")  # noqa: SLF001

        agent._broadcast_task_event("t-bcast", TaskStatus.WORKING)  # noqa: SLF001

        event = queue.get_nowait()
        assert event["task_id"] == "t-bcast"
        assert event["status"] == "working"
        assert "timestamp" in event

    @pytest.mark.asyncio
    async def test_terminal_sends_sentinel(self) -> None:
        agent = AgentServer("a", "b")
        queue = agent._subscribe_task("t-term")  # noqa: SLF001

        agent._broadcast_task_event("t-term", TaskStatus.COMPLETED, result={"ok": True})  # noqa: SLF001

        # First item: the event
        event = queue.get_nowait()
        assert event["status"] == "completed"
        assert event["result"] == {"ok": True}

        # Second item: sentinel (None)
        sentinel = queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_terminal_cleans_up_subscribers(self) -> None:
        agent = AgentServer("a", "b")
        agent._subscribe_task("t-cleanup")  # noqa: SLF001

        agent._broadcast_task_event("t-cleanup", TaskStatus.COMPLETED)  # noqa: SLF001

        assert "t-cleanup" not in agent._task_subscribers  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_no_subscribers_is_noop(self) -> None:
        agent = AgentServer("a", "b")
        # Should not raise
        agent._broadcast_task_event("t-nobody", TaskStatus.WORKING)  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_queue(self) -> None:
        agent = AgentServer("a", "b")
        queue = agent._subscribe_task("t-unsub")  # noqa: SLF001
        assert "t-unsub" in agent._task_subscribers  # noqa: SLF001

        agent._unsubscribe_task("t-unsub", queue)  # noqa: SLF001
        assert "t-unsub" not in agent._task_subscribers  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_broadcast_includes_result_for_completed(self) -> None:
        agent = AgentServer("a", "b")
        queue = agent._subscribe_task("t-result")  # noqa: SLF001

        agent._broadcast_task_event(  # noqa: SLF001
            "t-result", TaskStatus.COMPLETED, result={"summary": "done"}
        )

        event = queue.get_nowait()
        assert event["result"] == {"summary": "done"}

    @pytest.mark.asyncio
    async def test_broadcast_omits_result_when_none(self) -> None:
        agent = AgentServer("a", "b")
        queue = agent._subscribe_task("t-noresult")  # noqa: SLF001

        agent._broadcast_task_event("t-noresult", TaskStatus.WORKING)  # noqa: SLF001

        event = queue.get_nowait()
        assert "result" not in event
