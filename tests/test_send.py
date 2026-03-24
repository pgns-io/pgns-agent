# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentServer.send() outbound messaging."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pgns.sdk.errors import PigeonsError
from pgns_agent import AgentServer, Task, TaskMetadata
from pgns_agent._context import current_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_roost(*, secret: str | None = "whsec_dGVzdC1rZXk=") -> AsyncMock:
    """Return a mock Roost object with a configurable secret."""
    roost = AsyncMock()
    roost.secret = secret
    roost.id = "roost-target"
    return roost


def _make_agent() -> AgentServer:
    """Return an AgentServer with a mocked SDK client."""
    with patch("pgns.sdk.async_client.AsyncPigeonsClient") as mock_cls:
        agent = AgentServer("sender", "sends things", pgns_key="pk_test_send")
        # Replace the real client with a mock
        mock_client = mock_cls.return_value
        mock_client.get_roost = AsyncMock(return_value=_mock_roost())
        mock_client.send = AsyncMock(
            return_value=AsyncMock(id="pig-1", status="accepted", destinations=1)
        )
        return agent


# ---------------------------------------------------------------------------
# send() — success paths
# ---------------------------------------------------------------------------


class TestSendSuccess:
    @pytest.mark.asyncio
    async def test_sends_to_target_roost(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        result = await agent.send("roost-target", {"msg": "hello"})

        agent.client.get_roost.assert_awaited_once_with("roost-target")
        agent.client.send.assert_awaited_once()
        call_kwargs = agent.client.send.call_args.kwargs
        assert call_kwargs["event_type"] == "agent.task"
        assert call_kwargs["payload"] == {"msg": "hello"}
        assert call_kwargs["signing_secret"] == "whsec_dGVzdC1rZXk="
        assert result.id == "pig-1"

    @pytest.mark.asyncio
    async def test_custom_event_type(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        await agent.send("roost-target", {}, event_type="agent.status")

        call_kwargs = agent.client.send.call_args.kwargs
        assert call_kwargs["event_type"] == "agent.status"

    @pytest.mark.asyncio
    async def test_no_extra_headers_without_task_context(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        await agent.send("roost-target", {"data": 1})

        call_kwargs = agent.client.send.call_args.kwargs
        assert call_kwargs["extra_headers"] is None


# ---------------------------------------------------------------------------
# send() — correlation ID propagation
# ---------------------------------------------------------------------------


class TestCorrelationIdPropagation:
    @pytest.mark.asyncio
    async def test_propagates_correlation_id_from_task_context(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        task = Task(
            id="task-42",
            input={"text": "hi"},
            metadata=TaskMetadata(correlation_id="corr-abc-123"),
        )
        token = current_task.set(task)
        try:
            await agent.send("roost-target", {"reply": "ok"})
        finally:
            current_task.reset(token)

        call_kwargs = agent.client.send.call_args.kwargs
        assert call_kwargs["extra_headers"] == {"X-Pgns-CorrelationId": "corr-abc-123"}

    @pytest.mark.asyncio
    async def test_drops_invalid_correlation_id(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        # Correlation ID with spaces and exceeding reasonable length
        task = Task(
            id="task-44",
            input={},
            metadata=TaskMetadata(correlation_id="has spaces\tinvalid"),
        )
        token = current_task.set(task)
        try:
            await agent.send("roost-target", {"data": 1})
        finally:
            current_task.reset(token)

        call_kwargs = agent.client.send.call_args.kwargs
        assert call_kwargs["extra_headers"] is None

    @pytest.mark.asyncio
    async def test_drops_overlong_correlation_id(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        task = Task(
            id="task-45",
            input={},
            metadata=TaskMetadata(correlation_id="x" * 200),
        )
        token = current_task.set(task)
        try:
            await agent.send("roost-target", {"data": 1})
        finally:
            current_task.reset(token)

        call_kwargs = agent.client.send.call_args.kwargs
        assert call_kwargs["extra_headers"] is None

    @pytest.mark.asyncio
    async def test_no_header_when_correlation_id_is_none(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        task = Task(id="task-43", input={})
        token = current_task.set(task)
        try:
            await agent.send("roost-target", {"data": 1})
        finally:
            current_task.reset(token)

        call_kwargs = agent.client.send.call_args.kwargs
        assert call_kwargs["extra_headers"] is None


# ---------------------------------------------------------------------------
# send() — error cases
# ---------------------------------------------------------------------------


class TestSendErrors:
    @pytest.mark.asyncio
    async def test_raises_in_local_dev_mode(self) -> None:
        agent = AgentServer("local", "dev mode")

        with pytest.raises(RuntimeError, match="local dev mode"):
            await agent.send("roost-x", {"msg": "hi"})

    @pytest.mark.asyncio
    async def test_raises_when_roost_has_no_secret(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        agent.client.get_roost = AsyncMock(return_value=_mock_roost(secret=None))

        with pytest.raises(RuntimeError, match="no signing secret"):
            await agent.send("roost-no-secret", {"msg": "hi"})

    @pytest.mark.asyncio
    async def test_propagates_get_roost_api_error(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        agent.client.get_roost = AsyncMock(side_effect=PigeonsError("Not Found", status=404))

        with pytest.raises(PigeonsError, match="Not Found"):
            await agent.send("roost-missing", {"msg": "hi"})

    @pytest.mark.asyncio
    async def test_propagates_get_roost_network_error(self) -> None:
        agent = _make_agent()
        assert agent.client is not None

        agent.client.get_roost = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with pytest.raises(httpx.ConnectError):
            await agent.send("roost-unreachable", {"msg": "hi"})
