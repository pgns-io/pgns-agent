# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Adapter base class and AgentServer.use() wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from pgns_agent import Adapter, AgentServer, Task
from pgns_agent._server import DEFAULT_HANDLER_NAME

# ---------------------------------------------------------------------------
# Concrete adapter fixtures
# ---------------------------------------------------------------------------


class EchoAdapter(Adapter):
    """Synchronous adapter that echoes the input."""

    async def handle(self, task_input: dict[str, Any]) -> dict[str, Any]:
        return {"echo": task_input}


class MetadataAdapter(Adapter):
    """Adapter that returns framework-specific metadata."""

    async def handle(self, task_input: dict[str, Any]) -> dict[str, Any]:
        return {
            "output": f"processed {task_input.get('text', '')}",
            "metadata": {"model": "test-model", "token_count": 42},
        }


class StreamingAdapter(Adapter):
    """Streaming adapter that yields multiple chunks."""

    async def handle(self, task_input: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        words = task_input.get("text", "").split()
        for i, word in enumerate(words):
            yield {"token": word, "index": i}
        yield {"done": True, "tokens": len(words)}


class EmptyStreamAdapter(Adapter):
    """Streaming adapter that yields nothing."""

    async def handle(self, task_input: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        return
        yield  # makes this function an async generator


class SingleChunkStreamAdapter(Adapter):
    """Streaming adapter that yields exactly one chunk."""

    async def handle(self, task_input: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        yield {"result": "only-chunk"}


class ErrorAdapter(Adapter):
    """Adapter that raises during handle."""

    async def handle(self, task_input: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("adapter exploded")


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


class TestAdapterBaseClass:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Adapter()  # type: ignore[abstract]

    def test_subclass_must_implement_handle(self) -> None:
        class Incomplete(Adapter):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        adapter = EchoAdapter()
        assert isinstance(adapter, Adapter)


# ---------------------------------------------------------------------------
# agent.use() — registration
# ---------------------------------------------------------------------------


class TestUseRegistration:
    def test_registers_default_handler(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter())
        assert DEFAULT_HANDLER_NAME in agent.handlers

    def test_registers_named_skill(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter(), skill="echo")
        assert "echo" in agent.handlers

    def test_multiple_adapters_different_skills(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter(), skill="echo")
        agent.use(MetadataAdapter(), skill="meta")
        assert set(agent.handlers.keys()) == {"echo", "meta"}

    def test_adapter_plus_on_task(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter(), skill="echo")

        @agent.on_task
        async def fallback(task: Task) -> dict[str, bool]:
            return {"fallback": True}

        assert set(agent.handlers.keys()) == {"echo", DEFAULT_HANDLER_NAME}

    def test_duplicate_default_raises(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter())

        with pytest.raises(ValueError, match="already registered"):
            agent.use(MetadataAdapter())

    def test_duplicate_named_raises(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter(), skill="echo")

        with pytest.raises(ValueError, match="already registered"):
            agent.use(MetadataAdapter(), skill="echo")

    def test_explicit_default_name_raises(self) -> None:
        agent = AgentServer("a", "b")
        with pytest.raises(ValueError, match="reserved"):
            agent.use(EchoAdapter(), skill="default")

    def test_non_adapter_raises_type_error(self) -> None:
        agent = AgentServer("a", "b")
        with pytest.raises(TypeError, match="Adapter instance"):
            agent.use("not an adapter")  # type: ignore[arg-type]

    def test_dict_raises_type_error(self) -> None:
        agent = AgentServer("a", "b")
        with pytest.raises(TypeError, match="Adapter instance"):
            agent.use({"handle": lambda x: x})  # type: ignore[arg-type]

    def test_sync_handle_raises_type_error(self) -> None:
        class SyncAdapter(Adapter):
            def handle(self, task_input: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
                return {"result": "ok"}

        agent = AgentServer("a", "b")
        with pytest.raises(TypeError, match="must be async"):
            agent.use(SyncAdapter())

    def test_adapter_and_on_task_conflict_raises(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        with pytest.raises(ValueError, match="already registered"):
            agent.use(EchoAdapter())


# ---------------------------------------------------------------------------
# agent.use() — sync (dict) adapters via HTTP
# ---------------------------------------------------------------------------


class TestUseSyncAdapter:
    def test_echo_adapter(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter())

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {"msg": "hello"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "t1"
        assert data["result"] == {"echo": {"msg": "hello"}}

    def test_metadata_passes_through(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(MetadataAdapter())

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {"text": "hello"}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["output"] == "processed hello"
        assert result["metadata"]["model"] == "test-model"
        assert result["metadata"]["token_count"] == 42

    def test_named_skill_dispatch(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter(), skill="echo")

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "x", "skill": "echo"})
        assert resp.status_code == 200
        assert resp.json()["result"] == {"echo": "x"}

    def test_unknown_skill_falls_back_to_adapter_default(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter())

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "y", "skill": "unknown"})
        assert resp.status_code == 200
        assert resp.json()["result"]["echo"] == "y"

    def test_adapter_exception_returns_500(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(ErrorAdapter())

        client = TestClient(agent.app(), raise_server_exceptions=False)
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 500

    def test_null_input_normalized_to_empty_dict(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter())

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1"})
        assert resp.status_code == 200
        assert resp.json()["result"] == {"echo": {}}


# ---------------------------------------------------------------------------
# agent.use() — streaming (AsyncIterator) adapters via HTTP
# ---------------------------------------------------------------------------


class TestUseStreamingAdapter:
    def test_streaming_returns_last_chunk(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(StreamingAdapter())

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {"text": "hello world foo"}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result == {"done": True, "tokens": 3}

    def test_empty_stream_returns_null(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EmptyStreamAdapter())

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        assert resp.json()["result"] is None

    def test_single_chunk_stream(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(SingleChunkStreamAdapter())

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        assert resp.json()["result"] == {"result": "only-chunk"}

    def test_streaming_named_skill(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(StreamingAdapter(), skill="stream")

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {"text": "a b"}, "skill": "stream"})
        assert resp.status_code == 200
        assert resp.json()["result"]["done"] is True


# ---------------------------------------------------------------------------
# Agent Card integration
# ---------------------------------------------------------------------------


class TestAdapterAgentCard:
    def test_default_adapter_excluded_from_skills(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter())

        client = TestClient(agent.app())
        resp = client.get("/.well-known/agent.json")
        assert resp.json()["skills"] == []

    def test_named_adapter_appears_in_skills(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(EchoAdapter(), skill="echo")
        agent.use(MetadataAdapter(), skill="enrich")

        client = TestClient(agent.app())
        resp = client.get("/.well-known/agent.json")
        skill_ids = {s["id"] for s in resp.json()["skills"]}
        assert skill_ids == {"echo", "enrich"}
