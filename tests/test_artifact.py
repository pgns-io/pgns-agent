# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for artifact types, errors, ArtifactStore, and Task integration."""

from __future__ import annotations

import dataclasses

import pytest

from pgns_agent import (
    AgentServer,
    ArtifactAccessError,
    ArtifactError,
    ArtifactMediaType,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactStore,
    ArtifactTooLargeError,
    Task,
)

# ---------------------------------------------------------------------------
# ArtifactRef
# ---------------------------------------------------------------------------


class TestArtifactRef:
    def test_frozen(self) -> None:
        ref = ArtifactRef(
            artifact_id="art_abc",
            url="local://artifacts/art_abc",
            access_token="tok",
            media_type="application/json",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.artifact_id = "new"  # type: ignore[misc]

    def test_defaults(self) -> None:
        ref = ArtifactRef(
            artifact_id="art_1",
            url="/v1/artifacts/art_1",
            access_token="tok",
            media_type="text/plain",
        )
        assert ref.size_bytes is None
        assert ref.task_id is None
        assert ref.expires_at is None

    def test_equality(self) -> None:
        kwargs = {
            "artifact_id": "art_1",
            "url": "u",
            "access_token": "t",
            "media_type": "application/json",
        }
        assert ArtifactRef(**kwargs) == ArtifactRef(**kwargs)

    def test_repr_redacts_token(self) -> None:
        ref = ArtifactRef(
            artifact_id="art_1",
            url="u",
            access_token="secret_token_value",
            media_type="application/json",
        )
        r = repr(ref)
        assert "art_1" in r
        assert "secret_token_value" not in r


# ---------------------------------------------------------------------------
# ArtifactMediaType
# ---------------------------------------------------------------------------


class TestArtifactMediaType:
    def test_values(self) -> None:
        assert ArtifactMediaType.JSON == "application/json"
        assert ArtifactMediaType.TEXT == "text/plain"
        assert ArtifactMediaType.BINARY == "application/octet-stream"
        assert ArtifactMediaType.PDF == "application/pdf"
        assert ArtifactMediaType.PNG == "image/png"

    def test_is_str(self) -> None:
        assert isinstance(ArtifactMediaType.JSON, str)


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class TestArtifactErrors:
    def test_base_error(self) -> None:
        err = ArtifactError("boom")
        assert str(err) == "boom"
        assert isinstance(err, Exception)

    def test_not_found_reason(self) -> None:
        for reason in ("expired", "consumed", "not_found"):
            err = ArtifactNotFoundError(reason)
            assert err.reason == reason
            assert reason in str(err)

    def test_not_found_custom_message(self) -> None:
        err = ArtifactNotFoundError("expired", "Custom message")
        assert str(err) == "Custom message"
        assert err.reason == "expired"

    def test_not_found_inherits(self) -> None:
        assert issubclass(ArtifactNotFoundError, ArtifactError)
        assert issubclass(ArtifactNotFoundError, Exception)

    def test_access_error(self) -> None:
        err = ArtifactAccessError("denied")
        assert str(err) == "denied"
        assert isinstance(err, ArtifactError)

    def test_too_large(self) -> None:
        err = ArtifactTooLargeError(size_bytes=2_000_000, limit_bytes=1_000_000)
        assert err.size_bytes == 2_000_000
        assert err.limit_bytes == 1_000_000
        assert "2000000" in str(err)
        assert "1000000" in str(err)

    def test_too_large_custom_message(self) -> None:
        err = ArtifactTooLargeError(100, 50, "too big")
        assert str(err) == "too big"


# ---------------------------------------------------------------------------
# ArtifactStore (local dev)
# ---------------------------------------------------------------------------


class TestArtifactStore:
    @pytest.fixture()
    def store(self) -> ArtifactStore:
        return ArtifactStore()

    @pytest.mark.asyncio
    async def test_store_returns_ref(self, store: ArtifactStore) -> None:
        ref = await store._store_raw(b"hello", content_type="text/plain")
        assert ref.url.startswith("local://artifacts/")
        assert ref.artifact_id.startswith("art_")
        assert ref.media_type == "text/plain"
        assert ref.size_bytes == 5
        assert ref.access_token

    @pytest.mark.asyncio
    async def test_round_trip(self, store: ArtifactStore) -> None:
        data = b'{"key":"value"}'
        ref = await store._store_raw(data, content_type="application/json")
        raw, ct = await store._get_raw(ref.artifact_id, token=ref.access_token)
        assert raw == data
        assert ct == "application/json"

    @pytest.mark.asyncio
    async def test_get_invalid_token(self, store: ArtifactStore) -> None:
        ref = await store._store_raw(b"data", content_type="text/plain")
        with pytest.raises(ArtifactAccessError):
            await store._get_raw(ref.artifact_id, token="wrong-token")

    @pytest.mark.asyncio
    async def test_get_unknown_id(self, store: ArtifactStore) -> None:
        with pytest.raises(ArtifactNotFoundError) as exc_info:
            await store._get_raw("art_nonexistent")
        assert exc_info.value.reason == "not_found"

    @pytest.mark.asyncio
    async def test_list_by_task(self, store: ArtifactStore) -> None:
        await store._store_raw(b"a", content_type="text/plain", task_id="t1")
        await store._store_raw(b"b", content_type="text/plain", task_id="t1")
        await store._store_raw(b"c", content_type="text/plain", task_id="t2")
        refs = store.list_by_task("t1")
        assert len(refs) == 2
        assert all(r.task_id == "t1" for r in refs)


# ---------------------------------------------------------------------------
# Task integration (via TestClient)
# ---------------------------------------------------------------------------


class TestTaskArtifactMethods:
    @pytest.mark.asyncio
    async def test_store_artifact_outside_handler(self) -> None:
        task = Task(id="t1", input=None)
        with pytest.raises(RuntimeError, match="store_artifact"):
            await task.store_artifact({"x": 1})

    @pytest.mark.asyncio
    async def test_get_artifact_outside_handler(self) -> None:
        task = Task(id="t1", input=None)
        with pytest.raises(RuntimeError, match="get_artifact"):
            await task.get_artifact("local://artifacts/art_000000000000000000000000")

    @pytest.mark.asyncio
    async def test_get_artifacts_outside_handler(self) -> None:
        task = Task(id="t1", input=None)
        with pytest.raises(RuntimeError, match="get_artifacts"):
            await task.get_artifacts(["local://artifacts/art_000000000000000000000000"])

    def test_store_and_get_json(self) -> None:
        agent = AgentServer("test", "test")

        @agent.on_task
        async def handle(task: Task) -> dict:
            ref = await task.store_artifact({"key": "value"})
            data = await task.get_artifact(ref.url, token=ref.access_token)
            return {"round_trip": data}

        client = agent.test_client()
        resp = client.send_task("go")
        assert resp.status == "completed"
        assert resp.result == {"round_trip": {"key": "value"}}

    def test_store_and_get_text(self) -> None:
        agent = AgentServer("test", "test")

        @agent.on_task
        async def handle(task: Task) -> dict:
            ref = await task.store_artifact("hello world", media_type="text/plain")
            data = await task.get_artifact(ref.url, token=ref.access_token)
            return {"text": data}

        client = agent.test_client()
        resp = client.send_task("go")
        assert resp.status == "completed"
        assert resp.result == {"text": "hello world"}

    def test_store_and_get_binary(self) -> None:
        agent = AgentServer("test", "test")

        @agent.on_task
        async def handle(task: Task) -> dict:
            ref = await task.store_artifact(b"\x89PNG\r\n", media_type="image/png")
            data = await task.get_artifact(ref.url, token=ref.access_token)
            # Binary comes back as bytes; return length for JSON serialization
            return {"is_bytes": isinstance(data, bytes), "length": len(data)}

        client = agent.test_client()
        resp = client.send_task("go")
        assert resp.status == "completed"
        assert resp.result["is_bytes"] is True
        assert resp.result["length"] == 6

    def test_get_artifacts_concurrent(self) -> None:
        agent = AgentServer("test", "test")

        @agent.on_task
        async def handle(task: Task) -> dict:
            ref1 = await task.store_artifact({"a": 1})
            ref2 = await task.store_artifact({"b": 2})
            results = await task.get_artifacts(
                [ref1.url, ref2.url],
                tokens=[ref1.access_token, ref2.access_token],
            )
            return {"results": results}

        client = agent.test_client()
        resp = client.send_task("go")
        assert resp.status == "completed"
        assert resp.result["results"] == [{"a": 1}, {"b": 2}]

    def test_get_artifacts_token_mismatch(self) -> None:
        agent = AgentServer("test", "test")

        @agent.on_task
        async def handle(task: Task) -> dict:
            await task.get_artifacts(["u1", "u2"], tokens=["t1"])
            return {}

        client = agent.test_client()
        resp = client.send_task("go")
        assert resp.status == "failed"

    def test_cross_task_shared_store(self) -> None:
        agent = AgentServer("test", "test")
        captured_ref: dict = {}

        @agent.on_task
        async def handle(task: Task) -> dict:
            if task.input == "store":
                ref = await task.store_artifact({"shared": True})
                captured_ref["url"] = ref.url
                captured_ref["token"] = ref.access_token
                return {"stored": ref.artifact_id}
            else:
                data = await task.get_artifact(captured_ref["url"], token=captured_ref["token"])
                return {"fetched": data}

        client = agent.test_client()
        resp1 = client.send_task("store")
        assert resp1.status == "completed"

        resp2 = client.send_task("retrieve")
        assert resp2.status == "completed"
        assert resp2.result == {"fetched": {"shared": True}}

    def test_artifact_store_property(self) -> None:
        agent = AgentServer("test", "test")
        store = ArtifactStore()
        client = agent.test_client(artifact_store=store)
        assert client.artifact_store is store

    def test_injected_artifact_store(self) -> None:
        agent = AgentServer("test", "test")
        store = ArtifactStore()

        @agent.on_task
        async def handle(task: Task) -> dict:
            ref = await task.store_artifact({"injected": True})
            return {"id": ref.artifact_id}

        client = agent.test_client(artifact_store=store)
        resp = client.send_task("go")
        assert resp.status == "completed"
        # Verify the injected store has the artifact
        assert len(store._artifacts) == 1

    def test_get_artifacts_without_tokens(self) -> None:
        agent = AgentServer("test", "test")

        @agent.on_task
        async def handle(task: Task) -> dict:
            ref1 = await task.store_artifact({"a": 1})
            ref2 = await task.store_artifact({"b": 2})
            results = await task.get_artifacts([ref1.url, ref2.url])
            return {"results": results}

        client = agent.test_client()
        resp = client.send_task("go")
        assert resp.status == "completed"
        assert resp.result["results"] == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# ArtifactStore public API (put / get / list_all)
# ---------------------------------------------------------------------------


class TestArtifactStorePublicAPI:
    @pytest.fixture()
    def store(self) -> ArtifactStore:
        return ArtifactStore()

    def test_put_and_get_json(self, store: ArtifactStore) -> None:
        store.put("art_json", {"key": "value"}, "application/json")
        assert store.get("art_json") == {"key": "value"}

    def test_put_and_get_text(self, store: ArtifactStore) -> None:
        store.put("art_text", "hello world", "text/plain")
        assert store.get("art_text") == "hello world"

    def test_put_and_get_binary(self, store: ArtifactStore) -> None:
        data = b"\x89PNG\r\n"
        store.put("art_png", data, "image/png")
        assert store.get("art_png") == data

    def test_get_not_found(self, store: ArtifactStore) -> None:
        with pytest.raises(ArtifactNotFoundError) as exc_info:
            store.get("art_nonexistent")
        assert exc_info.value.reason == "not_found"

    def test_put_overwrites(self, store: ArtifactStore) -> None:
        store.put("art_ow", {"v": 1}, "application/json")
        store.put("art_ow", {"v": 2}, "application/json")
        assert store.get("art_ow") == {"v": 2}

    def test_list_all_empty(self, store: ArtifactStore) -> None:
        assert store.list_all() == {}

    def test_list_all_with_items(self, store: ArtifactStore) -> None:
        store.put("art_a", {"a": 1}, "application/json")
        store.put("art_b", "text", "text/plain")
        result = store.list_all()
        assert result == {"art_a": {"a": 1}, "art_b": "text"}

    def test_testclient_put_get_pattern(self) -> None:
        """Pre-populate store with put(), then retrieve inside handler."""
        agent = AgentServer("test", "test")
        store = ArtifactStore()
        art_id = "art_019d2a7b-73fe-7122-9bb6-6bfcc1d77333"
        store.put(art_id, {"doc": "data"}, "application/json")

        @agent.on_task
        async def handle(task: Task) -> dict:
            data = await task.get_artifact(
                f"local://artifacts/{art_id}",
            )
            return {"fetched": data}

        client = agent.test_client(artifact_store=store)
        resp = client.send_task("go")
        assert resp.status == "completed"
        assert resp.result == {"fetched": {"doc": "data"}}

    def test_testclient_store_get_pattern(self) -> None:
        """Issue spec example: handler stores, test asserts via store.get()."""
        agent = AgentServer("test", "test")

        @agent.on_task
        async def handle(task: Task) -> dict:
            ref = await task.store_artifact({"summary": "done"})
            return {"artifact": {"artifact_id": ref.artifact_id}}

        store = ArtifactStore()
        client = agent.test_client(artifact_store=store)
        resp = client.send_task({"document": "..."})
        ref = resp.result["artifact"]
        data = store.get(ref["artifact_id"])
        assert data["summary"] == "done"
