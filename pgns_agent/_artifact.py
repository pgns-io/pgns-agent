# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Artifact types, in-memory store, and escrow bridge."""

from __future__ import annotations

__all__ = ["ArtifactMediaType", "ArtifactRef", "ArtifactStore"]

import asyncio
import base64
import dataclasses
import enum
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pgns_agent._errors import (
    ArtifactAccessError,
    ArtifactNotFoundError,
    ArtifactTooLargeError,
)

if TYPE_CHECKING:
    from pgns.async_client import AsyncPigeonsClient

logger = logging.getLogger("pgns_agent")

_ARTIFACT_ID_RE = re.compile(r"^art_[0-9a-f-]{24,36}$")


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a stored artifact, returned by store_artifact()."""

    artifact_id: str
    url: str
    access_token: str = dataclasses.field(repr=False)
    media_type: str
    size_bytes: int | None = None
    task_id: str | None = None
    expires_at: str | None = None


class ArtifactMediaType(enum.StrEnum):
    """Common media types for artifact content."""

    JSON = "application/json"
    TEXT = "text/plain"
    BINARY = "application/octet-stream"
    PDF = "application/pdf"
    PNG = "image/png"


# ---------------------------------------------------------------------------
# In-memory artifact store for local dev / testing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class _StoredArtifact:
    """Internal record kept by :class:`ArtifactStore`."""

    data: bytes
    content_type: str
    raw_token: str
    task_id: str | None
    auto_delete: bool
    ref: ArtifactRef


class ArtifactStore:
    """In-memory artifact store for local dev and testing.

    Stores artifacts in a dict keyed by artifact_id with synthetic
    ``local://`` URLs.  Thread-safe via asyncio (single-threaded event loop).
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, _StoredArtifact] = {}

    # ----- public convenience API (sync, for tests) -------------------------

    def put(self, art_id: str, data: Any, media_type: str) -> None:
        """Store an artifact with an explicit ID (for test pre-population).

        Serializes *data* based on *media_type* using the same logic as
        Task.store_artifact(). Overwrites any existing artifact with the
        same *art_id*.

        Raises:
            ValueError: If *art_id* does not match the required format.
        """
        raw = _serialize(data, media_type)
        raw_token = base64.urlsafe_b64encode(os.urandom(32)).decode()
        ref = ArtifactRef(
            artifact_id=art_id,
            url=f"local://artifacts/{art_id}",
            access_token=raw_token,
            media_type=media_type,
            size_bytes=len(raw),
        )
        self._artifacts[art_id] = _StoredArtifact(
            data=raw,
            content_type=media_type,
            raw_token=raw_token,
            task_id=None,
            auto_delete=False,
            ref=ref,
        )

    def get(self, art_id: str) -> Any:
        """Return the deserialized artifact data.

        Unlike ``_get_raw()``, this is synchronous, skips token
        validation, and returns the deserialized value directly.

        Raises:
            ArtifactNotFoundError: If *art_id* is not in the store.
        """
        stored = self._artifacts.get(art_id)
        if stored is None:
            raise ArtifactNotFoundError("not_found")
        return _deserialize(stored.data, stored.content_type)

    def list_all(self) -> dict[str, Any]:
        """Return all stored artifacts as ``{artifact_id: deserialized_data}``."""
        return {
            art_id: _deserialize(stored.data, stored.content_type)
            for art_id, stored in self._artifacts.items()
        }

    # ----- internal async API (used by _ArtifactEscrow) --------------------

    async def _store_raw(
        self,
        data: bytes,
        *,
        content_type: str,
        task_id: str | None = None,
        auto_delete: bool = False,
    ) -> ArtifactRef:
        """Store *data* and return an :class:`ArtifactRef`."""
        artifact_id = f"art_{os.urandom(12).hex()}"
        raw_token = base64.urlsafe_b64encode(os.urandom(32)).decode()

        ref = ArtifactRef(
            artifact_id=artifact_id,
            url=f"local://artifacts/{artifact_id}",
            access_token=raw_token,
            media_type=content_type,
            size_bytes=len(data),
            task_id=task_id,
        )
        self._artifacts[artifact_id] = _StoredArtifact(
            data=data,
            content_type=content_type,
            raw_token=raw_token,
            task_id=task_id,
            auto_delete=auto_delete,
            ref=ref,
        )
        return ref

    async def _get_raw(self, artifact_id: str, *, token: str | None = None) -> tuple[bytes, str]:
        """Return ``(data, content_type)`` for *artifact_id*.

        Raises:
            ArtifactNotFoundError: If the artifact does not exist.
            ArtifactAccessError: If *token* does not match.
        """
        stored = self._artifacts.get(artifact_id)
        if stored is None:
            raise ArtifactNotFoundError("not_found")
        if token is not None and token != stored.raw_token:
            raise ArtifactAccessError("Invalid access token.")
        data, content_type = stored.data, stored.content_type
        if stored.auto_delete:
            del self._artifacts[artifact_id]
        return data, content_type

    def list_by_task(self, task_id: str) -> list[ArtifactRef]:
        """Return all artifact refs for a given *task_id*."""
        return [s.ref for s in self._artifacts.values() if s.task_id == task_id]


# ---------------------------------------------------------------------------
# Escrow bridge (not public API)
# ---------------------------------------------------------------------------


def _serialize(data: Any, media_type: str) -> bytes:
    """Serialize *data* to bytes based on *media_type*."""
    if isinstance(data, bytes):
        return data
    if media_type == "application/json":
        return json.dumps(data, separators=(",", ":")).encode()
    if media_type.startswith("text/"):
        return str(data).encode("utf-8")
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError(f"Cannot serialize {type(data).__name__} for media_type={media_type!r}")


def _deserialize(raw: bytes, content_type: str) -> Any:
    """Deserialize *raw* bytes based on *content_type*."""
    if content_type == "application/json":
        return json.loads(raw)
    if content_type.startswith("text/"):
        return raw.decode("utf-8")
    return raw


class _ArtifactEscrow:
    """Bridges Task.store/get_artifact to the SDK client or local dev ArtifactStore.

    In production (client is not None): delegates to AsyncPigeonsClient methods.
    In local dev (client is None): delegates to the shared ArtifactStore.
    """

    __slots__ = (
        "_client",
        "_task_id",
        "_correlation_id",
        "_store",
        "_metadata_cache",
        "_cache_lock",
    )

    def __init__(
        self,
        client: AsyncPigeonsClient | None,
        task_id: str,
        correlation_id: str | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        self._client = client
        self._task_id = task_id
        self._correlation_id = correlation_id
        self._store = store
        # Cache artifact metadata from list call to avoid repeated requests.
        self._metadata_cache: dict[str, str] | None = None  # artifact_id -> content_type
        self._cache_lock = asyncio.Lock()

    async def store_artifact(
        self,
        data: Any,
        *,
        media_type: str,
        ttl_seconds: int | None,
        auto_delete: bool,
    ) -> ArtifactRef:
        """Serialize and store an artifact, returning an :class:`ArtifactRef`."""
        raw = _serialize(data, media_type)

        if self._client is not None:
            try:
                resp = await self._client.create_artifact(
                    raw,
                    content_type=media_type,
                    task_id=self._task_id,
                    correlation_id=self._correlation_id,
                    auto_delete=auto_delete,
                )
            except Exception as exc:
                _map_sdk_error(exc)
                raise
            # Update cache so subsequent get_artifact resolves the content type.
            if self._metadata_cache is not None:
                self._metadata_cache[resp.artifact_id] = media_type
            ref = ArtifactRef(
                artifact_id=resp.artifact_id,
                url=resp.url,
                access_token=resp.access_token,
                media_type=media_type,
                size_bytes=resp.size_bytes,
                task_id=self._task_id,
                expires_at=resp.expires_at,
            )
            return ref

        # Local dev mode
        assert self._store is not None  # noqa: S101
        return await self._store._store_raw(
            raw,
            content_type=media_type,
            task_id=self._task_id,
            auto_delete=auto_delete,
        )

    async def get_artifact(self, url: str, *, token: str | None = None) -> Any:
        """Download and deserialize an artifact by URL."""
        artifact_id = urlparse(url).path.rsplit("/", 1)[-1]
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ArtifactNotFoundError("invalid_id", f"Invalid artifact ID: {artifact_id!r}")

        if self._client is not None:
            raw, content_type = await self._get_artifact_production(artifact_id, token=token)
            return _deserialize(raw, content_type)

        # Local dev mode
        assert self._store is not None  # noqa: S101
        raw_bytes, content_type = await self._store._get_raw(artifact_id, token=token)
        return _deserialize(raw_bytes, content_type)

    async def _get_artifact_production(
        self, artifact_id: str, *, token: str | None
    ) -> tuple[bytes, str]:
        """Fetch raw bytes and content type from the SDK, mapping HTTP errors."""
        try:
            return await self._client.get_artifact(artifact_id, token=token)  # type: ignore[union-attr]
        except Exception as exc:
            _map_sdk_error(exc)
            raise

    async def _resolve_content_type(self, artifact_id: str) -> str:
        """Look up content_type for *artifact_id*, caching the list call."""
        async with self._cache_lock:
            if self._metadata_cache is None:
                self._metadata_cache = {}
                try:
                    page = await self._client.list_artifacts(task_id=self._task_id)  # type: ignore[union-attr]
                    for art in page.data:
                        self._metadata_cache[art.id] = art.content_type
                    # Exhaust all pages.
                    while getattr(page, "has_more", False):
                        page = await self._client.list_artifacts(  # type: ignore[union-attr]
                            task_id=self._task_id,
                            cursor=page.cursor,  # type: ignore[union-attr]
                        )
                        for art in page.data:
                            self._metadata_cache[art.id] = art.content_type
                except Exception:
                    logger.warning(
                        "Failed to list artifacts for content-type resolution",
                        exc_info=True,
                    )

        return self._metadata_cache.get(artifact_id, "application/octet-stream")


def _map_sdk_error(exc: Exception) -> None:
    """Re-raise SDK HTTP errors as artifact-domain errors when possible."""
    # httpx.HTTPStatusError
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        return
    if status == 404:
        try:
            body = exc.response.json()  # type: ignore[attr-defined]
            reason = body.get("reason", "not_found")
        except Exception:
            reason = "not_found"
        raise ArtifactNotFoundError(reason) from exc
    if status == 403:
        raise ArtifactAccessError("Access denied.") from exc
    if status == 413:
        try:
            body = exc.response.json()  # type: ignore[attr-defined]
            raise ArtifactTooLargeError(
                size_bytes=int(body.get("size_bytes") or 0),
                limit_bytes=int(body.get("max_bytes") or 0),
            ) from exc
        except ArtifactTooLargeError:
            raise  # Re-raise to prevent the bare except from swallowing it.
        except Exception:
            raise ArtifactTooLargeError(size_bytes=0, limit_bytes=0) from exc
