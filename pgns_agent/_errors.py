# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Artifact-related error classes."""

from __future__ import annotations

__all__ = [
    "ArtifactAccessError",
    "ArtifactError",
    "ArtifactNotFoundError",
    "ArtifactTooLargeError",
]


class ArtifactError(Exception):
    """Base class for artifact-related errors."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact cannot be found, has expired, or was consumed."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or f"Artifact not found: {reason}")


class ArtifactAccessError(ArtifactError):
    """Raised when the caller lacks permission to access an artifact."""


class ArtifactTooLargeError(ArtifactError):
    """Raised when an artifact exceeds the plan's size limit."""

    def __init__(self, size_bytes: int, limit_bytes: int, message: str | None = None) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            message or f"Artifact too large: {size_bytes} bytes exceeds {limit_bytes} byte limit"
        )
