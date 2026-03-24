# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Agent Card model and auto-generation for A2A discovery."""

from __future__ import annotations

__all__ = ["AgentCard", "AgentCardAuthentication", "AgentCardProvider", "AgentCardSkill"]

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCardSkill:
    """A single skill advertised in an Agent Card.

    Attributes:
        id: Machine-readable identifier (matches the ``@on_task`` name).
        name: Human-readable display name.  Defaults to *id* if not provided.
        description: Optional longer description of what the skill does.
    """

    id: str
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", self.id)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "name": self.name}
        if self.description is not None:
            d["description"] = self.description
        return d


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCardAuthentication:
    """Authentication requirements advertised in the Agent Card.

    Attributes:
        schemes: Supported auth schemes (e.g. ``["bearer"]``).
        credentials: Optional URL or instruction for obtaining credentials.
    """

    schemes: tuple[str, ...] = ()
    credentials: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schemes": list(self.schemes)}
        if self.credentials is not None:
            d["credentials"] = self.credentials
        return d


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCardProvider:
    """Provider metadata advertised in the Agent Card.

    Attributes:
        organization: The organization that operates this agent.
        url: URL for the provider's homepage or documentation.
    """

    organization: str
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"organization": self.organization}
        if self.url is not None:
            d["url"] = self.url
        return d


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCard:
    """A2A Agent Card served at ``/.well-known/agent.json``.

    See the `A2A specification <https://google.github.io/A2A/>`_ for the full
    schema.  This implementation covers the fields that ``pgns-agent``
    auto-generates plus fields the developer can supply explicitly.

    Attributes:
        name: Agent name (from :class:`AgentServer` constructor).
        description: Human-readable description.
        url: The agent's base URL (derived at runtime from host/port).
        version: Semantic version string supplied by the developer.
        skills: Skills derived from ``@on_task`` registrations.
        authentication: Optional auth requirements.
        provider: Optional provider metadata.
    """

    name: str
    description: str
    url: str
    version: str
    skills: tuple[AgentCardSkill, ...] = ()
    authentication: AgentCardAuthentication | None = None
    provider: AgentCardProvider | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict matching the A2A Agent Card schema."""
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "skills": [s.to_dict() for s in self.skills],
        }
        if self.authentication is not None:
            d["authentication"] = self.authentication.to_dict()
        if self.provider is not None:
            d["provider"] = self.provider.to_dict()
        return d
