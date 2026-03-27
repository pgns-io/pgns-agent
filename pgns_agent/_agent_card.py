# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Agent Card model and auto-generation for A2A discovery."""

from __future__ import annotations

__all__ = [
    "AgentCapabilities",
    "AgentCard",
    "AgentCardAuthentication",
    "AgentCardProvider",
    "AgentCardSecurityScheme",
    "AgentCardSkill",
]

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCardSkill:
    """A single skill advertised in an Agent Card.

    Attributes:
        id: Machine-readable identifier (matches the ``@on_task`` name).
        name: Human-readable display name.  Defaults to *id* if not provided.
        description: Human-readable description of what the skill does.
            Required by A2A v1.0; serialized as ``""`` when *None*.
        tags: Freeform tags for categorisation (A2A v1.0).
    """

    id: str
    name: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", self.id)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description or "",
            "tags": list(self.tags),
        }
        if self.input_schema is not None:
            d["inputSchema"] = self.input_schema
        if self.output_schema is not None:
            d["outputSchema"] = self.output_schema
        return d


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCardAuthentication:
    """Authentication requirements advertised in the Agent Card.

    .. deprecated::
        Use :class:`AgentCardSecurityScheme` and the ``security_schemes``
        parameter on :class:`AgentServer` instead.

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
class AgentCardSecurityScheme:
    """A2A v1.0 security scheme entry, replacing :class:`AgentCardAuthentication`.

    Attributes:
        scheme: The authentication scheme identifier (e.g. ``"bearer"``).
        credentials: Optional URL or instruction for obtaining credentials.
    """

    scheme: str
    credentials: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"scheme": self.scheme}
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
class AgentCapabilities:
    """A2A v1.0 agent capabilities.

    Attributes:
        streaming: Whether the agent supports streaming responses.
        push_notifications: Whether the agent supports push notifications.
        extended_agent_card: Whether the agent supports extended agent cards.
    """

    streaming: bool = False
    push_notifications: bool = False
    extended_agent_card: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "extendedAgentCard": self.extended_agent_card,
        }


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
            Serialized under ``supportedInterfaces[0].url`` in the wire format.
        version: Semantic version string supplied by the developer.
        skills: Skills derived from ``@on_task`` registrations.
        capabilities: A2A v1.0 capabilities (defaults all false).
        default_input_modes: MIME types accepted as input (A2A v1.0).
        default_output_modes: MIME types produced as output (A2A v1.0).
        security_schemes: A2A v1.0 security schemes.
        authentication: Deprecated auth requirements (use *security_schemes*).
        provider: Optional provider metadata.
    """

    name: str
    description: str
    url: str
    version: str
    skills: tuple[AgentCardSkill, ...] = ()
    capabilities: AgentCapabilities = dataclasses.field(default_factory=AgentCapabilities)
    default_input_modes: tuple[str, ...] = ("application/json",)
    default_output_modes: tuple[str, ...] = ("application/json",)
    security_schemes: tuple[AgentCardSecurityScheme, ...] = ()
    authentication: AgentCardAuthentication | None = None
    provider: AgentCardProvider | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict matching the A2A v1.0 Agent Card schema.

        A2A v1.0 required fields (capabilities, defaultInputModes, defaultOutputModes)
        are always included.  Optional fields (securitySchemes, authentication, provider)
        are omitted when empty/None to keep the card minimal.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "supportedInterfaces": [
                {
                    "url": self.url,
                    "protocolBinding": "HTTP+JSON",
                    "protocolVersion": "1.0",
                }
            ],
            "version": self.version,
            "skills": [s.to_dict() for s in self.skills],
            "capabilities": self.capabilities.to_dict(),
            "defaultInputModes": list(self.default_input_modes),
            "defaultOutputModes": list(self.default_output_modes),
        }
        if self.security_schemes:
            d["securitySchemes"] = [s.to_dict() for s in self.security_schemes]
        if self.authentication is not None:
            d["authentication"] = self.authentication.to_dict()
        if self.provider is not None:
            d["provider"] = self.provider.to_dict()
        return d
