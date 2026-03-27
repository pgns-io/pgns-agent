# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""pgns-agent — wrap any agent function in a production-ready A2A server."""

from pgns_agent._adapter import Adapter
from pgns_agent._agent_card import (
    AgentCapabilities,
    AgentCard,
    AgentCardAuthentication,
    AgentCardProvider,
    AgentCardSecurityScheme,
    AgentCardSkill,
)
from pgns_agent._artifact import ArtifactMediaType, ArtifactRef, ArtifactStore
from pgns_agent._context import get_current_task
from pgns_agent._errors import (
    ArtifactAccessError,
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactTooLargeError,
)
from pgns_agent._server import DEFAULT_HANDLER_NAME, AgentServer
from pgns_agent._state import TaskState
from pgns_agent._task import Task, TaskMetadata, TaskStatus
from pgns_agent._version import __version__

__all__ = [
    "Adapter",
    "AgentCapabilities",
    "AgentCard",
    "AgentCardAuthentication",
    "AgentCardProvider",
    "AgentCardSecurityScheme",
    "AgentCardSkill",
    "AgentServer",
    "ArtifactAccessError",
    "ArtifactError",
    "ArtifactMediaType",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactStore",
    "ArtifactTooLargeError",
    "DEFAULT_HANDLER_NAME",
    "Task",
    "TaskMetadata",
    "TaskState",
    "TaskStatus",
    "get_current_task",
    "__version__",
]
