# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Agent-local task state tracking (ephemeral in v1)."""

from __future__ import annotations

__all__ = ["TaskState"]

import dataclasses
import datetime
from typing import Any

from pgns_agent._task import TaskStatus

_VALID_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.SUBMITTED: frozenset({TaskStatus.WORKING}),
    TaskStatus.WORKING: frozenset(
        {TaskStatus.WORKING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.INPUT_REQUIRED}
    ),
    TaskStatus.INPUT_REQUIRED: frozenset({TaskStatus.WORKING, TaskStatus.FAILED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
}

_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED})


@dataclasses.dataclass
class TaskState:
    """Mutable state of a task within the agent's process.

    Task state is agent-local and does not survive restarts (ephemeral v1).
    The pigeon stream is the authoritative history — this is a local cache
    of the current status for inspection and debugging.

    Attributes:
        task_id: Unique task identifier.
        status: Current lifecycle status.
        created_at: When the task was received.
        updated_at: When the status last changed.
        message: Optional human-readable message from the last transition.
        artifact: Optional artifact from the last transition.
    """

    task_id: str
    status: TaskStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    message: str | None = None
    artifact: Any = None

    @property
    def is_terminal(self) -> bool:
        """Whether the task is in a terminal state (completed or failed)."""
        return self.status in _TERMINAL_STATUSES

    def transition(
        self,
        new_status: TaskStatus,
        *,
        message: str | None = None,
        artifact: Any = None,
    ) -> None:
        """Transition to *new_status*, updating timestamps and optional fields.

        Raises:
            ValueError: If the transition is not permitted by the A2A state machine.
        """
        allowed = _VALID_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {self.status.value!r} → {new_status.value!r}. "
                f"Allowed targets: {sorted(s.value for s in allowed)}."
            )
        self.status = new_status
        self.updated_at = datetime.datetime.now(datetime.UTC)
        self.message = message
        self.artifact = artifact
