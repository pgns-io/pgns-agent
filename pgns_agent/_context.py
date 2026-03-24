# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Context variable for tracking the active task inside a handler."""

from __future__ import annotations

__all__ = ["get_current_task"]

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pgns_agent._task import Task

current_task: ContextVar[Task | None] = ContextVar("current_task", default=None)
"""Holds the :class:`Task` being processed by the current handler, or ``None``."""


def get_current_task() -> Task | None:
    """Return the active task, or ``None`` if called outside a handler."""
    return current_task.get()
