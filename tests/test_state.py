# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TaskState dataclass."""

from __future__ import annotations

import datetime

from pgns_agent import TaskState, TaskStatus


class TestTaskState:
    def test_initial_state(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-1",
            status=TaskStatus.SUBMITTED,
            created_at=now,
            updated_at=now,
        )
        assert state.task_id == "t-1"
        assert state.status is TaskStatus.SUBMITTED
        assert state.created_at == now
        assert state.updated_at == now
        assert state.message is None
        assert state.artifact is None

    def test_transition_updates_status(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-2",
            status=TaskStatus.SUBMITTED,
            created_at=now,
            updated_at=now,
        )
        state.transition(TaskStatus.WORKING)
        assert state.status is TaskStatus.WORKING
        assert state.updated_at > now

    def test_transition_updates_message_and_artifact(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-3",
            status=TaskStatus.WORKING,
            created_at=now,
            updated_at=now,
        )
        state.transition(TaskStatus.COMPLETED, message="done", artifact={"key": "val"})
        assert state.status is TaskStatus.COMPLETED
        assert state.message == "done"
        assert state.artifact == {"key": "val"}

    def test_transition_clears_previous_message(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-4",
            status=TaskStatus.WORKING,
            created_at=now,
            updated_at=now,
            message="in progress",
        )
        state.transition(TaskStatus.COMPLETED)
        assert state.message is None
        assert state.artifact is None

    def test_created_at_unchanged_after_transition(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-5",
            status=TaskStatus.SUBMITTED,
            created_at=now,
            updated_at=now,
        )
        state.transition(TaskStatus.WORKING)
        state.transition(TaskStatus.COMPLETED)
        assert state.created_at == now

    def test_input_required_to_working(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-6",
            status=TaskStatus.WORKING,
            created_at=now,
            updated_at=now,
        )
        state.transition(TaskStatus.INPUT_REQUIRED, message="need input")
        assert state.status is TaskStatus.INPUT_REQUIRED
        assert state.message == "need input"
        state.transition(TaskStatus.WORKING)
        assert state.status is TaskStatus.WORKING

    def test_input_required_to_failed(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        state = TaskState(
            task_id="t-7",
            status=TaskStatus.WORKING,
            created_at=now,
            updated_at=now,
        )
        state.transition(TaskStatus.INPUT_REQUIRED)
        state.transition(TaskStatus.FAILED)
        assert state.status is TaskStatus.FAILED
