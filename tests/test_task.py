# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for Task, TaskMetadata, and TaskStatus."""

from __future__ import annotations

import dataclasses

import pytest

from pgns_agent import Task, TaskMetadata, TaskStatus


class TestTaskMetadata:
    def test_defaults(self) -> None:
        meta = TaskMetadata()
        assert meta.correlation_id is None
        assert meta.source_agent is None

    def test_with_values(self) -> None:
        meta = TaskMetadata(correlation_id="corr-123", source_agent="agent-x")
        assert meta.correlation_id == "corr-123"
        assert meta.source_agent == "agent-x"

    def test_frozen(self) -> None:
        meta = TaskMetadata()
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.correlation_id = "new"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TaskMetadata(correlation_id="x")
        b = TaskMetadata(correlation_id="x")
        assert a == b


class TestTask:
    def test_minimal(self) -> None:
        task = Task(id="task-1", input={"text": "hello"})
        assert task.id == "task-1"
        assert task.input == {"text": "hello"}
        assert task.metadata == TaskMetadata()

    def test_with_metadata(self) -> None:
        meta = TaskMetadata(correlation_id="c-1", source_agent="bot")
        task = Task(id="task-2", input="raw string", metadata=meta)
        assert task.metadata.correlation_id == "c-1"
        assert task.metadata.source_agent == "bot"

    def test_frozen(self) -> None:
        task = Task(id="t", input=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            task.id = "other"  # type: ignore[misc]

    def test_input_accepts_any_type(self) -> None:
        assert Task(id="t", input=42).input == 42
        assert Task(id="t", input=[1, 2]).input == [1, 2]
        assert Task(id="t", input=None).input is None


class TestTaskControlMethods:
    @pytest.mark.asyncio
    async def test_update_status_without_ctrl_raises(self) -> None:
        task = Task(id="t", input=None)
        with pytest.raises(RuntimeError, match="inside a task handler"):
            await task.update_status("step 1")

    @pytest.mark.asyncio
    async def test_request_input_without_ctrl_raises(self) -> None:
        task = Task(id="t", input=None)
        with pytest.raises(RuntimeError, match="inside a task handler"):
            await task.request_input("what?")

    def test_ctrl_not_in_repr(self) -> None:
        task = Task(id="t", input=None)
        assert "_ctrl" not in repr(task)

    def test_ctrl_not_in_equality(self) -> None:
        a = Task(id="t", input=None)
        b = Task(id="t", input=None)
        assert a == b


class TestTaskStatus:
    def test_values(self) -> None:
        assert TaskStatus.SUBMITTED.value == "submitted"
        assert TaskStatus.WORKING.value == "working"
        assert TaskStatus.INPUT_REQUIRED.value == "input-required"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"

    def test_is_string(self) -> None:
        assert isinstance(TaskStatus.SUBMITTED, str)
        assert str(TaskStatus.COMPLETED) == "completed"

    def test_all_members(self) -> None:
        assert set(TaskStatus) == {
            TaskStatus.SUBMITTED,
            TaskStatus.WORKING,
            TaskStatus.INPUT_REQUIRED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        }
