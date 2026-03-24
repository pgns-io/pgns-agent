# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Adapter base class for framework integrations."""

from __future__ import annotations

__all__ = ["Adapter"]

import abc
from collections.abc import AsyncIterator
from typing import Any


class Adapter(abc.ABC):
    """Base class for framework adapters.

    An adapter wraps an existing agent framework's ``run()`` / ``invoke()``
    function into the pgns-agent task interface.  Subclasses implement
    :meth:`handle` to bridge between the framework's API and the pgns-agent
    contract.

    The :meth:`handle` method receives the task input dict and returns either:

    * A **dict** — synchronous, single-shot result.
    * An **AsyncIterator of dicts** — streaming; each yielded dict is sent as
      an SSE event to the caller (when the library's SSE transport is wired).

    Framework-specific metadata (token counts, tool calls, latency, etc.)
    should be included under a ``"metadata"`` key in the returned dict(s).
    The library passes it through to the task result without interpretation.

    Example — synchronous adapter::

        class LangChainAdapter(Adapter):
            def __init__(self, chain):
                self._chain = chain

            async def handle(self, task_input):
                result = await self._chain.ainvoke(task_input)
                return {"output": result}

    Example — streaming adapter::

        class StreamingAdapter(Adapter):
            def __init__(self, model):
                self._model = model

            async def handle(self, task_input):
                async for token in self._model.astream(task_input):
                    yield {"token": token}

    Wire an adapter into an :class:`~pgns_agent.AgentServer` with
    :meth:`~pgns_agent.AgentServer.use`::

        agent = AgentServer(name="my-agent", ...)
        agent.use(MyAdapter(chain))
        agent.listen(3000)
    """

    @abc.abstractmethod
    async def handle(
        self, task_input: dict[str, Any]
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Process a task input and return a result.

        Args:
            task_input: The deserialized JSON payload from the inbound task.

        Returns:
            A dict containing the result (optionally with a ``"metadata"``
            key for framework-specific metadata), **or** an async iterator
            of dicts for streaming responses.
        """
        ...
