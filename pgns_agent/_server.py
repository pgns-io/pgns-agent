# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""AgentServer — core entry point for the pgns-agent library."""

from __future__ import annotations

__all__ = ["AgentServer"]

import asyncio
import dataclasses
import datetime
import functools
import inspect
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, overload

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from pgns_agent._adapter import Adapter
from pgns_agent._agent_card import (
    AgentCard,
    AgentCardAuthentication,
    AgentCardProvider,
    AgentCardSkill,
)
from pgns_agent._context import current_task, get_current_task
from pgns_agent._state import _TERMINAL_STATUSES, TaskState
from pgns_agent._task import Task, TaskMetadata, TaskStatus, _TaskControl

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.types import ASGIApp

    from pgns.async_client import AsyncPigeonsClient
    from pgns.models import AgentCard as SdkAgentCard
    from pgns.models import Roost, SendResponse
    from pgns.webhook import Webhook
    from pgns_agent.testing import TestClient

logger = logging.getLogger("pgns_agent")

TaskHandler = Callable[[Task], Awaitable[Any]]
"""Async function that receives a :class:`Task` and returns a result."""

DEFAULT_PGNS_URL = "https://api.pgns.io"
DEFAULT_HANDLER_NAME = "default"
_DEFAULT_CARD_URL = "http://localhost"

_FALLBACK_VERSION = "0.1.0"

_CID_RE = re.compile(r"[\x21-\x7E]{1,128}")

# Maximum time (seconds) a handler can wait for caller input before timing out.
_INPUT_TIMEOUT: float = 300.0


def _safe_correlation_header(cid: str | None) -> dict[str, str] | None:
    """Return a ``X-Pgns-CorrelationId`` header dict if *cid* is valid, else ``None``."""
    if cid is None:
        return None
    if _CID_RE.fullmatch(cid):
        return {"X-Pgns-CorrelationId": cid}
    logger.warning(
        "Dropping invalid correlation_id (length=%d): not printable ASCII or too long",
        len(cid),
    )
    return None


def _agent_version() -> str:
    """Return the installed pgns-agent package version, or a fallback."""
    try:
        from importlib.metadata import version

        return version("pgns-agent")
    except Exception:
        return _FALLBACK_VERSION


class AgentServer:
    """Wraps agent handler functions in a production-ready A2A server.

    Example::

        agent = AgentServer(
            name="my-summarizer",
            description="Summarizes documents on request",
            pgns_key=os.environ.get("PGNS_API_KEY"),
        )

        @agent.on_task
        async def handle(task):
            return {"summary": await summarize(task.input)}

    When *pgns_key* is provided the server initialises an
    :class:`~pgns.sdk.async_client.AsyncPigeonsClient` for communicating with
    the pgns relay (roost provisioning, pigeon publishing, etc.).  When omitted
    the server runs in **local dev mode** — no network calls, console output
    only.
    """

    def __init__(
        self,
        name: str,
        description: str,
        *,
        version: str = "0.0.0",
        authentication: AgentCardAuthentication | None = None,
        provider: AgentCardProvider | None = None,
        pgns_key: str | None = None,
        pgns_url: str = DEFAULT_PGNS_URL,
    ) -> None:
        self._name = name
        self._description = description
        self._version = version
        self._authentication = authentication
        self._provider = provider
        self._pgns_url = pgns_url
        self._handlers: dict[str, TaskHandler] = {}
        self._tasks: dict[str, TaskState] = {}
        self._pending_input: dict[str, asyncio.Future[Any]] = {}
        self._app: Starlette | None = None
        self._agent_card: SdkAgentCard | None = None
        self._roost: Roost | None = None
        self._webhook: Webhook | None = None
        self._provisioned = False
        self._provision_lock = asyncio.Lock()
        self._task_subscribers: dict[str, list[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Late import so the SDK is only required when a key is supplied.
        if pgns_key is not None:
            try:
                from pgns.async_client import AsyncPigeonsClient
            except ImportError as exc:
                raise RuntimeError(
                    "pgns-agent requires the 'pgns' package when pgns_key is provided. "
                    "Install it with: pip install pgns-agent[pgns]"
                ) from exc

            self._client: AsyncPigeonsClient | None = AsyncPigeonsClient(pgns_url, api_key=pgns_key)
            logger.debug("Initialised pgns SDK client for %s", pgns_url)
        else:
            self._client = None
            logger.debug("Running in local dev mode (no pgns_key)")

    # -- Public properties ----------------------------------------------------

    @property
    def name(self) -> str:
        """Agent name used in the Agent Card and logging."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable description of what this agent does."""
        return self._description

    @property
    def version(self) -> str:
        """Semantic version string for the Agent Card."""
        return self._version

    @property
    def pgns_url(self) -> str:
        """Base URL of the pgns API."""
        return self._pgns_url

    @property
    def handlers(self) -> dict[str, TaskHandler]:
        """Snapshot of registered task handlers keyed by skill name."""
        return dict(self._handlers)

    @property
    def client(self) -> AsyncPigeonsClient | None:
        """The underlying :class:`AsyncPigeonsClient`, or ``None`` in local dev mode."""
        return self._client

    @property
    def agent_card(self) -> SdkAgentCard | None:
        """The provisioned :class:`AgentCard`, or ``None`` before provisioning."""
        return self._agent_card

    @property
    def roost(self) -> Roost | None:
        """The provisioned :class:`Roost`, or ``None`` before provisioning."""
        return self._roost

    @property
    def provisioned(self) -> bool:
        """Whether :meth:`provision` has completed successfully."""
        return self._provisioned

    @property
    def tasks(self) -> dict[str, TaskState]:
        """Snapshot of in-flight task states keyed by task ID.

        Task state is agent-local and ephemeral (v1) — it does not survive
        process restarts.  The pigeon stream is the authoritative history.
        """
        return {k: dataclasses.replace(v) for k, v in self._tasks.items()}

    # -- Lifecycle ------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client, releasing connection resources."""
        if self._client is not None:
            await self._client._http.aclose()

    async def __aenter__(self) -> AgentServer:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- Agent Card -----------------------------------------------------------

    def build_agent_card(self, url: str = _DEFAULT_CARD_URL) -> AgentCard:
        """Build an :class:`AgentCard` from the server's current state.

        Skills are derived from registered ``@on_task`` handlers.  The
        *default* handler (bare ``@on_task``) is **not** listed as a skill —
        it serves as a catch-all rather than an explicitly advertised
        capability.

        Args:
            url: The base URL where this agent is reachable.  When running
                via ``listen()`` this will be filled in automatically from the
                host/port.
        """
        skills: list[AgentCardSkill] = []
        for handler_name in self._handlers:
            if handler_name == DEFAULT_HANDLER_NAME:
                continue
            skills.append(AgentCardSkill(id=handler_name))

        return AgentCard(
            name=self._name,
            description=self._description,
            url=url,
            version=self._version,
            skills=tuple(skills),
            authentication=self._authentication,
            provider=self._provider,
        )

    def agent_card_route(self, url: str = _DEFAULT_CARD_URL) -> Route:
        """Return a Starlette :class:`Route` that serves the Agent Card.

        The route is ``/.well-known/agent.json`` and responds to ``GET``
        requests with a JSON representation of :meth:`build_agent_card`.
        The card is built lazily on each request so that handlers registered
        after this call are included.
        """

        async def _agent_card_endpoint(request: Request) -> JSONResponse:
            return JSONResponse(
                content=self.build_agent_card(url).to_dict(),
                headers={"Cache-Control": "public, max-age=3600"},
            )

        return Route("/.well-known/agent.json", endpoint=_agent_card_endpoint, methods=["GET"])

    # -- Outbound messaging ---------------------------------------------------

    async def send(
        self,
        target: str,
        payload: Any,
        *,
        event_type: str = "agent.task",
    ) -> SendResponse:
        """Send a message to another agent's roost (inbox).

        Fetches the target roost's signing secret, signs the payload, and
        delivers it via the pgns relay.  If a task is currently being handled,
        the task's ``correlation_id`` is automatically propagated via the
        ``X-Pgns-CorrelationId`` header.

        .. note::

            Each call issues a ``get_roost()`` API request to fetch the
            target's signing secret.  The secret is stable between calls
            (changes only on explicit rotation), so callers in hot loops
            should be aware of the per-call network round-trip.

        Args:
            target: Roost ID of the receiving agent's inbox.
            payload: JSON-serializable data to deliver.
            event_type: Event type header (default ``"agent.task"``).

        Returns:
            A :class:`~pgns.sdk.models.SendResponse` with the pigeon ID,
            status, and destination count.

        Raises:
            RuntimeError: If the server is running in local dev mode (no
                ``pgns_key``) or if the target roost has no signing secret.
        """
        if self._client is None:
            raise RuntimeError(
                "Cannot send messages in local dev mode. "
                "Provide a pgns_key when constructing AgentServer."
            )

        roost = await self._client.get_roost(target)
        if not roost.secret:
            raise RuntimeError(
                f"Target roost {target!r} has no signing secret configured. "
                "A signing secret is required for agent-to-agent messaging."
            )

        # Propagate correlation ID from the current task context, if present.
        task = get_current_task()
        cid = task.metadata.correlation_id if task is not None else None
        extra_headers = _safe_correlation_header(cid)

        logger.debug(
            "Sending to roost %r (event_type=%r, correlation_id=%s)",
            target,
            event_type,
            extra_headers.get("X-Pgns-CorrelationId") if extra_headers else None,
        )

        return await self._client.send(
            target,
            event_type=event_type,
            payload=payload,
            signing_secret=roost.secret,
            extra_headers=extra_headers,
        )

    # -- @on_task decorator ---------------------------------------------------

    @overload
    def on_task(self, fn: TaskHandler, /) -> TaskHandler: ...

    @overload
    def on_task(self, fn: str, /) -> Callable[[TaskHandler], TaskHandler]: ...

    def on_task(
        self,
        fn: TaskHandler | str,
        /,
    ) -> TaskHandler | Callable[[TaskHandler], TaskHandler]:
        """Register an async handler for inbound tasks.

        Can be used as a bare decorator (registers the *default* handler) or
        as a decorator factory with a skill name for multi-skill agents::

            # Default (single-skill) handler
            @agent.on_task
            async def handle(task): ...

            # Named handler for multi-skill agents
            @agent.on_task("summarize")
            async def summarize(task): ...
        """
        if callable(fn):
            self._register_handler(DEFAULT_HANDLER_NAME, fn)
            return fn

        name = fn
        if name == DEFAULT_HANDLER_NAME:
            raise ValueError(
                f"The name {DEFAULT_HANDLER_NAME!r} is reserved; "
                "use the bare @agent.on_task decorator to register the default handler."
            )

        def _decorator(handler: TaskHandler) -> TaskHandler:
            self._register_handler(name, handler)
            return handler

        return _decorator

    # -- Testing --------------------------------------------------------------

    def test_client(self, *, raise_server_exceptions: bool = False) -> TestClient:
        """Return a :class:`~pgns_agent.testing.TestClient` for unit testing.

        Convenience shortcut so tests can write::

            client = agent.test_client()
            resp = client.send_task({"text": "hello"})
            assert resp.status == "completed"
        """
        from pgns_agent.testing import TestClient

        return TestClient(self, raise_server_exceptions=raise_server_exceptions)

    # -- Adapter wiring -------------------------------------------------------

    def use(self, adapter: Adapter, *, skill: str | None = None) -> None:
        """Wire a framework adapter into this server.

        The adapter's :meth:`~pgns_agent.Adapter.handle` method is registered
        as a task handler.  When a matching task arrives, the adapter receives
        ``task.input`` and its return value becomes the task result.

        For streaming adapters (those that yield an ``collections.abc.AsyncIterator``),
        the library collects yielded chunks and returns the **last** chunk as
        the final result.  (Full SSE streaming is wired in the A2A Task
        Lifecycle milestone.)

        Args:
            adapter: An :class:`~pgns_agent.Adapter` subclass instance.
            skill: Optional skill name.  When omitted the adapter is
                registered as the default handler.  Pass a name to register
                it as a named skill (e.g. ``agent.use(adapter, skill="summarize")``).

        Raises:
            TypeError: If *adapter* is not an :class:`Adapter` instance.
            ValueError: If the skill slot is already occupied (same rules as
                :meth:`on_task`).
        """
        if not isinstance(adapter, Adapter):
            raise TypeError(
                f"Expected an Adapter instance, got {type(adapter).__name__}. "
                "Subclass pgns_agent.Adapter and implement the handle() method."
            )

        if skill is not None and skill == DEFAULT_HANDLER_NAME:
            raise ValueError(
                f"The name {DEFAULT_HANDLER_NAME!r} is reserved; "
                "omit the skill argument to register as the default handler."
            )

        if not inspect.iscoroutinefunction(adapter.handle) and not inspect.isasyncgenfunction(
            adapter.handle
        ):
            raise TypeError(
                f"{type(adapter).__name__}.handle() must be async. "
                "Define it with 'async def handle(self, task_input): ...'."
            )

        handler_name = skill if skill is not None else DEFAULT_HANDLER_NAME

        async def _adapter_handler(task: Task) -> Any:
            task_input = task.input if task.input is not None else {}
            result_or_iter: Any = adapter.handle(task_input)
            # Async generator → iterate; regular coroutine → await.
            if inspect.isasyncgen(result_or_iter):
                last: dict[str, Any] | None = None
                async for chunk in result_or_iter:
                    last = chunk
                return last
            return await result_or_iter

        self._register_handler(handler_name, _adapter_handler)
        logger.debug(
            "Wired adapter %s as handler %r",
            type(adapter).__name__,
            handler_name,
        )

    # -- Entry points ---------------------------------------------------------

    def app(self) -> Starlette:
        """Return the ASGI application for mounting into existing frameworks.

        The returned :class:`~starlette.applications.Starlette` instance
        exposes:

        * ``GET /.well-known/agent.json`` — A2A Agent Card discovery
        * ``POST /`` — task dispatch to registered handlers
        * ``GET /tasks/{task_id}/events`` — SSE stream for task status updates

        Handlers registered via :meth:`on_task` after the first call are still
        dispatched correctly because route closures read ``self._handlers`` at
        request time.
        """
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def handler(self) -> ASGIApp:
        """Return the raw ASGI callable for serverless deployment.

        The returned object implements the ASGI interface and can be wrapped
        with adapters such as Mangum (AWS Lambda) or similar.
        """
        return self.app()

    def listen(self, port: int = 8000, *, host: str = "127.0.0.1") -> None:
        """Start a standalone HTTP server (blocking).

        Requires ``uvicorn`` to be installed::

            pip install uvicorn
        """
        try:
            import uvicorn  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "agent.listen() requires uvicorn. Install it with: pip install uvicorn"
            ) from exc

        logger.info("Starting %s on %s:%d", self._name, host, port)
        uvicorn.run(self.app(), host=host, port=port)

    # -- Provisioning ---------------------------------------------------------

    async def provision(self) -> None:
        """Auto-provision an AgentCard and Roost on the pgns relay.

        In **local dev mode** (no *pgns_key*) this is a no-op — the server
        simply logs that it is running locally.

        When a *pgns_key* is present:

        1. List existing agent cards and look for one matching :attr:`name`.
           Create a new card if none is found.
        2. List existing roosts and look for one whose ``agent_card_id`` matches
           the card from step 1.  Create a new roost if none is found.
        3. Store the roost secret for HMAC verification of incoming webhooks.

        This method is idempotent — calling it multiple times is safe.
        """
        if self._provisioned:
            return

        async with self._provision_lock:
            # Double-check after acquiring the lock.
            if self._provisioned:
                return

            if self._client is None:
                logger.info(
                    "Local dev mode — skipping provisioning for %r (no pgns_key)", self._name
                )
                self._provisioned = True
                return

            from pgns.models import CreateAgentCard, CreateRoost

            client = self._client

            # -- Step 1: ensure AgentCard exists ------------------------------
            agent_cards = await client.list_agents()
            agent_card = next((c for c in agent_cards if c.name == self._name), None)

            if agent_card is None:
                agent_card = await client.create_agent(
                    CreateAgentCard(
                        name=self._name,
                        url="",  # placeholder until ASGI routes are wired
                        description=self._description,
                        version=_agent_version(),
                    )
                )
                logger.info("Created agent card %s (%s)", agent_card.name, agent_card.id)
            else:
                logger.info("Found existing agent card %s (%s)", agent_card.name, agent_card.id)

            self._agent_card = agent_card

            # -- Step 2: ensure Roost exists for this card --------------------
            roosts = await client.list_roosts()
            roost = next((r for r in roosts if r.agent_card_id == agent_card.id), None)

            if roost is None:
                roost = await client.create_roost(
                    CreateRoost(
                        name=f"{self._name}-inbox",
                        description=f"Auto-provisioned inbox for agent {self._name!r}",
                        agent_card_id=agent_card.id,
                    )
                )
                logger.info("Created roost %s (%s)", roost.name, roost.id)
            else:
                logger.info("Found existing roost %s (%s)", roost.name, roost.id)

            self._roost = roost

            # -- Step 3: initialise webhook verifier from roost secret --------
            if roost.secret:
                from pgns.webhook import Webhook

                self._webhook = Webhook(roost.secret)
                logger.debug("HMAC verification enabled for roost %s", roost.id)
            else:
                logger.warning(
                    "Roost %s has no secret — webhook verification is disabled", roost.id
                )

            self._provisioned = True
            logger.info("Provisioning complete for agent %r", self._name)

    # -- Webhook verification -------------------------------------------------

    def verify_webhook(self, body: str | bytes, headers: dict[str, str]) -> Any:
        """Verify an incoming webhook signature and return the parsed payload.

        In **local dev mode** the body is parsed as JSON without any signature
        check — zero friction for development.

        Raises :class:`~pgns.sdk.errors.WebhookVerificationError` when a
        signature is present but invalid (production mode only).
        """

        if self._client is None:
            # Local dev mode: no verification, just parse
            body_str = body.decode() if isinstance(body, bytes) else body
            payload = json.loads(body_str)
            logger.debug("Local dev mode — accepted webhook without verification")
            return payload

        if self._webhook is None:
            from pgns.errors import WebhookVerificationError

            raise WebhookVerificationError(
                "Webhook verification is not configured — roost has no secret. "
                "Re-provision or set a secret on the roost.",
                code="webhook_not_configured",
            )

        return self._webhook.verify(body, headers)

    # -- Internals ------------------------------------------------------------

    async def _publish_status_update(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        correlation_id: str | None = None,
        message: str | None = None,
        artifact: Any = None,
    ) -> None:
        """Publish a task-status pigeon to the agent's own roost.

        In **local dev mode** the transition is logged to the console.
        In production mode the pigeon is published via the SDK client.
        Publishing failures are logged but never propagated — they must not
        affect the task response.
        """
        payload: dict[str, Any] = {
            "task_id": task_id,
            "status": status.value,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        if message is not None:
            payload["message"] = message
        if artifact is not None:
            payload["artifact"] = artifact

        if self._client is None:
            logger.info("Task %s → %s", task_id, status.value)
            return

        if self._roost is None or not self._roost.secret:
            logger.warning(
                "Skipping status pigeon for task %s — no provisioned roost or secret",
                task_id,
            )
            return

        extra_headers = _safe_correlation_header(correlation_id)

        try:
            await self._client.send(
                self._roost.id,
                event_type="agent.task.status",
                payload=payload,
                signing_secret=self._roost.secret,
                extra_headers=extra_headers,
            )
        except Exception:
            logger.warning(
                "Failed to publish status pigeon for task %s → %s",
                task_id,
                status.value,
                exc_info=True,
            )

    async def _handle_update_status(
        self,
        task_id: str,
        correlation_id: str | None,
        message: str,
    ) -> None:
        """Publish a progress-update pigeon (called from ``task.update_status``)."""
        state = self._tasks.get(task_id)
        if state is None:
            logger.warning("update_status called for evicted task %s — ignoring.", task_id)
            return
        state.transition(TaskStatus.WORKING, message=message)
        asyncio.create_task(
            self._publish_status_update(
                task_id,
                TaskStatus.WORKING,
                correlation_id=correlation_id,
                message=message,
            )
        )

    async def _handle_request_input(
        self,
        task_id: str,
        correlation_id: str | None,
        prompt: str,
    ) -> Any:
        """Transition to input-required, suspend, and return caller's input."""
        state = self._tasks.get(task_id)
        if state is None:
            raise RuntimeError(f"Task {task_id!r} not found in agent state.")

        # -- input-required --------------------------------------------------
        state.transition(TaskStatus.INPUT_REQUIRED, message=prompt)
        await self._publish_status_update(
            task_id,
            TaskStatus.INPUT_REQUIRED,
            correlation_id=correlation_id,
            message=prompt,
        )

        # Create and register a Future that the resume request will resolve.
        if task_id in self._pending_input:
            raise RuntimeError(
                f"Task {task_id!r} already has a pending request_input — "
                "concurrent request_input is not supported."
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending_input[task_id] = future

        try:
            result = await asyncio.wait_for(future, timeout=_INPUT_TIMEOUT)
        except TimeoutError:
            state.transition(TaskStatus.FAILED, message="Timed out waiting for input.")
            await self._publish_status_update(
                task_id,
                TaskStatus.FAILED,
                correlation_id=correlation_id,
                message="Timed out waiting for input.",
            )
            raise
        finally:
            fut = self._pending_input.pop(task_id, None)
            if fut is not None and not fut.done():
                fut.cancel()

        # -- back to working -------------------------------------------------
        state.transition(TaskStatus.WORKING)
        await self._publish_status_update(
            task_id,
            TaskStatus.WORKING,
            correlation_id=correlation_id,
        )

        return result

    def _register_handler(self, name: str, handler: TaskHandler) -> None:
        if not name or not name.strip():
            raise ValueError("Handler name must be a non-empty string.")
        if name in self._handlers:
            raise ValueError(
                f"A task handler is already registered for {name!r}. "
                "Each skill name (or the default slot) can only have one handler."
            )
        self._handlers[name] = handler
        logger.debug("Registered task handler %r → %s", name, handler.__qualname__)

    _MAX_SUBSCRIBERS_PER_TASK = 32

    def _subscribe_task(self, task_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        """Subscribe to SSE events for a task. Returns a queue that receives events."""
        subs = self._task_subscribers.setdefault(task_id, [])
        if len(subs) >= self._MAX_SUBSCRIBERS_PER_TASK:
            raise ValueError(
                f"Subscriber limit ({self._MAX_SUBSCRIBERS_PER_TASK}) reached for task {task_id!r}."
            )
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        subs.append(queue)
        return queue

    def _unsubscribe_task(self, task_id: str, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        """Remove a subscriber queue for a task."""
        subs = self._task_subscribers.get(task_id)
        if subs is not None:
            try:
                subs.remove(queue)
            except ValueError:
                pass
            if not subs:
                del self._task_subscribers[task_id]

    def _broadcast_task_event(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = None,
    ) -> None:
        """Broadcast a task state change to all SSE subscribers."""
        subs = self._task_subscribers.get(task_id)
        if not subs:
            return

        event: dict[str, Any] = {
            "task_id": task_id,
            "status": status.value,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        if result is not None:
            event["result"] = result

        for queue in subs:
            queue.put_nowait(event)

        # Terminal state: send sentinel and clean up subscriber list.
        if status in _TERMINAL_STATUSES:
            for queue in subs:
                queue.put_nowait(None)
            del self._task_subscribers[task_id]

    async def _run_async_handler(
        self,
        task: Task,
        handler: TaskHandler,
        state: TaskState,
        cid: str | None,
        skill: str,
    ) -> None:
        """Execute a task handler in the background (async mode)."""
        # -- working ---------------------------------------------------------
        state.transition(TaskStatus.WORKING)
        self._broadcast_task_event(task.id, TaskStatus.WORKING)
        asyncio.create_task(
            self._publish_status_update(
                task.id,
                TaskStatus.WORKING,
                correlation_id=cid,
            )
        )

        token = current_task.set(task)
        try:
            result = await handler(task)
        except Exception:
            logger.exception("Handler %r raised an exception (async)", skill)
            state.transition(TaskStatus.FAILED)
            self._broadcast_task_event(task.id, TaskStatus.FAILED)
            await self._publish_status_update(
                task.id,
                TaskStatus.FAILED,
                correlation_id=cid,
            )
            self._tasks.pop(task.id, None)
            return
        except BaseException:
            state.transition(TaskStatus.FAILED)
            self._broadcast_task_event(task.id, TaskStatus.FAILED)
            await self._publish_status_update(
                task.id,
                TaskStatus.FAILED,
                correlation_id=cid,
            )
            self._tasks.pop(task.id, None)
            raise
        finally:
            current_task.reset(token)

        # -- completed -------------------------------------------------------
        state.transition(TaskStatus.COMPLETED)
        self._broadcast_task_event(task.id, TaskStatus.COMPLETED, result=result)
        await self._publish_status_update(
            task.id,
            TaskStatus.COMPLETED,
            correlation_id=cid,
            artifact=result,
        )
        self._tasks.pop(task.id, None)

    def _build_app(self) -> Starlette:
        """Construct the Starlette ASGI application."""
        from starlette.applications import Starlette

        async def task_endpoint(request: Request) -> JSONResponse:
            """Accept a task payload and dispatch to the appropriate handler.

            Implements the A2A task state machine with automatic transitions:

            1. ``submitted`` — on receipt
            2. ``working`` — when handler starts executing
            3. ``completed`` / ``failed`` — on handler return / throw

            Each transition publishes a status pigeon to the agent's own roost.
            """
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "Request body must be a JSON object."},
                    status_code=400,
                )

            # When delivered by the pgns worker, the body is the raw pigeon
            # payload (no "id" field).  Auto-construct a Task envelope from
            # the delivery headers so agent destinations work out of the box.
            if "id" not in body:
                pigeon_id = request.headers.get("x-pigeon-id")
                if pigeon_id:
                    metadata: dict[str, str] = {}
                    cid_header = request.headers.get("x-correlation-id")
                    if cid_header:
                        metadata["correlation_id"] = cid_header
                    body = {"id": pigeon_id, "input": body, "metadata": metadata}
                else:
                    return JSONResponse(
                        {"error": "Request body must be a JSON object with an 'id' field."},
                        status_code=400,
                    )

            skill = body.get("skill") or DEFAULT_HANDLER_NAME
            matched_handler = self._handlers.get(skill)
            # Fall back to the default handler when a named skill isn't found.
            if matched_handler is None and skill != DEFAULT_HANDLER_NAME:
                matched_handler = self._handlers.get(DEFAULT_HANDLER_NAME)
            if matched_handler is None:
                return JSONResponse(
                    {"error": f"No handler registered for skill {skill!r}."},
                    status_code=404,
                )

            task_id = body["id"]
            if not isinstance(task_id, str):
                return JSONResponse({"error": '"id" must be a string.'}, status_code=400)

            metadata_raw = body.get("metadata")
            if not isinstance(metadata_raw, dict):
                metadata_raw = {}

            # Validate correlation_id early — before it reaches logs or payloads.
            cid: str | None = metadata_raw.get("correlation_id")
            if cid is not None and not _CID_RE.fullmatch(cid):
                logger.warning(
                    "Discarding invalid correlation_id from task %s (length=%d)",
                    task_id,
                    len(cid),
                )
                cid = None

            # If the task is waiting for input, resume it instead of rejecting.
            existing = self._tasks.get(task_id)
            if existing is not None and not existing.is_terminal:
                if existing.status is TaskStatus.INPUT_REQUIRED:
                    future = self._pending_input.pop(task_id, None)
                    if future is not None and not future.done():
                        existing.transition(TaskStatus.WORKING)
                        future.set_result(body.get("input"))
                        return JSONResponse({"id": task_id, "status": "input-received"})
                    return JSONResponse(
                        {
                            "error": (
                                f"Task {task_id!r} is in input-required state "
                                "but has no pending future."
                            )
                        },
                        status_code=500,
                    )
                return JSONResponse(
                    {"error": f"Task {task_id!r} is already in-flight."},
                    status_code=409,
                )

            # -- submitted -------------------------------------------------------
            now = datetime.datetime.now(datetime.UTC)
            state = TaskState(
                task_id=task_id,
                status=TaskStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
            )
            self._tasks[task_id] = state
            asyncio.create_task(
                self._publish_status_update(
                    task_id,
                    TaskStatus.SUBMITTED,
                    correlation_id=cid,
                )
            )

            # Build the Task object before the async/sync branch so both
            # paths can reference it.
            ctrl = _TaskControl(
                update_status=functools.partial(self._handle_update_status, task_id, cid),
                request_input=functools.partial(self._handle_request_input, task_id, cid),
            )
            task = Task(
                id=task_id,
                input=body.get("input"),
                metadata=TaskMetadata(
                    correlation_id=cid,
                    source_agent=metadata_raw.get("source_agent"),
                ),
                _ctrl=ctrl,
            )

            # -- async mode (Prefer: respond-async, RFC 7240) -------------------
            prefer = request.headers.get("prefer", "")
            prefer_tokens = {t.strip().split(";")[0].strip() for t in prefer.split(",")}
            if "respond-async" in prefer_tokens:
                bg = asyncio.create_task(
                    self._run_async_handler(task, matched_handler, state, cid, skill)
                )
                self._background_tasks.add(bg)
                bg.add_done_callback(self._background_tasks.discard)
                return JSONResponse(
                    {"id": task.id, "status": "submitted"},
                    status_code=202,
                    headers={"Preference-Applied": "respond-async"},
                )

            # -- sync mode (default) --------------------------------------------

            # -- working ---------------------------------------------------------
            state.transition(TaskStatus.WORKING)
            self._broadcast_task_event(task.id, TaskStatus.WORKING)
            asyncio.create_task(
                self._publish_status_update(
                    task_id,
                    TaskStatus.WORKING,
                    correlation_id=cid,
                )
            )

            token = current_task.set(task)
            try:
                result = await matched_handler(task)
            except Exception:
                logger.exception("Handler %r raised an exception", skill)
                # -- failed ------------------------------------------------------
                if not state.is_terminal:
                    state.transition(TaskStatus.FAILED)
                    self._broadcast_task_event(task.id, TaskStatus.FAILED)
                    await self._publish_status_update(
                        task.id,
                        TaskStatus.FAILED,
                        correlation_id=cid,
                    )
                pending = self._pending_input.pop(task.id, None)
                if pending is not None and not pending.done():
                    pending.cancel()
                self._tasks.pop(task.id, None)
                return JSONResponse(
                    {"id": task.id, "status": "failed", "error": "Internal handler error."},
                    status_code=500,
                )
            except BaseException:
                # CancelledError, KeyboardInterrupt, etc. — still transition to
                # FAILED so the state machine isn't left in WORKING, then re-raise.
                if not state.is_terminal:
                    state.transition(TaskStatus.FAILED)
                    self._broadcast_task_event(task.id, TaskStatus.FAILED)
                    await self._publish_status_update(
                        task.id,
                        TaskStatus.FAILED,
                        correlation_id=cid,
                    )
                pending = self._pending_input.pop(task.id, None)
                if pending is not None and not pending.done():
                    pending.cancel()
                self._tasks.pop(task.id, None)
                raise
            finally:
                current_task.reset(token)

            # -- completed -------------------------------------------------------
            state.transition(TaskStatus.COMPLETED)
            self._broadcast_task_event(task.id, TaskStatus.COMPLETED, result=result)
            await self._publish_status_update(
                task.id,
                TaskStatus.COMPLETED,
                correlation_id=cid,
                artifact=result,
            )
            # Evict terminal entry — the pigeon stream is the authoritative history.
            self._tasks.pop(task.id, None)
            return JSONResponse({"id": task.id, "status": "completed", "result": result})

        async def sse_endpoint(request: Request) -> JSONResponse | StreamingResponse:
            """Stream SSE events for a task's state transitions.

            .. warning::

                This endpoint performs no authentication or authorization.
                It MUST be deployed behind an authenticating reverse proxy.
                COMPLETED events may include handler results containing
                sensitive data.
            """
            task_id = request.path_params["task_id"]
            state = self._tasks.get(task_id)
            if state is None:
                return JSONResponse(
                    {"error": f"Task {task_id!r} not found."},
                    status_code=404,
                )

            try:
                queue = self._subscribe_task(task_id)
            except ValueError:
                return JSONResponse(
                    {"error": "Too many subscribers for this task."},
                    status_code=429,
                )
            # Snapshot before the generator's deferred execution to avoid races.
            snapshot_status = state.status
            snapshot_updated = state.updated_at

            async def event_generator() -> AsyncIterator[str]:
                try:
                    # Send current status as the initial event.
                    initial: dict[str, Any] = {
                        "task_id": task_id,
                        "status": snapshot_status.value,
                        "timestamp": snapshot_updated.isoformat(),
                    }
                    yield f"event: task.status\ndata: {json.dumps(initial)}\n\n"

                    if snapshot_status in _TERMINAL_STATUSES:
                        return

                    while True:
                        event = await queue.get()
                        if event is None:
                            return
                        yield f"event: task.status\ndata: {json.dumps(event)}\n\n"
                finally:
                    self._unsubscribe_task(task_id, queue)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        return Starlette(
            routes=[
                self.agent_card_route(),
                Route("/tasks/{task_id}/events", sse_endpoint, methods=["GET"]),
                Route("/", task_endpoint, methods=["POST"]),
            ],
        )
