# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""AgentServer — core entry point for the pgns-agent library."""

from __future__ import annotations

__all__ = ["AgentServer"]

import asyncio
import contextlib
import dataclasses
import datetime
import functools
import inspect
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypedDict, overload

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from pgns_agent._adapter import Adapter
from pgns_agent._agent_card import (
    AgentCapabilities,
    AgentCard,
    AgentCardAuthentication,
    AgentCardProvider,
    AgentCardSecurityScheme,
    AgentCardSkill,
)
from pgns_agent._artifact import ArtifactStore, _ArtifactEscrow
from pgns_agent._context import _current_trace, current_task, get_current_task
from pgns_agent._state import _TERMINAL_STATUSES, TaskState
from pgns_agent._task import Task, TaskMetadata, TaskStatus, _TaskControl
from pgns_agent._trace import _StageHandle

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


class _SkillMeta(TypedDict, total=False):
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


_CID_RE = re.compile(r"[\x21-\x7E]{1,128}")
_TASK_ID_RE = re.compile(r"[\x21-\x7E]{1,256}")

# Maximum time (seconds) a handler can wait for caller input before timing out.
_INPUT_TIMEOUT: float = 300.0

# How long to remember completed/failed task IDs to prevent post-completion
# reprocessing when delivery retries arrive after a task already finished.
_COMPLETED_TASK_TTL = datetime.timedelta(minutes=10)


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
        capabilities: AgentCapabilities | None = None,
        default_input_modes: tuple[str, ...] | None = None,
        default_output_modes: tuple[str, ...] | None = None,
        security_schemes: tuple[AgentCardSecurityScheme, ...] | None = None,
        authentication: AgentCardAuthentication | None = None,
        provider: AgentCardProvider | None = None,
        pgns_key: str | None = None,
        pgns_url: str = DEFAULT_PGNS_URL,
        tracing: bool = False,
    ) -> None:
        self._name = name
        self._tracing = tracing
        self._description = description
        self._version = version
        self._capabilities = capabilities or AgentCapabilities()
        self._default_input_modes = default_input_modes or ("application/json",)
        self._default_output_modes = default_output_modes or ("application/json",)

        # Deprecation shim: convert authentication= to security_schemes=
        if authentication is not None and security_schemes is None:
            import warnings

            warnings.warn(
                "authentication= is deprecated; use security_schemes= instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self._security_schemes = tuple(
                AgentCardSecurityScheme(scheme=s, credentials=authentication.credentials)
                for s in authentication.schemes
            )
            self._authentication = authentication
        else:
            self._security_schemes = security_schemes or ()
            self._authentication = None  # explicit schemes win
        self._provider = provider
        self._pgns_url = pgns_url
        self._handlers: dict[str, TaskHandler] = {}
        self._skill_meta: dict[str, _SkillMeta] = {}
        self._tasks: dict[str, TaskState] = {}
        self._completed_tasks: dict[str, datetime.datetime] = {}
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

        # In-memory artifact store for local dev mode.
        # Shared across all tasks so artifacts stored by one handler
        # can be retrieved by another (local-only).
        self._artifact_store: ArtifactStore = ArtifactStore()

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
    def tracing(self) -> bool:
        """Whether automatic trace capture is enabled."""
        return self._tracing

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
            meta = self._skill_meta.get(handler_name, {})
            skills.append(
                AgentCardSkill(
                    id=handler_name,
                    input_schema=meta.get("input_schema"),
                    output_schema=meta.get("output_schema"),
                )
            )

        return AgentCard(
            name=self._name,
            description=self._description,
            url=url,
            version=self._version,
            skills=tuple(skills),
            capabilities=self._capabilities,
            default_input_modes=self._default_input_modes,
            default_output_modes=self._default_output_modes,
            security_schemes=self._security_schemes,
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
        propagate_trace: bool = True,
    ) -> SendResponse:
        """Send a message to another agent's roost (inbox).

        Fetches the target roost's signing secret, signs the payload, and
        delivers it via the pgns relay.  If a task is currently being handled,
        the task's ``correlation_id`` is automatically propagated via the
        ``X-Pgns-CorrelationId`` header.

        When tracing is enabled and a trace context is active, the current
        stage's trace data is automatically injected into the outbound payload
        under the ``_trace`` key.  Downstream ``pgns-agent`` instances extract
        this in ``_dispatch_task`` to build a full pipeline trace.

        .. note::

            Each call issues a ``get_roost()`` API request to fetch the
            target's signing secret.  The secret is stable between calls
            (changes only on explicit rotation), so callers in hot loops
            should be aware of the per-call network round-trip.

        Args:
            target: Roost ID of the receiving agent's inbox.
            payload: JSON-serializable data to deliver.
            event_type: Event type header (default ``"agent.task"``).
            propagate_trace: Whether to inject trace data into the outbound
                payload.  Set to ``False`` when sending to non-pgns-agent
                receivers that don't understand the ``_trace`` key.
                Defaults to ``True``.

                .. warning::

                    When ``True``, internal operational data (agent name,
                    timing, status) is included in the outbound payload
                    under ``_trace``.  Always pass ``propagate_trace=False``
                    when sending to external webhooks, third-party APIs, or
                    any receiver outside your agent network.

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

        # Inject trace data into outbound payload when tracing is active.
        # Idempotent: skip if _trace already present (e.g. Loft migration).
        if (
            propagate_trace
            and self._tracing
            and isinstance(payload, dict)
            and "_trace" not in payload
        ):
            stage = _current_trace.get()
            if stage is not None:
                payload = {**payload, "_trace": [stage._to_wire()]}
                logger.debug(
                    "Injected trace into outbound payload for roost %r",
                    target,
                )

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

    @staticmethod
    def _extract_schema(schema: type | None) -> dict[str, Any] | None:
        """Extract JSON Schema from a type, with optional Pydantic support."""
        if schema is None:
            return None
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        raise TypeError(
            f"Unsupported schema type: {type(schema).__name__}. "
            "Pass a Pydantic BaseModel subclass (requires pydantic)."
        )

    @overload
    def on_task(self, fn: TaskHandler, /) -> TaskHandler: ...

    @overload
    def on_task(self, fn: str, /) -> Callable[[TaskHandler], TaskHandler]: ...

    @overload
    def on_task(self, fn: TaskHandler, /, *, schema: type, output_schema: type) -> TaskHandler: ...

    @overload
    def on_task(
        self, fn: str, /, *, schema: type, output_schema: type
    ) -> Callable[[TaskHandler], TaskHandler]: ...

    @overload
    def on_task(
        self, *, schema: type, output_schema: type
    ) -> Callable[[TaskHandler], TaskHandler]: ...

    def on_task(
        self,
        fn: TaskHandler | str | None = None,
        /,
        *,
        schema: type | None = None,
        output_schema: type | None = None,
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

            # With input/output schema (requires pydantic)
            @agent.on_task("summarize", schema=SummarizeInput, output_schema=SummarizeOutput)
            async def summarize(task): ...

            # Default handler with schema
            @agent.on_task(schema=SummarizeInput)
            async def handle(task): ...

        Args:
            schema: Optional Pydantic ``BaseModel`` subclass. When provided,
                ``model_json_schema()`` is called at registration time and the
                result is served as ``inputSchema`` on the skill in the agent card.
            output_schema: Optional Pydantic ``BaseModel`` subclass. When provided,
                ``model_json_schema()`` is called at registration time and the
                result is served as ``outputSchema`` on the skill in the agent card.
        """
        meta: _SkillMeta | None = None
        input_json = self._extract_schema(schema)
        output_json = self._extract_schema(output_schema)
        if input_json is not None or output_json is not None:
            meta = _SkillMeta()
            if input_json is not None:
                meta["input_schema"] = input_json
            if output_json is not None:
                meta["output_schema"] = output_json

        if callable(fn):
            self._register_handler(DEFAULT_HANDLER_NAME, fn, meta=meta)
            return fn

        # fn is a string (named skill) or None (keyword-only call)
        name = fn if fn is not None else DEFAULT_HANDLER_NAME
        if isinstance(fn, str) and name == DEFAULT_HANDLER_NAME:
            raise ValueError(
                f"The name {DEFAULT_HANDLER_NAME!r} is reserved; "
                "use the bare @agent.on_task decorator to register the default handler."
            )

        def _decorator(handler: TaskHandler) -> TaskHandler:
            self._register_handler(name, handler, meta=meta)
            return handler

        return _decorator

    # -- Testing --------------------------------------------------------------

    def set_artifact_store(self, store: ArtifactStore) -> None:
        """Replace the in-memory artifact store (for testing)."""
        self._artifact_store = store

    def test_client(
        self,
        *,
        raise_server_exceptions: bool = False,
        artifact_store: ArtifactStore | None = None,
    ) -> TestClient:
        """Return a :class:`~pgns_agent.testing.TestClient` for unit testing.

        Convenience shortcut so tests can write::

            client = agent.test_client()
            resp = client.send_task({"text": "hello"})
            assert resp.status == "completed"
        """
        from pgns_agent.testing import TestClient

        return TestClient(
            self,
            raise_server_exceptions=raise_server_exceptions,
            artifact_store=artifact_store,
        )

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
            # handle() may be a coroutine (returns dict or async gen)
            # or an async generator function (returns async gen directly).
            if inspect.isawaitable(result_or_iter):
                result_or_iter = await result_or_iter
            if inspect.isasyncgen(result_or_iter):
                last: dict[str, Any] | None = None
                async for chunk in result_or_iter:
                    last = chunk
                return last
            return result_or_iter

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
        * ``POST /message:send`` — A2A v1.0 SendMessageRequest endpoint
        * ``POST /`` — task dispatch via pigeon delivery protocol
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
        """Auto-provision an AgentCard on the pgns relay.

        In **local dev mode** (no *pgns_key*) this is a no-op -- the server
        simply logs that it is running locally.

        When a *pgns_key* is present:

        1. List existing agent cards and look for one matching :attr:`name`.
           Create a new card if none is found.  The server auto-creates a
           backing roost for every new agent card.
        2. Fetch the backing roost by name and store its secret for HMAC
           verification of incoming webhooks.

        This method is idempotent -- calling it multiple times is safe.
        """
        if self._provisioned:
            return

        async with self._provision_lock:
            # Double-check after acquiring the lock.
            if self._provisioned:
                return

            if self._client is None:
                logger.info(
                    "Local dev mode -- skipping provisioning for %r (no pgns_key)", self._name
                )
                self._provisioned = True
                return

            from pgns.models import CreateAgentCard

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
                        capabilities=self._capabilities.to_dict(),
                        default_input_modes=list(self._default_input_modes),
                        default_output_modes=list(self._default_output_modes),
                        security_schemes=[s.to_dict() for s in self._security_schemes] or None,
                    )
                )
                logger.info("Created agent card %s (%s)", agent_card.name, agent_card.id)
            else:
                logger.info("Found existing agent card %s (%s)", agent_card.name, agent_card.id)

            self._agent_card = agent_card

            # -- Step 2: fetch backing roost (auto-created by server) ---------
            roost = await client.get_roost_by_name(f"{self._name}-inbox")
            logger.info("Found backing roost %s (%s)", roost.name, roost.id)
            self._roost = roost

            # -- Step 3: initialise webhook verifier from roost secret --------
            if roost.secret:
                from pgns.webhook import Webhook

                self._webhook = Webhook(roost.secret)
                logger.debug("HMAC verification enabled for roost %s", roost.id)
            else:
                logger.warning(
                    "Roost %s has no secret -- webhook verification is disabled", roost.id
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

    @contextlib.asynccontextmanager
    async def _tracing_scope(self) -> AsyncIterator[_StageHandle | None]:
        """Set up and tear down the tracing ContextVar around handler execution.

        Yields the ``_StageHandle`` (or ``None`` when tracing is disabled).
        The caller is responsible for calling ``stage._finalize(...)`` on
        success and error paths; this context manager only manages the
        ``_current_trace`` ContextVar lifecycle.
        """
        stage: _StageHandle | None = None
        token = None
        if self._tracing:
            stage = _StageHandle(agent_name=self._name)
            token = _current_trace.set(stage)
        try:
            yield stage
        finally:
            if token is not None:
                _current_trace.reset(token)

    def _evict_task(self, task_id: str) -> None:
        """Remove task from active state and record a tombstone for dedup."""
        self._tasks.pop(task_id, None)
        now = datetime.datetime.now(datetime.UTC)
        self._completed_tasks[task_id] = now
        # Reap expired tombstones
        cutoff = now - _COMPLETED_TASK_TTL
        self._completed_tasks = {k: v for k, v in self._completed_tasks.items() if v >= cutoff}

    async def _publish_status_update(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        correlation_id: str | None = None,
        message: str | None = None,
        artifact: Any = None,
        duration_ms: float | None = None,
        trace_error: str | None = None,
    ) -> None:
        """Publish a task-status pigeon to the agent's own roost.

        In **local dev mode** the transition is logged to the console.
        In production mode the pigeon is published via the SDK client.
        Publishing failures are logged but never propagated — they must not
        affect the task response.

        When *duration_ms* or *trace_error* are provided, they are included
        in the payload under their own keys — no dict merge, so existing
        fields like ``task_id`` or ``status`` can never be overwritten.
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
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if trace_error is not None:
            payload["trace_error"] = trace_error

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

    def _register_handler(
        self, name: str, handler: TaskHandler, *, meta: _SkillMeta | None = None
    ) -> None:
        if not name or not name.strip():
            raise ValueError("Handler name must be a non-empty string.")
        if name in self._handlers:
            raise ValueError(
                f"A task handler is already registered for {name!r}. "
                "Each skill name (or the default slot) can only have one handler."
            )
        self._handlers[name] = handler
        if meta is not None:
            self._skill_meta[name] = meta
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

        async with self._tracing_scope() as stage:
            token = current_task.set(task)
            try:
                result = await handler(task)
            except Exception as exc:
                if stage is not None:
                    stage._finalize(status="failed", error=str(exc))
                logger.exception("Handler %r raised an exception (async)", skill)
                state.transition(TaskStatus.FAILED)
                self._broadcast_task_event(task.id, TaskStatus.FAILED)
                await self._publish_status_update(
                    task.id,
                    TaskStatus.FAILED,
                    correlation_id=cid,
                    duration_ms=stage.duration_ms if stage else None,
                    trace_error="handler error",
                )
                self._evict_task(task.id)
                return
            except BaseException as exc:
                if stage is not None:
                    stage._finalize(status="failed", error=str(exc))
                state.transition(TaskStatus.FAILED)
                self._broadcast_task_event(task.id, TaskStatus.FAILED)
                await self._publish_status_update(
                    task.id,
                    TaskStatus.FAILED,
                    correlation_id=cid,
                    duration_ms=stage.duration_ms if stage else None,
                    trace_error="handler error",
                )
                self._evict_task(task.id)
                raise
            finally:
                current_task.reset(token)

        # Finalize trace on success.
        if stage is not None:
            stage._finalize(status="completed")

        # -- completed -------------------------------------------------------
        state.transition(TaskStatus.COMPLETED)
        self._broadcast_task_event(task.id, TaskStatus.COMPLETED, result=result)
        await self._publish_status_update(
            task.id,
            TaskStatus.COMPLETED,
            correlation_id=cid,
            artifact=result,
            duration_ms=stage.duration_ms if stage else None,
        )
        self._evict_task(task.id)

    async def _dispatch_task(
        self,
        task_id: str,
        task_input: Any,
        skill: str,
        metadata_raw: dict[str, Any],
        prefer_async: bool,
    ) -> tuple[int, dict[str, Any]]:
        """Core dispatch logic shared by pigeon and A2A endpoints.

        Returns ``(status_code, response_body_dict)``.
        """
        matched_handler = self._handlers.get(skill)
        # Fall back to the default handler when a named skill isn't found.
        if matched_handler is None and skill != DEFAULT_HANDLER_NAME:
            matched_handler = self._handlers.get(DEFAULT_HANDLER_NAME)
        if matched_handler is None:
            return 404, {"error": f"No handler registered for skill {skill!r}."}

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
                    future.set_result(task_input)
                    return 200, {"id": task_id, "status": "input-received"}
                return 500, {
                    "error": (
                        f"Task {task_id!r} is in input-required state but has no pending future."
                    )
                }
            logger.warning(
                "Duplicate delivery for task %s (status=%s) — "
                "task is still in-flight. If your handler routinely "
                "takes >30s, increase reply_timeout_ms.",
                task_id,
                existing.status.value,
            )
            return 409, {"error": f"Task {task_id!r} is already in-flight."}

        # Check tombstone cache — task may have already completed before
        # this retry arrived.
        completed_at = self._completed_tasks.get(task_id)
        if completed_at is not None:
            cutoff = datetime.datetime.now(datetime.UTC) - _COMPLETED_TASK_TTL
            if completed_at >= cutoff:
                logger.debug(
                    "Retry for task %s arrived after completion "
                    "(completed_at=%s), returning cached success.",
                    task_id,
                    completed_at.isoformat(),
                )
                return 200, {"id": task_id, "status": "completed"}
            # Expired tombstone — clean it up and allow reprocessing
            del self._completed_tasks[task_id]

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

        # Strip _trace from input — it's a reserved namespace used for
        # trace propagation and must never reach handler code.  Use a
        # shallow copy to preserve the HMAC-verified original dict.
        if isinstance(task_input, dict):
            task_input = {k: v for k, v in task_input.items() if k != "_trace"}

        # Build the Task object before the async/sync branch so both
        # paths can reference it.
        escrow = _ArtifactEscrow(
            client=self._client,
            task_id=task_id,
            correlation_id=cid,
            store=self._artifact_store,
        )
        ctrl = _TaskControl(
            update_status=functools.partial(self._handle_update_status, task_id, cid),
            request_input=functools.partial(self._handle_request_input, task_id, cid),
            store_artifact=escrow.store_artifact,
            get_artifact=escrow.get_artifact,
        )
        task = Task(
            id=task_id,
            input=task_input,
            metadata=TaskMetadata(
                correlation_id=cid,
                source_agent=metadata_raw.get("source_agent"),
            ),
            _ctrl=ctrl,
        )

        # -- async mode (Prefer: respond-async, RFC 7240) -------------------
        if prefer_async:
            bg = asyncio.create_task(
                self._run_async_handler(task, matched_handler, state, cid, skill)
            )
            self._background_tasks.add(bg)
            bg.add_done_callback(self._background_tasks.discard)
            return 202, {"id": task.id, "status": "submitted"}

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

        async with self._tracing_scope() as stage:
            token = current_task.set(task)
            try:
                result = await matched_handler(task)
            except Exception as exc:
                # Finalize trace on error before any early returns.
                if stage is not None:
                    stage._finalize(status="failed", error=str(exc))

                # Detect rate-limit errors from upstream LLM providers and
                # return 503 so the delivery worker backs off instead of
                # immediately retrying (which amplifies the rate limit).
                exc_cls = type(exc).__name__
                is_rate_limit = "RateLimitError" in exc_cls or (
                    hasattr(exc, "status_code") and exc.status_code == 429
                )
                if is_rate_limit:
                    logger.warning(
                        "Handler %r hit rate limit: %s",
                        skill,
                        exc,
                    )
                    retry_after = (
                        getattr(exc, "headers", {}).get("retry-after", "60")
                        if hasattr(exc, "headers")
                        else "60"
                    )
                    # Trace is finalized as "failed" above but we intentionally
                    # don't publish a status pigeon here — the 503 tells the
                    # delivery worker to retry, so this task will be re-created.
                    self._evict_task(task.id)
                    pending = self._pending_input.pop(task.id, None)
                    if pending is not None and not pending.done():
                        pending.cancel()
                    return 503, {
                        "id": task.id,
                        "status": "rate_limited",
                        "error": "rate limited",
                        "_retry_after": str(retry_after),
                    }

                logger.exception("Handler %r raised an exception", skill)
                # -- failed --------------------------------------------------
                if not state.is_terminal:
                    state.transition(TaskStatus.FAILED)
                    self._broadcast_task_event(task.id, TaskStatus.FAILED)
                    await self._publish_status_update(
                        task.id,
                        TaskStatus.FAILED,
                        correlation_id=cid,
                        duration_ms=stage.duration_ms if stage else None,
                        trace_error="handler error",
                    )
                pending = self._pending_input.pop(task.id, None)
                if pending is not None and not pending.done():
                    pending.cancel()
                self._evict_task(task.id)
                return 500, {"id": task.id, "status": "failed", "error": "Internal handler error."}
            except BaseException as exc:
                # Finalize trace on BaseException (CancelledError, etc.).
                if stage is not None:
                    stage._finalize(status="failed", error=str(exc))

                # CancelledError, KeyboardInterrupt, etc. — still transition to
                # FAILED so the state machine isn't left in WORKING, then re-raise.
                if not state.is_terminal:
                    state.transition(TaskStatus.FAILED)
                    self._broadcast_task_event(task.id, TaskStatus.FAILED)
                    await self._publish_status_update(
                        task.id,
                        TaskStatus.FAILED,
                        correlation_id=cid,
                        duration_ms=stage.duration_ms if stage else None,
                        trace_error="handler error",
                    )
                pending = self._pending_input.pop(task.id, None)
                if pending is not None and not pending.done():
                    pending.cancel()
                self._evict_task(task.id)
                raise
            finally:
                current_task.reset(token)

        # Finalize trace on success.
        if stage is not None:
            stage._finalize(status="completed")

        # -- completed -------------------------------------------------------
        state.transition(TaskStatus.COMPLETED)
        self._broadcast_task_event(task.id, TaskStatus.COMPLETED, result=result)
        await self._publish_status_update(
            task.id,
            TaskStatus.COMPLETED,
            correlation_id=cid,
            artifact=result,
            duration_ms=stage.duration_ms if stage else None,
        )
        # Evict terminal entry — the pigeon stream is the authoritative history.
        self._evict_task(task.id)
        return 200, {"id": task.id, "status": "completed", "result": result}

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

            task_id = body.get("id")
            if not isinstance(task_id, str):
                return JSONResponse({"error": '"id" must be a string.'}, status_code=400)
            if not _TASK_ID_RE.fullmatch(task_id):
                return JSONResponse(
                    {"error": '"id" must be 1-256 printable ASCII characters.'},
                    status_code=400,
                )

            skill = body.get("skill") or DEFAULT_HANDLER_NAME
            metadata_raw = body.get("metadata")
            if not isinstance(metadata_raw, dict):
                metadata_raw = {}

            prefer = request.headers.get("prefer", "")
            prefer_tokens = {t.strip().split(";")[0].strip() for t in prefer.split(",")}
            prefer_async = "respond-async" in prefer_tokens

            status_code, response_body = await self._dispatch_task(
                task_id=task_id,
                task_input=body.get("input"),
                skill=skill,
                metadata_raw=metadata_raw,
                prefer_async=prefer_async,
            )

            headers: dict[str, str] = {}
            if status_code == 202:
                headers["Preference-Applied"] = "respond-async"
            if status_code == 503:
                headers["Retry-After"] = (
                    response_body.pop("_retry_after", "60")
                    if "_retry_after" in response_body
                    else "60"
                )

            return JSONResponse(response_body, status_code=status_code, headers=headers or None)

        async def a2a_message_endpoint(request: Request) -> JSONResponse:
            """Accept an A2A SendMessageRequest envelope and dispatch to handler.

            Parses the A2A v1.0 ``SendMessageRequest`` format, extracts text
            content from message parts, and dispatches to the default handler
            via :meth:`_dispatch_task`.  Returns an A2A ``Task`` response.

            .. warning::

                This endpoint performs no authentication or authorization.
                It MUST be deployed behind an authenticating reverse proxy.
                The pigeon ``POST /`` path relies on HMAC-signed delivery
                from the pgns relay; this endpoint has no equivalent guard.
            """
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "Request body must be a JSON object."}, status_code=400
                )

            # Validate A2A envelope structure.
            message = body.get("message")
            if not isinstance(message, dict) or "parts" not in message:
                return JSONResponse(
                    {"error": "Invalid A2A envelope: 'message' with 'parts' is required."},
                    status_code=400,
                )

            parts = message.get("parts")
            if not isinstance(parts, list) or not parts:
                return JSONResponse(
                    {"error": "Invalid A2A envelope: 'message.parts' must be a non-empty array."},
                    status_code=400,
                )

            # Extract text content from parts (capped to prevent abuse).
            _MAX_PARTS = 100
            _MAX_TEXT_BYTES = 1_048_576  # 1 MB

            text_parts: list[str] = []
            total_size = 0
            for part in parts[:_MAX_PARTS]:
                if not isinstance(part, dict):
                    continue
                kind = part.get("kind")
                if kind == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        total_size += len(text.encode("utf-8", errors="replace"))
                        if total_size > _MAX_TEXT_BYTES:
                            return JSONResponse(
                                {"error": "A2A message text exceeds 1 MB limit."},
                                status_code=400,
                            )
                        text_parts.append(text)
                    else:
                        logger.warning(
                            "A2A text part has non-string 'text' field (type=%s), skipping",
                            type(text).__name__,
                        )
                else:
                    logger.debug("Ignoring non-text A2A part (kind=%s)", kind)

            if not text_parts:
                return JSONResponse(
                    {"error": "No text parts found in A2A message."},
                    status_code=400,
                )

            extracted_text = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]

            # Determine blocking mode from configuration.
            config = body.get("configuration") or {}
            blocking_raw = config.get("blocking", True)
            blocking = blocking_raw is True  # strict bool check

            # Use client-supplied messageId for idempotency, fall back to uuid4.
            message_id = message.get("messageId")
            task_id = (
                str(message_id) if isinstance(message_id, str) and message_id else uuid.uuid4().hex
            )

            status_code, dispatch_result = await self._dispatch_task(
                task_id=task_id,
                task_input=extracted_text,
                skill=DEFAULT_HANDLER_NAME,
                metadata_raw={},
                prefer_async=not blocking,
            )

            # Transform dispatch result into A2A Task response format.
            return _to_a2a_response(task_id, status_code, dispatch_result)

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
                Route("/message:send", a2a_message_endpoint, methods=["POST"]),
                Route("/", task_endpoint, methods=["POST"]),
            ],
        )


def _to_a2a_response(
    task_id: str, status_code: int, dispatch_result: dict[str, Any]
) -> JSONResponse:
    """Transform an internal dispatch result into an A2A Task response.

    For 503 responses, propagates the ``Retry-After`` header from the
    dispatch result so A2A clients back off instead of retrying immediately.
    """
    if status_code == 200:
        status = dispatch_result.get("status", "completed")
        if status == "completed":
            result = dispatch_result.get("result")
            result_text = json.dumps(result)
            response_body: dict[str, Any] = {
                "id": task_id,
                "status": {"state": "completed"},
            }
            if result is not None:
                response_body["artifacts"] = [{"parts": [{"kind": "text", "text": result_text}]}]
            return JSONResponse(response_body)
        # Non-completed 200s (e.g. input-received, cached completed)
        return JSONResponse(
            {
                "id": task_id,
                "status": {"state": status},
            }
        )

    if status_code == 202:
        return JSONResponse(
            {"id": task_id, "status": {"state": "submitted"}},
            status_code=202,
        )

    # Error cases (400, 404, 409, 500, 503)
    error_msg = dispatch_result.get("error", "Unknown error")
    headers: dict[str, str] = {}
    if status_code == 503:
        headers["Retry-After"] = str(dispatch_result.pop("_retry_after", "60"))
    return JSONResponse(
        {
            "id": task_id,
            "status": {
                "state": "failed",
                "message": {"role": "agent", "parts": [{"kind": "text", "text": error_msg}]},
            },
        },
        status_code=status_code,
        headers=headers or None,
    )
