"""FastAPI mountable routes for the Phase B stepping server.

The HTTP/SSE transport that lets a browser drive an interactive step-through
session. Pairs with :mod:`agent_timetravel.stepping` (the primitive) — this module
provides the server-side glue: a runner registry, an SSE-backed approval
channel, and three endpoints.

Routes
------
* ``POST /api/v1/sessions``               - start a stepping session
* ``GET  /api/v1/sessions``               - list sessions (newest-first)
* ``GET  /api/v1/sessions/{id}``          - session detail
* ``GET  /api/v1/sessions/{id}/stream``   - SSE stream of pending Steps
* ``POST /api/v1/sessions/{id}/decide``   - post a Decision to resume the agent
* ``DELETE /api/v1/sessions/{id}``        - remove a session row

Architecture
------------
The agent runs server-side in a background ``asyncio.Task`` spawned by
``POST /sessions``. The task opens a :func:`timetravel.replay.replay` context
with an :class:`SSEApprovalChannel` attached; each paused call pushes a
:class:`~agent_timetravel.stepping.Step` onto the channel. The SSE endpoint drains
the channel's pending-step queue; ``POST /decide`` resolves it. The task
stays alive across requests — it does **not** block the spawning POST
(unlike :mod:`agent_timetravel.eval_api`'s fire-and-wait ``POST /evals``).

Runner registry
---------------
The server does not know how to run an arbitrary agent. The developer
registers entrypoints with :func:`register_runner` before starting the
server::

    from agent_timetravel.stepping_api import register_runner

    async def my_agent(session):
        await agent.run()   # pauses at each LLM call via the active session

    register_runner("deep-research", my_agent)

The ``agent_ref`` in ``POST /sessions`` resolves to a registered runner.

Concurrency
-----------
One process holds all live sessions in :data:`_LIVE_SESSIONS`. This is a
single-process design — multi-worker uvicorn would need an external
coordination store (Redis), which is out of scope for the local-first
Phase B. WAL-mode SQLite + per-call connections keep the request handlers
responsive while the runner task is blocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_timetravel.agents import AgentDefinition, TimeTravel, mask_secrets
from agent_timetravel.enums import ReplayMode
from agent_timetravel.replay import ReplaySession
from agent_timetravel.stepping import (
    Decision,
    DecisionKind,
    InteractiveSession,
    RunControlBreakpoint,
    RunControlIntent,
    Step,
    SteppingStopped,
)
from agent_timetravel.storage import TraceStore

__all__ = [
    "AgentListResponse",
    "AgentStartRequest",
    "DecisionRequest",
    "EvaluateRequest",
    "EvaluateResponse",
    "EvaluatorResult",
    "RestartFromRequest",
    "RunControlView",
    "RunnerFn",
    "SSEApprovalChannel",
    "SessionManager",
    "StartSessionRequest",
    "mount_stepping",
    "register_evaluator",
    "register_runner",
]


# ----------------------------------------------------------------------
# Runner registry
# ----------------------------------------------------------------------
#: A runner is the developer-supplied coroutine that drives the agent.
#: It receives the bound :class:`ReplaySession` so it can inspect the cursor
#: or branch_id if needed; typically it just calls ``await agent.run()``
#: inside the ``replay()`` context the server has already opened.
RunnerFn = Callable[[ReplaySession], Awaitable[Any]]

#: Module-global registry. Populated by :func:`register_runner` before the
#: server starts. Kept global (not on app.state) so a developer's call to
#: ``register_runner`` at import time works regardless of which app instance
#: mounts the routes — mirrors how framework instrumentation registers
#: itself globally.
_RUNNERS: dict[str, RunnerFn] = {}


def register_runner(ref: str, runner: RunnerFn) -> None:
    """Register an agent runner under ``ref``.

    Called once at import time by the developer's wiring code. The same
    ``ref`` is then sent in ``POST /api/v1/sessions`` bodies to start a
    session. Re-registering under an existing ref overwrites — useful for
    hot-reload during development.
    """
    if not ref:
        raise ValueError("runner ref must be non-empty")
    _RUNNERS[ref] = runner


def get_runner(ref: str) -> RunnerFn | None:
    """Resolve a registered runner by ref. Returns ``None`` if unknown."""
    return _RUNNERS.get(ref)


# ----------------------------------------------------------------------
# Custom evaluator registry (Phase 3.4)
# ----------------------------------------------------------------------
#: A custom evaluator takes the step's result text + optional context and
#: returns a pass/fail verdict with detail. Registered by the developer so the
#: workbench UI can run ad-hoc quality checks on a step's output without
#: writing a full eval suite.
CustomEvaluatorFn = Callable[..., Awaitable["EvaluatorResult"]]

#: Module-global registry, mirrors ``_RUNNERS``. Populated at import time.
_EVALUATORS: dict[str, CustomEvaluatorFn] = {}


def register_evaluator(name: str, evaluator: CustomEvaluatorFn) -> None:
    """Register a custom evaluator under ``name`` (Phase 3.4).

    Called once at import time. The same ``name`` is sent in
    ``POST /api/v1/evaluate`` bodies to invoke it on a step's output.
    Re-registering overwrites — useful for hot-reload during development.
    """
    if not name:
        raise ValueError("evaluator name must be non-empty")
    _EVALUATORS[name] = evaluator


def get_evaluator(name: str) -> CustomEvaluatorFn | None:
    """Resolve a registered evaluator by name. Returns ``None`` if unknown."""
    return _EVALUATORS.get(name)


@dataclass(frozen=True)
class EvaluatorResult:
    """The verdict a custom evaluator returns for one step output."""

    passed: bool
    detail: str = ""


# ----------------------------------------------------------------------
# SSE approval channel — bridges the agent to the HTTP/SSE transport
# ----------------------------------------------------------------------
@dataclass
class SSEApprovalChannel:
    """Approval channel that surfaces Steps via a queue for SSE streaming.

    The agent side calls :meth:`submit` (inherited from the protocol) and
    blocks on the decision. The SSE handler drains :meth:`pending_steps`
    and the POST /decide handler calls :meth:`decide` to unblock it.

    Three queues, all capacity-bounded to surface backpressure early
    rather than buffering unbounded steps if the browser is slow or
    disconnected:

    * ``_pending`` — Steps pushed by the agent, awaiting a decision.
    * ``_decisions`` — Decisions posted by the browser, awaiting collection.
    * ``_events`` — Lifecycle events (paused/resumed/done/errored) for the
      SSE stream to publish alongside Steps so the UI can render state
      transitions without polling.
    """

    _pending: asyncio.Queue[Step] = field(
        default_factory=lambda: asyncio.Queue(maxsize=8)
    )
    _decisions: asyncio.Queue[Decision] = field(
        default_factory=lambda: asyncio.Queue(maxsize=8)
    )
    _events: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=32)
    )
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    #: Server-owned run-control intent (Phase 1.2). The runner's gate reads
    #: this in-memory copy (mirroring the persisted row) so "pause after
    #: current" / "run until breakpoint" survive a page refresh AND take
    #: effect at the next gate without a DB round-trip.
    run_control: RunControlIntent = field(default_factory=RunControlIntent)
    #: Optional persistence hook supplied by the session runner. Run-control
    #: is consumed inside the gate, so the durable row must be updated at the
    #: same point rather than waiting for another HTTP request.
    persist_run_control: Callable[[RunControlIntent], None] | None = field(
        default=None, repr=False
    )
    #: Reason for the most recent surfaced run-control pause. This is reset
    #: for every gate and copied into the SSE event by :meth:`submit`.
    last_pause_reason: str | None = field(default=None, init=False, repr=False)
    #: Latest in-flight call events retained for an SSE reconnect. The browser
    #: may refresh after the original stream consumed these queue entries.
    _replay_events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the server loop before a worker-thread tool reaches a gate."""
        self._loop = loop

    def set_run_control(self, intent: RunControlIntent) -> None:
        """Replace the in-memory run-control intent (mirrors the DB row).

        Called by ``PATCH /api/v1/sessions/{id}/run-control``. The next gate
        invocation sees the new value immediately.
        """
        self.run_control = intent

    def _consume_run_control(self, intent: RunControlIntent) -> None:
        self.run_control = intent
        if self.persist_run_control is not None:
            self.persist_run_control(intent)

    def _run_from_thread(self, coroutine: Coroutine[Any, Any, Decision]) -> Decision:
        """Schedule a browser-mediated pause from a synchronous tool worker."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("SSE approval channel has no bound event loop")
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is loop:
            coroutine.close()
            raise RuntimeError(
                "a synchronous TimeTravel tool cannot block the server event loop; "
                "run the tool in a worker thread or expose it as an async tool"
            )
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()

    async def submit(self, step: Step) -> Decision:
        """Push the pre-call gate and await its execution decision.

        When server-owned run-control intent is set (Phase 1.2) the channel
        short-circuits the browser round-trip:

        * ``run_until_breakpoint`` — auto-APPROVE unless ``step`` matches a
          registered breakpoint (then surface to the browser).
        * ``pause_after_current`` — clear the flag (it was consumed by the
          *previous* approved step) and surface *this* step to the browser.

        When no intent is set, the behaviour is unchanged: every step
        surfaces to the browser via the paused event + decision queue.
        """
        decision = self._maybe_auto_decide(step)
        if decision is not None:
            return decision
        await self._pending.put(step)
        paused_event = {
            "type": "paused",
            "cursor": step.cursor,
            "kind": step.kind.value,
            "pause_reason": self.last_pause_reason,
            # Include the full payload so the browser can render the pending
            # call (messages/params/tool args) without a second round-trip.
            "step": step.payload,
        }
        self._replay_events = [paused_event]
        await self._events.put(paused_event)
        decision = await self._decisions.get()
        resumed_event: dict[str, Any]
        if decision.kind is DecisionKind.STOP:
            resumed_event = {"type": "resumed", "decision": decision.kind.value}
            self._replay_events = []
        else:
            resumed_event = {
                "type": "dispatching",
                "cursor": step.cursor,
                "decision": decision.kind.value,
            }
            if self._replay_events and self._replay_events[0].get("cursor") == step.cursor:
                self._replay_events.append(resumed_event)
            else:
                self._replay_events = [resumed_event]
        await self._events.put(resumed_event)
        return decision

    def _maybe_auto_decide(self, step: Step) -> Decision | None:
        """Apply server-owned run-control intent; return a decision to skip the gate.

        Returns ``None`` when the step must surface to the browser (no intent,
        or a breakpoint fired). Mutates ``self.run_control`` as intent is
        consumed so each flag fires exactly once.

        * ``pause_after_current`` — this is the "after current" step. Clear
          the flag and surface to the browser (return ``None``).
        * ``run_until_breakpoint`` — auto-APPROVE unless this step hits a
          breakpoint, in which case clear the flag and surface.
        """
        self.last_pause_reason = None
        rc = self.run_control
        if rc.pause_after_current:
            # Consume the one-shot: this step is the one we wanted to pause at.
            self._consume_run_control(RunControlIntent(
                pause_after_current=False,
                run_until_breakpoint=rc.run_until_breakpoint,
                breakpoints=rc.breakpoints,
            ))
            self.last_pause_reason = "pause_after_current"
            return None
        breakpoint_hit = self._hits_breakpoint(step)
        if rc.run_until_breakpoint and not breakpoint_hit:
            return Decision(kind=DecisionKind.APPROVE)
        if rc.run_until_breakpoint and breakpoint_hit:
            # Breakpoint fired — stop running and surface to the browser.
            self._consume_run_control(RunControlIntent(
                pause_after_current=False,
                run_until_breakpoint=False,
                breakpoints=rc.breakpoints,
            ))
            self.last_pause_reason = "breakpoint"
        return None

    def _hits_breakpoint(self, step: Step) -> bool:
        """True if a persisted rule or legacy payload marker matches.

        The marker remains supported for compatibility with older callers, but
        normal browser-configured breakpoints are evaluated here, inside the
        runner's gate, so a disconnected or refreshed browser cannot bypass
        the stop.
        """
        return bool(step.payload.get("breakpoint")) or any(
            rule.matches(step) for rule in self.run_control.breakpoints
        )

    def submit_sync(self, step: Step) -> Decision:
        """Thread-safe pre-tool pause used by ``@timetravel.tool`` workers."""
        return self._run_from_thread(self.submit(step))

    def decide(self, decision: Decision) -> None:
        """POST /decide entry point — non-blocking, raises if backpressured."""
        # Validation mirrors stepping.decide_with_validation; we re-run it
        # here so a malformed HTTP body fails at the boundary with a 400
        # rather than corrupting the agent's dispatch.
        from agent_timetravel.stepping import decide_with_validation

        decide_with_validation(decision)
        self._decisions.put_nowait(decision)

    async def next_step(self) -> Step:
        """SSE consumer side — await the next pending step."""
        return await self._pending.get()

    async def next_event(self) -> dict[str, Any]:
        """SSE consumer side — await the next lifecycle event or step-wrap."""
        return await self._events.get()

    def replay_events_if_idle(self) -> list[dict[str, Any]]:
        """Return the current in-flight gate events for a replacement stream.

        The caller suppresses matching queue entries after sending this
        snapshot. Keeping the snapshot independent of queue state handles a
        browser refresh while another SSE connection is still draining the
        shared queue.
        """
        return list(self._replay_events)

    def emit(self, event: dict[str, Any]) -> None:
        """Non-blocking event publish for runner-driven state (done/errored)."""
        self._events.put_nowait(event)

    def emit_delta(self, cursor: int, chunk: str) -> None:
        """Lossy publish for high-frequency stream fragments (reasoning deltas).

        Unlike :meth:`emit`, a full event queue drops the fragment instead of
        raising: the ``step_completed`` event carries the authoritative full
        text, so a dropped delta only costs a brief UI stutter, never a
        crashed run. Fragments are appended to the reconnect replay while
        they belong to the in-flight gate so a mid-generation page refresh
        restores the reasoning stream too.
        """
        event = {"type": "reasoning_delta", "cursor": cursor, "chunk": chunk}
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            return
        if self._replay_events and self._replay_events[0].get("cursor") == cursor:
            self._replay_events.append(event)

    async def complete(
        self, step: Step, result: str, usage: dict[str, int] | None = None
    ) -> Decision:
        """Surface a response and hold the agent until the developer advances."""
        event: dict[str, Any] = {
            "type": "step_completed",
            "cursor": step.cursor,
            "kind": step.kind.value,
            "result": result,
        }
        if usage is not None:
            event["usage"] = usage
        if self._replay_events and self._replay_events[0].get("cursor") == step.cursor:
            self._replay_events.append(event)
        else:
            self._replay_events = [event]
        await self._events.put(event)
        decision = await self._decisions.get()
        await self._events.put({"type": "resumed", "decision": decision.kind.value})
        self._replay_events = []
        if decision.kind is DecisionKind.STOP:
            raise SteppingStopped(step)
        return decision

    def complete_sync(
        self,
        step: Step,
        result: str,
        usage: dict[str, int] | None = None,
    ) -> Decision:
        """Thread-safe post-tool review used by ``@timetravel.tool`` workers."""
        return self._run_from_thread(self.complete(step, result, usage))

    def drain_events(self) -> list[dict[str, Any]]:
        """Non-blocking drain of any queued lifecycle events.

        Used by the SSE stream's terminal-handling path to flush straggler
        events after a ``done``/``errored`` so the UI sees every transition
        before the stream closes.
        """
        out: list[dict[str, Any]] = []
        while not self._events.empty():
            out.append(self._events.get_nowait())
        return out


# ----------------------------------------------------------------------
# Live session manager — holds the task + channel per session_id
# ----------------------------------------------------------------------
@dataclass
class LiveSession:
    """In-memory state for one running stepping session.

    The DB row (:class:`InteractiveSession`) is the persisted bookkeeping;
    this is the live handle the request handlers consult to reach the
    blocked task and its channel.
    """

    session_id: str
    task: asyncio.Task[None]
    channel: SSEApprovalChannel


class SessionManager:
    """Process-global registry of live stepping sessions.

    Why not app.state? The runner task outlives any single request — it's
    started by ``POST /sessions`` and awaited/cancelled when the run
    completes or the session is deleted. Tying it to app.state would work,
    but a module singleton is clearer about the "one process, many sessions"
    invariant and matches the runner registry's design.

    Not thread-safe by design — asyncio tasks live on one event loop. The
    request handlers are async (FastAPI runs them on the loop, not in the
    threadpool) so no lock is needed.
    """

    def __init__(self) -> None:
        self._live: dict[str, LiveSession] = {}

    def add(self, session_id: str, live: LiveSession) -> None:
        self._live[session_id] = live

    def get(self, session_id: str) -> LiveSession | None:
        return self._live.get(session_id)

    def remove(self, session_id: str) -> LiveSession | None:
        return self._live.pop(session_id, None)

    def all_ids(self) -> list[str]:
        return list(self._live)


#: Module-global manager instance.
_SESSIONS: SessionManager = SessionManager()


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


# ----------------------------------------------------------------------
# View models — wire shape
# ----------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    """Body of ``POST /api/v1/sessions``."""

    trace_id: str = Field(..., description="OTel trace id (32-hex) to step through.")
    runner_ref: str = Field(
        ..., description="Ref of a runner registered via timetravel.stepping_api.register_runner."
    )
    mode: str = Field(
        ReplayMode.INTERACTIVE.value,
        description="ReplayMode to use. Defaults to 'interactive'.",
    )
    branch_at: int | None = Field(
        None,
        description="Optional span index to branch from. None replays the seed trace.",
    )
    label: str = Field("", description="Human-readable label for the session.")


class StartSessionResponse(BaseModel):
    """Response of ``POST /api/v1/sessions``."""

    session_id: str
    trace_id: str
    branch_id: str
    status: str


class AgentStartRequest(BaseModel):
    """Body of ``POST /api/v1/agents/{ref}/sessions``."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    mode: str = ReplayMode.INTERACTIVE.value
    branch_at: int | None = None
    label: str = ""


class AgentListItem(BaseModel):
    ref: str
    name: str
    description: str
    framework: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    tags: tuple[str, ...]
    capabilities: dict[str, bool]
    available: bool
    availability_reason: str | None = None


class AgentListResponse(BaseModel):
    items: list[AgentListItem]
    total: int


class SessionDetailView(BaseModel):
    """Wire shape for a single session row."""

    session_id: str
    trace_id: str
    branch_id: str
    runner_ref: str
    agent_ref: str | None = None
    input_payload: dict[str, Any] | None = None
    result_payload: Any = None
    status: str
    error_message: str | None = None
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    """Wire shape for the session list endpoint."""

    items: list[SessionDetailView]
    total: int
    limit: int
    offset: int


class DecisionRequest(BaseModel):
    """Body of ``POST /api/v1/sessions/{id}/decide``."""

    kind: str = Field(..., description="One of: approve, edit, stop, step_once, mock, skip.")
    messages: list[dict[str, Any]] | None = Field(
        None, description="For edit: replacement message list."
    )
    params: dict[str, Any] | None = Field(
        None, description="For edit: kwargs to merge into the LLM call."
    )
    args: list[Any] | None = Field(
        None, description="For edit: replacement tool positional args."
    )
    kwargs: dict[str, Any] | None = Field(
        None, description="For edit: replacement tool keyword args."
    )
    model: str | None = Field(None, description="For edit: model-name override.")
    mock_result: Any = Field(None, description="For mock: JSON-safe replacement tool result.")
    reason: str | None = Field(
        None,
        description="Optional human-readable reason for the decision (e.g. a "
        "reject rationale). Logged for auditability; not interpreted by the runner.",
    )


class RunControlView(BaseModel):
    """Wire shape for server-owned run-control intent (Phase 1.2).

    Mirrors :class:`agent_timetravel.stepping.RunControlIntent`. ``PATCH``ing this
    object mid-run lets the UI say "pause after this step" or "run until
    breakpoint" in a way that survives a page refresh or SSE reconnect —
    the intent lives on the session row, not in volatile browser state.
    """

    pause_after_current: bool = Field(
        default=False,
        description="If true, the runner re-pauses before the step *after* the "
        "currently-approved one.",
    )
    run_until_breakpoint: bool = Field(
        default=False,
        description="If true, the runner auto-approves until a breakpoint fires.",
    )
    breakpoints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rules evaluated by the server before each intercepted call.",
    )


class EvaluateRequest(BaseModel):
    """Body of ``POST /api/v1/evaluate`` (Phase 3.4).

    Invokes a registered custom evaluator on a step's output text. The
    evaluator is identified by ``name`` (registered via
    :func:`register_evaluator`); ``result`` is the model's response to check.
    """

    name: str = Field(..., description="Registered evaluator name.")
    result: str = Field(..., description="The step output text to evaluate.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context dict forwarded to the evaluator.",
    )


class EvaluateResponse(BaseModel):
    """Response of ``POST /api/v1/evaluate``."""

    name: str
    passed: bool
    detail: str = ""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _is_valid_uuid(value: str) -> bool:
    """Cheap UUID-format check (not a strict UUID parse — fast pre-validation)."""
    try:
        UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _session_to_view(s: InteractiveSession) -> SessionDetailView:
    return SessionDetailView(
        session_id=s.session_id,
        trace_id=s.trace_id,
        branch_id=s.branch_id,
        runner_ref=s.runner_ref,
        agent_ref=s.agent_ref,
        input_payload=s.input_payload,
        result_payload=s.result_payload,
        status=s.status,
        error_message=s.error_message,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


def _resolve_mode(mode_str: str) -> ReplayMode:
    """Parse a mode string, raising HTTPException(400) on a bad value."""
    try:
        return ReplayMode(mode_str)
    except ValueError as exc:
        valid = ", ".join(m.value for m in ReplayMode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown mode '{mode_str}'; expected one of: {valid}",
        ) from exc


def _build_decision(body: DecisionRequest) -> Decision:
    """Convert the wire body to a stepping.Decision, validating the kind."""
    try:
        kind = DecisionKind(body.kind)
    except ValueError as exc:
        valid = ", ".join(k.value for k in DecisionKind)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown decision kind '{body.kind}'; expected one of: {valid}",
        ) from exc
    return Decision(
        kind=kind,
        messages=body.messages,
        params=body.params,
        args=body.args,
        kwargs=body.kwargs,
        model=body.model,
        mock_result=body.mock_result,
        reason=body.reason,
    )


class RestartFromRequest(BaseModel):
    """Body of ``POST /api/v1/sessions/{id}/restart-from``."""

    branch_at: int = Field(
        ...,
        ge=0,
        description="Span index to timetravel to. Spans [0, branch_at) are "
        "inherited from the source session's branch; the new session "
        "re-runs the runner live from there.",
    )
    label: str = Field("", description="Human-readable label for the new session.")
    inputs: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional replacement inputs. Required when restarting a decorated "
            "agent whose original inputs contain Pydantic secret inputs."
        ),
    )


# ----------------------------------------------------------------------
# Runner task spawner — shared by start_session + restart_from
# ----------------------------------------------------------------------
def _spawn_runner_task(
    *,
    store: TraceStore,
    session_obj: ReplaySession,
    runner: RunnerFn,
    runner_ref: str,
    agent_ref: str | None = None,
    input_payload: dict[str, Any] | None = None,
) -> str:
    """Spawn the background ``asyncio.Task`` that drives an interactive run.

    Shared by ``POST /sessions`` and ``POST /sessions/{id}/restart-from`` so
    the ContextVar binding, status transitions, and SSE event sequence stay
    in one place. Returns the new session_id.

    The ContextVar binding happens INSIDE the task — see the inlined comment
    for why a token minted in the request handler's context can't be reset
    from the child task (``ValueError: was created in a different Context``).
    """
    session_id = str(uuid4())
    channel = SSEApprovalChannel()
    session_obj.approval = channel

    def persist_run_control(intent: RunControlIntent) -> None:
        """Persist a gate-consumed intent without changing session status."""
        row = store.get_interactive_session(session_id)
        if row is None:
            return
        store.upsert_interactive_session(
            InteractiveSession(
                session_id=row.session_id,
                trace_id=row.trace_id,
                branch_id=row.branch_id,
                runner_ref=row.runner_ref,
                agent_ref=row.agent_ref,
                input_payload=row.input_payload,
                result_payload=row.result_payload,
                status=row.status,
                error_message=row.error_message,
                created_at=row.created_at,
                updated_at=_now_iso(),
                run_control=intent,
            )
        )

    channel.persist_run_control = persist_run_control
    now = _now_iso()
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.replay import _active_session
    # pylint: enable=import-outside-toplevel

    async def _runner_wrapper() -> None:
        """Drive the runner, update status, never raise to the caller."""
        row = InteractiveSession(
            session_id=session_id,
            trace_id=session_obj.trace_id,
            branch_id=str(session_obj.branch_id),
            runner_ref=runner_ref,
            agent_ref=agent_ref,
            input_payload=input_payload,
            status="running",
            created_at=now,
            updated_at=now,
        )
        store.upsert_interactive_session(row)
        channel.bind_loop(asyncio.get_running_loop())
        token = _active_session.set(session_obj)
        try:
            result = await runner(session_obj)
            result_payload = mask_secrets(result)
            _set_status(store, session_id, "done", result_payload=result_payload)
            channel.emit({"type": "done", "result": result_payload})
        except SteppingStopped as exc:
            # Normal, developer-initiated termination.
            _set_status(store, session_id, "done")
            channel.emit({"type": "done", "reason": "stopped", "cursor": exc.step.cursor})
        except asyncio.CancelledError:
            # Deletion or shutdown — propagate after marking the row.
            _set_status(store, session_id, "done")
            channel.emit({"type": "done", "reason": "cancelled"})
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # The runner is arbitrary developer code; we must catch
            # everything to surface the failure via the session row +
            # SSE 'errored' event rather than letting the task die
            # silently. Mirrors evaluate.py:1294 and enrichment.py:218.
            _set_status(store, session_id, "errored", error_message=str(exc))
            channel.emit({"type": "errored", "message": str(exc)})
        finally:
            _active_session.reset(token)
            _SESSIONS.remove(session_id)

    task = asyncio.create_task(_runner_wrapper(), name=f"timetravel-stepping-{session_id}")
    _SESSIONS.add(
        session_id,
        LiveSession(session_id=session_id, task=task, channel=channel),
    )
    return session_id


# ----------------------------------------------------------------------
# Mount
# ----------------------------------------------------------------------
def mount_stepping(app: FastAPI, registry: TimeTravel | None = None) -> None:
    """Register the stepping-server API routes on ``app``.

    Mirrors :func:`agent_timetravel.timeline.mount_timeline` and
    :func:`agent_timetravel.eval_api.mount_eval` — same app.state.store accessor, same
    exception conventions. Mount after eval so the read API is available
    first.
    """

    app.state.agent_registry = registry

    @app.get("/api/v1/agents", tags=["agents"])
    def list_agents(request: Request) -> AgentListResponse:
        """List decorator-registered agents owned by this app."""
        owner: TimeTravel | None = getattr(request.app.state, "agent_registry", None)
        definitions = list(owner) if owner is not None else []
        items = [
            AgentListItem(
                ref=definition.ref,
                name=definition.name,
                description=definition.description,
                framework=definition.framework,
                input_schema=definition.input_schema,
                output_schema=definition.output_schema,
                tags=definition.tags,
                capabilities=definition.capabilities,
                available=definition.available,
                availability_reason=definition.availability_reason,
            )
            for definition in definitions
        ]
        return AgentListResponse(items=items, total=len(items))

    @app.post(
        "/api/v1/agents/{agent_ref}/sessions",
        tags=["agents"],
        status_code=status.HTTP_201_CREATED,
    )
    async def start_agent_session(
        request: Request,
        agent_ref: str,
        body: AgentStartRequest,
    ) -> StartSessionResponse:
        """Validate inputs and start a decorator-owned interactive run."""
        owner: TimeTravel | None = getattr(request.app.state, "agent_registry", None)
        definition: AgentDefinition | None = owner.get(agent_ref) if owner else None
        if definition is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown agent ref '{agent_ref}'",
            )
        if not definition.available:
            reason = definition.availability_reason or "the integration is unavailable"
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": f"agent '{agent_ref}' is unavailable",
                    "availability_reason": reason,
                },
            )
        try:
            validated = definition.validate_inputs(body.inputs)
        except Exception as exc:  # Pydantic validation is client input.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        mode = _resolve_mode(body.mode)
        trace_id = body.trace_id
        if trace_id is None:
            from secrets import token_hex

            from agent_timetravel.models import Trace

            trace_id = token_hex(16)
            store: TraceStore = request.app.state.store
            store.upsert_trace(Trace(trace_id=trace_id))
        store = request.app.state.store
        try:
            root = ReplaySession.for_root(store, trace_id, mode=mode, label=body.label)
            session_obj = (
                root
                if body.branch_at is None
                else root.fork(branch_at=body.branch_at, mode=mode, label=body.label)
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        raw_inputs = validated.model_dump(mode="python")

        agent_runner = definition.compile_runner(validated)

        session_id = _spawn_runner_task(
            store=store,
            session_obj=session_obj,
            runner=agent_runner,
            runner_ref=agent_ref,
            agent_ref=agent_ref,
            input_payload=mask_secrets(raw_inputs),
        )
        return StartSessionResponse(
            session_id=session_id,
            trace_id=session_obj.trace_id,
            branch_id=str(session_obj.branch_id),
            status="running",
        )

    @app.post(
        "/api/v1/sessions",
        tags=["stepping"],
        status_code=status.HTTP_201_CREATED,
    )
    async def start_session(
        request: Request,
        body: StartSessionRequest,
    ) -> StartSessionResponse:
        """Start an interactive stepping session.

        Spawns a background ``asyncio.Task`` that runs the registered runner
        inside a ``replay(mode=INTERACTIVE, approval=sse_channel)`` context.
        Returns immediately with the session_id; the agent's progress flows
        out via ``GET /sessions/{id}/stream``.
        """
        store: TraceStore = request.app.state.store
        runner = get_runner(body.runner_ref)
        if runner is None:
            available = ", ".join(sorted(_RUNNERS)) or "(none registered)"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown runner_ref '{body.runner_ref}'; registered: {available}",
            )
        mode = _resolve_mode(body.mode)

        # Build the replay session synchronously so a bad trace_id 404s here
        # rather than inside the spawned task (where the error would surface
        # only via the SSE stream's terminal event).
        try:
            root = ReplaySession.for_root(store, body.trace_id, mode=mode, label=body.label)
        except Exception as exc:
            detail = str(exc)
            code = (
                status.HTTP_404_NOT_FOUND
                if "not found" in detail.lower()
                else status.HTTP_400_BAD_REQUEST
            )
            raise HTTPException(status_code=code, detail=detail) from exc

        session_obj = (
            root
            if body.branch_at is None
            else root.fork(branch_at=body.branch_at, mode=mode, label=body.label)
        )

        session_id = _spawn_runner_task(
            store=store,
            session_obj=session_obj,
            runner=runner,
            runner_ref=body.runner_ref,
        )
        return StartSessionResponse(
            session_id=session_id,
            trace_id=session_obj.trace_id,
            branch_id=str(session_obj.branch_id),
            status="running",
        )

    @app.post(
        "/api/v1/sessions/{session_id}/restart-from",
        tags=["stepping"],
        status_code=status.HTTP_201_CREATED,
    )
    async def restart_from(
        request: Request,
        session_id: str,
        body: RestartFromRequest,
    ) -> StartSessionResponse:
        """Fork a session at ``branch_at`` and start a fresh interactive run.

        Implements the "timetravel and play again" loop: the developer inspects
        a completed or in-progress session, picks a step to timetravel to, and
        starts a new session that inherits spans ``[0, branch_at)`` from the
        source session's branch. The new session runs the same runner under
        a fresh INTERACTIVE + channel context.

        The source session is not modified. The new branch's captured spans
        are persisted under a fresh ``branch_id``; the existing
        :func:`~agent_timetravel.diff` surface can compare the two timelines.
        """
        if not _is_valid_uuid(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a UUID",
            )
        store: TraceStore = request.app.state.store
        source = store.get_interactive_session(session_id)
        if source is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"session not found: {session_id}",
            )
        owner: TimeTravel | None = getattr(request.app.state, "agent_registry", None)
        definition = owner.get(source.agent_ref) if owner and source.agent_ref else None
        runner: RunnerFn | None = None
        restart_input_payload: dict[str, Any] | None = source.input_payload
        if definition is not None:
            if not definition.available:
                reason = definition.availability_reason or "the integration is unavailable"
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "message": f"agent '{definition.ref}' is unavailable",
                        "availability_reason": reason,
                    },
                )
            if definition.has_secret_inputs and body.inputs is None:
                fields = ", ".join(definition.secret_input_fields)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"restart requires an inputs override because the decorated "
                        f"agent has secret input fields: {fields}"
                    ),
                )
            restart_inputs = body.inputs if body.inputs is not None else source.input_payload or {}
            try:
                validated_restart = definition.validate_inputs(restart_inputs)
            except Exception as exc:  # Pydantic validation is client input.
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            runner = definition.compile_runner(validated_restart)
            restart_input_payload = mask_secrets(validated_restart.model_dump(mode="python"))

            runner_ref = source.runner_ref
        else:
            runner = get_runner(source.runner_ref)
            runner_ref = source.runner_ref
        if runner is None:
            # The runner was un-registered between the source run and now.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"runner '{source.runner_ref}' is no longer registered; "
                    "re-register it before restarting."
                ),
            )
        if body.branch_at < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="branch_at must be >= 0",
            )

        # Rebuild the *source branch* before forking. A UI step cursor can
        # include tool gates, whereas the replay cache only counts captured
        # spans, so a completed tool may legitimately point past that cache.
        # In that case the closest valid tail is the safe restart point.
        try:
            root = ReplaySession.for_root(
                store,
                source.trace_id,
                mode=ReplayMode.INTERACTIVE,
                label=body.label or f"restart-from-{session_id[:8]}",
            )
            source_branch_id = UUID(source.branch_id)
            source_session = ReplaySession(
                store=store,
                trace_id=source.trace_id,
                branch_id=source_branch_id,
                mode=ReplayMode.INTERACTIVE,
                label=root.label,
            )
            source_session._spans_cache = store.get_spans(  # pylint: disable=protected-access
                source.trace_id,
                branch_id=source_branch_id,
            )
            branch_at = min(body.branch_at, len(source_session.recorded_spans()))
            new_session = source_session.fork(
                branch_at=branch_at,
                mode=ReplayMode.INTERACTIVE,
                label=body.label or f"restart-from-{session_id[:8]}",
            )
        except Exception as exc:
            detail = str(exc)
            code = (
                status.HTTP_404_NOT_FOUND
                if "not found" in detail.lower() or "out of range" in detail.lower()
                else status.HTTP_400_BAD_REQUEST
            )
            raise HTTPException(status_code=code, detail=detail) from exc

        new_session_id = _spawn_runner_task(
            store=store,
            session_obj=new_session,
            runner=runner,
            runner_ref=runner_ref,
            agent_ref=source.agent_ref,
            input_payload=restart_input_payload,
        )
        return StartSessionResponse(
            session_id=new_session_id,
            trace_id=new_session.trace_id,
            branch_id=str(new_session.branch_id),
            status="running",
        )

    @app.get("/api/v1/sessions", tags=["stepping"])
    def list_sessions(
        request: Request,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> SessionListResponse:
        """List stepping sessions, newest-first."""
        store: TraceStore = request.app.state.store
        items, total = store.list_interactive_sessions(limit=limit, offset=offset)
        return SessionListResponse(
            items=[_session_to_view(s) for s in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/sessions/{session_id}", tags=["stepping"])
    def get_session(request: Request, session_id: str) -> SessionDetailView:
        """Return one session row."""
        if not _is_valid_uuid(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a UUID",
            )
        store: TraceStore = request.app.state.store
        row = store.get_interactive_session(session_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"session not found: {session_id}",
            )
        return _session_to_view(row)

    @app.get("/api/v1/sessions/{session_id}/stream", tags=["stepping"])
    async def stream_session(
        request: Request,
        session_id: str,
    ) -> StreamingResponse:
        """Server-Sent Events stream of Steps + lifecycle events.

        Emits one SSE ``data:`` line per event. Event types:

        * ``paused``  — agent is blocked at a step; payload includes the Step.
        * ``resumed`` — a decision was applied; the agent continues.
        * ``reasoning_delta`` — live reasoning fragment while a forwarded LLM
          call streams (lossy; ``step_completed`` carries the full text).
        * ``done``    — runner completed (normally, via STOP, or cancelled).
        * ``errored`` — runner raised; payload includes the message.

        The stream stays open until the runner task finishes. If the runner
        finished before EventSource attached, replay the durable terminal row
        as one event so the browser can still leave its loading state.
        """
        if not _is_valid_uuid(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a UUID",
            )
        live = _SESSIONS.get(session_id)
        if live is None:
            store: TraceStore = request.app.state.store
            row = store.get_interactive_session(session_id)
            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"session not found: {session_id}",
                )
            if row.status == "done":
                terminal_event = {
                    "type": "done",
                    "result": row.result_payload,
                    "result_payload": row.result_payload,
                }
            elif row.status == "errored":
                message = row.error_message or "interactive session failed"
                terminal_event = {
                    "type": "errored",
                    "message": message,
                    "error_message": message,
                }
            else:
                terminal_event = {
                    "type": "errored",
                    "message": (
                        f"session {session_id} has no live runner; durable status is {row.status!r}"
                    ),
                    "error_message": "session has no live runner",
                }

            async def terminal_stream() -> AsyncIterator[str]:
                yield _format_sse(terminal_event)

            return StreamingResponse(
                terminal_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        channel = live.channel

        async def event_stream() -> AsyncIterator[str]:
            # A browser refresh can consume the original queue entries while
            # the runner remains blocked on its decision. Replay the in-flight
            # gate for every replacement stream. Matching queue entries are
            # discarded below so the first stream does not duplicate events.
            replayed = channel.replay_events_if_idle()
            for replay in replayed:
                yield _format_sse(replay)
            # Drain the channel until the runner emits a terminal event.
            # Each event is one SSE block: ``data: <json>\n\n``.
            #
            # We intentionally do NOT poll ``request.is_disconnected()`` in
            # the hot loop. The earlier draft used
            # ``asyncio.wait_for(channel.next_event(), timeout=1.0)`` with a
            # disconnect check on timeout, but that pattern interacts badly
            # with Starlette's StreamingResponse buffering under
            # ``ASGITransport`` (the response never flushes between
            # timeout-and-continue cycles). Direct ``await next_event()``
            # is simpler and flushes promptly. Disconnect handling falls to
            # the runner's task lifecycle: when the client goes away the
            # next-event await simply stays pending until the session is
            # deleted or the runner completes, at which point the stream
            # closes normally.
            while True:
                event = await channel.next_event()
                if replayed:
                    try:
                        replayed.remove(event)
                    except ValueError:
                        pass
                    else:
                        continue
                if event.get("type") == "paused":
                    # ``_pending`` is retained for the programmatic
                    # ``next_step`` API, while the browser consumes the same
                    # step from this event. Drain it here so a long browser
                    # session cannot fill the compatibility queue.
                    await channel.next_step()
                yield _format_sse(event)
                if event.get("type") in ("done", "errored"):
                    # Drain any straggler events then close.
                    for straggler in channel.drain_events():
                        yield _format_sse(straggler)
                    return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
            },
        )

    @app.post("/api/v1/sessions/{session_id}/decide", tags=["stepping"])
    async def decide_session(
        request: Request,
        session_id: str,
        body: DecisionRequest,
    ) -> dict[str, str]:
        """Post a Decision to resume the blocked agent.

        Returns 202 immediately; the agent unblocks asynchronously on its
        own task. Validation errors (bad kind, inconsistent EDIT) return 400.
        """
        if not _is_valid_uuid(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a UUID",
            )
        live = _SESSIONS.get(session_id)
        if live is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no live session: {session_id}",
            )
        if live.task.done():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="session is not paused (runner has finished)",
            )
        decision = _build_decision(body)
        try:
            live.channel.decide(decision)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return {"status": "accepted", "decision": decision.kind.value}

    @app.get(
        "/api/v1/sessions/{session_id}/run-control",
        tags=["stepping", "run-control"],
    )
    def get_run_control(request: Request, session_id: str) -> RunControlView:
        """Return the server-owned run-control intent for a session.

        The runner reads this intent at each stepping gate; the UI reads it on
        load to hydrate its "pause after current" / "run until breakpoint"
        toggle state from the server (Phase 1.2).
        """
        if not _is_valid_uuid(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a UUID",
            )
        store: TraceStore = request.app.state.store
        row = store.get_interactive_session(session_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no interactive session: {session_id}",
            )
        intent = row.run_control
        return RunControlView(
            pause_after_current=intent.pause_after_current,
            run_until_breakpoint=intent.run_until_breakpoint,
            breakpoints=[rule.to_dict() for rule in intent.breakpoints],
        )

    @app.patch(
        "/api/v1/sessions/{session_id}/run-control",
        tags=["stepping", "run-control"],
    )
    def patch_run_control(
        request: Request,
        session_id: str,
        body: RunControlView,
    ) -> RunControlView:
        """Update the server-owned run-control intent for a session.

        Persisted immediately so a page refresh doesn't lose it. The runner
        picks the new intent up at the *next* gate (the in-flight step, if
        any, is unaffected). The intent is also pushed onto the live channel
        so a paused step re-evaluates without a second round-trip.
        """
        if not _is_valid_uuid(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a UUID",
            )
        store: TraceStore = request.app.state.store
        row = store.get_interactive_session(session_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no interactive session: {session_id}",
            )
        new_intent = RunControlIntent(
            pause_after_current=body.pause_after_current,
            run_until_breakpoint=body.run_until_breakpoint,
            breakpoints=tuple(
                rule
                for item in body.breakpoints
                for rule in (RunControlBreakpoint.from_dict(item),)
                if rule is not None
            ),
        )
        store.upsert_interactive_session(
            InteractiveSession(
                session_id=row.session_id,
                trace_id=row.trace_id,
                branch_id=row.branch_id,
                runner_ref=row.runner_ref,
                agent_ref=row.agent_ref,
                input_payload=row.input_payload,
                result_payload=row.result_payload,
                status=row.status,
                error_message=row.error_message,
                created_at=row.created_at,
                updated_at=_now_iso(),
                run_control=new_intent,
            )
        )
        # Push the fresh intent onto the live channel so a runner blocked at a
        # gate sees it without waiting for the next submit cycle.
        live = _SESSIONS.get(session_id)
        if live is not None:
            live.channel.set_run_control(new_intent)
        return RunControlView(
            pause_after_current=new_intent.pause_after_current,
            run_until_breakpoint=new_intent.run_until_breakpoint,
            breakpoints=[rule.to_dict() for rule in new_intent.breakpoints],
        )

    @app.delete(
        "/api/v1/sessions/{session_id}",
        tags=["stepping"],
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_session(request: Request, session_id: str) -> None:
        """Cancel a live session and remove its row.

        Does NOT delete the captured spans — they remain queryable under
        ``branch_id`` via the timeline API. The runner task is cancelled;
        if it's mid-pause it unwinds via ``asyncio.CancelledError``.
        """
        if not _is_valid_uuid(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a UUID",
            )
        store: TraceStore = request.app.state.store
        live = _SESSIONS.remove(session_id)
        if live is not None and not live.task.done():
            live.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await live.task
        store.delete_interactive_session(session_id)

    @app.post("/api/v1/evaluate", tags=["stepping", "evaluators"])
    async def evaluate(body: EvaluateRequest) -> EvaluateResponse:
        """Invoke a registered custom evaluator on a step's output (Phase 3.4).

        Lets the workbench UI run ad-hoc quality checks (e.g. "does the
        response cite a source?") without writing a full eval suite. The
        evaluator is resolved by name from the module-global registry.
        """
        evaluator = get_evaluator(body.name)
        if evaluator is None:
            registered = ", ".join(sorted(_EVALUATORS)) or "(none registered)"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no evaluator named '{body.name}'; registered: {registered}",
            )
        try:
            result = await evaluator(body.result, body.context)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Developer-supplied evaluator; surface the failure rather than 500.
            return EvaluateResponse(name=body.name, passed=False, detail=f"error: {exc}")
        return EvaluateResponse(
            name=body.name, passed=result.passed, detail=result.detail
        )

    @app.get("/api/v1/evaluators", tags=["stepping", "evaluators"])
    def list_evaluators() -> list[str]:
        """Return names of developer-registered evaluators.

        Only names cross the HTTP boundary. Evaluator callables remain in the
        trusted Python process, so the browser can select a check but cannot
        submit or execute Python source.
        """
        return sorted(_EVALUATORS)


def _set_status(
    store: TraceStore,
    session_id: str,
    status_: str,
    *,
    error_message: str | None = None,
    result_payload: Any = None,  # noqa: ANN401
) -> None:
    """Update a session row's status + timestamp in place.

    Preserves the row's ``run_control`` intent — :func:`_set_status` runs on
    every lifecycle transition and must not clobber a pending "pause after
    current" / "run until breakpoint" (Phase 1.2).
    """
    row = store.get_interactive_session(session_id)
    if row is None:
        return
    store.upsert_interactive_session(
        InteractiveSession(
            session_id=row.session_id,
            trace_id=row.trace_id,
            branch_id=row.branch_id,
            runner_ref=row.runner_ref,
            agent_ref=row.agent_ref,
            input_payload=row.input_payload,
            result_payload=(result_payload if result_payload is not None else row.result_payload),
            status=status_,
            error_message=error_message,
            created_at=row.created_at,
            updated_at=_now_iso(),
            run_control=row.run_control,
        )
    )


def _format_sse(event: dict[str, Any]) -> str:
    """Serialize ``event`` as one SSE block (``data: <json>\\n\\n``)."""
    return f"data: {json.dumps(event, default=str)}\n\n"
