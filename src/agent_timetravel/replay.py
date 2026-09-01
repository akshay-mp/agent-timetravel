"""Phase 3 — Time-travel replay engine.

This module is **pure logic**. It depends only on :class:`agent_timetravel.storage.TraceStore`
and :mod:`agent_timetravel.models`; it does **not** import OpenAI, LangGraph, or any
network client. The interceptors in :mod:`agent_timetravel.openai_intercept`,
:mod:`agent_timetravel.tool_intercept`, and :mod:`agent_timetravel.adapters.langgraph` layer on
top of the contracts defined here.

Two concepts:

* :class:`ReplaySession` — owns ``trace_id``, ``branch_id``, the cursor, and
  the replay mode. It walks the recorded spans in order and either serves a
  cached response (frozen) or authorises forwarding live (branch/full).
* :class:`Responder` — the contract an interceptor implements. Given a call
  signature (``model`` + params + ``messages_hash`` + ``tools_hash``) the
  responder decides *serve-from-fixture vs. forward-live* and advances the
  cursor accordingly.

The high-level entry point is :func:`replay`, a context manager that:

1. Loads the seed trace (Phase 1 ingest).
2. Optionally forks a branch under a new ``branch_id``.
3. Stashes the active session in contextvars so the OpenAI / tool
   interceptors can find it without being passed it explicitly.
4. Tears down on exit.

See ``docs/phases/phase-3.md`` §1 for the rationale and the decisions
documented here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from agent_timetravel.enums import ReplayMode
from agent_timetravel.models import Span, Trace

if TYPE_CHECKING:
    from agent_timetravel.stepping import ApprovalChannel
    from agent_timetravel.storage import TraceStore

__all__ = [
    "CallSignature",
    "RecordedResponse",
    "ReplayError",
    "ReplaySession",
    "Responder",
    "replay",
]


#: Sentinel for "argument not supplied" — distinct from ``None`` which is a
#: valid explicit value for ``approval`` (drop the channel on a fork). We keep
#: it module-private rather than reusing ``dataclasses.MISSING`` so the type
#: annotation can be a plain union without dragging ``dataclasses`` into the
#: public signature.
class _Unset:
    """Type marker for the :data:`_UNSET` sentinel."""


_UNSET: _Unset = _Unset()


#: Contextvar holding the active :class:`ReplaySession` for the current task.
#:
#: OpenAI / tool interceptors consult this to decide whether to consult the
#: fixture cache or forward live. A ``None`` value means "no replay in
#: flight" — the client behaves normally.
_active_session: ContextVar[ReplaySession | None] = ContextVar(
    "timetravel_active_session", default=None
)


def active_session() -> ReplaySession | None:
    """Return the :class:`ReplaySession` bound to the current task, or ``None``.

    Intercept machinery uses this to short-circuit when no replay is in
    flight — keeping the patching code zero-cost in production code paths.
    """
    return _active_session.get()


class ReplayError(RuntimeError):
    """Raised when a replay invariant is violated (e.g. cursor exhausted)."""


@dataclass(frozen=True, slots=True)
class CallSignature:
    """The identifying projection of an inbound call.

    Two calls with the same signature are *required* to be served from the
    same recorded span when the session is frozen. Interceptors build this
    from the inbound request (model + messages + params + tools) and hand it
    to :meth:`Responder.respond_or_forward`.

    ``messages_hash`` is the canonical hash of the request messages (see
    :func:`agent_timetravel.models.hash_payload`). ``tools_hash`` is the hash of the
    tool schema list. Together they identify the *call* without re-reading
    the (potentially large) request body.
    """

    model: str
    messages_hash: str
    tools_hash: str | None = None


@dataclass(frozen=True, slots=True)
class RecordedResponse:
    """A cached response served from a recorded span.

    ``payload`` is the verbatim body the response span captured at ingest
    time. Interceptors cast it to their framework's native return type.
    """

    payload: dict[str, Any]
    span_id: str
    timetravel_id: UUID
    model: str | None = None


class Responder(Protocol):
    """The contract an interceptor (OpenAI / tool / framework adapter) meets.

    A responder is asked: "given this call signature, do you have a recorded
    answer?". Three outcomes:

    * The signature matches the recorded span at the cursor → return it as a
      :class:`RecordedResponse`; the caller must serve it verbatim.
    * The session is in branching mode and the cursor is exhausted (or the
      signature diverges) → return ``None`` to authorise a live forward.
      The caller must capture the new span under the session's branch_id.
    * The session is frozen and the signature does not match → raise
      :class:`ReplayError`. Frozen replay is *deterministic*; a divergence is
      a programming error, not a fallback trigger.

    Implementations live in the interceptor modules; the engine's own
    :class:`ReplaySession` is the canonical implementation.
    """

    def respond_or_forward(self, signature: CallSignature) -> RecordedResponse | None:
        """Return a recorded response if served-from-fixture, else ``None``."""


@dataclass
class ReplaySession:
    """One replay run against a seed trace.

    Owns:

    * ``trace_id`` — the seed trace being replayed.
    * ``branch_id`` — the branch under which *new* spans are recorded.
      Equal to ``root_branch_id`` in ``FROZEN`` mode (no new spans are
      written); a fresh UUID otherwise.
    * ``cursor`` — 0-based index into the seed trace's span list. Spans
      ``[0, cursor)`` have been consumed; the next call signature is matched
      against ``spans[cursor]``.
    * ``mode`` — how to handle call divergences (see :class:`ReplayMode`).
    * ``approval`` — optional :class:`~agent_timetravel.stepping.ApprovalChannel`.
      When ``mode is INTERACTIVE`` and this is non-``None``, every
      intercepted LLM/tool call pauses at the stepping gate and awaits a
      :class:`~agent_timetravel.stepping.Decision` from the channel. ``None`` for
      FROZEN/BRANCH/FULL — stepping is a no-op there.

    The session is **reentrant and isolated by branch_id** — the Phase 5.5
    eval harness relies on this. Two concurrent sessions in the same Python
    process never share mutable cursor state because the cursor lives on the
    dataclass instance, not on the store. The same isolation covers the
    approval channel: it rides on the session, which rides on the
    :data:`_active_session` ContextVar, so concurrent interactive sessions
    don't cross-talk.
    """

    # pylint: disable=too-many-instance-attributes

    store: TraceStore
    trace_id: str
    branch_id: UUID
    mode: ReplayMode = ReplayMode.FROZEN
    label: str = ""
    approval: ApprovalChannel | None = None
    _spans_cache: list[Span] = field(default_factory=list, repr=False)
    _cursor: int = field(default=0)
    _forked_at: int | None = field(default=None)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def for_root(
        cls,
        store: TraceStore,
        trace_id: str,
        *,
        mode: ReplayMode = ReplayMode.FROZEN,
        label: str = "",
        approval: ApprovalChannel | None = None,
    ) -> ReplaySession:
        """Open a session against a seed trace with no branching.

        Loads the recorded spans eagerly. The seed must exist; otherwise
        :class:`ReplayError` is raised (replay cannot proceed without a
        fixture set).

        ``approval`` attaches a stepping channel; see
        :func:`~agent_timetravel.stepping.gate_async` for the pause semantics.
        """
        trace = store.get_trace(trace_id)
        if trace is None:
            raise ReplayError(f"trace not found: {trace_id}")
        # Fresh decorator sessions may create an empty trace before the
        # runner starts. Materialize its root branch now so the timeline,
        # tree, and diff APIs can address the exact session branch id.
        from agent_timetravel.models import Branch

        store.ensure_branch(
            Branch(
                branch_id=trace.root_branch_id,
                trace_id=trace_id,
                mode=mode.value,
                label=label,
            )
        )
        session = cls(
            store=store,
            trace_id=trace_id,
            branch_id=trace.root_branch_id,
            mode=mode,
            label=label,
            approval=approval,
        )
        session._spans_cache = list(trace.spans)
        return session

    def fork(
        self,
        *,
        branch_at: int,
        mode: ReplayMode,
        label: str = "",
        approval: ApprovalChannel | _Unset | None = _UNSET,
    ) -> ReplaySession:
        """Branch this session at ``branch_at``.

        Returns a *new* session positioned at ``cursor = branch_at`` whose
        ``branch_id`` is a fresh UUID. The prefix ``[0, branch_at)`` is
        inherited from the parent branch **in-memory only** — no span
        rows are duplicated. The new branch's tail (live-captured spans
        via :meth:`record_new`) is persisted under ``branch_id``; callers
        reconstruct the full timeline by unioning ``branch_id == parent
        OR branch_id == new``.

        Why no persistence of prefix clones? Span rows are immutable and
        carry a parent via ``Branch.parent_branch_id`` — duplicating
        thousands of rows per fork would inflate the DB and would conflict
        with the inherited-prefix union exposed by
        :meth:`~agent_timetravel.storage.TraceStore.get_spans`.

        Forking mode:

        * ``BRANCH`` — the new session will forward live from cursor.
        * ``FROZEN`` — the new session will keep replaying fixtures (useful
          for diffing two replays of the same trace).
        * ``FULL_RERUN`` — the new session re-executes every span live,
          ignoring fixtures.

        ``approval`` defaults to inheriting the parent's channel (a fork of
        an interactive session stays interactive). Pass ``None`` explicitly
        to drop the channel on the fork.
        """
        if branch_at < 0 or branch_at > len(self._spans_cache):
            raise ReplayError(
                f"branch_at={branch_at} out of range [0, {len(self._spans_cache)}]"
            )
        # pylint: disable=import-outside-toplevel
        from agent_timetravel.models import Branch
        # pylint: enable=import-outside-toplevel

        new_branch_id = uuid4()
        branch = Branch(
            branch_id=new_branch_id,
            trace_id=self.trace_id,
            parent_branch_id=self.branch_id,
            branch_at_index=branch_at,
            mode=mode.value,
            label=label or f"fork-at-{branch_at}",
        )
        self.store.insert_branch(branch)
        inherited_approval: ApprovalChannel | None = (
            self.approval if isinstance(approval, _Unset) else approval
        )
        new_session = ReplaySession(
            store=self.store,
            trace_id=self.trace_id,
            branch_id=new_branch_id,
            mode=mode,
            label=label,
            approval=inherited_approval,
        )
        # The cache is shared in-memory: cloned prefix + inherited parent tail.
        # No DB writes here — see docstring for rationale.
        # pylint: disable=protected-access
        new_session._spans_cache = list(self._spans_cache)
        new_session._cursor = branch_at
        new_session._forked_at = branch_at
        return new_session

    # ------------------------------------------------------------------
    # Responder protocol
    # ------------------------------------------------------------------
    @property
    def cursor(self) -> int:
        """Index of the next span to consume (0-based)."""
        return self._cursor

    @property
    def forked_at(self) -> int | None:
        """Where this session diverged from its parent branch, or ``None``."""
        return self._forked_at

    def recorded_spans(self) -> list[Span]:
        """Return the (read-only) seed spans for this branch."""
        return list(self._spans_cache)

    def advance_cursor_to(self, index: int) -> None:
        """Move the cursor to ``index`` after a non-LLM (tool/MCP) cache hit.

        Used by :mod:`agent_timetravel.tool_intercept` when it matches a TOOL span
        by name+args-hash (out-of-order w.r.t. the LLM ``messages_hash``
        cursor). The lookup is forward-only: ``index`` must be within
        ``[cursor, len(spans)]``; spans below the current cursor are
        already consumed and cannot be re-served.
        """
        if index < self._cursor:
            raise ReplayError(
                f"advance_cursor_to({index}) < current cursor={self._cursor}; "
                "tool spans cannot timetravel an already-consumed cursor"
            )
        if index > len(self._spans_cache):
            raise ReplayError(
                f"advance_cursor_to({index}) > recorded span count "
                f"{len(self._spans_cache)}"
            )
        self._cursor = index

    def respond_or_forward(self, signature: CallSignature) -> RecordedResponse | None:
        """Match ``signature`` against the cursor and decide serve vs. forward.

        The match policy is hash-equivalence on ``messages_hash`` (and
        ``tools_hash`` if both sides carry one). The cursor advances by one
        only on a successful fixture hit; a forward live leaves the cursor
        alone (the *new* span will be appended by the interceptor capture).
        """
        if self._cursor >= len(self._spans_cache):
            # Past the recorded tail — only branching authorised forwards.
            if self.mode is ReplayMode.FROZEN:
                raise ReplayError(
                    f"frozen replay exhausted cursor={self._cursor}; no fixture"
                )
            return None

        candidate = self._spans_cache[self._cursor]
        if not _signature_matches(candidate, signature):
            # Divergence. In frozen mode this is an error (deterministic
            # violation); in branch/full it authorises a live re-execution.
            if self.mode is ReplayMode.FROZEN:
                raise ReplayError(
                    "frozen replay divergence at cursor="
                    f"{self._cursor}: expected "
                    f"messages_hash={candidate.messages_hash!r}, "
                    f"got {signature.messages_hash!r}"
                )
            return None

        self._cursor += 1
        return RecordedResponse(
            payload=candidate.raw_attributes,
            span_id=candidate.span_id,
            timetravel_id=candidate.timetravel_id,
            model=candidate.model_name,
        )

    def record_new(self, span: Span) -> None:
        """Persist a span captured by the interceptor during a live forward.

        Always uses ``self.branch_id`` so branch-bound spans are queryable as
        a distinct timeline via ``trace_id`` + ``branch_id`` filtering.
        """
        self.store.insert_span(span, branch_id=self.branch_id)
        for index, cached in enumerate(self._spans_cache):
            if cached.timetravel_id == span.timetravel_id:
                self._spans_cache[index] = span
                return
        self._spans_cache.append(span)
        self._cursor = len(self._spans_cache)

    # ------------------------------------------------------------------
    # Branch introspection
    # ------------------------------------------------------------------
    def trace_summary(self) -> Trace:
        """Return a :class:`~agent_timetravel.models.Trace` view of this branch's spans.

        Used by tests and the timeline API to query a branch as its own
        distinct timeline.
        """
        return Trace(
            trace_id=self.trace_id,
            root_branch_id=self.branch_id,
            spans=list(self._spans_cache),
        )


# ----------------------------------------------------------------------
# Signature matching
# ----------------------------------------------------------------------
def _signature_matches(span: Span, signature: CallSignature) -> bool:
    """Check whether ``span`` is the recorded form of the inbound call.

    A span matches when its ``messages_hash`` equals the signature's
    (canonical, ``hash_payload``-computed) hash, and the same holds for
    ``tools_hash`` when the caller supplies one. Model name is intentionally
    *not* matched — branching often swaps models, and the messages hash is
    already a strong identity signal.
    """
    if span.messages_hash != signature.messages_hash:
        return False
    return not (
        signature.tools_hash is not None and span.tools_hash != signature.tools_hash
    )


# ----------------------------------------------------------------------
# Context manager
# ----------------------------------------------------------------------
@contextmanager
def replay(
    store: TraceStore,
    trace_id: str,
    *,
    branch_at: int | None = None,
    mode: ReplayMode = ReplayMode.FROZEN,
    label: str = "",
    approval: ApprovalChannel | None = None,
) -> Iterator[ReplaySession]:
    """Bind a :class:`ReplaySession` for the duration of the ``with`` block.

    Behaviour:

    * ``branch_at is None`` and ``mode == FROZEN`` — replay the seed trace
      verbatim with no new spans written. The simplest "no egress" guarantee.
    * ``branch_at is None`` and ``mode in (BRANCH, FULL_RERUN)`` — open a
      new branch positioned at span 0 with the given mode.
    * ``branch_at = N`` — fork a branch from span ``N`` (spans ``[0, N)``
      are cloned). The new session's ``branch_id`` differs from the seed
      trace's ``root_branch_id``.
    * ``approval`` — attach a stepping channel. Combined with
      ``mode=INTERACTIVE`` this pauses every intercepted LLM/tool call at
      :func:`~agent_timetravel.stepping.gate_async` until the channel yields a
      :class:`~agent_timetravel.stepping.Decision`. Ignored for non-INTERACTIVE modes.

    Sessions are stored in a :class:`contextvars.ContextVar`, so nested or
    concurrent ``with replay(...)`` blocks are isolated per task. The Phase
    5.5 eval harness depends on this.
    """
    root = ReplaySession.for_root(store, trace_id, mode=mode, label=label, approval=approval)
    session = (
        root
        if branch_at is None
        else root.fork(branch_at=branch_at, mode=mode, label=label, approval=approval)
    )

    token = _active_session.set(session)
    try:
        yield session
    finally:
        _active_session.reset(token)


# ----------------------------------------------------------------------
# Type checkers + formatters
# ----------------------------------------------------------------------
def _check_activated_for_intercept() -> ReplaySession | None:
    """Internal: return the active session, or ``None``. Interceptors call this."""
    return _active_session.get()
