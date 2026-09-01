"""Unit tests for the Phase 3 replay engine (Track 3A).

Covers:

* ``ReplaySession.for_root`` — loads an existing trace.
* ``ReplaySession.fork`` — clones spans [0, branch_at) under a new UUID.
* ``respond_or_forward`` — frozen serve / branch forward / cursor advance.
* ``advance_cursor_to`` — tool-span lookup out-of-order vs LLM cursor.
* Hash-equivalence matching (``_signature_matches``).
* Frozen divergence raises :class:`ReplayError`.
* Branch isolation — concurrent sessions have independent cursors.
* ``replay(...)`` ctxmgr — sets/resets the active-session ContextVar.
* Missing trace → ``ReplayError`` at construction.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from agent_timetravel.enums import ReplayMode, SpanKind, SpanStatus
from agent_timetravel.models import Span, Trace, hash_payload
from agent_timetravel.replay import (
    CallSignature,
    RecordedResponse,
    ReplayError,
    ReplaySession,
    active_session,
)
from agent_timetravel.replay import (
    replay as replay_ctx,
)
from agent_timetravel.storage import TraceStore


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _llm_span(
    trace_id: str,
    *,
    span_id: str,
    messages: list[dict[str, str]],
    model: str = "qwen3:32b",
    response_content: str = "hello",
) -> Span:
    """Build an LLM span with computed ``messages_hash`` and a stored response."""
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        name="chat.completions.create",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        model_name=model,
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        messages_hash=hash_payload(messages),
        raw_attributes={
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.response": {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    """Empty SQLite store in a tmp directory."""
    return TraceStore(str(tmp_path / "agent_timetravel.db"))


@pytest.fixture
def trace_id() -> str:
    return "abcd1234abcd1234abcd1234abcd1234"


@pytest.fixture
def seeded_store(
    store: TraceStore, trace_id: str
) -> tuple[TraceStore, list[Span]]:
    """Seed ``store`` with a 2-LLM-span trace and return (store, spans)."""
    msgs_a = [{"role": "user", "content": "hello"}]
    msgs_b = [{"role": "user", "content": "follow up"}]
    spans = [
        _llm_span(trace_id, span_id="a" * 16, messages=msgs_a, response_content="hi"),
        _llm_span(trace_id, span_id="b" * 16, messages=msgs_b, response_content="bye"),
    ]
    trace = Trace(trace_id=trace_id, spans=spans)
    store.upsert_trace(trace)
    for s in spans:
        store.insert_span(s)
    return store, spans


def _sig(span: Span, *, messages_hash_override: str | None = None) -> CallSignature:
    """Build a ``CallSignature`` from an LLM ``span``, narrowing optionals.

    The seeded LLM spans always populate ``model_name`` and ``messages_hash``;
    this helper is where mypy learns that — every call goes through here.
    """
    assert span.model_name is not None
    assert span.messages_hash is not None
    return CallSignature(
        model=span.model_name,
        messages_hash=messages_hash_override or span.messages_hash,
        tools_hash=None,
    )


# ----------------------------------------------------------------------
# for_root
# ----------------------------------------------------------------------
def test_for_root_loads_existing_trace(
    seeded_store: tuple[TraceStore, list[Span]],
    trace_id: str,
) -> None:
    """``for_root`` eagerly hydrates the trace and exposes recorded_spans()."""
    store, spans = seeded_store
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.FROZEN, label="root"
    )
    assert session.trace_id == trace_id
    assert session.branch_id is not None
    assert isinstance(session.branch_id, UUID)
    assert session.mode is ReplayMode.FROZEN
    assert session.cursor == 0
    assert session.label == "root"
    assert session.forked_at is None
    recorded = session.recorded_spans()
    assert len(recorded) == 2
    assert [s.span_id for s in recorded] == [s.span_id for s in spans]


def test_for_root_raises_on_missing_trace(store: TraceStore, trace_id: str) -> None:
    """Missing traces surface as a ``ReplayError`` (not a KeyError)."""
    with pytest.raises(ReplayError, match=r"not found|no spans|missing"):
        ReplaySession.for_root(store, trace_id, mode=ReplayMode.FROZEN)


# ----------------------------------------------------------------------
# respond_or_forward
# ----------------------------------------------------------------------
def test_respond_or_forward_serves_cached_and_advances_cursor(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """Frozen replay: matching signature returns RecordedResponse, cursor += 1."""
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.FROZEN)
    sig = _sig(spans[0])
    recorded = session.respond_or_forward(sig)
    assert recorded is not None
    assert isinstance(recorded, RecordedResponse)
    assert recorded.span_id == spans[0].span_id
    assert recorded.payload == spans[0].raw_attributes
    assert session.cursor == 1


def test_respond_or_forward_frozen_raises_on_divergence(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """Frozen mode: a non-matching signature at the cursor is an error."""
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.FROZEN)
    wrong_sig = CallSignature(
        model=spans[0].model_name or "fallback",
        messages_hash="0" * 64,
        tools_hash=None,
    )
    with pytest.raises(ReplayError, match="divergence"):
        session.respond_or_forward(wrong_sig)


def test_respond_or_forward_frozen_raises_on_exhausted_cursor(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """Frozen mode: calling past the recorded tail raises."""
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.FROZEN)
    for s in spans:
        assert session.respond_or_forward(_sig(s)) is not None
    # Cursor exhausted — one more call should raise.
    overflow = CallSignature(model="x", messages_hash="y", tools_hash=None)
    with pytest.raises(ReplayError, match="exhausted"):
        session.respond_or_forward(overflow)


def test_respond_or_forward_branch_returns_none_on_divergence(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """BRANCH mode: divergence authorises a forward (returns None, no raise)."""
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.BRANCH)
    wrong = CallSignature(model="x", messages_hash="0" * 64, tools_hash=None)
    assert session.respond_or_forward(wrong) is None
    # Cursor does NOT advance on a forward (interceptor captures separately).
    assert session.cursor == 0


# ----------------------------------------------------------------------
# _signature_matches (model-agnostic)
# ----------------------------------------------------------------------
def test_signature_matches_ignores_model_name(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """A signature with a different model name but same messages hash still matches.

    This is the contract for cold-swap branching: replay a trace against a
    different LLM to compare outputs.
    """
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.FROZEN)
    sig = CallSignature(
        model="totally-different-model",
        messages_hash=spans[0].messages_hash or "fallback",
        tools_hash=None,
    )
    assert session.respond_or_forward(sig) is not None


def test_signature_matches_checks_tools_hash_when_present(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """If the signature supplies ``tools_hash`` it must match the span."""
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.FROZEN)
    sig = CallSignature(
        model=spans[0].model_name or "fallback",
        messages_hash=spans[0].messages_hash or "fallback",
        tools_hash="9" * 64,
    )
    with pytest.raises(ReplayError, match="divergence"):
        session.respond_or_forward(sig)


# ----------------------------------------------------------------------
# fork
# ----------------------------------------------------------------------
def test_fork_clones_prefix_under_new_branch_id(
    seeded_store: tuple[TraceStore, list[Span]],
    trace_id: str,
) -> None:
    """Forking at index 1 clones spans [0, 1) and persists them under a new branch.

    Phase 3 exit criterion: ``len(cloned) == branch_at`` and all cloned spans
    have fresh ``timetravel_id``s.
    """
    store, spans = seeded_store
    root = ReplaySession.for_root(store, trace_id, mode=ReplayMode.FROZEN)
    original_branch = root.branch_id

    forked = root.fork(branch_at=1, mode=ReplayMode.BRANCH, label="my fork")

    assert forked.branch_id != original_branch
    assert forked.mode is ReplayMode.BRANCH
    assert forked.label == "my fork"
    assert forked.forked_at == 1
    assert forked.cursor == 1

    # Branch row was inserted and links back to the parent.
    branches = store.list_branches(trace_id)
    branch_row = next(b for b in branches if b.branch_id == forked.branch_id)
    assert branch_row.parent_branch_id == original_branch
    assert branch_row.branch_at_index == 1

    # Prefix spans are inherited in-memory (no DB duplication).
    recorded = forked.recorded_spans()
    assert len(recorded) == 2
    # Storage-level: the fork has no own rows yet (only the Branch row exists).
    # The prefix surfaces through the inherited-union query (root spans + ).
    own_persisted = [
        s for s in store.get_spans(trace_id, branch_id=forked.branch_id)
        if s.span_id not in {sp.span_id for sp in spans}
    ]
    assert own_persisted == []


def test_fork_full_rerun_inherits_full_prefix_in_cache(
    seeded_store: tuple[TraceStore, list[Span]],
    trace_id: str,
) -> None:
    """FULL_RERUN forks inherit the whole prefix in cache (no DB duplication).

    The session is ready to drive from cursor 1 with full knowledge of the
    seed timeline, even though the live tail will be re-executed.
    """
    store, _spans = seeded_store
    root = ReplaySession.for_root(store, trace_id, mode=ReplayMode.FROZEN)
    forked = root.fork(branch_at=1, mode=ReplayMode.FULL_RERUN)
    assert len(forked.recorded_spans()) == 2
    persisted = store.get_spans(trace_id, branch_id=forked.branch_id)
    own = [s for s in persisted if s.span_id not in {sp.span_id for sp in _spans}]
    assert own == []


def test_fork_rejects_out_of_range_branch_at(
    seeded_store: tuple[TraceStore, list[Span]],
    trace_id: str,
) -> None:
    """``branch_at`` must be in ``[0, len(spans)]``."""
    store, _spans = seeded_store
    root = ReplaySession.for_root(store, trace_id, mode=ReplayMode.FROZEN)
    with pytest.raises(ReplayError, match="out of range"):
        root.fork(branch_at=99, mode=ReplayMode.BRANCH)
    with pytest.raises(ReplayError, match="out of range"):
        root.fork(branch_at=-1, mode=ReplayMode.BRANCH)


# ----------------------------------------------------------------------
# advance_cursor_to — tool-span lookup
# ----------------------------------------------------------------------
def test_advance_cursor_to_forwards_only(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """``advance_cursor_to`` accepts values ``>=`` current cursor only."""
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.FROZEN)
    session.advance_cursor_to(2)
    assert session.cursor == 2
    with pytest.raises(ReplayError, match="cannot timetravel"):
        session.advance_cursor_to(0)


def test_advance_cursor_to_bounds_checks(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """Index past the recorded span count raises."""
    store, spans = seeded_store
    session = ReplaySession.for_root(store, spans[0].trace_id, mode=ReplayMode.FROZEN)
    with pytest.raises(ReplayError, match=">"):
        session.advance_cursor_to(99)


# ----------------------------------------------------------------------
# record_new
# ----------------------------------------------------------------------
def test_record_new_appends_under_branch_id(
    seeded_store: tuple[TraceStore, list[Span]],
    trace_id: str,
) -> None:
    """``record_new`` persists a span under ``session.branch_id`` and advances cursor."""
    store, _spans = seeded_store
    session = ReplaySession.for_root(store, trace_id, mode=ReplayMode.BRANCH)
    live_span = _llm_span(
        trace_id,
        span_id="c" * 16,
        messages=[{"role": "user", "content": "live"}],
        response_content="live-resp",
    )
    session.record_new(live_span)
    persisted = store.get_spans(trace_id, branch_id=session.branch_id)
    # The live span is queryable under the session's branch.
    persisted_ids = {s.span_id for s in persisted}
    assert live_span.span_id in persisted_ids
    # Cursor has jumped to the new tail.
    assert session.cursor == len(session.recorded_spans())


def test_record_new_replaces_cached_span_for_repeated_timetravel_id(
    store: TraceStore,
    trace_id: str,
) -> None:
    """Repeated updates refresh one cached span without moving the cursor."""
    store.upsert_trace(Trace(trace_id=trace_id))
    session = ReplaySession.for_root(store, trace_id, mode=ReplayMode.BRANCH)
    live_span = _llm_span(
        trace_id,
        span_id="c" * 16,
        messages=[{"role": "user", "content": "live"}],
        response_content="first",
    )

    session.record_new(live_span)
    first_cursor = session.cursor
    live_span.raw_attributes["gen_ai.response"]["choices"][0]["message"][
        "content"
    ] = "second"
    session.record_new(live_span)

    assert len(session.recorded_spans()) == 1
    assert session.recorded_spans()[0].raw_attributes["gen_ai.response"]["choices"][0][
        "message"
    ]["content"] == "second"
    assert session.cursor == first_cursor


# ----------------------------------------------------------------------
# replay() context manager
# ----------------------------------------------------------------------
def test_replay_ctxmgr_sets_and_resets_active_session(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """``active_session()`` is set inside the block and cleared on exit."""
    store, spans = seeded_store
    assert active_session() is None
    with replay_ctx(store, spans[0].trace_id, mode=ReplayMode.FROZEN) as session:
        assert active_session() is session
    assert active_session() is None


def test_replay_ctxmgr_resets_even_on_exception(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """The active-session token is reset even when the body raises."""
    store, spans = seeded_store
    with pytest.raises(RuntimeError, match="boom"), \
            replay_ctx(store, spans[0].trace_id, mode=ReplayMode.FROZEN):
        raise RuntimeError("boom")
    assert active_session() is None


def test_replay_ctxmgr_forks_when_branch_at_given(
    seeded_store: tuple[TraceStore, list[Span]],
    trace_id: str,
) -> None:
    """``branch_at=N`` opens a forked session positioned at N."""
    store, _spans = seeded_store
    with replay_ctx(
        store, trace_id, branch_at=1, mode=ReplayMode.BRANCH, label="ctx"
    ) as session:
        assert session.forked_at == 1
        assert session.cursor == 1
        assert session.label == "ctx"


# ----------------------------------------------------------------------
# Branch isolation — concurrent sessions via contextvars
# ----------------------------------------------------------------------
def test_branch_isolation_concurrent_contextvars(
    seeded_store: tuple[TraceStore, list[Span]],
) -> None:
    """Two sessions on the same trace have independent cursors.

    Phase 5.5 eval-harness contract: replay sessions are scoped per task via
    :class:`contextvars.ContextVar`, so concurrent forks can't trample each
    other's cursor state.
    """
    import contextvars  # pylint: disable=import-outside-toplevel

    store, spans = seeded_store
    trace_id_val = spans[0].trace_id

    # Spawn each session in its own Context (simulates asyncio.Task).
    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()

    def _run_a() -> int:
        with replay_ctx(store, trace_id_val, mode=ReplayMode.FROZEN) as session:
            session.respond_or_forward(_sig(spans[0]))
            return session.cursor

    def _run_b() -> int:
        with replay_ctx(store, trace_id_val, mode=ReplayMode.FROZEN) as session:
            # Don't consume anything in B — its cursor stays at 0.
            return session.cursor

    cursor_a: int = ctx_a.run(_run_a)
    cursor_b: int = ctx_b.run(_run_b)
    assert cursor_a == 1
    assert cursor_b == 0
    # And after both contexts ran, the outer active_session is still None.
    assert active_session() is None
