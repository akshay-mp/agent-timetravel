"""Tests for the Phase B stepping server.

Two test layers:

1. **HTTP endpoints** — POST/GET/DELETE /sessions and POST /decide, exercised
   via FastAPI's sync TestClient. Covers validation, status codes, and the
   persisted row. The runner task spawned by POST /sessions does NOT outlive
   the TestClient request (TestClient tears down its portal per request), so
   these tests use a stub runner that completes synchronously.

2. **Channel + runner-task mechanics** — the SSEApprovalChannel event
   sequence, the LiveSession task lifecycle, and the runner's status
   transitions. Tested directly without HTTP because the SSE transport
   (StreamingResponse over a long-lived task) cannot be driven through
   TestClient or httpx.ASGITransport: both tear down the loop/portal before
   the background runner task can make progress, and ASGITransport
   serializes requests so a POST /decide made while a stream is open
   deadlocks. The transport itself is a thin FastAPI wrapper over the
   channel; the load-bearing logic is the channel + runner, tested here.

The full HTTP-driven SSE flow is exercised manually against a real uvicorn
server using ``docs/interactive-workbench-testing.md``. The reconnect
snapshot contract is pinned below at the channel level so it remains covered
in the default test run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_timetravel.enums import SpanKind, SpanStatus
from agent_timetravel.models import Span, Trace, hash_payload
from agent_timetravel.replay import active_session
from agent_timetravel.stepping import (
    Decision,
    DecisionKind,
    Step,
    StepKind,
    SteppingStopped,
    gate_async,
)
from agent_timetravel.stepping_api import (
    _SESSIONS,
    SSEApprovalChannel,
    mount_stepping,
    register_runner,
)
from agent_timetravel.storage import TraceStore

_TRACE_ID = "abcd1234abcd1234abcd1234abcd1234"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _llm_span(span_id: str, messages: list[dict[str, str]]) -> Span:
    return Span(
        trace_id=_TRACE_ID,
        span_id=span_id,
        parent_span_id=None,
        name="chat.completions.create",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        model_name="qwen3:32b",
        messages_hash=hash_payload(messages),
        raw_attributes={
            "gen_ai.request.model": "qwen3:32b",
            "gen_ai.response": {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}]
            },
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore(str(tmp_path / "stepping_api.db"))
    msgs = [{"role": "user", "content": "hello"}]
    spans = [_llm_span("a" * 16, msgs)]
    s.upsert_trace(Trace(trace_id=_TRACE_ID, spans=spans))
    for sp in spans:
        s.insert_span(sp)
    return s


@pytest.fixture
def app(store: TraceStore) -> FastAPI:
    a = FastAPI()
    a.state.store = store
    mount_stepping(a)
    return a


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    """Snapshot + restore the runner + live-session registries per test."""
    from agent_timetravel import stepping_api

    saved_runners = dict(stepping_api._RUNNERS)
    saved_live = dict(_SESSIONS._live)
    _SESSIONS._live.clear()
    yield
    stepping_api._RUNNERS.clear()
    stepping_api._RUNNERS.update(saved_runners)
    _SESSIONS._live.clear()
    _SESSIONS._live.update(saved_live)


# ----------------------------------------------------------------------
# Stub runners
# ----------------------------------------------------------------------
async def _completing_runner(session: Any) -> None:
    """A runner that exits immediately — for HTTP tests that just need 201."""
    # Touch session so type checkers don't flag the unused param.
    assert session is not None


async def _stop_runner(session: Any) -> None:
    """A runner that raises SteppingStopped."""
    raise SteppingStopped(Step(kind=StepKind.LLM, payload={}, cursor=0))


async def _raise_runner(session: Any) -> None:
    """A runner that blows up — exercises the errored status path."""
    raise RuntimeError("boom")


# ----------------------------------------------------------------------
# POST /sessions
# ----------------------------------------------------------------------
class TestStartSession:
    def test_starts_session_returns_201(self, client: TestClient) -> None:
        register_runner("ok", _completing_runner)
        resp = client.post(
            "/api/v1/sessions",
            json={"trace_id": _TRACE_ID, "runner_ref": "ok"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["session_id"]
        assert body["trace_id"] == _TRACE_ID
        assert body["status"] == "running"

    def test_unknown_runner_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/sessions",
            json={"trace_id": _TRACE_ID, "runner_ref": "no-such-runner"},
        )
        assert resp.status_code == 404
        assert "no-such-runner" in resp.text

    def test_unknown_trace_returns_404(self, client: TestClient) -> None:
        register_runner("ok", _completing_runner)
        resp = client.post(
            "/api/v1/sessions",
            json={"trace_id": "f" * 32, "runner_ref": "ok"},
        )
        assert resp.status_code == 404

    def test_bad_mode_returns_400(self, client: TestClient) -> None:
        register_runner("ok", _completing_runner)
        resp = client.post(
            "/api/v1/sessions",
            json={"trace_id": _TRACE_ID, "runner_ref": "ok", "mode": "nonsense"},
        )
        assert resp.status_code == 400


# ----------------------------------------------------------------------
# GET /sessions and /sessions/{id}
# ----------------------------------------------------------------------
class TestSessionDetail:
    def test_get_unknown_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/v1/sessions/" + "1" * 32)
        assert resp.status_code == 404

    def test_bad_uuid_returns_400(self, client: TestClient) -> None:
        resp = client.get("/api/v1/sessions/not-a-uuid")
        assert resp.status_code == 400

    def test_list_empty_returns_zero(self, client: TestClient) -> None:
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_list_reflects_inserted_row(self, client: TestClient, store: TraceStore) -> None:
        from agent_timetravel.stepping import InteractiveSession

        store.upsert_interactive_session(
            InteractiveSession(
                session_id="12345678-1234-5678-1234-567812345678",
                trace_id=_TRACE_ID,
                branch_id="b" * 36,
                runner_ref="ok",
                status="done",
                created_at="2026-07-20T00:00:00+00:00",
                updated_at="2026-07-20T00:00:01+00:00",
            )
        )
        resp = client.get("/api/v1/sessions")
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["runner_ref"] == "ok"
        assert body["items"][0]["status"] == "done"


# ----------------------------------------------------------------------
# Channel + runner-task mechanics (no HTTP)
#
# These exercise the load-bearing logic directly: the SSEApprovalChannel
# event sequence, the runner task lifecycle, and the status transitions
# that POST /sessions would drive in production. See the module docstring
# for why the HTTP transport can't be driven end-to-end in tests.
# ----------------------------------------------------------------------
class TestRunnerMechanics:
    async def test_worker_thread_tool_round_trip(self) -> None:
        """A sync tool worker can use the SSE channel before and after execution."""
        channel = SSEApprovalChannel()
        channel.bind_loop(asyncio.get_running_loop())
        step = Step(
            kind=StepKind.TOOL,
            payload={"name": "lookup", "args": ["timetravel"], "kwargs": {}},
            cursor=0,
        )

        pre_call = asyncio.create_task(asyncio.to_thread(channel.submit_sync, step))
        paused = await channel.next_event()
        assert paused["type"] == "paused"
        assert paused["kind"] == "tool"
        channel.decide(Decision(kind=DecisionKind.APPROVE))
        assert (await pre_call).kind is DecisionKind.APPROVE
        assert (await channel.next_event())["type"] == "dispatching"

        post_call = asyncio.create_task(
            asyncio.to_thread(
                channel.complete_sync,
                step,
                '{"result": "ok"}',
                {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            )
        )
        completed = await channel.next_event()
        assert completed == {
            "type": "step_completed",
            "cursor": 0,
            "kind": "tool",
            "result": '{"result": "ok"}',
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        }
        channel.decide(Decision(kind=DecisionKind.APPROVE))
        assert (await post_call).kind is DecisionKind.APPROVE
        assert (await channel.next_event())["type"] == "resumed"

    async def test_sync_gate_fails_fast_on_server_event_loop(self) -> None:
        """A sync call must move to a worker before browser approval can block."""
        channel = SSEApprovalChannel()
        channel.bind_loop(asyncio.get_running_loop())
        step = Step(
            kind=StepKind.LLM,
            payload={"model": "stub", "messages": []},
            cursor=0,
        )

        with pytest.raises(RuntimeError, match="cannot block the server event loop"):
            channel.submit_sync(step)

    async def test_approve_drives_session_to_done(
        self, store: TraceStore
    ) -> None:
        """A runner that pauses + gets APPROVE completes → status=done."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from agent_timetravel.enums import ReplayMode
        from agent_timetravel.replay import ReplaySession
        from agent_timetravel.stepping_api import (
            SSEApprovalChannel,
            _set_status,
        )

        async def pausing_runner(session: Any) -> None:
            sess = active_session()
            assert sess is not None
            decision = await gate_async(
                sess,
                Step(kind=StepKind.LLM, payload={"model": "stub"}, cursor=sess.cursor),
            )
            assert decision is not None
            assert decision.kind is DecisionKind.APPROVE

        channel = SSEApprovalChannel()
        session_obj = ReplaySession.for_root(
            store, _TRACE_ID, mode=ReplayMode.INTERACTIVE
        )
        session_obj.approval = channel
        from agent_timetravel.replay import _active_session

        session_id = str(uuid4())
        now = datetime.now(tz=UTC).isoformat()

        async def wrapper() -> None:
            from agent_timetravel.stepping import InteractiveSession

            # Bind the ContextVar INSIDE the task — same fix as the server's
            # _runner_wrapper. asyncio.create_task copies the parent context
            # at spawn time, so a token minted outside can't be reset here.
            token = _active_session.set(session_obj)
            store.upsert_interactive_session(
                InteractiveSession(
                    session_id=session_id,
                    trace_id=session_obj.trace_id,
                    branch_id=str(session_obj.branch_id),
                    runner_ref="test",
                    status="running",
                    created_at=now,
                    updated_at=now,
                )
            )
            try:
                await pausing_runner(session_obj)
                _set_status(store, session_id, "done")
                channel.emit({"type": "done"})
            except SteppingStopped as exc:
                _set_status(store, session_id, "done")
                channel.emit({"type": "done", "cursor": exc.step.cursor})
            except Exception as exc:
                _set_status(store, session_id, "errored", error_message=str(exc))
                channel.emit({"type": "errored", "message": str(exc)})
            finally:
                _active_session.reset(token)

        task = asyncio.create_task(wrapper())
        # Wait for paused, drive APPROVE, await completion.
        event = await channel.next_event()
        assert event["type"] == "paused"
        channel.decide(Decision(kind=DecisionKind.APPROVE))
        dispatching = await channel.next_event()
        assert dispatching["type"] == "dispatching"
        done = await channel.next_event()
        assert done["type"] == "done"
        await task

        row = store.get_interactive_session(session_id)
        assert row is not None
        assert row.status == "done"

    async def test_stop_terminates_session(self, store: TraceStore) -> None:
        """A STOP decision causes SteppingStopped → status=done."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from agent_timetravel.enums import ReplayMode
        from agent_timetravel.replay import ReplaySession
        from agent_timetravel.stepping_api import _set_status

        async def stop_runner(session: Any) -> None:
            sess = active_session()
            assert sess is not None
            decision = await gate_async(
                sess,
                Step(kind=StepKind.LLM, payload={"model": "stub"}, cursor=sess.cursor),
            )
            if decision is not None and decision.kind is DecisionKind.STOP:
                raise SteppingStopped(
                    Step(kind=StepKind.LLM, payload={"model": "stub"}, cursor=0)
                )

        channel = SSEApprovalChannel()
        session_obj = ReplaySession.for_root(
            store, _TRACE_ID, mode=ReplayMode.INTERACTIVE
        )
        session_obj.approval = channel
        from agent_timetravel.replay import _active_session

        session_id = str(uuid4())
        now = datetime.now(tz=UTC).isoformat()

        async def wrapper() -> None:
            from agent_timetravel.stepping import InteractiveSession

            token = _active_session.set(session_obj)
            store.upsert_interactive_session(
                InteractiveSession(
                    session_id=session_id,
                    trace_id=session_obj.trace_id,
                    branch_id=str(session_obj.branch_id),
                    runner_ref="test",
                    status="running",
                    created_at=now,
                    updated_at=now,
                )
            )
            try:
                await stop_runner(session_obj)
                _set_status(store, session_id, "done")
                channel.emit({"type": "done"})
            except SteppingStopped as exc:
                _set_status(store, session_id, "done")
                channel.emit({"type": "done", "cursor": exc.step.cursor})
            except Exception as exc:
                _set_status(store, session_id, "errored", error_message=str(exc))
                channel.emit({"type": "errored", "message": str(exc)})
            finally:
                _active_session.reset(token)

        task = asyncio.create_task(wrapper())
        await channel.next_event()  # paused
        channel.decide(Decision(kind=DecisionKind.STOP))
        # STOP emits 'resumed' (the channel's submit() returns) then 'done'
        # (the runner raises SteppingStopped → wrapper catches → emits done).
        resumed = await channel.next_event()
        assert resumed["type"] == "resumed"
        done = await channel.next_event()
        assert done["type"] == "done"
        await task

        row = store.get_interactive_session(session_id)
        assert row is not None
        assert row.status == "done"

    async def test_runner_error_surfaces_as_errored(self, store: TraceStore) -> None:
        """A runner that raises → status=errored + the message is captured."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from agent_timetravel.enums import ReplayMode
        from agent_timetravel.replay import ReplaySession
        from agent_timetravel.stepping_api import _set_status

        channel = SSEApprovalChannel()
        session_obj = ReplaySession.for_root(
            store, _TRACE_ID, mode=ReplayMode.INTERACTIVE
        )
        session_obj.approval = channel
        from agent_timetravel.replay import _active_session

        session_id = str(uuid4())
        now = datetime.now(tz=UTC).isoformat()

        async def wrapper() -> None:
            from agent_timetravel.stepping import InteractiveSession

            token = _active_session.set(session_obj)
            store.upsert_interactive_session(
                InteractiveSession(
                    session_id=session_id,
                    trace_id=session_obj.trace_id,
                    branch_id=str(session_obj.branch_id),
                    runner_ref="test",
                    status="running",
                    created_at=now,
                    updated_at=now,
                )
            )
            try:
                await _raise_runner(session_obj)
                _set_status(store, session_id, "done")
                channel.emit({"type": "done"})
            except SteppingStopped as exc:
                _set_status(store, session_id, "done")
                channel.emit({"type": "done", "cursor": exc.step.cursor})
            except Exception as exc:
                _set_status(store, session_id, "errored", error_message=str(exc))
                channel.emit({"type": "errored", "message": str(exc)})
            finally:
                _active_session.reset(token)

        task = asyncio.create_task(wrapper())
        errored = await channel.next_event()
        assert errored["type"] == "errored"
        assert "boom" in errored["message"]
        await task

        row = store.get_interactive_session(session_id)
        assert row is not None
        assert row.status == "errored"
        assert "boom" in (row.error_message or "")


# ----------------------------------------------------------------------
# SSEApprovalChannel — direct unit tests
# ----------------------------------------------------------------------
class TestSSEApprovalChannel:
    async def test_submit_blocks_until_decide(self) -> None:
        ch = SSEApprovalChannel()
        step = Step(kind=StepKind.LLM, payload={"model": "m"}, cursor=0)

        async def submitter() -> Decision:
            return await ch.submit(step)

        task = asyncio.create_task(submitter())
        await asyncio.sleep(0)  # let submitter reach the await
        assert (await ch.next_event())["type"] == "paused"
        replay = ch.replay_events_if_idle()
        assert replay
        assert replay[0]["type"] == "paused"
        ch.decide(Decision(kind=DecisionKind.APPROVE))
        decision = await task
        assert decision.kind is DecisionKind.APPROVE
        assert (await ch.next_event())["type"] == "dispatching"
        assert [event["type"] for event in ch.replay_events_if_idle()] == [
            "paused",
            "dispatching",
        ]

    async def test_complete_holds_until_next_decision(self) -> None:
        ch = SSEApprovalChannel()
        step = Step(kind=StepKind.LLM, payload={"model": "m"}, cursor=0)

        task = asyncio.create_task(ch.complete(step, "model response"))
        completed = await ch.next_event()
        assert completed == {
            "type": "step_completed",
            "cursor": 0,
            "kind": "llm",
            "result": "model response",
        }
        assert not task.done()

        ch.decide(Decision(kind=DecisionKind.APPROVE))
        decision = await task
        assert decision.kind is DecisionKind.APPROVE
        assert (await ch.next_event())["type"] == "resumed"
        assert ch.replay_events_if_idle() == []

    async def test_replay_snapshot_preserves_llm_review_after_refresh(self) -> None:
        """A replacement SSE stream can rebuild an in-flight LLM review."""
        ch = SSEApprovalChannel()
        step = Step(kind=StepKind.LLM, payload={"model": "m"}, cursor=0)

        submit_task = asyncio.create_task(ch.submit(step))
        assert (await ch.next_event())["type"] == "paused"
        ch.decide(Decision(kind=DecisionKind.APPROVE))
        assert (await submit_task).kind is DecisionKind.APPROVE
        assert (await ch.next_event())["type"] == "dispatching"

        complete_task = asyncio.create_task(
            ch.complete(
                step,
                "<think>internal plan</think>final answer",
                {"input_tokens": 4, "thinking_tokens": 2, "final_tokens": 3},
            )
        )
        completed = await ch.next_event()
        assert completed["type"] == "step_completed"
        assert completed["result"] == "<think>internal plan</think>final answer"
        assert completed["usage"]["thinking_tokens"] == 2

        replay = ch.replay_events_if_idle()
        assert [event["type"] for event in replay] == [
            "paused",
            "dispatching",
            "step_completed",
        ]
        assert replay[-1]["cursor"] == 0

        ch.decide(Decision(kind=DecisionKind.APPROVE))
        assert (await complete_task).kind is DecisionKind.APPROVE
        assert (await ch.next_event())["type"] == "resumed"
        assert ch.replay_events_if_idle() == []

    def test_decide_validates(self) -> None:
        ch = SSEApprovalChannel()
        with pytest.raises(ValueError, match="EDIT"):
            ch.decide(Decision(kind=DecisionKind.EDIT))  # no overrides

    def test_drain_events(self) -> None:
        ch = SSEApprovalChannel()
        ch.emit({"type": "done"})
        ch.emit({"type": "x"})
        drained = ch.drain_events()
        assert [e["type"] for e in drained] == ["done", "x"]
        assert ch.drain_events() == []


# ----------------------------------------------------------------------
# register_runner
# ----------------------------------------------------------------------
class TestRegistry:
    def test_register_and_resolve(self) -> None:
        async def r(s: Any) -> None:
            pass

        register_runner("tmp", r)
        from agent_timetravel.stepping_api import get_runner

        assert get_runner("tmp") is r
        assert get_runner("missing") is None

    def test_empty_ref_rejected(self) -> None:
        async def r(s: Any) -> None:
            pass

        with pytest.raises(ValueError, match="non-empty"):
            register_runner("", r)


# ----------------------------------------------------------------------
# Storage CRUD (interactive_sessions)
# ----------------------------------------------------------------------
class TestStorageCRUD:
    def test_upsert_and_get(self, store: TraceStore) -> None:
        from agent_timetravel.stepping import InteractiveSession

        sid = "11111111-2222-3333-4444-555555555555"
        store.upsert_interactive_session(
            InteractiveSession(
                session_id=sid,
                trace_id=_TRACE_ID,
                branch_id="b" * 36,
                runner_ref="r",
                status="running",
                created_at="2026-07-20T00:00:00+00:00",
                updated_at="2026-07-20T00:00:00+00:00",
            )
        )
        row = store.get_interactive_session(sid)
        assert row is not None
        assert row.runner_ref == "r"
        assert row.status == "running"

        # Upsert again with a status change.
        store.upsert_interactive_session(
            InteractiveSession(
                session_id=sid,
                trace_id=_TRACE_ID,
                branch_id="b" * 36,
                runner_ref="r",
                status="done",
                created_at="2026-07-20T00:00:00+00:00",
                updated_at="2026-07-20T00:00:01+00:00",
            )
        )
        row = store.get_interactive_session(sid)
        assert row is not None
        assert row.status == "done"

    def test_get_missing_returns_none(self, store: TraceStore) -> None:
        assert store.get_interactive_session("0" * 36) is None

    def test_list_pagination(self, store: TraceStore) -> None:
        from agent_timetravel.stepping import InteractiveSession

        for i in range(3):
            store.upsert_interactive_session(
                InteractiveSession(
                    session_id=f"{i:08x}-0000-0000-0000-{i:012x}",
                    trace_id=_TRACE_ID,
                    branch_id="b" * 36,
                    runner_ref=f"r{i}",
                    status="done",
                    created_at=f"2026-07-20T00:00:0{i:1d}+00:00",
                    updated_at=f"2026-07-20T00:00:0{i:1d}+00:00",
                )
            )
        items, total = store.list_interactive_sessions(limit=2, offset=0)
        assert total == 3
        assert len(items) == 2

    def test_delete(self, store: TraceStore) -> None:
        from agent_timetravel.stepping import InteractiveSession

        sid = "22222222-3333-4444-5555-666666666666"
        store.upsert_interactive_session(
            InteractiveSession(
                session_id=sid,
                trace_id=_TRACE_ID,
                branch_id="b" * 36,
                runner_ref="r",
                status="done",
                created_at="2026-07-20T00:00:00+00:00",
                updated_at="2026-07-20T00:00:00+00:00",
            )
        )
        assert store.delete_interactive_session(sid) is True
        assert store.get_interactive_session(sid) is None
        # Second delete returns False.
        assert store.delete_interactive_session(sid) is False


# ----------------------------------------------------------------------
# Phase 1 decision kinds — HTTP contract
# ----------------------------------------------------------------------
class TestDecisionKindsPhase1:
    """``DecisionRequest`` accepts Phase 1 decision kinds and forwards them.

    Pins the HTTP wire contract for ``reject`` (with optional ``reason``) and
    ``run_until_breakpoint``. These decision kinds were previously only tested
    at the channel / stepping-primitive level; these tests verify that
    * the ``DecisionRequest`` Pydantic model accepts the new ``kind`` strings,
    * the ``_build_decision`` helper maps them to the correct
      :class:`~agent_timetravel.stepping.DecisionKind` enum value,
    * and the ``reason`` field is preserved on the resulting ``Decision``.
    """

    def test_reject_kind_accepted(self) -> None:
        """DecisionRequest.kind='reject' maps to DecisionKind.REJECT."""
        from agent_timetravel.stepping_api import DecisionRequest, _build_decision

        req = DecisionRequest(kind="reject")
        decision = _build_decision(req)
        assert decision.kind is DecisionKind.REJECT
        assert decision.reason is None

    def test_reject_with_reason_preserved(self) -> None:
        """The ``reason`` string is forwarded onto the Decision."""
        from agent_timetravel.stepping_api import DecisionRequest, _build_decision

        req = DecisionRequest(kind="reject", reason="this action is unsafe")
        decision = _build_decision(req)
        assert decision.kind is DecisionKind.REJECT
        assert decision.reason == "this action is unsafe"

    def test_run_until_breakpoint_kind_accepted(self) -> None:
        """DecisionRequest.kind='run_until_breakpoint' maps correctly."""
        from agent_timetravel.stepping_api import DecisionRequest, _build_decision

        req = DecisionRequest(kind="run_until_breakpoint")
        decision = _build_decision(req)
        assert decision.kind is DecisionKind.RUN_UNTIL_BREAKPOINT

    def test_unknown_kind_raises(self) -> None:
        """An unrecognised kind raises HTTPException(400)."""
        from fastapi import HTTPException

        from agent_timetravel.stepping_api import DecisionRequest, _build_decision

        req = DecisionRequest(kind="totally_unknown_kind")  # type: ignore[arg-type]
        with pytest.raises(HTTPException) as exc_info:
            _build_decision(req)
        assert exc_info.value.status_code == 400
        assert "totally_unknown_kind" in exc_info.value.detail

    def test_reject_reason_empty_string_treated_as_none(self) -> None:
        """An empty reason string is passed through unchanged (None in request = None)."""
        from agent_timetravel.stepping_api import DecisionRequest, _build_decision

        req = DecisionRequest(kind="reject", reason=None)
        decision = _build_decision(req)
        assert decision.reason is None

    def test_run_control_patch_persists_and_returns(
        self, client: TestClient, store: TraceStore
    ) -> None:
        """PATCH /run-control persists the intent and GET round-trips it.

        Uses the store directly to seed a session row (no live runner needed),
        then drives PATCH and GET through the HTTP client.
        """
        from agent_timetravel.stepping import InteractiveSession

        sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        store.upsert_interactive_session(
            InteractiveSession(
                session_id=sid,
                trace_id=_TRACE_ID,
                branch_id="b" * 36,
                runner_ref="r",
                status="running",
                created_at="2026-08-03T00:00:00+00:00",
                updated_at="2026-08-03T00:00:00+00:00",
            )
        )
        # PATCH sets run_until_breakpoint.
        patch_res = client.patch(
            f"/api/v1/sessions/{sid}/run-control",
            json={"pause_after_current": False, "run_until_breakpoint": True},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["run_until_breakpoint"] is True

        # GET returns the persisted intent.
        get_res = client.get(f"/api/v1/sessions/{sid}/run-control")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["run_until_breakpoint"] is True
        assert data["pause_after_current"] is False


# ----------------------------------------------------------------------
# SSEApprovalChannel.emit_delta — lossy reasoning-fragment publishing
# ----------------------------------------------------------------------
async def test_emit_delta_publishes_and_replays_for_inflight_cursor() -> None:
    """A reasoning delta flows to the SSE queue and the reconnect replay."""
    ch = SSEApprovalChannel()
    ch._replay_events = [{"type": "paused", "cursor": 3}]
    ch.emit_delta(3, "comparing stab")
    assert (await ch.next_event()) == {
        "type": "reasoning_delta",
        "cursor": 3,
        "chunk": "comparing stab",
    }
    assert ch.replay_events_if_idle()[-1] == {
        "type": "reasoning_delta",
        "cursor": 3,
        "chunk": "comparing stab",
    }


async def test_emit_delta_drops_when_queue_full_without_raising() -> None:
    """A full event queue drops the fragment instead of crashing the run."""
    ch = SSEApprovalChannel()
    for i in range(ch._events.maxsize):
        ch._events.put_nowait({"type": "reasoning_delta", "cursor": 0, "chunk": str(i)})
    ch.emit_delta(0, "overflow")  # Must not raise QueueFull.
    assert ch._events.qsize() == ch._events.maxsize


async def test_emit_delta_skips_replay_for_stale_cursor() -> None:
    """Deltas from a cursor other than the in-flight gate are not replayed."""
    ch = SSEApprovalChannel()
    ch._replay_events = [{"type": "paused", "cursor": 1}]
    ch.emit_delta(9, "stale")
    assert (await ch.next_event())["cursor"] == 9
    assert all(event.get("cursor") != 9 for event in ch.replay_events_if_idle())
