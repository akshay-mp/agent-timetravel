"""Phase 3 Track 3B.2 — unit tests for the OpenAI interceptor.

Strategy
--------
We test the **dispatch logic** (``_dispatch_sync`` / ``_dispatch_async``)
directly by passing a fake ``orig_create`` callable that returns a canned
response. This isolates the frozen-serve / branch-forward / streaming-
failsclosed decisions without touching the real OpenAI HTTP client.

For the ``patch()`` context manager we install a *fake* ``openai.resources.
chat.completions`` module via ``sys.modules`` so install/uninstall and
idempotency can be exercised against a deterministic stub — real OpenAI
network behaviour is out of scope for unit tests.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from agent_timetravel.enums import ReplayMode, SpanKind, SpanStatus
from agent_timetravel.models import Span, Trace, hash_payload
from agent_timetravel.openai_intercept import (
    InterceptError,
    _dispatch_async,
    _dispatch_sync,
    _step_async,
    _step_sync,
    extract_signature,
    patch,
)
from agent_timetravel.replay import ReplayError, ReplaySession, active_session
from agent_timetravel.replay import replay as replay_ctx
from agent_timetravel.stepping import Decision, DecisionKind, SteppingStopped
from agent_timetravel.storage import TraceStore


# ----------------------------------------------------------------------
# Helpers / fixtures
# ----------------------------------------------------------------------
def _llm_span(
    trace_id: str,
    *,
    span_id: str,
    messages: list[dict[str, str]],
    model: str = "qwen3:32b",
    response_content: str = "hello",
) -> Span:
    """Build an LLM span carrying a stored ChatCompletion payload."""
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
    return TraceStore(str(tmp_path / "agent_timetravel.db"))


@pytest.fixture
def trace_id() -> str:
    return "abcd1234abcd1234abcd1234abcd1234"


@pytest.fixture
def seeded_store(
    store: TraceStore, trace_id: str
) -> tuple[TraceStore, list[Span], list[dict[str, str]]]:
    """Seed a 1-LLM-span trace and return (store, spans, messages)."""
    messages = [{"role": "user", "content": "hello"}]
    span = _llm_span(trace_id, span_id="a" * 16, messages=messages, response_content="hi")
    trace = Trace(trace_id=trace_id, spans=[span])
    store.upsert_trace(trace)
    store.insert_span(span)
    return store, [span], messages


def _fake_chat_completion(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }


class _SyncApproval:
    """Small synchronous approval channel for direct interceptor tests."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        self.steps: list[Any] = []

    def submit_sync(self, step: Any) -> Decision:
        self.steps.append(step)
        return self.decision


# ----------------------------------------------------------------------
# extract_signature
# ----------------------------------------------------------------------
def test_extract_signature_hashes_messages_and_tools() -> None:
    """``extract_signature`` returns deterministic hashes for a call."""
    msgs = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "search"}}]

    sig_no_tools = extract_signature(model="qwen3:32b", messages=msgs)
    sig_with_tools = extract_signature(model="qwen3:32b", messages=msgs, tools=tools)

    assert sig_no_tools.model == "qwen3:32b"
    assert sig_no_tools.messages_hash == hash_payload(msgs)
    assert sig_no_tools.tools_hash is None
    assert sig_with_tools.tools_hash == hash_payload(tools)
    # Different model name doesn't change messages_hash (model is logged separately).
    sig_other_model = extract_signature(model="gpt-4o", messages=msgs)
    assert sig_other_model.messages_hash == sig_no_tools.messages_hash


def test_extract_signature_empty_messages_yields_stable_hash() -> None:
    """Missing or empty messages produce a stable hash (not a crash)."""
    sig = extract_signature(model="x", messages=[])
    assert sig.messages_hash == hash_payload([])


def test_step_async_captures_structured_sampling_snapshot(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 3.1 keeps reproducibility-critical sampling fields on the step."""
    store, _spans, _messages = seeded_store
    session = ReplaySession.for_root(store, trace_id, mode=ReplayMode.BRANCH)
    captured: list[Any] = []

    async def capture_gate(_session: Any, step: Any) -> None:
        captured.append(step)
        return None

    monkeypatch.setattr("agent_timetravel.stepping.gate_async", capture_gate)

    kwargs: dict[str, Any] = {
        "model": "unsloth/gemma-4-12b-it-GGUF",
        "messages": [{"role": "user", "content": "compare RLHF and DPO"}],
        "temperature": 0.2,
        "seed": 42,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "tool_choice": "auto",
        "top_p": 0.9,
        "stream": True,
    }

    returned, step = asyncio.run(_step_async(session, kwargs))

    assert returned is kwargs
    assert captured == [step]
    assert step.payload["sampling"] == {
        "temperature": 0.2,
        "seed": 42,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "tool_choice": "auto",
        "top_p": 0.9,
    }
    assert step.payload["params"]["stream"] is True


def test_step_sync_captures_sampling_and_applies_edit(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """Sync OpenAI calls expose the same payload and honor gate edits."""
    store, _spans, _messages = seeded_store
    approval = _SyncApproval(
        Decision(
            kind=DecisionKind.EDIT,
            messages=[{"role": "user", "content": "edited prompt"}],
            model="edited-model",
            params={"temperature": 0.7},
        )
    )
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    kwargs: dict[str, Any] = {
        "model": "original-model",
        "messages": [{"role": "user", "content": "original prompt"}],
        "temperature": 0.2,
        "seed": 42,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "tool_choice": "auto",
    }

    returned, step = _step_sync(session, kwargs)

    assert approval.steps == [step]
    assert step.payload["sampling"] == {
        "temperature": 0.2,
        "seed": 42,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "tool_choice": "auto",
    }
    assert returned["messages"] == [{"role": "user", "content": "edited prompt"}]
    assert returned["model"] == "edited-model"
    assert returned["temperature"] == 0.7


def test_step_sync_honors_stop_decision(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """A sync STOP decision prevents the OpenAI call from proceeding."""
    store, _spans, _messages = seeded_store
    approval = _SyncApproval(Decision(kind=DecisionKind.STOP))
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )

    with pytest.raises(SteppingStopped):
        _step_sync(session, {"model": "model", "messages": []})


@pytest.mark.asyncio
async def test_dispatch_sync_worker_publishes_sse_review_usage(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """A worker-thread sync call completes through the browser channel."""
    from agent_timetravel.stepping_api import SSEApprovalChannel

    store, _spans, _messages = seeded_store
    channel = SSEApprovalChannel()
    channel.bind_loop(asyncio.get_running_loop())
    session = ReplaySession.for_root(
        store,
        trace_id,
        mode=ReplayMode.INTERACTIVE,
        approval=channel,
    )

    def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        return _fake_chat_completion("live-model", "worker result")

    call = asyncio.create_task(
        asyncio.to_thread(
            _dispatch_sync,
            object(),
            session,
            orig_create,
            (),
            {"model": "qwen3:32b", "messages": [{"role": "user", "content": "new prompt"}]},
        )
    )

    paused = await channel.next_event()
    assert paused["type"] == "paused"
    channel.decide(Decision(kind=DecisionKind.APPROVE))
    assert (await channel.next_event())["type"] == "dispatching"

    completed = await channel.next_event()
    assert completed["type"] == "step_completed"
    assert completed["usage"] == {
        "input_tokens": 9,
        "output_tokens": 3,
        "thinking_tokens": 0,
        "final_tokens": 3,
        "total_tokens": 12,
        "estimated": False,
    }
    channel.decide(Decision(kind=DecisionKind.APPROVE))
    response = await call
    body = response.model_dump() if hasattr(response, "model_dump") else response
    assert body["choices"][0]["message"]["content"] == "worker result"


# ----------------------------------------------------------------------
# _dispatch_sync (frozen serve / branch forward / streaming fail-closed)
# ----------------------------------------------------------------------
def test_dispatch_sync_serves_cached_payload_in_frozen(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """In FROZEN mode the dispatcher serves the recorded payload — no live call."""
    store, _spans, messages = seeded_store
    session = ReplaySession.for_root(store, trace_id, mode=ReplayMode.FROZEN)
    calls: list[Any] = []

    def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_chat_completion("live-caller", "LIVE")

    kwargs: dict[str, Any] = {"model": "qwen3:32b", "messages": messages}
    response = _dispatch_sync(object(), session, orig_create, (), kwargs)

    assert not calls  # No live call — entirely served from cache.
    body = response.model_dump() if hasattr(response, "model_dump") else response
    # The cached payload stores "hi" (from the seeded span).
    assert body["choices"][0]["message"]["content"] == "hi"
    assert session.cursor == 1


def test_dispatch_sync_branch_forwards_live_and_captures(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """In BRANCH mode a cache miss forwards live and records a new span."""
    store, _spans, _messages = seeded_store
    session = ReplaySession.for_root(store, trace_id, mode=ReplayMode.BRANCH)

    captured: dict[str, Any] = {}

    def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        captured["called"] = True
        return _fake_chat_completion("live-model", "live-body")

    # Different messages so the cache misses.
    new_messages = [{"role": "user", "content": "different prompt"}]
    kwargs: dict[str, Any] = {"model": "qwen3:32b", "messages": new_messages}
    response = _dispatch_sync(object(), session, orig_create, (), kwargs)

    assert captured.get("called") is True
    body = response.model_dump() if hasattr(response, "model_dump") else response
    assert body["choices"][0]["message"]["content"] == "live-body"
    # The live-captured span is appended; cursor moves past the new tail
    # (seed had 1 span, +1 captured → cache length 2 → cursor is 2).
    assert session.cursor == 2


def test_dispatch_sync_streaming_in_frozen_raises(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """``stream=True`` in FROZEN mode fails closed (Phase 5 streaming replay)."""
    store, _spans, messages = seeded_store
    session = ReplaySession.for_root(store, trace_id, mode=ReplayMode.FROZEN)

    def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("live call should not happen in frozen streaming test")

    kwargs: dict[str, Any] = {
        "model": "qwen3:32b",
        "messages": messages,
        "stream": True,
    }
    with pytest.raises(ReplayError, match="frozen streaming replay not yet supported"):
        _dispatch_sync(object(), session, orig_create, (), kwargs)


# ----------------------------------------------------------------------
# _dispatch_async
# ----------------------------------------------------------------------
def test_dispatch_async_serves_cached_payload_in_frozen(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """Async path mirrors sync: frozen replay returns the cached payload."""
    store, _spans, messages = seeded_store
    session = ReplaySession.for_root(store, trace_id, mode=ReplayMode.FROZEN)
    calls: list[Any] = []

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_chat_completion("live", "LIVE")

    kwargs: dict[str, Any] = {"model": "qwen3:32b", "messages": messages}
    response = asyncio.run(_dispatch_async(object(), session, orig_create, (), kwargs))
    assert not calls  # No live call — entirely served from cache.
    body = response.model_dump() if hasattr(response, "model_dump") else response
    assert body["choices"][0]["message"]["content"] == "hi"


# ----------------------------------------------------------------------
# patch() lifecycle — uses a stub openai module installed in sys.modules
# ----------------------------------------------------------------------
@contextmanager
def _fake_openai_module() -> Iterator[dict[str, Any]]:
    """Install a fake ``openai.resources.chat.completions`` module in sys.modules.

    Provides a ``Completions`` and ``AsyncCompletions`` class whose ``create``
    methods are *replaced* by ``patch()`` and then restored. This is the
    minimal surface ``patch()`` touches.
    """
    # Save any previously-imported real submodule so we can restore it.
    saved = {
        key: sys.modules.get(key)
        for key in (
            "openai",
            "openai.resources",
            "openai.resources.chat",
            "openai.resources.chat.completions",
            "openai.types",
            "openai.types.chat",
        )
    }
    try:
        # Build the fake class objects. Create a fresh attribute each call so
        # the ``__timetravel_patched__`` marker from a previous test doesn't leak.
        class Completions:
            def create(self, *args: Any, **kwargs: Any) -> Any:
                return {"_kind": "sync-original"}

        class AsyncCompletions:
            async def create(self, *args: Any, **kwargs: Any) -> Any:
                return {"_kind": "async-original"}

        completions_mod = types.ModuleType("openai.resources.chat.completions")
        completions_mod.Completions = Completions  # type: ignore[attr-defined]
        completions_mod.AsyncCompletions = AsyncCompletions  # type: ignore[attr-defined]

        chat_mod = types.ModuleType("openai.resources.chat")
        chat_mod.completions = completions_mod  # type: ignore[attr-defined]

        resources_mod = types.ModuleType("openai.resources")
        resources_mod.chat = chat_mod  # type: ignore[attr-defined]

        openai_mod = types.ModuleType("openai")
        openai_mod.resources = resources_mod  # type: ignore[attr-defined]

        sys.modules["openai"] = openai_mod
        sys.modules["openai.resources"] = resources_mod
        sys.modules["openai.resources.chat"] = chat_mod
        sys.modules["openai.resources.chat.completions"] = completions_mod
        # No types module → _chat_completion_module() returns None (deterministic).
        sys.modules.pop("openai.types", None)
        sys.modules.pop("openai.types.chat", None)

        yield {
            "Completions": Completions,
            "AsyncCompletions": AsyncCompletions,
        }
    finally:
        # Restore saved modules (or remove if they were absent).
        for key, mod in saved.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod


def test_patch_installs_and_restores() -> None:
    """``patch()`` installs the patched ``create`` and restores the original in exit."""
    with _fake_openai_module() as fake:
        Completions = fake["Completions"]
        original_create = Completions.create

        with patch():
            patched = Completions.create
            assert patched is not original_create
            assert getattr(patched, "__timetravel_patched__", False) is True

        # Restored.
        assert Completions.create is original_create


def test_patch_is_idempotent_nested() -> None:
    """Nested ``with patch():`` calls do not double-patch or fail to restore."""
    with _fake_openai_module() as fake:
        Completions = fake["Completions"]
        original = Completions.create

        with patch():
            inner_patched = Completions.create
            with patch():
                # Second patch is a no-op: same object as outer patch.
                assert Completions.create is inner_patched
                assert getattr(Completions.create, "__timetravel_patched__", False) is True
            # After inner exits we're still patched (outer is still active).
            assert Completions.create is inner_patched
        # Outer exit fully restores.
        assert Completions.create is original


def test_overlapping_patch_contexts_restore_only_after_the_last_exit() -> None:
    """A concurrent session keeps the process-global methods patched."""
    with _fake_openai_module() as fake:
        Completions = fake["Completions"]
        original = Completions.create
        first_entered = threading.Event()
        both_entered = threading.Barrier(2)
        first_exited = threading.Event()
        retained_patch: list[bool] = []

        def first() -> None:
            with patch():
                first_entered.set()
                both_entered.wait(timeout=5)
            first_exited.set()

        def second() -> None:
            first_entered.wait(timeout=5)
            with patch():
                both_entered.wait(timeout=5)
                first_exited.wait(timeout=5)
                retained_patch.append(Completions.create is not original)

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        second_thread.start()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert retained_patch == [True]
        assert Completions.create is original


def test_patch_restores_even_on_exception() -> None:
    """If the body raises, ``patch()`` still restores the originals."""
    with _fake_openai_module() as fake:
        Completions = fake["Completions"]
        original = Completions.create
        with pytest.raises(RuntimeError, match="boom"), patch():
            raise RuntimeError("boom")
        assert Completions.create is original


def test_patch_without_openai_raises_intercept_error() -> None:
    """If ``openai`` is uninstallable, ``patch()`` raises InterceptError.

    We simulate this by hiding the real openai from the importer for the
    duration of the test.
    """
    saved_openai = sys.modules.pop("openai", None)
    saved_resources = sys.modules.pop("openai.resources", None)
    saved_chat = sys.modules.pop("openai.resources.chat", None)
    saved_completions = sys.modules.pop("openai.resources.chat.completions", None)
    # Block the import path too.
    import builtins

    real_import = builtins.__import__

    def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai.resources.chat.completions":
            raise ImportError("simulated missing openai")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocking_import
    try:
        with pytest.raises(InterceptError, match="requires the `openai` package"), \
                patch():
            pass
    finally:
        builtins.__import__ = real_import
        for key, mod in [
            ("openai", saved_openai),
            ("openai.resources", saved_resources),
            ("openai.resources.chat", saved_chat),
            ("openai.resources.chat.completions", saved_completions),
        ]:
            if mod is not None:
                sys.modules[key] = mod


def test_patch_passthrough_when_no_active_session() -> None:
    """When no replay session is active, ``patch()`` forwards to the original create."""
    with _fake_openai_module() as fake:
        Completions = fake["Completions"]
        instance = Completions()
        with patch():
            # No active replay context — call goes straight to the original.
            assert active_session() is None
            result = instance.create(model="x", messages=[])
            assert result == {"_kind": "sync-original"}


def test_patch_routes_through_replay_when_active(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """An active replay context routes ``create`` through the dispatcher."""
    store, _spans, messages = seeded_store
    with _fake_openai_module() as fake:
        Completions = fake["Completions"]
        instance = Completions()
        with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
            assert active_session() is not None
            response = instance.create(model="qwen3:32b", messages=messages)
            body = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            # Served from cache — payload content is "hi", not original.
            assert body["choices"][0]["message"]["content"] == "hi"


# ----------------------------------------------------------------------
# Live streaming capture (reasoning deltas)
# ----------------------------------------------------------------------
class _FakeAsyncStream:
    """Minimal async-iterable of ChatCompletionChunk-like objects."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _FakeAsyncStream:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _stream_chunk(
    content: str | None = None,
    *,
    reasoning: str | None = None,
    usage: dict[str, int] | None = None,
    finish_reason: str | None = None,
) -> Any:
    """Build a chunk shaped like the SDK's ChatCompletionChunk (duck-typed)."""
    empty = content is None and reasoning is None and finish_reason is None
    choice: Any | None = None
    if not empty:
        delta = types.SimpleNamespace(
            content=content, reasoning_content=reasoning, reasoning=None, tool_calls=None
        )
        choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return types.SimpleNamespace(
        id="chatcmpl-stream",
        created=123,
        model="unsloth/gemma-4-12b-it-GGUF",
        usage=usage,
        choices=[choice] if choice is not None else [],
    )


def _tool_call_chunk(
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    """Build a chunk carrying a tool-call delta fragment."""
    call = types.SimpleNamespace(
        index=index,
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )
    delta = types.SimpleNamespace(
        content=None, reasoning_content=None, reasoning=None, tool_calls=[call]
    )
    choice = types.SimpleNamespace(delta=delta, finish_reason=None)
    return types.SimpleNamespace(
        id="chatcmpl-stream",
        created=123,
        model="unsloth/gemma-4-12b-it-GGUF",
        usage=None,
        choices=[choice],
    )


class _StreamingApproval:
    """Approval channel double that records reasoning deltas and completions."""

    def __init__(self) -> None:
        self.deltas: list[tuple[int, str]] = []
        self.completed: list[tuple[str, dict[str, int] | None]] = []
        self.step_cursor: int | None = None

    async def submit(self, step: Any) -> Decision:
        self.step_cursor = step.cursor
        return Decision(kind=DecisionKind.APPROVE)

    async def complete(
        self, step: Any, result: str, usage: dict[str, int] | None = None
    ) -> Decision:
        self.completed.append((result, usage))
        return Decision(kind=DecisionKind.APPROVE)

    def emit_delta(self, cursor: int, chunk: str) -> None:
        self.deltas.append((cursor, chunk))


def _dispatch_live(
    session: ReplaySession,
    orig_create: Any,
    messages: list[dict[str, str]] | None = None,
) -> Any:
    """Run one live-forwarded async dispatch with distinct-from-cache messages."""
    kwargs: dict[str, Any] = {
        "model": "unsloth/gemma-4-12b-it-GGUF",
        "messages": messages or [{"role": "user", "content": "compare RLHF and DPO"}],
    }
    return asyncio.run(_dispatch_async(object(), session, orig_create, (), kwargs))


def _body(response: Any) -> dict[str, Any]:
    return response.model_dump() if hasattr(response, "model_dump") else response


def test_dispatch_async_streams_inline_thinking_deltas(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live forwards stream <think> fragments and reassemble the same response."""
    import agent_timetravel.openai_intercept as intercept_module

    store, _spans, _messages = seeded_store
    monkeypatch.setattr(intercept_module, "_REASONING_DELTA_INTERVAL_S", 0.0)
    approval = _StreamingApproval()
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    seen: list[dict[str, Any]] = []
    chunks = [
        _stream_chunk("<th"),  # Opener split across chunk boundaries.
        _stream_chunk("ink>should compare stability first"),
        _stream_chunk("</think>"),
        _stream_chunk("Final answer: DPO is cheaper.", finish_reason="stop"),
        _stream_chunk(usage={"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}),
    ]

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return _FakeAsyncStream(chunks)

    response = _dispatch_live(session, orig_create)

    # The request was transparently upgraded to a usage-reporting stream.
    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"]["include_usage"] is True
    # Reasoning fragments were published while the stream was consumed…
    streamed = "".join(chunk for _cursor, chunk in approval.deltas)
    assert streamed == "should compare stability first"
    assert approval.deltas
    assert all(cursor == approval.step_cursor for cursor, _chunk in approval.deltas)
    # …and the caller still receives the reassembled non-streaming response.
    body = _body(response)
    assert body["choices"][0]["message"]["content"] == (
        "<think>should compare stability first</think>Final answer: DPO is cheaper."
    )
    assert body["usage"]["total_tokens"] == 14
    # The verify loop saw the full <think> block and real provider usage.
    result, usage = approval.completed[0]
    assert result.startswith("<think>should compare stability first</think>")
    assert usage is not None
    assert usage["estimated"] is False
    assert usage["total_tokens"] == 14


def test_dispatch_async_streams_separate_reasoning_field(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Providers that emit reasoning_content deltas stream without <think> tags."""
    import agent_timetravel.openai_intercept as intercept_module

    store, _spans, _messages = seeded_store
    monkeypatch.setattr(intercept_module, "_REASONING_DELTA_INTERVAL_S", 0.0)
    approval = _StreamingApproval()
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    chunks = [
        _stream_chunk(reasoning="deep thought "),
        _stream_chunk(reasoning="about DPO"),
        _stream_chunk("DPO skips the reward model.", finish_reason="stop"),
        _stream_chunk(usage={"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}),
    ]

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        return _FakeAsyncStream(chunks)

    response = _dispatch_live(session, orig_create)

    streamed = "".join(chunk for _cursor, chunk in approval.deltas)
    assert streamed == "deep thought about DPO"
    body = _body(response)
    assert body["choices"][0]["message"]["content"] == "DPO skips the reward model."
    assert body["choices"][0]["message"]["reasoning_content"] == "deep thought about DPO"
    result, _usage = approval.completed[0]
    assert result == "<think>deep thought about DPO</think>\nDPO skips the reward model."


def test_dispatch_async_no_thinking_model_emits_no_deltas(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """A plain model streams no reasoning; the response round-trips untouched."""
    store, _spans, _messages = seeded_store
    approval = _StreamingApproval()
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    chunks = [
        _stream_chunk("plain "),
        _stream_chunk("answer", finish_reason="stop"),
    ]

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        return _FakeAsyncStream(chunks)

    response = _dispatch_live(session, orig_create)

    assert approval.deltas == []
    assert _body(response)["choices"][0]["message"]["content"] == "plain answer"


def test_dispatch_async_stream_capture_kill_switch(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENT_TIMETRAVEL_DISABLE_STREAM_CAPTURE forwards the call verbatim."""
    store, _spans, _messages = seeded_store
    monkeypatch.setenv("AGENT_TIMETRAVEL_DISABLE_STREAM_CAPTURE", "1")
    approval = _StreamingApproval()
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    seen: list[dict[str, Any]] = []

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return _fake_chat_completion("live", "LIVE")

    response = _dispatch_live(session, orig_create)

    assert "stream" not in seen[0]
    assert _body(response)["choices"][0]["message"]["content"] == "LIVE"
    assert approval.deltas == []


def test_dispatch_async_plain_channel_forwards_without_streaming(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """Channels without emit_delta (e.g. in-process test doubles) stay as-is."""

    class _PlainApproval(_StreamingApproval):
        emit_delta = None  # type: ignore[assignment]

    store, _spans, _messages = seeded_store
    approval = _PlainApproval()
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    seen: list[dict[str, Any]] = []

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return _fake_chat_completion("live", "LIVE")

    response = _dispatch_live(session, orig_create)

    assert "stream" not in seen[0]
    assert _body(response)["choices"][0]["message"]["content"] == "LIVE"


def test_dispatch_async_reassembles_tool_call_deltas(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """Tool-call fragments streamed across chunks rebuild the full call."""
    store, _spans, _messages = seeded_store
    approval = _StreamingApproval()
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    chunks = [
        _tool_call_chunk(0, call_id="call_1", name="search"),
        _tool_call_chunk(0, arguments='{"q": '),
        _tool_call_chunk(0, arguments='"rlhf"}'),
        _stream_chunk(finish_reason="tool_calls"),
    ]

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        return _FakeAsyncStream(chunks)

    response = _dispatch_live(session, orig_create)

    body = _body(response)
    assert body["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "rlhf"}'},
        }
    ]
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_capture_only_observes_without_gating_or_spans(
    seeded_store: tuple[TraceStore, list[Span], list[dict[str, str]]],
    trace_id: str,
) -> None:
    """Wire-level observe mode: forward verbatim, stash raw, no gate/span."""
    from agent_timetravel.openai_intercept import capture_only, last_wire_raw

    store, _spans, messages = seeded_store
    approval = _StreamingApproval()
    session = ReplaySession.for_root(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=approval
    )
    seen: list[dict[str, Any]] = []

    async def orig_create(_self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return _fake_chat_completion("live", "LIVE")

    kwargs: dict[str, Any] = {"model": "qwen3:32b", "messages": messages}
    spans_before = len(store.get_spans(trace_id))

    def scenario() -> Any:
        with capture_only():
            response = asyncio.run(
                _dispatch_async(object(), session, orig_create, (), kwargs)
            )
            wire = last_wire_raw()
        return response, wire

    response, wire = asyncio.run(asyncio.to_thread(scenario))

    assert seen == [{"model": "qwen3:32b", "messages": messages}]  # Verbatim.
    assert approval.step_cursor is None  # No gate surfaced.
    assert approval.completed == []
    assert len(store.get_spans(trace_id)) == spans_before  # No span added.
    body = _body(response)
    assert body["choices"][0]["message"]["content"] == "LIVE"
    assert wire is not None
    assert wire["gen_ai.response"]["choices"][0]["message"]["content"] == "LIVE"
