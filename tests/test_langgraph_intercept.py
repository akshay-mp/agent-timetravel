"""Generic LangGraph / langchain-core workbench interceptor tests.

Gated on ``langchain_core`` being importable (the ``langgraph`` extra).

Exercises the class-level ``BaseChatModel`` / ``BaseTool`` patches:

* gate approve / edit / stop for LLM and tool calls (sync + async);
* replay hit from a recorded span (zero model calls) in FROZEN mode;
* frozen divergence failing closed with :class:`~timetravel.replay.ReplayError`;
* live-forward capture of LLM and TOOL spans in INTERACTIVE sessions;
* the ``replay_chat_model`` / ``@timetravel.tool()`` double-handling guards;
* patch restore and nested-patch idempotency.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from pathlib import Path
from typing import Any

import pytest

from agent_timetravel.enums import ReplayMode, SpanKind
from agent_timetravel.models import Span, Trace, hash_payload
from agent_timetravel.replay import replay as replay_ctx
from agent_timetravel.stepping import (
    AsyncioChannel,
    Decision,
    DecisionKind,
    Step,
    ThreadBridgeChannel,
)
from agent_timetravel.storage import TraceStore

if not importlib.util.find_spec("langchain_core"):
    pytest.skip("langchain-core not installed", allow_module_level=True)

# pylint: disable=import-outside-toplevel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool

from agent_timetravel.langgraph_intercept import _is_replay_wrapper, _is_timetravel_tool, patch

# pylint: enable=import-outside-toplevel


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _fake_model() -> tuple[BaseChatModel, list[list[dict[str, Any]]]]:
    """Build a deterministic echo model plus its call tracker.

    The tracker lives in a closure — ``BaseChatModel`` is a pydantic model,
    so undeclared per-instance attributes cannot be assigned.
    """
    tracker: list[list[dict[str, Any]]] = []

    class _FakeChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            tracker.append([m.model_dump() for m in messages])
            return _echo(messages)

        async def _agenerate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            tracker.append([m.model_dump() for m in messages])
            return _echo(messages)

    return _FakeChatModel(), tracker


def _echo(messages: Any) -> ChatResult:
    content = f"echo:{messages[-1].content}"
    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content=content,
                    usage_metadata={
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "total_tokens": 8,
                    },
                )
            )
        ]
    )


def _tool_call_model(
    tool_calls: list[dict[str, Any]],
) -> tuple[BaseChatModel, list[list[dict[str, Any]]]]:
    """Build a model whose responses are pure tool-call decisions, plus tracker."""

    def _result() -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[dict(call) for call in tool_calls],
                        usage_metadata={
                            "input_tokens": 5,
                            "output_tokens": 3,
                            "total_tokens": 8,
                        },
                    )
                )
            ]
        )

    tracker: list[list[dict[str, Any]]] = []

    class _ToolCallModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-tools"

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            tracker.append([m.model_dump() for m in messages])
            return _result()

        async def _agenerate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            tracker.append([m.model_dump() for m in messages])
            return _result()

    return _ToolCallModel(), tracker


class _ForbiddenChannel(AsyncioChannel):
    """Channel that fails the test if any step reaches the gate."""

    async def submit(self, step: Step) -> Decision:
        raise AssertionError(f"gate must not fire, got step {step.kind}")


def _seed_trace(store: TraceStore, trace_id: str, spans: list[Span]) -> None:
    store.upsert_trace(Trace(trace_id=trace_id, spans=spans))
    for span in spans:
        store.insert_span(span)


def _recorded_llm_span(
    trace_id: str,
    messages: list[dict[str, Any]],
    *,
    content: str = "recorded",
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id="a" * 16,
        parent_span_id=None,
        name="langchain.fake",
        kind=SpanKind.LLM,
        model_name="fake",
        messages_hash=hash_payload(messages),
        raw_attributes={
            "gen_ai.request.model": "fake",
            "gen_ai.response": {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        },
    )


def _start_approver(channel: ThreadBridgeChannel, decision: Decision) -> threading.Thread:
    """Daemon thread resolving the next paused step with ``decision``."""

    def approve() -> None:
        while channel.take_step() is None:
            continue
        channel.decide(decision)

    thread = threading.Thread(target=approve, daemon=True)
    thread.start()
    return thread


async def _approve_all(channel: AsyncioChannel, decision: Decision) -> None:
    """Drain the channel, answering every step with ``decision``."""
    while True:
        await channel.next_step()
        channel.decide(decision)


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(str(tmp_path / "langgraph.db"))


@pytest.fixture
def trace_id() -> str:
    return "abcd1234abcd1234abcd1234abcd1234"


@pytest.fixture(autouse=True)
def _unpatched() -> None:
    """Force-restore the class patches even if a test leaks its context."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel import langgraph_intercept as intercept_module
    # pylint: enable=import-outside-toplevel

    yield
    with intercept_module._PATCH_LOCK:
        for (cls, name), original in intercept_module._ORIGINALS.items():
            setattr(cls, name, original)
        intercept_module._ORIGINALS.clear()
        intercept_module._PATCH_DEPTH = 0


# ----------------------------------------------------------------------
# Passthrough + patch lifecycle
# ----------------------------------------------------------------------
def test_patch_passthrough_without_session() -> None:
    model, tracker = _fake_model()
    with patch():
        message = model.invoke([HumanMessage(content="hi")])
    assert message.content == "echo:hi"
    assert len(tracker) == 1


def test_patch_restores_originals() -> None:
    original = BaseChatModel.invoke
    with patch():
        assert BaseChatModel.invoke is not original
    assert BaseChatModel.invoke is original


def test_patch_is_idempotent_nested() -> None:
    original = BaseChatModel.invoke
    with patch():
        with patch():
            assert BaseChatModel.invoke is not original
        assert BaseChatModel.invoke is not original
    assert BaseChatModel.invoke is original


# ----------------------------------------------------------------------
# LLM path
# ----------------------------------------------------------------------
def test_async_invoke_gates_and_captures(store: TraceStore, trace_id: str) -> None:
    _seed_trace(store, trace_id, [])
    model, tracker = _fake_model()
    channel = AsyncioChannel()

    async def scenario() -> str:
        approver = asyncio.create_task(_approve_all(channel, Decision(kind=DecisionKind.APPROVE)))
        with patch(), replay_ctx(
            store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
        ) as session:
            await model.ainvoke([HumanMessage(content="hello")])
        approver.cancel()
        return str(session.branch_id)

    from uuid import UUID
    branch_id = UUID(asyncio.run(scenario()))
    assert len(tracker) == 1
    spans = store.get_spans(trace_id, branch_id=branch_id)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == SpanKind.LLM
    assert span.name == "langchain.fake"
    assert span.messages_hash == hash_payload(
        [HumanMessage(content="hello").model_dump()]
    )
    assert span.raw_attributes["gen_ai.response"]["choices"][0]["message"][
        "content"
    ] == "echo:hello"
    assert span.prompt_tokens == 5
    assert span.completion_tokens == 3


def test_async_invoke_edit_rewrites_outbound_messages(
    store: TraceStore, trace_id: str
) -> None:
    _seed_trace(store, trace_id, [])
    model, tracker = _fake_model()
    channel = AsyncioChannel()
    edited = [{"role": "user", "content": "edited prompt"}]

    async def scenario() -> str:
        async def editor() -> None:
            await _approve_all(
                channel,
                Decision(kind=DecisionKind.EDIT, messages=edited),
            )

        approver = asyncio.create_task(editor())
        with patch(), replay_ctx(
            store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
        ) as session:
            await model.ainvoke([HumanMessage(content="original")])
        approver.cancel()
        return str(session.branch_id)

    from uuid import UUID

    branch_id = UUID(asyncio.run(scenario()))
    assert tracker[0][-1]["content"] == "edited prompt"
    # The captured span must describe the edited call, not the original one.
    span = store.get_spans(trace_id, branch_id=branch_id)[0]
    assert span.messages_hash == hash_payload(edited)


def test_async_invoke_stop_unwinds(store: TraceStore, trace_id: str) -> None:
    from agent_timetravel.stepping import SteppingStopped

    _seed_trace(store, trace_id, [])
    model, tracker = _fake_model()
    channel = AsyncioChannel()

    async def scenario() -> None:
        async def stopper() -> None:
            await _approve_all(channel, Decision(kind=DecisionKind.STOP))

        approver = asyncio.create_task(stopper())
        try:
            with patch(), replay_ctx(
                store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
            ):
                await model.ainvoke([HumanMessage(content="halt")])
        finally:
            approver.cancel()

    with pytest.raises(SteppingStopped):
        asyncio.run(scenario())
    assert tracker == []


def test_frozen_replay_serves_recorded_message(store: TraceStore, trace_id: str) -> None:
    messages = [HumanMessage(content="replay me").model_dump()]
    seed = _recorded_llm_span(trace_id, messages, content="recorded-answer")
    _seed_trace(store, trace_id, [seed])
    model, tracker = _fake_model()

    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
        message = asyncio.run(model.ainvoke([HumanMessage(content="replay me")]))

    assert message.content == "recorded-answer"
    assert message.usage_metadata is not None
    assert message.usage_metadata["total_tokens"] == 8
    assert tracker == []  # zero outbound calls


def test_frozen_divergence_fails_closed(store: TraceStore, trace_id: str) -> None:
    from agent_timetravel.replay import ReplayError

    messages = [HumanMessage(content="expected").model_dump()]
    _seed_trace(store, trace_id, [_recorded_llm_span(trace_id, messages)])
    model, tracker = _fake_model()

    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN), pytest.raises(
        ReplayError
    ):
        asyncio.run(model.ainvoke([HumanMessage(content="divergent")]))
    assert tracker == []  # the divergence never reached the live model


def test_manual_replay_wrapper_skips_the_gate(store: TraceStore, trace_id: str) -> None:
    from agent_timetravel.adapters.langgraph import replay_chat_model

    _seed_trace(store, trace_id, [])
    inner, inner_tracker = _fake_model()
    model = replay_chat_model(inner)
    channel = _ForbiddenChannel()

    async def scenario() -> str:
        with patch(), replay_ctx(
            store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
        ):
            message = await model.ainvoke([HumanMessage(content="wrapped")])
        return message.content

    assert asyncio.run(scenario()) == "echo:wrapped"
    assert _is_replay_wrapper(model)
    assert len(inner_tracker) == 1


# ----------------------------------------------------------------------
# Tool path
# ----------------------------------------------------------------------
def test_tool_mock_returns_result_without_live_call(
    store: TraceStore, trace_id: str
) -> None:
    _seed_trace(store, trace_id, [])
    calls: list[Any] = []

    def lookup(query: str) -> str:
        """Look something up."""
        calls.append(query)
        return "live"

    tool = StructuredTool.from_function(func=lookup, name="lookup")
    channel = ThreadBridgeChannel()

    with patch(), replay_ctx(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
    ):
        approver = _start_approver(
            channel, Decision(kind=DecisionKind.MOCK, mock_result="mocked")
        )
        # StructuredTool.ainvoke runs the sync invoke in an executor, so the
        # pause surfaces on the thread bridge — the production sync-tool path.
        result = asyncio.run(tool.ainvoke({"query": "q"}))
        approver.join(timeout=2)

    assert result == "mocked"
    assert calls == []


def test_tool_reject_returns_structured_refusal(
    store: TraceStore, trace_id: str
) -> None:
    _seed_trace(store, trace_id, [])

    def danger(command: str) -> str:
        """Run a dangerous command."""
        return "ran"

    tool = StructuredTool.from_function(func=danger, name="danger")
    channel = ThreadBridgeChannel()

    with patch(), replay_ctx(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
    ):
        approver = _start_approver(
            channel, Decision(kind=DecisionKind.REJECT, reason="too risky")
        )
        result = asyncio.run(tool.ainvoke({"command": "rm -rf"}))
        approver.join(timeout=2)
    assert result == {
        "timetravel": "tool rejected",
        "tool": "danger",
        "reason": "too risky",
    }


def test_tool_approve_captures_tool_span(store: TraceStore, trace_id: str) -> None:
    _seed_trace(store, trace_id, [])
    calls: list[str] = []

    def search(query: str) -> str:
        """Search for results."""
        calls.append(query)
        return f"results-for:{query}"

    tool = StructuredTool.from_function(func=search, name="search")
    channel = ThreadBridgeChannel()

    with patch(), replay_ctx(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
    ) as session:
        approver = _start_approver(channel, Decision(kind=DecisionKind.APPROVE))
        result = asyncio.run(tool.ainvoke({"query": "rlhf"}))
        approver.join(timeout=2)
        branch_id = session.branch_id
    assert result == "results-for:rlhf"
    assert calls == ["rlhf"]
    span = store.get_spans(trace_id, branch_id=branch_id)[0]
    assert span.kind == SpanKind.TOOL
    assert span.name == "search"
    assert span.raw_attributes["gen_ai.tool.output"] == "results-for:rlhf"


def test_timetravel_tool_wrapper_defers_to_its_own_dispatch(
    store: TraceStore, trace_id: str
) -> None:
    from agent_timetravel.tool_intercept import tool as timetravel_tool

    _seed_trace(store, trace_id, [])
    calls: list[str] = []

    @timetravel_tool(name="owned")
    def owned(query: str) -> str:
        """An owned tool."""
        calls.append(query)
        return "owned-result"

    tool = StructuredTool.from_function(func=owned, name="owned")
    assert _is_timetravel_tool(tool)

    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE) as session:
        result = asyncio.run(tool.ainvoke({"query": "q"}))
        branch_id = session.branch_id

    assert result == "owned-result"
    assert calls == ["q"]
    # Exactly one TOOL span — the @timetravel.tool() path — despite both
    # interception layers being active.
    tool_spans = [
        s for s in store.get_spans(trace_id, branch_id=branch_id)
        if s.kind == SpanKind.TOOL
    ]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "owned"


# ----------------------------------------------------------------------
# Review regression tests: tool-call-preserving replay, edited-arg
# execution, Command round-trip, non-JSON tool args.
# ----------------------------------------------------------------------
def test_frozen_replay_preserves_tool_calls(store: TraceStore, trace_id: str) -> None:
    """A recorded tool-call decision replays as a real tool-call message."""
    _seed_trace(store, trace_id, [])
    decision = [
        {
            "name": "write_todos",
            "args": {"todos": ["plan", "research"]},
            "id": "call_1",
            "type": "tool_call",
        }
    ]

    capture_model, capture_tracker = _tool_call_model(decision)
    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE):
        asyncio.run(capture_model.ainvoke([HumanMessage(content="plan please")]))
    assert len(capture_tracker) == 1

    spans = store.get_trace(trace_id).spans
    assert len(spans) == 1
    assistant = spans[0].raw_attributes["gen_ai.response"]["choices"][0]["message"]
    assert assistant["tool_calls"][0]["name"] == "write_todos"
    assert assistant["tool_calls"][0]["args"] == {"todos": ["plan", "research"]}

    replay_model, replay_tracker = _tool_call_model(decision)
    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
        message = asyncio.run(replay_model.ainvoke([HumanMessage(content="plan please")]))

    assert replay_tracker == []  # zero outbound calls
    assert message.content == ""
    assert [
        {key: call[key] for key in ("name", "args", "id")} for call in message.tool_calls
    ] == [
        {"name": "write_todos", "args": {"todos": ["plan", "research"]}, "id": "call_1"}
    ]


def test_frozen_replay_parses_openai_wire_tool_calls(store: TraceStore, trace_id: str) -> None:
    """Spans captured by the OpenAI interceptor (function/arguments) replay too."""
    messages = [HumanMessage(content="wire").model_dump()]
    span = _recorded_llm_span(trace_id, messages)
    span.raw_attributes["gen_ai.response"]["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call_9",
            "type": "function",
            "function": {"name": "search", "arguments": "{\"query\": \"dpo\"}"},
        }
    ]
    _seed_trace(store, trace_id, [span])
    model, tracker = _fake_model()

    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
        message = asyncio.run(model.ainvoke([HumanMessage(content="wire")]))

    assert tracker == []
    assert message.tool_calls[0]["name"] == "search"
    assert message.tool_calls[0]["args"] == {"query": "dpo"}
    assert message.tool_calls[0]["id"] == "call_9"


def test_llm_result_text_summarizes_tool_call_steps() -> None:
    from agent_timetravel.langgraph_intercept import _llm_result_text

    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_todos", "args": {"todos": ["a"]}, "id": "c1", "type": "tool_call"}
        ],
    )
    assert _llm_result_text(message).startswith("→ write_todos(")
    assert _llm_result_text(AIMessage(content="plain answer")) == "plain answer"


def test_edited_tool_args_reach_the_live_tool(store: TraceStore, trace_id: str) -> None:
    """EDIT rewrites what the tool actually executes, not just the view."""
    _seed_trace(store, trace_id, [])
    calls: list[str] = []

    def search(query: str) -> str:
        """Search for results."""
        calls.append(query)
        return f"results-for:{query}"

    tool = StructuredTool.from_function(func=search, name="search")
    channel = ThreadBridgeChannel()

    with patch(), replay_ctx(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
    ):
        approver = _start_approver(
            channel,
            Decision(kind=DecisionKind.EDIT, args=[{"query": "edited-query"}]),
        )
        result = asyncio.run(tool.ainvoke({"query": "original"}))
        approver.join(timeout=2)

    assert result == "results-for:edited-query"
    assert calls == ["edited-query"]
    span = store.get_trace(trace_id).spans[0]
    assert span.kind == SpanKind.TOOL
    assert span.raw_attributes["gen_ai.tool.input"]["args"][0] == {
        "query": "edited-query"
    }


@pytest.mark.skipif(
    not importlib.util.find_spec("langgraph"), reason="langgraph not installed"
)
def test_command_tool_output_round_trips_on_replay(
    store: TraceStore, trace_id: str
) -> None:
    """A live ``Command`` stores an envelope; replay rebuilds a ``Command``."""
    from langgraph.types import Command

    _seed_trace(store, trace_id, [])

    # Return annotation is Any: StructuredTool resolves annotations in the
    # function's module globals, where Command isn't imported.
    def write_todos(todos: list[str]) -> Any:
        """Update the todo state."""
        return Command(update={"todos": todos})

    tool = StructuredTool.from_function(func=write_todos, name="write_todos")

    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE):
        live = asyncio.run(tool.ainvoke({"todos": ["plan"]}))

    assert isinstance(live, Command)
    assert live.update == {"todos": ["plan"]}
    stored = store.get_trace(trace_id).spans[0].raw_attributes["gen_ai.tool.output"]
    assert "__timetravel_command__" in stored
    assert stored["__timetravel_command__"]["update"] == {"todos": ["plan"]}

    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
        replayed = asyncio.run(tool.ainvoke({"todos": ["plan"]}))

    assert isinstance(replayed, Command)
    assert replayed.update == {"todos": ["plan"]}


def test_non_json_tool_inputs_serialize_and_forward_unchanged(
    store: TraceStore, trace_id: str
) -> None:
    """ToolRuntime-style objects serialize for the view; the live call gets the original.

    langchain 1.x injects runtime state into the parsed tool input dict (the
    ``ToolRuntime`` objects deepagents tools receive), so the fixture rides in
    the input — exactly where the real crash came from.
    """
    _seed_trace(store, trace_id, [])

    class _Runtime:
        def __repr__(self) -> str:
            return "<ToolRuntime fixture>"

    received: dict[str, Any] = {}

    class _LooseTool(BaseTool):
        name: str = "loose"
        description: str = "Accepts anything."

        def _run(
            self, payload: Any = None, runtime: Any = None, run_manager: Any = None, **_: Any
        ) -> str:
            received["payload"] = payload
            received["runtime"] = runtime
            return "ok"

    tool = _LooseTool()
    channel = ThreadBridgeChannel()

    with patch(), replay_ctx(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
    ):
        approver = _start_approver(channel, Decision(kind=DecisionKind.APPROVE))
        runtime = _Runtime()
        result = tool.invoke({"payload": "probe", "runtime": runtime})
        approver.join(timeout=2)

    assert result == "ok"
    assert received["payload"] == "probe"
    assert received["runtime"] is runtime  # identity preserved through the gate
    span = store.get_trace(trace_id).spans[0]
    assert span.raw_attributes["gen_ai.tool.input"]["args"][0]["runtime"] == (
        "<ToolRuntime fixture>"
    )


def test_edited_tool_args_restore_runtime_objects(store: TraceStore, trace_id: str) -> None:
    """Editing a runtime-injected tool keeps the ToolRuntime object.

    The debugger shows ``ToolRuntime`` as its repr; the edited JSON echoes
    that string back. The dispatch must restore the original object (only
    the genuinely edited values change) so the live tool still executes.
    """
    _seed_trace(store, trace_id, [])

    class _Runtime:
        tool_call_id = "call_42"

        def __repr__(self) -> str:
            return "<ToolRuntime fixture>"

    received: dict[str, Any] = {}

    class _RuntimeTool(BaseTool):
        name: str = "stateful"
        description: str = "Uses injected runtime."

        def _run(self, payload: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
            runtime = kwargs.get("runtime")
            received["runtime"] = runtime
            received["tool_call_id"] = getattr(runtime, "tool_call_id", None)
            received["payload"] = payload
            return "ok"

    tool = _RuntimeTool()
    channel = ThreadBridgeChannel()
    runtime = _Runtime()

    with patch(), replay_ctx(
        store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
    ):
        approver = _start_approver(
            channel,
            Decision(
                kind=DecisionKind.EDIT,
                # Exactly what the browser sends back: args with the edited
                # payload and the runtime object's repr string.
                args=[{"payload": "edited", "runtime": repr(runtime)}],
            ),
        )
        result = tool.invoke({"payload": "original", "runtime": runtime})
        approver.join(timeout=2)

    assert result == "ok"
    assert received["payload"] == "edited"  # the edit executed
    assert received["runtime"] is runtime  # untouched object restored
    assert received["tool_call_id"] == "call_42"


def test_named_non_tool_messages_are_stripped_for_strict_servers(
    store: TraceStore, trace_id: str
) -> None:
    """vLLM-style servers 422 on `name` for non-tool roles; strip on forward."""
    from langchain_core.messages import ToolMessage

    _seed_trace(store, trace_id, [])
    model, tracker = _fake_model()

    with patch(), replay_ctx(store, trace_id, mode=ReplayMode.BRANCH):
        asyncio.run(
            model.ainvoke(
                [
                    HumanMessage(content="hi", name="researcher"),
                    ToolMessage(content="result", name="search", tool_call_id="c1"),
                ]
            )
        )

    assert len(tracker) == 1
    sent = tracker[0]
    assert sent[0]["name"] is None  # human message stripped
    assert sent[1]["name"] == "search"  # tool message keeps its name


# ----------------------------------------------------------------------
# Provider reasoning (reasoning_content) display + persistence
# ----------------------------------------------------------------------
def test_llm_result_text_wraps_reasoning_content() -> None:
    """Separate reasoning_content lands in the <think> display convention."""
    from agent_timetravel.langgraph_intercept import _llm_result_text

    message = AIMessage(
        content="DPO skips the reward model.",
        additional_kwargs={"reasoning_content": "compare parameter counts"},
    )
    assert _llm_result_text(message) == (
        "<think>compare parameter counts</think>\nDPO skips the reward model."
    )


def test_llm_result_text_reasoning_only_message() -> None:
    """A message whose text lives entirely in reasoning_content still shows."""
    from agent_timetravel.langgraph_intercept import _llm_result_text

    message = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "pure reasoning"},
    )
    assert _llm_result_text(message) == "<think>pure reasoning</think>"


def test_llm_result_text_keeps_inline_think_block() -> None:
    """Inline <think> content is left untouched — no double wrapping."""
    from agent_timetravel.langgraph_intercept import _llm_result_text

    message = AIMessage(
        content="<think>inline</think>answer",
        additional_kwargs={"reasoning_content": "ignored"},
    )
    assert _llm_result_text(message) == "<think>inline</think>answer"


def test_async_invoke_persists_reasoning_in_span(
    store: TraceStore, trace_id: str
) -> None:
    """A live capture keeps reasoning_content in the recorded wire message."""
    from langchain_core.messages import HumanMessage

    from agent_timetravel.enums import ReplayMode
    from agent_timetravel.langgraph_intercept import patch
    from agent_timetravel.replay import replay as replay_ctx

    _seed_trace(store, trace_id, [])
    model, tracker = _fake_model()
    # Inject reasoning through additional_kwargs on the async path the
    # scenario actually exercises (ainvoke -> _agenerate).
    original_agenerate = type(model)._agenerate  # type: ignore[attr-defined]

    async def _agenerate_with_reasoning(self: Any, messages: Any, **kwargs: Any) -> Any:
        result = await original_agenerate(self, messages, **kwargs)
        result.generations[0].message.additional_kwargs["reasoning_content"] = (
            "plan before answering"
        )
        return result

    type(model)._agenerate = _agenerate_with_reasoning  # type: ignore[method-assign]
    channel = AsyncioChannel()

    async def scenario() -> str:
        approver = asyncio.create_task(
            _approve_all(channel, Decision(kind=DecisionKind.APPROVE))
        )
        with patch(), replay_ctx(
            store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
        ) as session:
            await model.ainvoke([HumanMessage(content="hello")])
        approver.cancel()
        return str(session.branch_id)

    branch_id = asyncio.run(scenario())
    spans = store.get_spans(trace_id, branch_id=branch_id)
    stored = spans[0].raw_attributes["gen_ai.response"]["choices"][0]["message"]
    assert stored["reasoning_content"] == "plan before answering"


def test_materialise_message_restores_reasoning(
    store: TraceStore, trace_id: str
) -> None:
    """Frozen replay rebuilds reasoning into additional_kwargs for display."""
    from langchain_core.messages import HumanMessage

    from agent_timetravel.enums import ReplayMode
    from agent_timetravel.enums import SpanStatus
    from agent_timetravel.langgraph_intercept import (
        _materialise_message,
        patch,
    )
    from agent_timetravel.models import Span, hash_payload
    from agent_timetravel.replay import RecordedResponse

    span = Span(
        trace_id=trace_id,
        span_id="b" * 16,
        parent_span_id=None,
        name="langchain.fake",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        model_name="fake",
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
        messages_hash=hash_payload(
            [HumanMessage(content="hello").model_dump()]
        ),
        raw_attributes={
            "gen_ai.response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "final answer",
                            "reasoning_content": "recorded reasoning",
                        }
                    }
                ]
            }
        },
    )
    message = _materialise_message(
        RecordedResponse(
            payload=span.raw_attributes,
            span_id=span.span_id,
            timetravel_id=span.trace_id,
            model=span.model_name,
        )
    )
    assert message.additional_kwargs["reasoning_content"] == "recorded reasoning"
    from agent_timetravel.langgraph_intercept import _llm_result_text

    assert _llm_result_text(message) == (
        "<think>recorded reasoning</think>\nfinal answer"
    )


def test_async_invoke_merges_wire_reasoning_into_span_and_result(
    store: TraceStore, trace_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire-captured reasoning reaches both the span payload and the UI text."""
    from langchain_core.messages import HumanMessage

    from agent_timetravel.enums import ReplayMode
    from agent_timetravel.langgraph_intercept import patch
    from agent_timetravel.openai_intercept import _wire_raw_holder, capture_only
    from agent_timetravel.replay import replay as replay_ctx

    _seed_trace(store, trace_id, [])
    model, _tracker = _fake_model()
    channel = AsyncioChannel()

    async def ainvoke_with_wire(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        # Simulate the OpenAI SDK layer observing the call under capture-only.
        holder = _wire_raw_holder()
        if holder is not None:
            holder["raw"] = {
                "gen_ai.request.model": "wire-model",
                "gen_ai.response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "wire-level reasoning",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"q": "tt"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 4,
                        "total_tokens": 11,
                    },
                },
            }
        from langchain_core.messages import AIMessage as _AIM

        return _AIM(content="", tool_calls=[
            {"name": "search", "args": {"q": "tt"}, "id": "call_1", "type": "tool_call"}
        ])

    monkeypatch.setattr(
        "langchain_core.language_models.chat_models.BaseChatModel.ainvoke",
        ainvoke_with_wire,
    )

    async def scenario() -> str:
        approver = asyncio.create_task(
            _approve_all(channel, Decision(kind=DecisionKind.APPROVE))
        )
        with patch(), replay_ctx(
            store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel
        ) as session:
            await model.ainvoke([HumanMessage(content="hello")])
        approver.cancel()
        return str(session.branch_id)

    branch_id = asyncio.run(scenario())
    spans = store.get_spans(trace_id, branch_id=branch_id)
    stored = spans[0].raw_attributes["gen_ai.response"]["choices"][0]["message"]
    # The wire payload (reasoning + OpenAI-form tool calls + usage) was stored.
    assert stored["reasoning_content"] == "wire-level reasoning"
    assert stored["tool_calls"][0]["function"]["name"] == "search"
    assert spans[0].total_tokens == 11
