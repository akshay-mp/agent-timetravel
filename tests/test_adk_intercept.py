"""Generic Google ADK workbench interceptor tests.

Gated on ``google.adk`` being importable (the ``adk`` extra).

Exercises the class-level ``BaseLlm`` / ``BaseTool`` patches:

* gate approve / edit / stop for model calls, mock for tool calls;
* replay hit from a recorded span (zero model calls) in FROZEN mode;
* frozen divergence failing closed with :class:`~timetravel.replay.ReplayError`;
* live-forward capture of LLM and TOOL spans in INTERACTIVE sessions;
* the ``replay_llm`` double-handling guard and late-defined subclass hook;
* patch restore and nested-patch idempotency.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from agent_timetravel.enums import ReplayMode, SpanKind
from agent_timetravel.models import Span, Trace, hash_payload
from agent_timetravel.openai_intercept import _to_jsonable
from agent_timetravel.replay import RecordedResponse
from agent_timetravel.replay import replay as replay_ctx
from agent_timetravel.stepping import (
    AsyncioChannel,
    Decision,
    DecisionKind,
)
from agent_timetravel.storage import TraceStore

if not (importlib.util.find_spec("google") and importlib.util.find_spec("google.adk")):
    pytest.skip("google-adk not installed", allow_module_level=True)

# pylint: disable=import-outside-toplevel
from google.adk.models import BaseLlm, LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import BaseTool as AdkBaseTool
from google.genai import types

from agent_timetravel.adapters.adk import _messages_from_adk
from agent_timetravel.adk_intercept import (
    _canonical_messages_from_adk,
    _capture_live_llm_span,
    _function_calls_of,
    _materialise_response,
    _recorded_result_text,
    patch,
)

# pylint: enable=import-outside-toplevel


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _llm_request(text: str) -> LlmRequest:
    """An ADK request with a single user text turn."""
    return LlmRequest(
        model="adk-test",
        contents=[types.Content(role="user", parts=[types.Part(text=text)])],
    )


def _response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=5,
            candidates_token_count=3,
            total_token_count=8,
        ),
    )


def _fake_llm() -> tuple[BaseLlm, list[list[str]]]:
    """Build a deterministic echo model plus its inbound-contents tracker.

    The tracker lives in a closure — ``BaseLlm`` is a pydantic model, so
    undeclared per-instance attributes cannot be assigned.
    """
    tracker: list[list[str]] = []

    class _FakeLlm(BaseLlm):
        model: str = "adk-test"

        async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
            texts = [
                part.text
                for content in llm_request.contents
                for part in (content.parts or [])
                if part.text
            ]
            tracker.append(texts)
            yield _response(f"echo:{texts[-1] if texts else ''}")

    return _FakeLlm(), tracker


class _LateLlm(BaseLlm):
    """Defined only inside the late-subclass test (via exec-free subclassing)."""

    model: str = "late-test"

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        yield _response("late-echo")


def _fake_tool() -> tuple[AdkBaseTool, list[dict[str, Any]]]:
    """Build a deterministic echo tool plus its call tracker."""
    calls: list[dict[str, Any]] = []

    class _FakeTool(AdkBaseTool):
        name = "lookup"
        description = "echoes its args"

        async def run_async(self, *, args: dict[str, Any], tool_context: Any = None) -> Any:
            calls.append(dict(args))
            return {"result": f"ran:{args.get('query', '')}"}

    return _FakeTool(name="lookup", description="echoes its args"), calls


def _seed_trace(store: TraceStore, trace_id: str, spans: list[Span]) -> None:
    store.upsert_trace(Trace(trace_id=trace_id, spans=spans))
    for span in spans:
        store.insert_span(span)


def _recorded_llm_span(
    trace_id: str,
    messages: list[dict[str, Any]],
    *,
    content: str = "recorded",
    tools_hash: str | None = None,
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id="a" * 16,
        parent_span_id=None,
        name="adk.adk-test",
        kind=SpanKind.LLM,
        model_name="adk-test",
        messages_hash=hash_payload(_to_jsonable(messages)),
        tools_hash=tools_hash,
        raw_attributes={
            "gen_ai.request.model": "adk-test",
            "gen_ai.response": {
                "choices": [{"message": {"role": "assistant", "content": content}}],
            },
        },
    )


async def _approve_all(channel: AsyncioChannel, decision: Decision) -> None:
    """Drain the channel, answering every step with ``decision``."""
    while True:
        await channel.next_step()
        channel.decide(decision)


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(str(tmp_path / "adk.db"))


@pytest.fixture
def trace_id() -> str:
    return "abcd1234abcd1234abcd1234abcd1234"


@pytest.fixture(autouse=True)
def _unpatched() -> None:
    """Force-restore the class patches even if a test leaks its context."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel import adk_intercept as intercept_module
    # pylint: enable=import-outside-toplevel

    yield
    with intercept_module._PATCH_LOCK:
        intercept_module._restore_all()
        intercept_module._PATCH_DEPTH = 0


# ----------------------------------------------------------------------
# Passthrough + patch lifecycle
# ----------------------------------------------------------------------
def test_patch_passthrough_without_session() -> None:
    model, tracker = _fake_llm()

    async def scenario() -> list[str]:
        with patch():
            return [
                response.content.parts[0].text
                async for response in model.generate_content_async(_llm_request("hi"))
            ]

    assert asyncio.run(scenario()) == ["echo:hi"]
    assert len(tracker) == 1


def test_patch_restores_originals() -> None:
    fake_original = _LateLlm.generate_content_async
    with patch():
        assert _LateLlm.generate_content_async is not fake_original
    assert _LateLlm.generate_content_async is fake_original


def test_patch_is_idempotent_nested() -> None:
    fake_original = _LateLlm.generate_content_async
    with patch():
        with patch():
            assert _LateLlm.generate_content_async is not fake_original
        assert _LateLlm.generate_content_async is not fake_original
    assert _LateLlm.generate_content_async is fake_original


def test_late_defined_subclass_is_wrapped() -> None:
    """A subclass defined during the patch window is intercepted via the hook."""
    tracker: list[list[str]] = []

    async def scenario() -> str:
        # Defined *inside* the window, so only the __init_subclass__ hook
        # (not the subclass walk at install time) can cover it.
        with patch():

            class _WindowLlm(BaseLlm):
                model: str = "window"

                async def generate_content_async(
                    self, llm_request: Any, stream: bool = False
                ) -> Any:
                    tracker.append(["window"])
                    yield _response("window-echo")

            return [
                response.content.parts[0].text
                async for response in _WindowLlm().generate_content_async(_llm_request("hi"))
            ]

    assert asyncio.run(scenario()) == ["window-echo"]
    assert tracker == [["window"]]

    # After restore, a *newly defined* subclass is untouched.
    class _PostLlm(BaseLlm):
        model: str = "post"

        async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
            tracker.append(["post"])
            yield _response("post-echo")

    async def after() -> str:
        return [
            response.content.parts[0].text
            async for response in _PostLlm().generate_content_async(_llm_request("hi"))
        ]

    assert asyncio.run(after()) == ["post-echo"]
    assert [entry[0] for entry in tracker] == ["window", "post"]


# ----------------------------------------------------------------------
# LLM path
# ----------------------------------------------------------------------
def test_generate_gates_and_captures(store: TraceStore, trace_id: str) -> None:
    _seed_trace(store, trace_id, [])
    model, tracker = _fake_llm()
    channel = AsyncioChannel()

    async def scenario() -> str:
        approver = asyncio.create_task(_approve_all(channel, Decision(kind=DecisionKind.APPROVE)))
        with (
            patch(),
            replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel) as session,
        ):
            responses = [
                response async for response in model.generate_content_async(_llm_request("hello"))
            ]
            assert [r.content.parts[0].text for r in responses] == ["echo:hello"]
        approver.cancel()
        return str(session.branch_id)

    branch_id = UUID(asyncio.run(scenario()))
    assert len(tracker) == 1
    spans = store.get_spans(trace_id, branch_id=branch_id)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == SpanKind.LLM
    assert span.name == "adk.adk-test"
    assert span.messages_hash == hash_payload(
        _canonical_messages_from_adk(_llm_request("hello"))
    )
    assert (
        span.raw_attributes["gen_ai.response"]["choices"][0]["message"]["content"] == "echo:hello"
    )
    assert span.prompt_tokens == 5
    assert span.completion_tokens == 3


def test_edit_rewrites_outbound_contents(store: TraceStore, trace_id: str) -> None:
    _seed_trace(store, trace_id, [])
    model, tracker = _fake_llm()
    channel = AsyncioChannel()
    edited = [{"role": "user", "content": "edited prompt"}]

    async def scenario() -> str:
        approver = asyncio.create_task(
            _approve_all(channel, Decision(kind=DecisionKind.EDIT, messages=edited))
        )
        with (
            patch(),
            replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel) as session,
        ):
            async for _response in model.generate_content_async(_llm_request("original")):
                pass
        approver.cancel()
        return str(session.branch_id)

    branch_id = UUID(asyncio.run(scenario()))
    assert tracker[0] == ["edited prompt"]
    # The captured span must describe the edited call, not the original one.
    span = store.get_spans(trace_id, branch_id=branch_id)[0]
    assert span.messages_hash == hash_payload(
        _canonical_messages_from_adk(_llm_request("edited prompt"))
    )


def test_stop_unwinds(store: TraceStore, trace_id: str) -> None:
    from agent_timetravel.stepping import SteppingStopped

    _seed_trace(store, trace_id, [])
    model, tracker = _fake_llm()
    channel = AsyncioChannel()

    async def scenario() -> None:
        approver = asyncio.create_task(_approve_all(channel, Decision(kind=DecisionKind.STOP)))
        try:
            with (
                patch(),
                replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel),
            ):
                async for _response in model.generate_content_async(_llm_request("halt")):
                    pass
        finally:
            approver.cancel()

    with pytest.raises(SteppingStopped):
        asyncio.run(scenario())
    assert tracker == []


def test_frozen_replay_serves_recorded(store: TraceStore, trace_id: str) -> None:
    recorded_messages = _canonical_messages_from_adk(_llm_request("hello"))
    _seed_trace(store, trace_id, [_recorded_llm_span(trace_id, recorded_messages)])
    model, tracker = _fake_llm()

    async def scenario() -> list[str]:
        with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
            return [
                response.content.parts[0].text
                async for response in model.generate_content_async(_llm_request("hello"))
            ]

    assert asyncio.run(scenario()) == ["recorded"]
    assert tracker == [], "FROZEN replay must make zero outbound calls"


def test_frozen_divergence_fails_closed(store: TraceStore, trace_id: str) -> None:
    from agent_timetravel.replay import ReplayError

    recorded_messages = _canonical_messages_from_adk(_llm_request("hello"))
    _seed_trace(store, trace_id, [_recorded_llm_span(trace_id, recorded_messages)])
    model, _tracker = _fake_llm()

    async def scenario() -> None:
        with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
            async for _response in model.generate_content_async(_llm_request("divergent")):
                pass

    with pytest.raises(ReplayError):
        asyncio.run(scenario())


def test_branch_replay_captures_divergence(store: TraceStore, trace_id: str) -> None:
    recorded_messages = _canonical_messages_from_adk(_llm_request("hello"))
    _seed_trace(store, trace_id, [_recorded_llm_span(trace_id, recorded_messages)])
    model, tracker = _fake_llm()

    async def scenario() -> list[str]:
        with patch(), replay_ctx(store, trace_id, mode=ReplayMode.BRANCH) as session:
            served = [
                response.content.parts[0].text
                async for response in model.generate_content_async(_llm_request("hello"))
            ]
            assert served == ["recorded"]
            assert tracker == []
            live = [
                response.content.parts[0].text
                async for response in model.generate_content_async(_llm_request("new turn"))
            ]
            assert (
                session.recorded_spans()[-1].raw_attributes["gen_ai.response"]["choices"][0][
                    "message"
                ]["content"]
                == "echo:new turn"
            )
            return live

    assert asyncio.run(scenario()) == ["echo:new turn"]
    assert len(tracker) == 1


def test_replay_llm_wrapper_is_not_double_patched(store: TraceStore, trace_id: str) -> None:
    """The manual ``replay_llm`` wrapper keeps owning the replay contract."""
    from agent_timetravel.adapters.adk import replay_llm

    recorded_messages = _messages_from_adk(_llm_request("hello"))
    _seed_trace(store, trace_id, [_recorded_llm_span(trace_id, recorded_messages)])
    model, _tracker = _fake_llm()
    wrapped = replay_llm(model)

    # No session: the wrapper delegates transparently under an active patch.
    async def passthrough() -> list[str]:
        with patch():
            return [
                response.content.parts[0].text
                async for response in wrapped.generate_content_async(_llm_request("hello"))
            ]

    assert asyncio.run(passthrough()) == ["echo:hello"]

    # With a FROZEN session and no approval channel wired to the gate,
    # the interceptor must not step in front of the wrapper (the wrapper
    # serves the recorded fixture itself).
    async def frozen() -> list[str]:
        with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
            return [
                response.content.parts[0].text
                async for response in wrapped.generate_content_async(_llm_request("hello"))
            ]

    assert asyncio.run(frozen()) == ["recorded"]


def test_replay_llm_wrapper_branch_records_single_span(store: TraceStore, trace_id: str) -> None:
    """A replay_llm wrapper's live forward must not double-capture a span.

    The interceptor patches the inner model's class too; without the
    wrapper-forward guard a BRANCH divergence records both the adapter's
    span and the interceptor's, corrupting the branch timeline.
    """
    from agent_timetravel.adapters.adk import replay_llm

    recorded_messages = _messages_from_adk(_llm_request("hello"))
    _seed_trace(store, trace_id, [_recorded_llm_span(trace_id, recorded_messages)])
    model, _tracker = _fake_llm()
    wrapped = replay_llm(model)

    async def scenario() -> str:
        with patch(), replay_ctx(store, trace_id, mode=ReplayMode.BRANCH) as session:
            async for _response in wrapped.generate_content_async(_llm_request("divergent")):
                pass
            return str(session.branch_id)

    branch_id = UUID(asyncio.run(scenario()))
    live_spans = [
        span
        for span in store.get_spans(trace_id, branch_id=branch_id)
        if span.raw_attributes.get("gen_ai.response", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content")
        == "echo:divergent"
    ]
    assert len(live_spans) == 1, "the live forward must be captured exactly once"
    assert live_spans[0].name == "agent_timetravel.adapter.adk-test"


def test_abandoned_stream_still_records_span(store: TraceStore, trace_id: str) -> None:
    """Closing a stream mid-flight still records what the model produced."""
    _seed_trace(store, trace_id, [])
    model, tracker = _fake_llm()
    channel = AsyncioChannel()

    async def scenario() -> str:
        approver = asyncio.create_task(_approve_all(channel, Decision(kind=DecisionKind.APPROVE)))
        with (
            patch(),
            replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel) as session,
        ):
            stream = model.generate_content_async(_llm_request("partial"))
            async for _response in stream:
                break
            await stream.aclose()
        approver.cancel()
        return str(session.branch_id)

    from uuid import UUID

    branch_id = UUID(asyncio.run(scenario()))
    assert tracker == [["partial"]]
    spans = store.get_spans(trace_id, branch_id=branch_id)
    assert len(spans) == 1
    assert (
        spans[0].raw_attributes["gen_ai.response"]["choices"][0]["message"]["content"]
        == "echo:partial"
    )


def test_tool_calling_trace_replays_through_interceptor(store: TraceStore, trace_id: str) -> None:
    """A span recorded with a tools_hash (as the manual adapter now stores)
    matches a request that carries tool declarations."""
    request = _llm_request("hello")
    request.config.tools = [
        types.Tool(
            function_declarations=[types.FunctionDeclaration(name="lookup", description="d")]
        )
    ]
    tools_hash = hash_payload(_to_jsonable(request.config.tools))
    recorded_messages = _canonical_messages_from_adk(request)
    _seed_trace(
        store,
        trace_id,
        [_recorded_llm_span(trace_id, recorded_messages, tools_hash=tools_hash)],
    )
    model, tracker = _fake_llm()

    async def scenario() -> list[str]:
        with patch(), replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
            return [
                response.content.parts[0].text
                async for response in model.generate_content_async(request)
            ]

    assert asyncio.run(scenario()) == ["recorded"]
    assert tracker == []


def test_function_response_history_matches_flat_openinference_shape() -> None:
    """ADK tool-loop history hashes like its flattened OI representation."""
    request = LlmRequest(
        model="adk-test",
        contents=[
            types.Content(role="user", parts=[types.Part(text="Weather?")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="weather", args={"city": "Boston"}, id="call_1"
                        )
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="weather", response={"temp": 72}, id="call_1"
                        )
                    )
                ],
            ),
        ],
    )
    expected = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {"city": "Boston"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temp":72}',
            "name": "weather",
            "tool_call_id": "call_1",
        },
    ]
    assert _canonical_messages_from_adk(request) == expected
    assert hash_payload(_canonical_messages_from_adk(request)) == hash_payload(expected)


def test_live_function_call_capture_preserves_id() -> None:
    result = _response("")
    result.content.parts = [
        types.Part(
            function_call=types.FunctionCall(
                name="weather", args={"city": "Boston"}, id="call_1"
            )
        )
    ]
    assert _function_calls_of(result)[0]["id"] == "call_1"


def test_streaming_capture_refreshes_typed_usage(
    store: TraceStore, trace_id: str
) -> None:
    store.upsert_trace(Trace(trace_id=trace_id, spans=[]))
    first = _response("partial")
    first.usage_metadata = None
    final = _response("final")

    with replay_ctx(store, trace_id, mode=ReplayMode.BRANCH) as session:
        span = _capture_live_llm_span(
            session,
            model_name="adk-test",
            messages=_canonical_messages_from_adk(_llm_request("hello")),
            signature=SimpleNamespace(messages_hash="messages", tools_hash=None),
            result=first,
        )
        _capture_live_llm_span(
            session,
            model_name="adk-test",
            messages=_canonical_messages_from_adk(_llm_request("hello")),
            signature=SimpleNamespace(messages_hash="messages", tools_hash=None),
            result=final,
            span=span,
        )

    assert span.prompt_tokens == 5
    assert span.completion_tokens == 3
    assert span.total_tokens == 8
    assert len(store.get_spans(trace_id, branch_id=session.branch_id)) == 1


def test_flat_output_materialises_text_and_tool_call() -> None:
    recorded = RecordedResponse(
        payload={
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "I will check that.",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "weather",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments":
                '{"city":"Boston"}',
        },
        span_id="a" * 16,
        timetravel_id=UUID("00000000-0000-0000-0000-000000000001"),
    )

    response = _materialise_response(recorded, "adk-test")
    parts = response.content.parts
    assert parts[0].text == "I will check that."
    assert parts[1].function_call.name == "weather"
    assert parts[1].function_call.args == {"city": "Boston"}
    assert parts[1].function_call.id == "call_1"
    assert _recorded_result_text(recorded) == "I will check that."


def test_flat_output_materialises_ordered_tool_use_without_text() -> None:
    recorded = RecordedResponse(
        payload={
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.contents.0.message_content.type": "tool_use",
            "llm.output_messages.0.message.contents.0.tool_call.id": "call_2",
            "llm.output_messages.0.message.contents.0.tool_call.function.name": "lookup",
            "llm.output_messages.0.message.contents.0.tool_call.function.arguments": "{}",
        },
        span_id="b" * 16,
        timetravel_id=UUID("00000000-0000-0000-0000-000000000002"),
    )

    response = _materialise_response(recorded, "adk-test")
    assert response.content.parts[0].function_call.name == "lookup"
    assert _recorded_result_text(recorded) == '→ lookup({})'


def test_staticmethod_override_passthrough() -> None:
    """A @staticmethod-defined generate_content_async is forwarded correctly."""
    calls: list[str] = []

    class _StaticLlm(BaseLlm):
        model: str = "static"

        @staticmethod
        async def generate_content_async(llm_request: Any, stream: bool = False) -> Any:
            calls.append("static")
            yield _response("static-echo")

    async def scenario() -> list[str]:
        with patch():
            return [
                response.content.parts[0].text
                async for response in _StaticLlm().generate_content_async(_llm_request("hi"))
            ]

    assert asyncio.run(scenario()) == ["static-echo"]
    assert calls == ["static"]


# ----------------------------------------------------------------------
# Tool path
# ----------------------------------------------------------------------
def test_tool_gates_and_captures(store: TraceStore, trace_id: str) -> None:
    _seed_trace(store, trace_id, [])
    tool, calls = _fake_tool()
    channel = AsyncioChannel()

    async def scenario() -> str:
        approver = asyncio.create_task(_approve_all(channel, Decision(kind=DecisionKind.APPROVE)))
        with (
            patch(),
            replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel) as session,
        ):
            result = await tool.run_async(args={"query": "weather"}, tool_context=None)
        approver.cancel()
        assert result == {"result": "ran:weather"}
        return str(session.branch_id)

    from uuid import UUID

    branch_id = UUID(asyncio.run(scenario()))
    assert calls == [{"query": "weather"}]
    spans = store.get_spans(trace_id, branch_id=branch_id)
    assert len(spans) == 1
    assert spans[0].kind == SpanKind.TOOL
    assert spans[0].name == "lookup"


def test_tool_mock_returns_without_calling(store: TraceStore, trace_id: str) -> None:
    _seed_trace(store, trace_id, [])
    tool, calls = _fake_tool()
    channel = AsyncioChannel()

    async def scenario() -> Any:
        approver = asyncio.create_task(
            _approve_all(
                channel,
                Decision(kind=DecisionKind.MOCK, mock_result={"result": "mocked"}),
            )
        )
        try:
            with (
                patch(),
                replay_ctx(store, trace_id, mode=ReplayMode.INTERACTIVE, approval=channel),
            ):
                return await tool.run_async(args={"query": "weather"}, tool_context=None)
        finally:
            approver.cancel()

    assert asyncio.run(scenario()) == {"result": "mocked"}
    assert calls == [], "MOCK must not invoke the live tool"


def test_tool_passthrough_without_session() -> None:
    tool, calls = _fake_tool()

    async def scenario() -> Any:
        with patch():
            return await tool.run_async(args={"query": "hi"}, tool_context=None)

    assert asyncio.run(scenario()) == {"result": "ran:hi"}
    assert len(calls) == 1
