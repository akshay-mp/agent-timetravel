"""Phase 6 — Google ADK adapter contract tests (gated on `google-adk`).

Skipped unless ``google-adk`` is importable. Exercises the full replay
contract against a stand-in ADK ``BaseLlm``:

* FROZEN replay returns the recorded payload with zero outbound calls.
* BRANCH replay forwards divergent calls and records a new span.
* No active session → the wrapper is transparent (delegates to wrapped).

Install the extra to run them::

    pip install agent-timetravel[adk]
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_timetravel.adapters.adk import _llm_response_to_text
from agent_timetravel.enums import ReplayMode, SpanKind, SpanStatus
from agent_timetravel.models import Span, Trace, hash_payload
from agent_timetravel.replay import (
    replay as replay_ctx,
)
from agent_timetravel.storage import TraceStore

_HAS_ADK = (
    importlib.util.find_spec("google") is not None
    and importlib.util.find_spec("google.adk") is not None
)
pytestmark = pytest.mark.skipif(not _HAS_ADK, reason="google-adk not installed")


_MESSAGES = [{"role": "user", "content": "hello"}]


def _recorded_llm_span(trace_id: str, *, content: str = "recorded") -> Span:
    return Span(
        trace_id=trace_id,
        span_id="a" * 16,
        parent_span_id=None,
        name="adk.llm",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        model_name="adk-test",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        messages_hash=hash_payload(_MESSAGES),
        raw_attributes={
            "gen_ai.request.model": "adk-test",
            "gen_ai.response": {
                "choices": [{"message": {"role": "assistant", "content": content}}],
            },
        },
    )


def _adk_request(parts_text: str = "hello") -> Any:
    """A duck-typed stand-in for ADK's ``LlmRequest``."""
    return SimpleNamespace(
        contents=[SimpleNamespace(role="user", parts=[parts_text])],
        config=SimpleNamespace(tools=None),
    )


def _wrapped_llm() -> tuple[Any, list[Any]]:
    """Build a wrapped ADK-style LLM plus its outbound-call log."""
    from agent_timetravel.adapters.adk import replay_llm

    calls: list[Any] = []

    class _Wrapped:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.model = "adk-test"

        async def generate_content_async(
            self, request: Any, stream: bool = False
        ) -> Any:
            calls.append(request)
            yield self._response("live-partial")
            yield self._response("live-text")

        @staticmethod
        def _response(text: str) -> Any:
            return SimpleNamespace(
                content=SimpleNamespace(
                    role="model", parts=[SimpleNamespace(text=text)]
                )
            )

    return replay_llm(_Wrapped()), calls


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(str(tmp_path / "adk.db"))


@pytest.fixture
def trace_id() -> str:
    return "abcd1234abcd1234abcd1234abcd1234"


@pytest.fixture
def seeded(store: TraceStore, trace_id: str) -> tuple[TraceStore, Span]:
    span = _recorded_llm_span(trace_id, content="recorded-text")
    store.upsert_trace(Trace(trace_id=trace_id, spans=[span]))
    store.insert_span(span)
    return store, span


def test_frozen_replay_returns_recorded_payload(
    seeded: tuple[TraceStore, Span], trace_id: str
) -> None:
    """FROZEN replay returns recorded payload and makes zero live calls."""
    store, _span = seeded
    wrapped, calls = _wrapped_llm()
    with replay_ctx(store, trace_id, mode=ReplayMode.FROZEN):
        result = asyncio.run(_collect(wrapped.generate_content_async(_adk_request())))
    assert len(result) == 1
    assert _llm_response_to_text(result[0]) == "recorded-text"
    assert calls == [], "FROZEN replay must make zero outbound calls"


def test_branch_replay_forwards_divergent_call(
    seeded: tuple[TraceStore, Span], trace_id: str
) -> None:
    """BRANCH replay forwards a divergent call and captures a new span."""
    store, _span = seeded
    wrapped, calls = _wrapped_llm()
    with replay_ctx(store, trace_id, mode=ReplayMode.BRANCH) as session:
        # Recorded message set: serve from fixture.
        frozen = asyncio.run(_collect(wrapped.generate_content_async(_adk_request())))
        assert len(frozen) == 1
        assert _llm_response_to_text(frozen[0]) == "recorded-text"
        assert calls == []
        # Divergence: a new message set never matches a recorded span.
        divergent = asyncio.run(
            _collect(wrapped.generate_content_async(_adk_request("a different turn")))
        )
        assert calls, "BRANCH divergence must forward to the wrapped model"
        assert [_llm_response_to_text(response) for response in divergent] == [
            "live-partial",
            "live-text",
        ]
        captured = session.recorded_spans()[-1]
        response = captured.raw_attributes["gen_ai.response"]
        assert response["choices"][0]["message"]["content"] == "live-text"


def test_no_session_delegates_to_wrapped(
    seeded: tuple[TraceStore, Span],
) -> None:
    """Without an active session, the wrapper is transparent."""
    wrapped, calls = _wrapped_llm()
    result = asyncio.run(_collect(wrapped.generate_content_async(_adk_request())))
    assert len(result) == 2
    assert len(calls) == 1


def test_abandoned_forward_stream_is_captured(
    store: TraceStore, trace_id: str
) -> None:
    """A consumer stopping after the first chunk still leaves a branch span."""
    store.upsert_trace(Trace(trace_id=trace_id, spans=[]))
    wrapped, _calls = _wrapped_llm()

    async def scenario() -> list[Any]:
        with replay_ctx(store, trace_id, mode=ReplayMode.BRANCH) as session:
            stream = wrapped.generate_content_async(_adk_request())
            first = [response async for response in _take_one(stream)]
            await stream.aclose()
            assert len(session.recorded_spans()) == 1
            return first

    result = asyncio.run(scenario())
    assert _llm_response_to_text(result[0]) == "live-partial"


def test_manual_streaming_capture_preserves_typed_usage(
    store: TraceStore, trace_id: str
) -> None:
    """Manual adapter streams update the single cached span's typed usage."""
    from google.adk.models import BaseLlm

    from agent_timetravel.adapters.adk import replay_llm

    def usage_response(text: str) -> Any:
        return SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
            usage_metadata=SimpleNamespace(
                prompt_token_count=5,
                candidates_token_count=3,
                total_token_count=8,
            ),
        )

    class _UsageLlm(BaseLlm):
        model: str = "usage-test"

        async def generate_content_async(
            self, request: Any, stream: bool = False
        ) -> Any:
            yield usage_response("partial")
            yield usage_response("final")

    store.upsert_trace(Trace(trace_id=trace_id, spans=[]))
    wrapped = replay_llm(_UsageLlm())

    async def scenario() -> Any:
        with replay_ctx(store, trace_id, mode=ReplayMode.BRANCH) as session:
            [response async for response in wrapped.generate_content_async(_adk_request())]
            return session.branch_id

    branch_id = asyncio.run(scenario())
    spans = store.get_spans(trace_id, branch_id=branch_id)
    assert len(spans) == 1
    assert spans[0].prompt_tokens == 5
    assert spans[0].completion_tokens == 3
    assert spans[0].total_tokens == 8


async def _take_one(generator: Any) -> Any:
    async for response in generator:
        yield response
        return


def test_replay_llm_preserves_wrapped_identity() -> None:
    """The wrapper must retain the exact BaseLlm instance it delegates to."""
    from google.adk.models import BaseLlm

    from agent_timetravel.adapters.adk import replay_llm

    class _IdentityLlm(BaseLlm):
        model: str = "identity-test"

        async def generate_content_async(
            self, request: Any, stream: bool = False
        ) -> Any:
            yield SimpleNamespace(content=SimpleNamespace(parts=[]))

    wrapped = _IdentityLlm()
    wrapper = replay_llm(wrapped)

    assert wrapper._timetravel_wrapped is wrapped


async def _collect(generator: Any) -> list[Any]:
    """Drive an ADK async generator while the replay ContextVar is active."""
    return [response async for response in generator]
