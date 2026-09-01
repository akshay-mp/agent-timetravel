"""Phase 6 — shared helpers for the four framework adapters.

Plan §6 Phase 6 lists four new per-framework replay adapters (ADK, CrewAI,
PydanticAI, SmolAgents) on top of the Phase-3 LangGraph adapter. Each
adapter follows the same three-step skeleton:

1. Lazy-import the framework inside the factory function (so
   ``agent-timetravel --version`` stays fast when frameworks aren't installed).
2. Subclass or wrap the framework's chat-model surface (``BaseLlm``,
   ``BaseChatModel``, ``Model``, ``HfApiModel``…).
3. For each inference call:
   - Look up the active :class:`~timetravel.replay.ReplaySession` via
     :func:`~timetravel.replay.active_session`.
   - Build a :class:`~timetravel.replay.CallSignature` (re-using
     :func:`~agent_timetravel.openai_intercept.extract_signature` when the framework
     accepts ``model=…, messages=[…], tools=[…]`` — ask all five do today).
   - Call ``session.respond_or_forward(signature)``:
     * returns a :class:`~timetravel.replay.RecordedResponse` → materialise
       a framework-native response and return it (zero egress);
     * returns ``None`` → assert not frozen, forward to the wrapped model,
       then call :func:`build_live_span` + ``session.record_new(span)``.

This module hosts the **framework-agnostic** halves of that skeleton:
``build_live_span`` (boring Span-shape construction repeated five times)
and the frozen-mode guard. Each framework file keeps its own
``_materialise`` (per-FW response shape) and duck-typed wrapper strategy.
"""

from __future__ import annotations

from secrets import token_hex
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_timetravel.replay import ReplaySession

__all__ = ["assert_not_frozen", "build_live_span"]


def build_live_span(
    session: ReplaySession,
    *,
    model_name: str,
    messages: list[Any] | None,
    content: str,
    raw_extras: dict[str, Any] | None = None,
    tool_name: str | None = None,
    kind_str: str = "LLM",
    tools_hash: str | None = None,
) -> Any:
    """Construct a ``Span`` for a live call made during a replay.

    Called by every adapter after forwarding a divergent call to the wrapped
    model. The returned span is handed to ``session.record_new(span)`` so
    subsequent calls in the same replay branch share its stored shape.

    Parameters
    ----------
    session:
        The active replay session the new span belongs to.
    model_name:
        The model the live call hit (``self._timetravel_wrapped._llm_type`` etc).
    messages:
        Outbound messages list (already in JSONable form if the framework
        pre-coerced; otherwise the adapter should ``model_dump()``/``dict()``
        its way to plain dicts). ``None`` for tool spans.
    content:
        The textual content the model returned (or tool result string).
    raw_extras:
        Optional extra ``raw_attributes`` keys the framework wants to
        preserve verbatim (about as opaque-key-value as OpenInference emits).
    tool_name:
        For tool spans — sets ``name`` to ``tool.{tool_name}``.
    kind_str:
        ``SpanKind`` enum value name (``LLM`` / ``TOOL`` / ``AGENT``).
    tools_hash:
        Hash of the outbound tool declarations, when the framework signature
        carries tools. Adapters that sign their calls with a tools hash must
        store it here — the replay matcher requires span and signature to
        agree on ``tools_hash`` for tool-carrying calls.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import SpanKind, SpanStatus
    from agent_timetravel.models import Span, hash_payload
    # pylint: enable=import-outside-toplevel

    payload_messages = messages or []
    kind = SpanKind[kind_str]
    if kind is SpanKind.TOOL:
        raw: dict[str, Any] = {
            "tool.name": tool_name or "timetravel-replay",
            "tool.output": content or "",
            **(raw_extras or {}),
        }
        name = f"tool.{tool_name or 'timetravel'}"
    else:
        raw = {
            "gen_ai.request.model": model_name,
            "gen_ai.response": {
                "choices": [{"message": {"role": "assistant", "content": content or ""}}],
            },
            **(raw_extras or {}),
        }
        name = f"agent_timetravel.adapter.{model_name or 'unknown'}"

    return Span(
        trace_id=session.trace_id,
        span_id=token_hex(8),
        parent_span_id=None,
        name=name,
        kind=kind,
        status=SpanStatus.OK,
        model_name=model_name if kind is SpanKind.LLM else None,
        messages_hash=hash_payload(payload_messages),
        tools_hash=tools_hash,
        raw_attributes=raw,
    )


def assert_not_frozen(session: ReplaySession) -> None:
    """Raise :class:`ReplayError` if the active session is in FROZEN mode.

    All adapters use this when ``respond_or_forward`` returned ``None``;
    FROZEN divergence is the strict-determinism contract that eval suites
    (Phase 5.5) rely on.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import ReplayMode
    from agent_timetravel.replay import ReplayError
    # pylint: enable=import-outside-toplevel

    if session.mode is ReplayMode.FROZEN:
        raise ReplayError(
            f"frozen replay diverged at cursor={session.cursor}; "
            "no recorded fixture to serve — switch to BRANCH or FULL_RERUN "
            "to authorise a live forward"
        )
