"""Generic LangGraph / langchain-core workbench interceptor.

Patches ``langchain_core.language_models.chat_models.BaseChatModel.invoke`` /
``ainvoke`` and ``langchain_core.tools.BaseTool.invoke`` / ``ainvoke`` at the
class level for the duration of a workbench run, so **any** LangGraph graph —
including one that constructs its chat models inside node functions at call
time — is stepped, replayed, and captured without the developer wrapping
anything manually.

``invoke`` / ``ainvoke`` are the template-method choke points every chat
model and tool shares: subclasses override ``_generate`` / ``_run``, not
``invoke``, and ``bind_tools`` bindings delegate to ``bound.invoke``. During a
:class:`func:`timetravel.replay`` context each intercepted call:

* pauses at the stepping gate (:func:`agent_timetravel.stepping.gate_async` /
  :func:`gate_sync`) so the workbench can approve / edit / mock / stop it;
* is served from a matching recorded span when one sits at the cursor
  (zero outbound traffic) — or forwarded live and captured in
  BRANCH / FULL_RERUN / INTERACTIVE modes;
* raises :class:`~timetravel.replay.ReplayError` on a frozen divergence.

With no active replay session every call passes through unchanged. Models
already wrapped by :func:`agent_timetravel.adapters.langgraph.replay_chat_model` are
skipped (their ``_generate`` owns the replay contract), and tools whose
underlying function carries the ``@timetravel.tool()`` marker keep that wrapper's
dispatch instead.

The module never imports ``langchain_core`` at load — only inside
:func:`patch` — matching the lazy-import contract of
:mod:`agent_timetravel.openai_intercept`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Any

from agent_timetravel.openai_intercept import (
    _build_llm_step,
    _extract_usage,
    extract_signature,
)

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage

    from agent_timetravel.replay import RecordedResponse, ReplaySession
    from agent_timetravel.stepping import Decision, Step

__all__ = ["InterceptError", "patch"]


class InterceptError(RuntimeError):
    """Raised when the LangGraph interceptor cannot be installed or honoured."""


_PATCH_LOCK = Lock()
_PATCH_DEPTH = 0
_ORIGINALS: dict[tuple[type, str], Any] = {}

#: Non-secret model attributes projected into the step payload so the UI can
#: show temperature / max_tokens / seed without dumping the whole pydantic
#: model (which would drag in api keys as SecretStr fields).
_SAMPLING_ATTRS = (
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "timeout",
)


# ----------------------------------------------------------------------
# Patch installation
# ----------------------------------------------------------------------
@contextmanager
def patch() -> Iterator[None]:
    """Monkey-patch langchain-core's chat models and tools for the block.

    Restores the original methods on exit even if the body raises. Installing
    a second time while one is already active is a no-op (nested workbench
    runs don't double-restore). Raises :class:`InterceptError` when
    ``langchain_core`` is not installed.
    """
    # pylint: disable=import-outside-toplevel
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - exercised only without langchain
        raise InterceptError(
            "timetravel workbench LangGraph activation requires `langchain-core`; "
            "install it with `pip install agent-timetravel[langgraph]`."
        ) from exc

    from agent_timetravel.replay import active_session
    # pylint: enable=import-outside-toplevel

    orig_invoke = BaseChatModel.invoke
    orig_ainvoke = BaseChatModel.ainvoke
    orig_tool_invoke = BaseTool.invoke
    orig_tool_ainvoke = BaseTool.ainvoke

    def patched_invoke(
        self: Any, input: Any, config: Any = None, *, stop: Any = None, **kwargs: Any
    ) -> Any:
        session = active_session()
        if session is None or _is_replay_wrapper(self):
            return orig_invoke(self, input, config, stop=stop, **kwargs)
        return _dispatch_llm_sync(self, session, orig_invoke, input, config, stop, kwargs)

    async def patched_ainvoke(
        self: Any, input: Any, config: Any = None, *, stop: Any = None, **kwargs: Any
    ) -> Any:
        session = active_session()
        if session is None or _is_replay_wrapper(self):
            return await orig_ainvoke(self, input, config, stop=stop, **kwargs)
        return await _dispatch_llm_async(self, session, orig_ainvoke, input, config, stop, kwargs)

    def patched_tool_invoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        session = active_session()
        if session is None or _is_timetravel_tool(self):
            return orig_tool_invoke(self, input, config, **kwargs)
        return _dispatch_tool_sync(
            session,
            lambda call_input, call_kwargs: orig_tool_invoke(
                self, call_input, config, **call_kwargs
            ),
            tool_name=_tool_display_name(self),
            input=input,
            kwargs=kwargs,
        )

    async def patched_tool_ainvoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        session = active_session()
        if session is None or _is_timetravel_tool(self):
            return await orig_tool_ainvoke(self, input, config, **kwargs)

        async def forward(call_input: Any, call_kwargs: dict[str, Any]) -> Any:
            return await orig_tool_ainvoke(self, call_input, config, **call_kwargs)

        return await _dispatch_tool_async(
            session,
            forward,
            tool_name=_tool_display_name(self),
            input=input,
            kwargs=kwargs,
        )

    for patched in (patched_invoke, patched_ainvoke, patched_tool_invoke, patched_tool_ainvoke):
        patched.__timetravel_patched__ = True  # type: ignore[attr-defined]

    global _PATCH_DEPTH

    with _PATCH_LOCK:
        if _PATCH_DEPTH:
            _PATCH_DEPTH += 1
        else:
            installed: list[tuple[type, str, Any]] = []
            try:
                for cls, name, replacement in (
                    (BaseChatModel, "invoke", patched_invoke),
                    (BaseChatModel, "ainvoke", patched_ainvoke),
                    (BaseTool, "invoke", patched_tool_invoke),
                    (BaseTool, "ainvoke", patched_tool_ainvoke),
                ):
                    installed.append((cls, name, getattr(cls, name)))
                    setattr(cls, name, replacement)
            except Exception:
                for cls, name, original in installed:
                    setattr(cls, name, original)
                raise
            for cls, name, original in installed:
                _ORIGINALS[(cls, name)] = original
            _PATCH_DEPTH = 1

    try:
        # The wire-level SDK hook runs in observe-only mode around framework
        # model calls (see capture_only) so provider reasoning survives the
        # langchain-openai conversion, which drops ``reasoning_content``.
        # pylint: disable=import-outside-toplevel
        from agent_timetravel.openai_intercept import patch as patch_openai
        # pylint: enable=import-outside-toplevel

        with patch_openai():
            yield
    finally:
        with _PATCH_LOCK:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0:
                for (cls, name), original in _ORIGINALS.items():
                    setattr(cls, name, original)
                _ORIGINALS.clear()


# ----------------------------------------------------------------------
# LLM path
# ----------------------------------------------------------------------
def _is_replay_wrapper(model: Any) -> bool:
    """True when the model is a manual ``replay_chat_model`` wrapper."""
    return getattr(model, "_timetravel_wrapped", None) is not None


def _model_name(model: Any) -> str:
    return str(getattr(model, "model_name", None) or model._llm_type)


def _tool_display_name(tool: Any) -> str:
    return str(getattr(tool, "name", None) or type(tool).__name__)


def _strip_names(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop ``name`` from non-tool messages (see :func:`_convert_messages`).

    Strict OpenAI-compatible servers (vLLM-based ones such as Unsloth's)
    reject ``name`` on any role but ``tool`` — ``422 "name" is only valid on
    role="tool" messages`` — while deepagents-style frameworks tag messages
    with names. Tool messages keep theirs.
    """
    return [
        message
        if message.type == "tool" or not getattr(message, "name", None)
        else message.model_copy(update={"name": None})
        for message in messages
    ]


def _convert_messages(model: Any, input: Any) -> list[BaseMessage] | None:
    """Normalise an invoke input to messages; ``None`` when unconvertible.

    The stripped form is used consistently for the step view, the replay
    hash, and the outbound call.
    """
    try:
        return _strip_names(model._convert_input(input).to_messages())
    except Exception:
        return None


def _llm_call_kwargs(
    model: Any,
    messages: list[BaseMessage],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the ``{model, messages, tools, **params}`` step-view kwargs."""
    params = {
        key: value
        for key, value in kwargs.items()
        if key not in ("tools", "stop", "model")
    }
    for attr in _SAMPLING_ATTRS:
        value = getattr(model, attr, None)
        if value is not None and attr not in params:
            params[attr] = value
    return {
        "model": _model_name(model),
        "messages": [m.model_dump() for m in messages],
        "tools": kwargs.get("tools"),
        **params,
    }


def _apply_llm_decision(
    decision: Decision,
    call_kwargs: dict[str, Any],
    invoke_kwargs: dict[str, Any],
    messages: list[BaseMessage],
) -> tuple[Any, dict[str, Any]]:
    """Apply an EDIT decision; returns ``(forward_input, forward_kwargs)``.

    Edited messages update ``call_kwargs`` too, so the replay signature (and
    the captured span's ``messages_hash``) describe the call that actually
    goes out — an edited prompt is a divergence the recorder must see.
    """
    # pylint: disable=import-outside-toplevel
    from langchain_core.messages.utils import convert_to_messages
    # pylint: enable=import-outside-toplevel

    forward_input: Any = messages
    forward_kwargs = dict(invoke_kwargs)
    if decision.messages is not None:
        call_kwargs["messages"] = decision.messages
        forward_input = _strip_names(convert_to_messages(decision.messages))
    if decision.params is not None:
        forward_kwargs.update(decision.params)
    if decision.model is not None:
        forward_kwargs["model"] = decision.model
    return forward_input, forward_kwargs


def _message_content(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    try:
        return str(content)
    except Exception:
        return ""


def _reasoning_of(message: AIMessage) -> str:
    """Provider reasoning langchain keeps out of ``content``.

    OpenAI-compatible servers (llama.cpp among them) deliver thinking as a
    separate ``reasoning_content`` field; langchain aggregates it into
    ``additional_kwargs`` rather than the text content.
    """
    # pylint: disable=import-outside-toplevel
    additional = getattr(message, "additional_kwargs", None) or {}
    # pylint: enable=import-outside-toplevel
    for source in (getattr(message, "reasoning_content", None), additional.get("reasoning_content"), additional.get("reasoning")):
        if isinstance(source, str) and source.strip():
            return source.strip()
    return ""


def _with_thinking(content: str, message: AIMessage) -> str:
    """Normalise provider reasoning into the ``<think>`` display convention.

    Mirrors the OpenAI interceptor's response-text shape so the workbench
    reasoning panel lights up regardless of which interceptor captured the
    call.
    """
    reasoning = _reasoning_of(message)
    if not reasoning or "<think>" in content.lower():
        return content
    if content.strip():
        return f"<think>{reasoning}</think>\n{content}"
    return f"<think>{reasoning}</think>"


def _usage_metadata(message: AIMessage) -> dict[str, int]:
    meta = getattr(message, "usage_metadata", None)
    if not isinstance(meta, dict):
        return {}
    return {
        key: meta[key]
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance(meta.get(key), int)
    }


def _tool_calls_of(message: AIMessage) -> list[dict[str, Any]]:
    """Normalised langchain tool_calls (``name``/``args``/``id``/``type``)."""
    calls = getattr(message, "tool_calls", None) or []
    return [call for call in calls if isinstance(call, dict)]


def _materialise_message(recorded: RecordedResponse) -> AIMessage:
    """Rebuild an ``AIMessage`` from a recorded span payload.

    Understands the GenAI semconv nested response (what this interceptor and
    the OpenAI interceptor capture) and OpenInference flat message keys.
    Assistant tool calls are preserved in both the langchain-native form
    (``message.tool_calls``) and the OpenAI wire form
    (``function.name`` / ``function.arguments``), so a frozen replay
    reproduces the agent's tool-call decisions — not just the text.
    """
    # pylint: disable=import-outside-toplevel
    from langchain_core.messages import AIMessage
    # pylint: enable=import-outside-toplevel

    payload = recorded.payload or {}
    response = (
        payload.get("gen_ai.response")
        or payload.get("raw_response")
        or payload.get("response")
        or {}
    )
    content = ""
    recorded_message: dict[str, Any] = {}
    if isinstance(response, dict):
        choices = response.get("choices") or []
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        if isinstance(message, dict):
            recorded_message = message
            if isinstance(message.get("content"), str):
                content = message["content"]
    if not content:
        for key, value in payload.items():
            if (
                key.startswith("llm.output_messages.")
                and key.endswith(".message.content")
                and isinstance(value, str)
            ):
                content = value
                break

    tool_calls = _parse_recorded_tool_calls(recorded_message)
    usage = response.get("usage") if isinstance(response, dict) else None
    usage_metadata = None
    if isinstance(usage, dict):
        usage_metadata = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    additional_kwargs: dict[str, Any] = {}
    reasoning = (
        recorded_message.get("reasoning_content")
        if isinstance(recorded_message, dict)
        else None
    )
    if isinstance(reasoning, str) and reasoning.strip():
        additional_kwargs["reasoning_content"] = reasoning
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        usage_metadata=usage_metadata,
        additional_kwargs=additional_kwargs,
    )


def _parse_recorded_tool_calls(recorded_message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise stored tool calls into langchain's ``tool_calls`` shape."""
    # pylint: disable=import-outside-toplevel
    import json
    # pylint: enable=import-outside-toplevel

    raw_calls = recorded_message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    parsed: list[dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        if isinstance(call.get("function"), dict):  # OpenAI wire form
            name = call["function"].get("name")
            arguments = call["function"].get("arguments")
            args: dict[str, Any] = (
                json.loads(arguments)
                if isinstance(arguments, str) and arguments.strip()
                else {}
            )
        else:  # langchain-native form
            name = call.get("name")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if not isinstance(name, str):
            continue
        parsed.append(
            {
                "name": name,
                "args": args,
                "id": call.get("id") or "",
                "type": "tool_call",
            }
        )
    return parsed


def _capture_live_llm_span(
    session: ReplaySession,
    *,
    model: Any,
    result: AIMessage,
    signature: Any,
    wire_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an LLM span for a live-forwarded call; return its raw payload.

    ``wire_raw`` is the observe-only SDK capture (:func:`capture_only`). When
    present it is the authoritative payload — it carries the provider's own
    reasoning and usage that the langchain message conversion dropped.
    """
    # pylint: disable=import-outside-toplevel
    from secrets import token_hex

    from agent_timetravel.enums import SpanKind, SpanStatus
    from agent_timetravel.models import Span
    # pylint: enable=import-outside-toplevel

    if isinstance(wire_raw, dict) and wire_raw.get("gen_ai.response"):
        raw = dict(wire_raw)
        raw.setdefault("gen_ai.request.model", _model_name(model))
        # Flatten nested usage into the span-column keys when the capture
        # didn't already do it (the OpenAI-side capture normally does).
        response = raw["gen_ai.response"]
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict) and "gen_ai.usage.total_tokens" not in raw:
            for side in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(side)
                if isinstance(value, int):
                    raw[f"gen_ai.usage.{side}"] = value
    else:
        content = _message_content(result)
        usage = _usage_metadata(result)
        reasoning = _reasoning_of(result)
        # Store tool calls in langchain-native form so frozen replay can rebuild
        # the exact assistant decision (name, args, id) — see _materialise_message.
        stored_calls = [
            {
                "name": call.get("name"),
                "args": call.get("args") or {},
                "id": call.get("id") or "",
                "type": "tool_call",
            }
            for call in _tool_calls_of(result)
        ]
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            # OpenAI wire form: keep reasoning in the recorded message so a
            # frozen replay (and the timeline) keeps the thinking too.
            assistant["reasoning_content"] = reasoning
        if stored_calls:
            assistant["tool_calls"] = stored_calls
        raw = {
            "gen_ai.request.model": _model_name(model),
            "gen_ai.response": {"choices": [{"message": assistant}]},
        }
        if usage:
            raw["gen_ai.response"]["usage"] = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            for side, key in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                raw[f"gen_ai.usage.{side}"] = usage[key]
    span = Span(
        trace_id=session.trace_id,
        span_id=token_hex(8),
        parent_span_id=None,
        name=f"langchain.{model._llm_type}",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        model_name=_model_name(model),
        prompt_tokens=raw.get("gen_ai.usage.prompt_tokens"),
        completion_tokens=raw.get("gen_ai.usage.completion_tokens"),
        total_tokens=raw.get("gen_ai.usage.total_tokens"),
        messages_hash=signature.messages_hash,
        tools_hash=signature.tools_hash,
        raw_attributes=raw,
    )
    session.record_new(span)
    return raw


def _llm_usage(
    payload: dict[str, Any],
    call_kwargs: dict[str, Any],
    message: AIMessage,
) -> dict[str, int]:
    """Token accounting with the shared OpenAI-path estimator as fallback.

    ``usage_metadata`` on the live message wins (it is the provider's own
    count); otherwise the recorded payload's ``gen_ai.response.usage`` is
    consulted; a zero/absent usage falls back to the clearly-marked
    estimate so local model servers still account.
    """
    meta = _usage_metadata(message)
    effective = (
        {
            "gen_ai.response": {
                "usage": {
                    "prompt_tokens": meta.get("input_tokens", 0),
                    "completion_tokens": meta.get("output_tokens", 0),
                    "total_tokens": meta.get("total_tokens", 0),
                }
            }
        }
        if meta
        else payload
    )
    return _extract_usage(
        effective,
        {"messages": call_kwargs["messages"]},
        _with_thinking(_message_content(message), message),
    )


def _llm_result_text(message: AIMessage, extra_reasoning: str | None = None) -> str:
    """What the debugger shows as the step result.

    Text content when present; otherwise a one-line summary of the assistant's
    tool-call decisions, so a planning turn that emits only tool calls still
    shows something meaningful instead of an empty preview. Provider reasoning
    is normalised into the ``<think>`` convention the workbench splits into
    its Thinking panel — either from the langchain message or, when the
    conversion dropped it, from the wire-level capture (``extra_reasoning``).
    """
    content = _with_thinking(_message_content(message), message)
    reasoning = (_reasoning_of(message) or (extra_reasoning or "")).strip()
    if reasoning and "<think>" not in content.lower():
        content = (
            f"<think>{reasoning}</think>\n{content}"
            if content.strip()
            else f"<think>{reasoning}</think>"
        )
    if content.strip():
        return content
    # pylint: disable=import-outside-toplevel
    import json
    # pylint: enable=import-outside-toplevel

    lines = [
        f"→ {call.get('name')}({json.dumps(call.get('args') or {}, default=str)})"
        for call in _tool_calls_of(message)
    ]
    return "\n".join(lines)


def _wire_reasoning(wire_raw: dict[str, Any] | None) -> str | None:
    """Reasoning text from an observe-only wire capture, when present."""
    if not isinstance(wire_raw, dict):
        return None
    response = wire_raw.get("gen_ai.response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    value = message.get("reasoning_content") if isinstance(message, dict) else None
    return value if isinstance(value, str) and value.strip() else None


def _complete_llm_step_sync(
    session: ReplaySession,
    step: Step,
    payload: dict[str, Any],
    call_kwargs: dict[str, Any],
    message: AIMessage,
    wire_raw: dict[str, Any] | None = None,
) -> None:
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import complete_step_sync
    # pylint: enable=import-outside-toplevel

    complete_step_sync(
        session,
        step,
        _llm_result_text(message, _wire_reasoning(wire_raw)),
        usage=_llm_usage(payload, call_kwargs, message),
    )


async def _complete_llm_step(
    session: ReplaySession,
    step: Step,
    payload: dict[str, Any],
    call_kwargs: dict[str, Any],
    message: AIMessage,
    wire_raw: dict[str, Any] | None = None,
) -> None:
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import complete_step
    # pylint: enable=import-outside-toplevel

    await complete_step(
        session,
        step,
        _llm_result_text(message, _wire_reasoning(wire_raw)),
        usage=_llm_usage(payload, call_kwargs, message),
    )


def _dispatch_llm_sync(
    model: Any,
    session: ReplaySession,
    orig_invoke: Callable[..., Any],
    input: Any,
    config: Any,
    stop: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Gate → replay-or-forward for a sync ``BaseChatModel.invoke``."""
    messages = _convert_messages(model, input)
    if messages is None:
        return orig_invoke(model, input, config, stop=stop, **kwargs)
    call_kwargs = _llm_call_kwargs(model, messages, kwargs)
    step = _build_llm_step(session, call_kwargs)
    decision = _gate_llm_sync(session, step)
    forward_input, forward_kwargs = messages, kwargs
    if decision is not None:
        forward_input, forward_kwargs = _apply_llm_decision(
            decision, call_kwargs, kwargs, messages
        )
    signature = extract_signature(**call_kwargs)
    recorded = session.respond_or_forward(signature)
    if recorded is not None:
        message = _materialise_message(recorded)
        _complete_llm_step_sync(session, step, recorded.payload, call_kwargs, message)
        return message
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.openai_intercept import capture_only
    # pylint: enable=import-outside-toplevel

    with capture_only() as wire_holder:
        result = orig_invoke(model, forward_input, config, stop=stop, **forward_kwargs)
    wire_raw = (wire_holder or {}).get("raw")
    raw = _capture_live_llm_span(
        session, model=model, result=result, signature=signature, wire_raw=wire_raw
    )
    _complete_llm_step_sync(session, step, raw, call_kwargs, result, wire_raw)
    return result


async def _dispatch_llm_async(
    model: Any,
    session: ReplaySession,
    orig_ainvoke: Callable[..., Any],
    input: Any,
    config: Any,
    stop: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Gate → replay-or-forward for an async ``BaseChatModel.ainvoke``."""
    messages = _convert_messages(model, input)
    if messages is None:
        return await orig_ainvoke(model, input, config, stop=stop, **kwargs)
    call_kwargs = _llm_call_kwargs(model, messages, kwargs)
    step = _build_llm_step(session, call_kwargs)
    decision = await _gate_llm_async(session, step)
    forward_input, forward_kwargs = messages, kwargs
    if decision is not None:
        forward_input, forward_kwargs = _apply_llm_decision(
            decision, call_kwargs, kwargs, messages
        )
    signature = extract_signature(**call_kwargs)
    recorded = session.respond_or_forward(signature)
    if recorded is not None:
        message = _materialise_message(recorded)
        await _complete_llm_step(session, step, recorded.payload, call_kwargs, message)
        return message
    # Observe-only wire capture: langchain-openai drops provider reasoning
    # (``reasoning_content``) during message conversion, so grab the raw SDK
    # response underneath and merge what the AIMessage lost.
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.openai_intercept import capture_only
    # pylint: enable=import-outside-toplevel

    with capture_only() as wire_holder:
        result = await orig_ainvoke(model, forward_input, config, stop=stop, **forward_kwargs)
    wire_raw = (wire_holder or {}).get("raw")
    raw = _capture_live_llm_span(
        session, model=model, result=result, signature=signature, wire_raw=wire_raw
    )
    await _complete_llm_step(session, step, raw, call_kwargs, result, wire_raw)
    return result


def _gate_llm_sync(session: ReplaySession, step: Step) -> Decision | None:
    """Interactive stepping gate for a sync intercepted call."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind, SteppingStopped, gate_sync
    # pylint: enable=import-outside-toplevel

    decision = gate_sync(session, step)
    if decision is not None and decision.kind is DecisionKind.STOP:
        raise SteppingStopped(step)
    return decision


async def _gate_llm_async(session: ReplaySession, step: Step) -> Decision | None:
    """Interactive stepping gate for an async intercepted call."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind, SteppingStopped, gate_async
    # pylint: enable=import-outside-toplevel

    decision = await gate_async(session, step)
    if decision is not None and decision.kind is DecisionKind.STOP:
        raise SteppingStopped(step)
    return decision


# ----------------------------------------------------------------------
# Tool path
# ----------------------------------------------------------------------
def _is_timetravel_tool(tool: Any) -> bool:
    """True when the tool's function already dispatches via ``@timetravel.tool``."""
    return getattr(getattr(tool, "func", None), "__timetravel_tool_name__", None) is not None


def _json_safe(value: Any) -> Any:
    """Coerce a tool input to something JSON-transportable.

    langchain 1.x injects runtime objects (e.g. ``ToolRuntime``) into tool
    args; they must never break the SSE step payload or span storage. The
    coercion is recursive so JSON-able structure stays inspectable — only the
    genuinely unserialisable leaves degrade to ``repr``. The forwarded call
    keeps the original objects; this only shapes the debugger view, hashes,
    and captured payloads.
    """
    # pylint: disable=import-outside-toplevel
    import json

    from agent_timetravel.openai_intercept import _to_jsonable
    # pylint: enable=import-outside-toplevel

    coerced = _to_jsonable(value)
    if isinstance(coerced, dict):
        return {str(key): _json_safe(item) for key, item in coerced.items()}
    if isinstance(coerced, (list, tuple)):
        return [_json_safe(item) for item in coerced]
    try:
        json.dumps(coerced)
        return coerced
    except (TypeError, ValueError):
        return repr(coerced)


def _safe_tool_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    return ([_json_safe(a) for a in args], {str(k): _json_safe(v) for k, v in kwargs.items()})


def _tool_step(
    session: ReplaySession, tool_name: str, args: list[Any], kwargs: dict[str, Any]
) -> Step:
    """Build the TOOL step payload (shape shared with ``@timetravel.tool``)."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import Step, StepKind
    # pylint: enable=import-outside-toplevel

    return Step(
        kind=StepKind.TOOL,
        payload={"name": tool_name, "args": list(args), "kwargs": dict(kwargs)},
        cursor=session.cursor,
    )


def _no_call_result(decision: Decision, tool_name: str) -> Any:
    """Result substituted for MOCK / SKIP / REJECT — no live tool call."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind
    # pylint: enable=import-outside-toplevel

    if decision.kind is DecisionKind.MOCK:
        return decision.mock_result
    if decision.kind is DecisionKind.SKIP:
        return {"timetravel": "tool skipped", "tool": tool_name}
    if decision.kind is DecisionKind.REJECT:
        return {
            "timetravel": "tool rejected",
            "tool": tool_name,
            "reason": (
                decision.reason
                if decision.reason is not None
                else "rejected by developer"
            ),
        }
    return None


def _tool_result_text(output: Any) -> str:
    # pylint: disable=import-outside-toplevel
    import json
    # pylint: enable=import-outside-toplevel

    try:
        return json.dumps(output, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return repr(output)


def _tool_replay_lookup(
    session: ReplaySession,
    tool_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> RecordedResponse | None:
    """Name+args-hash lookup against the recorded spans (tool semantics)."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import SpanKind
    from agent_timetravel.tool_intercept import _find_tool_span, _tool_args_hash
    # pylint: enable=import-outside-toplevel

    return _find_tool_span(
        session,
        name=tool_name,
        kind=SpanKind.TOOL,
        args_hash=_tool_args_hash(args, kwargs),
    )


def _encode_tool_output(output: Any) -> Any:
    """JSON-safe storage form, tagging langgraph ``Command`` for rebuild.

    A plain ``_json_safe`` coercion would flatten a ``Command`` to a dict (or
    repr), and a later replay would hand the graph that raw dict instead of a
    real state-update — silently changing graph behaviour. The envelope keeps
    the constructor fields so :func:`_decode_tool_output` can rebuild it.
    """
    output_type = type(output)
    if output_type.__name__ == "Command" and output_type.__module__.startswith(
        ("langgraph",)
    ):
        return {
            "__timetravel_command__": {
                "update": _json_safe(getattr(output, "update", None)),
                "resume": _json_safe(getattr(output, "resume", None)),
                "goto": _json_safe(getattr(output, "goto", None)),
                "graph": _json_safe(getattr(output, "graph", None)),
            }
        }
    return _json_safe(output)


def _decode_tool_output(stored: Any) -> Any:
    """Rebuild a langgraph ``Command`` from its stored envelope."""
    if not isinstance(stored, dict) or "__timetravel_command__" not in stored:
        return stored
    envelope = stored["__timetravel_command__"]
    # pylint: disable=import-outside-toplevel
    try:
        from langgraph.types import Command
    except ImportError:  # pragma: no cover - envelope without langgraph
        return envelope
    # pylint: enable=import-outside-toplevel
    return Command(
        update=envelope.get("update"),
        resume=envelope.get("resume"),
        goto=envelope.get("goto") or (),
        graph=envelope.get("graph") or None,
    )


def _capture_live_tool(
    session: ReplaySession,
    tool_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
) -> None:
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import SpanKind
    from agent_timetravel.tool_intercept import _capture_live_tool_span, _tool_args_hash
    # pylint: enable=import-outside-toplevel

    # langgraph state-updating tools return ``Command`` objects; storage
    # requires JSON-native payloads. The envelope round-trips back into a
    # ``Command`` on replay; the graph itself received the original object
    # from the forward call.
    _capture_live_tool_span(
        session,
        tool_name=tool_name,
        tool_kind=SpanKind.TOOL,
        args=args,
        kwargs=kwargs,
        args_hash=_tool_args_hash(args, kwargs),
        output=_encode_tool_output(output),
    )


def _frozen_tool_miss(session: ReplaySession, tool_name: str) -> None:
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import ReplayMode
    from agent_timetravel.replay import ReplayError
    # pylint: enable=import-outside-toplevel

    if session.mode is ReplayMode.FROZEN:
        raise ReplayError(
            f"frozen LangGraph replay diverged at tool `{tool_name}` "
            f"(cursor={session.cursor}); no recorded fixture to serve"
        )


def _complete_tool_sync(session: ReplaySession, step: Step, output: Any) -> None:
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import complete_step_sync
    # pylint: enable=import-outside-toplevel

    complete_step_sync(session, step, _tool_result_text(output))


async def _complete_tool_async(session: ReplaySession, step: Step, output: Any) -> None:
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import complete_step
    # pylint: enable=import-outside-toplevel

    await complete_step(session, step, _tool_result_text(output))


def _dispatch_tool_sync(
    session: ReplaySession,
    forward: Callable[[Any, dict[str, Any]], Any],
    *,
    tool_name: str,
    input: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Gate → replay → forward for a sync ``BaseTool`` call.

    An EDIT decision rewrites both the debugger view *and* the arguments the
    live tool receives — the edited call is what executes.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind
    # pylint: enable=import-outside-toplevel

    call_input, call_kwargs = input, dict(kwargs)
    view_args, view_kwargs = _safe_tool_inputs((call_input,), call_kwargs)
    step = _tool_step(session, tool_name, view_args, view_kwargs)
    decision = _gate_llm_sync(session, step)
    if decision is not None:
        if decision.kind in (
            DecisionKind.MOCK,
            DecisionKind.SKIP,
            DecisionKind.REJECT,
        ):
            output = _no_call_result(decision, tool_name)
            _complete_tool_sync(session, step, output)
            return output
        if decision.kind is DecisionKind.EDIT:
            call_input, call_kwargs = _apply_tool_edit(decision, call_input, call_kwargs)
            view_args, view_kwargs = _safe_tool_inputs((call_input,), call_kwargs)
            step = _tool_step(session, tool_name, view_args, view_kwargs)

    recorded = _tool_replay_lookup(session, tool_name, tuple(view_args), view_kwargs)
    if recorded is not None:
        output = _decode_tool_output(recorded.payload.get("output"))
        _complete_tool_sync(session, step, output)
        return output

    _frozen_tool_miss(session, tool_name)
    output = forward(call_input, call_kwargs)
    _capture_live_tool(session, tool_name, tuple(view_args), view_kwargs, output)
    _complete_tool_sync(session, step, output)
    return output


async def _dispatch_tool_async(
    session: ReplaySession,
    forward: Callable[[Any, dict[str, Any]], Awaitable[Any]],
    *,
    tool_name: str,
    input: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Gate → replay → forward for an async ``BaseTool`` call (see sync dual)."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind
    # pylint: enable=import-outside-toplevel

    call_input, call_kwargs = input, dict(kwargs)
    view_args, view_kwargs = _safe_tool_inputs((call_input,), call_kwargs)
    step = _tool_step(session, tool_name, view_args, view_kwargs)
    decision = await _gate_llm_async(session, step)
    if decision is not None:
        if decision.kind in (
            DecisionKind.MOCK,
            DecisionKind.SKIP,
            DecisionKind.REJECT,
        ):
            output = _no_call_result(decision, tool_name)
            await _complete_tool_async(session, step, output)
            return output
        if decision.kind is DecisionKind.EDIT:
            call_input, call_kwargs = _apply_tool_edit(decision, call_input, call_kwargs)
            view_args, view_kwargs = _safe_tool_inputs((call_input,), call_kwargs)
            step = _tool_step(session, tool_name, view_args, view_kwargs)

    recorded = _tool_replay_lookup(session, tool_name, tuple(view_args), view_kwargs)
    if recorded is not None:
        output = _decode_tool_output(recorded.payload.get("output"))
        await _complete_tool_async(session, step, output)
        return output

    _frozen_tool_miss(session, tool_name)
    output = await forward(call_input, call_kwargs)
    _capture_live_tool(session, tool_name, tuple(view_args), view_kwargs, output)
    await _complete_tool_async(session, step, output)
    return output


def _apply_tool_edit(
    decision: Decision,
    input: Any,
    kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply an EDIT decision to the tool input/kwargs that will execute.

    The step payload carries the tool input as ``args[0]``, so the edited
    positional list replaces the input when present; edited kwargs merge over
    the originals. Non-JSON injected objects the editor echoed back as their
    repr (e.g. langchain's ``ToolRuntime``) are restored from the original
    call, so editing a runtime-injected tool's visible args still executes.
    """
    edited_input = input
    if decision.args is not None and len(decision.args) >= 1:
        edited_input = _restore_unserializable(input, decision.args[0])
    edited_kwargs = dict(kwargs)
    if decision.kwargs is not None:
        for key, value in decision.kwargs.items():
            edited_kwargs[key] = _restore_unserializable(kwargs.get(key), value)
    return edited_input, edited_kwargs


def _restore_unserializable(original: Any, edited: Any) -> Any:
    """Deep-restore original objects the JSON editor could not round-trip.

    The debugger shows non-JSON injected args as their ``repr``; an edited
    payload therefore echoes those strings back verbatim. Wherever an edited
    leaf is exactly the repr of a non-JSON original, the original object is
    reinstated — untouched runtime state survives the edit while genuinely
    edited values pass through unchanged.
    """
    if isinstance(original, dict) and isinstance(edited, dict):
        restored = dict(edited)
        for key, value in original.items():
            if key in restored:
                restored[key] = _restore_unserializable(value, restored[key])
        return restored
    if (
        isinstance(original, (list, tuple))
        and isinstance(edited, list)
        and len(original) == len(edited)
    ):
        return [
            _restore_unserializable(o, e)
            for o, e in zip(original, edited, strict=True)
        ]
    if original is None or isinstance(original, (str, int, float, bool)):
        return edited
    if isinstance(edited, str) and edited in (repr(original), str(original)):
        return original
    return edited
