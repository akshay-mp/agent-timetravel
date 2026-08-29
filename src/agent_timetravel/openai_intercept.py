"""Phase 3 — OpenAI Chat Completions interceptor.

Track 3B.2 of plan §6 Phase 3. Patches ``openai.resources.chat.completions.
Completions.create`` and ``AsyncCompletions.create`` (non-streaming) so
that during a :func:`timetravel.replay` context:

* Calls matching a recorded LLM span ``<= cursor`` are served from the
  recorded ``raw_attributes`` payload (zero outbound traffic).
* Calls *beyond* the cursor (``BRANCH`` / ``FULL_RERUN`` only) are
  forwarded live and the new span captured under the replay branch.
* Live-forwarded async calls with an approval channel attached are
  transparently upgraded to ``stream=True`` so reasoning fragments reach
  the workbench UI while the model works; the chunks are reassembled into
  the non-streaming ``ChatCompletion`` the caller expects (disable with
  ``AGENT_TIMETRAVEL_DISABLE_STREAM_CAPTURE``).

The interceptor never imports ``openai`` at module load — it does so lazily
on :func:`patch` so projects without ``openai`` installed can still use
the rest of TimeTravel. It degrades to a no-op when no replay is active.

Reentrancy: the interceptor consults :func:`timetravel.replay.active_session`,
a :class:`contextvars.ContextVar`, so concurrent replay sessions in the
Phase 5.5 eval harness are isolated per task. Install/uninstall is also
idempotent — nested ``with patch():`` calls do not double-restore.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_timetravel.replay import CallSignature, ReplaySession

__all__ = ["InterceptError", "capture_only", "extract_signature", "last_wire_raw", "patch"]


class InterceptError(RuntimeError):
    """Raised when the interceptor cannot satisfy a frozen-replay contract."""


_PATCH_LOCK = Lock()
_PATCH_DEPTH = 0
_PATCHED_SYNC_CLASS: Any = None
_PATCHED_ASYNC_CLASS: Any = None
_ORIGINAL_SYNC_CREATE: Any = None
_ORIGINAL_ASYNC_CREATE: Any = None

#: Observe-only mode, owned by the LangGraph interceptor. While set, a
#: patched SDK ``create`` neither gates nor records — it forwards verbatim
#: and stashes the raw wire response for the framework-level dispatcher to
#: merge (langchain drops ``reasoning_content`` before the model-level
#: interceptor can see it). See :func:`capture_only` / :func:`last_wire_raw`.
_CAPTURE_ONLY: Any = None
_WIRE_RAW: Any = None


def _capture_only_var() -> Any:
    global _CAPTURE_ONLY
    if _CAPTURE_ONLY is None:
        # pylint: disable=import-outside-toplevel
        from contextvars import ContextVar
        # pylint: enable=import-outside-toplevel

        _CAPTURE_ONLY = ContextVar("timetravel_capture_only", default=False)
    return _CAPTURE_ONLY


def _wire_raw_var() -> Any:
    global _WIRE_RAW
    if _WIRE_RAW is None:
        # pylint: disable=import-outside-toplevel
        from contextvars import ContextVar
        # pylint: enable=import-outside-toplevel

        _WIRE_RAW = ContextVar("timetravel_wire_raw", default=None)
    return _WIRE_RAW


@contextmanager
def capture_only() -> Iterator[dict[str, Any]]:
    """Forward SDK calls verbatim while stashing the raw wire response.

    Yields a shared holder dict so the stashed payload stays visible across
    task/context boundaries (ContextVar ``set`` inside a task would not
    propagate back to the framework dispatcher that opened this block).
    """
    holder: dict[str, Any] = {"raw": None}
    token = _capture_only_var().set(holder)
    try:
        yield holder
    finally:
        _capture_only_var().reset(token)


def _wire_raw_holder() -> dict[str, Any] | None:
    holder = _capture_only_var().get()
    return holder if isinstance(holder, dict) else None


def last_wire_raw() -> dict[str, Any] | None:
    """Raw ``gen_ai.*`` payload of the last capture-only SDK call, if any."""
    holder = _wire_raw_holder()
    raw = holder.get("raw") if holder else None
    return raw if isinstance(raw, dict) else None


# ----------------------------------------------------------------------
# Signature extraction
# ----------------------------------------------------------------------
def extract_signature(**kwargs: Any) -> CallSignature:
    """Build a :class:`~timetravel.replay.CallSignature` from a Chat Completions call.

    Matches the SDK call style ``create(model=..., messages=..., tools=...)``.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.models import hash_payload
    from agent_timetravel.replay import CallSignature
    # pylint: enable=import-outside-toplevel

    model = str(kwargs.get("model", ""))
    messages = _to_jsonable(kwargs.get("messages") or [])
    tools_raw = kwargs.get("tools") or None
    tools_jsonable = _to_jsonable(tools_raw) if tools_raw is not None else None

    return CallSignature(
        model=model,
        messages_hash=hash_payload(messages),
        tools_hash=hash_payload(tools_jsonable) if tools_jsonable is not None else None,
    )


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce untyped inputs (pydantic, dataclasses) to plain JSON."""
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


# ----------------------------------------------------------------------
# Frozen response re-materialisation
# ----------------------------------------------------------------------
def _materialise_chat_completion(payload: dict[str, Any], sdk_module: Any) -> Any:
    """Re-build an SDK ``ChatCompletion`` from a stored raw payload.

    Handles three recorded-span conventions:
    * GenAI semconv: ``gen_ai.response`` (full chat-completion JSON).
    * Older exporters: ``raw_response`` / ``response``.
    * OpenInference flat: ``llm.output_messages.0.message.content`` —
      synthesised into a minimal chat-completion shape (the OpenInference
      instrumentor stores flat keys, not a nested response object).
    Falls back to a minimal but valid response so privacy-skinned exporters
    still replay.
    """
    response_json = (
        payload.get("gen_ai.response")
        or payload.get("raw_response")
        or payload.get("response")
    )
    if response_json is None:
        # OpenInference flat format — synthesise from the flat keys.
        response_json = _minimal_response(payload)
    if sdk_module is None:
        return response_json
    construct = getattr(sdk_module, "model_validate", None)
    if construct is None:
        return response_json
    return construct(response_json)


def _extract_output_content(payload: dict[str, Any]) -> str:
    """Pull the assistant content from OpenInference flat keys.

    OpenInference stores ``llm.output_messages.0.message.content`` as a flat
    string rather than nesting it under a response object. Returns the first
    non-empty output message content, or "" if none found.
    """
    for key, val in payload.items():
        if (
            key.startswith("llm.output_messages.")
            and key.endswith(".message.content")
            and isinstance(val, str)
            and val
        ):
            return val
    return ""


def _minimal_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal but valid Chat Completion payload from typed fields.

    Handles both GenAI semconv (``gen_ai.*``) and OpenInference flat
    (``llm.*``) attribute conventions.
    """
    # Content: try GenAI first, then OpenInference flat keys.
    content = ""
    response = payload.get("gen_ai.response")
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices, list):
            first = choices[0] if choices else {}
            msg = first.get("message") if isinstance(first, dict) else {}
            c = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(c, str):
                content = c
    if not content:
        content = _extract_output_content(payload)

    return {
        "id": "timetravel-replay",
        "object": "chat.completion",
        "created": 0,
        "model": payload.get("gen_ai.response.model")
        or payload.get("llm.model_name")
        or "timetravel-replay",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": payload.get("llm.finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": payload.get("gen_ai.usage.prompt_tokens")
            or payload.get("llm.token_count.prompt", 0),
            "completion_tokens": payload.get("gen_ai.usage.completion_tokens")
            or payload.get("llm.token_count.completion", 0),
            "total_tokens": payload.get("gen_ai.usage.total_tokens")
            or payload.get("llm.token_count.total", 0),
        },
    }


# ----------------------------------------------------------------------
# Live span capture
# ----------------------------------------------------------------------
def _capture_live_span(
    session: ReplaySession,
    *,
    kwargs: dict[str, Any],
    response: Any,
    signature_model: str,
) -> None:
    """Build a :class:`~agent_timetravel.models.Span` for a live-forwarded call.

    Persisted under ``session.branch_id`` so the live tail queries as a
    distinct branch timeline.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import SpanKind, SpanStatus
    from agent_timetravel.models import Span, hash_payload
    # pylint: enable=import-outside-toplevel

    raw = _response_to_raw(response, kwargs)
    span = Span(
        trace_id=session.trace_id,
        span_id=_gen_span_id_hex(),
        parent_span_id=None,
        name="chat.completions.create",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        model_name=signature_model,
        prompt_tokens=raw.get("gen_ai.usage.prompt_tokens"),
        completion_tokens=raw.get("gen_ai.usage.completion_tokens"),
        total_tokens=raw.get("gen_ai.usage.total_tokens"),
        messages_hash=hash_payload(_to_jsonable(kwargs.get("messages") or [])),
        tools_hash=(
            hash_payload(_to_jsonable(kwargs.get("tools")))
            if kwargs.get("tools")
            else None
        ),
        raw_attributes=raw,
    )
    session.record_new(span)


def _response_to_raw(response: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Flatten a live SDK ChatCompletion into the TimeTravel raw_attributes shape."""
    # langchain-openai calls ``client.with_raw_response.create(...)`` and gets
    # a ``LegacyAPIResponse`` wrapper — unwrap to the inner ChatCompletion or
    # the payload (and the provider reasoning it carries) is lost.
    # pylint: disable=import-outside-toplevel
    parse = getattr(response, "parse", None)
    # pylint: enable=import-outside-toplevel
    if parse is not None and not isinstance(response, dict):
        with suppress(Exception):
            response = parse()
    payload = _to_jsonable(response)
    raw: dict[str, Any] = {}
    raw["gen_ai.request.model"] = str(kwargs.get("model", ""))
    if isinstance(payload, dict):
        raw["gen_ai.response"] = payload
        model = payload.get("model")
        if isinstance(model, str):
            raw["gen_ai.response.model"] = model
        usage = payload.get("usage") or {}
        if isinstance(usage, dict):
            for side in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = usage.get(side)
                if isinstance(val, int):
                    raw[f"gen_ai.usage.{side}"] = val
    return raw


def _gen_span_id_hex() -> str:
    """Generate a fresh, valid OTel 16-hex-char span id."""
    # pylint: disable=import-outside-toplevel
    from secrets import token_hex
    # pylint: enable=import-outside-toplevel

    return token_hex(8)


# ----------------------------------------------------------------------
# Patch installation
# ----------------------------------------------------------------------
@contextmanager
def patch() -> Iterator[None]:
    """Monkey-patch ``openai`` Chat Completions for the duration of the block.

    Restores the original methods on exit even if the body raises. Installing
    a second time while one is already active is a no-op (so nested replays
    don't double-restore).
    """
    # pylint: disable=import-outside-toplevel
    try:
        import openai.resources.chat.completions as completions_mod
    except ImportError as exc:  # pragma: no cover - exercised only without openai
        raise InterceptError(
            "agent_timetravel.replay requires the `openai` package; install it or use the "
            "adapter path (agent_timetravel.adapters.<framework>)."
        ) from exc

    from agent_timetravel.replay import active_session
    # pylint: enable=import-outside-toplevel

    CompletionsCls = completions_mod.Completions
    AsyncCompletionsCls = getattr(completions_mod, "AsyncCompletions", None)

    global _PATCH_DEPTH, _PATCHED_SYNC_CLASS, _PATCHED_ASYNC_CLASS
    global _ORIGINAL_SYNC_CREATE, _ORIGINAL_ASYNC_CREATE

    with _PATCH_LOCK:
        if _PATCH_DEPTH:
            _PATCH_DEPTH += 1
        else:
            orig_sync_create = CompletionsCls.create
            orig_async_create = (
                AsyncCompletionsCls.create if AsyncCompletionsCls is not None else None
            )

            def patched_sync_create(self: Any, *args: Any, **kwargs: Any) -> Any:
                session = active_session()
                if session is None:
                    return orig_sync_create(self, *args, **kwargs)
                return _dispatch_sync(self, session, orig_sync_create, args, kwargs)

            async def patched_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
                session = active_session()
                if session is None:
                    return await orig_async_create(self, *args, **kwargs)  # type: ignore[misc]
                return await _dispatch_async(self, session, orig_async_create, args, kwargs)

            patched_sync_create.__timetravel_patched__ = True  # type: ignore[attr-defined]
            patched_async_create.__timetravel_patched__ = True  # type: ignore[attr-defined]
            try:
                CompletionsCls.create = patched_sync_create  # type: ignore[method-assign]
                if AsyncCompletionsCls is not None:
                    AsyncCompletionsCls.create = patched_async_create
            except Exception:
                CompletionsCls.create = orig_sync_create  # type: ignore[method-assign]
                if AsyncCompletionsCls is not None and orig_async_create is not None:
                    AsyncCompletionsCls.create = orig_async_create
                raise

            _PATCHED_SYNC_CLASS = CompletionsCls
            _PATCHED_ASYNC_CLASS = AsyncCompletionsCls
            _ORIGINAL_SYNC_CREATE = orig_sync_create
            _ORIGINAL_ASYNC_CREATE = orig_async_create
            _PATCH_DEPTH = 1

    try:
        yield
    finally:
        with _PATCH_LOCK:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0:
                _PATCHED_SYNC_CLASS.create = _ORIGINAL_SYNC_CREATE
                if _PATCHED_ASYNC_CLASS is not None and _ORIGINAL_ASYNC_CREATE is not None:
                    _PATCHED_ASYNC_CLASS.create = _ORIGINAL_ASYNC_CREATE
                _PATCHED_SYNC_CLASS = None
                _PATCHED_ASYNC_CLASS = None
                _ORIGINAL_SYNC_CREATE = None
                _ORIGINAL_ASYNC_CREATE = None


# ----------------------------------------------------------------------
# Dispatch (the actual frozen vs. forward logic)
# ----------------------------------------------------------------------
def _dispatch_sync(
    self: Any,
    session: ReplaySession,
    orig_create: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Decide serve-vs-forward for a sync ``create`` call.

    Streaming frozen replay fails closed — streaming-chunk caching lands
    in Phase 5 polish. ``mode=branch`` forwards live (then captures).
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import ReplayMode
    from agent_timetravel.replay import ReplayError
    # pylint: enable=import-outside-toplevel

    if _capture_only_var().get():
        response = orig_create(self, *args, **kwargs)
        holder = _wire_raw_holder()
        if holder is not None:
            holder["raw"] = _response_to_raw(response, kwargs)
        return response
    if kwargs.get("stream") and session.mode is ReplayMode.FROZEN:
        raise ReplayError(
            "frozen streaming replay not yet supported (Phase 5); "
            "use non-streaming calls or mode=branch"
        )
    kwargs, step = _step_sync(session, kwargs)
    signature = extract_signature(**kwargs)
    recorded = session.respond_or_forward(signature)
    if recorded is not None:
        response = _materialise_chat_completion(recorded.payload, _chat_completion_module())
        _complete_step_sync(session, recorded.payload, signature.model, step, kwargs)
        return response
    response = orig_create(self, *args, **kwargs)
    _capture_live_span(
        session, kwargs=kwargs, response=response, signature_model=signature.model
    )
    _complete_step_sync(session, _response_to_raw(response, kwargs), signature.model, step, kwargs)
    return response


async def _dispatch_async(
    self: Any,
    session: ReplaySession,
    orig_create: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Async dual of :func:`_dispatch_sync`."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.enums import ReplayMode
    from agent_timetravel.replay import ReplayError
    # pylint: enable=import-outside-toplevel

    if _capture_only_var().get():
        # Framework-owned dispatch (LangGraph): observe and hand the raw
        # wire payload to the dispatcher — no gate, no replay, no spans.
        response = await orig_create(self, *args, **kwargs)
        holder = _wire_raw_holder()
        if holder is not None:
            holder["raw"] = _response_to_raw(response, kwargs)
        return response
    if kwargs.get("stream") and session.mode is ReplayMode.FROZEN:
        raise ReplayError(
            "frozen streaming replay not yet supported (Phase 5); "
            "use non-streaming calls or mode=branch"
        )
    kwargs, step = await _step_async(session, kwargs)
    signature = extract_signature(**kwargs)
    recorded = session.respond_or_forward(signature)
    if recorded is not None:
        response = _materialise_chat_completion(recorded.payload, _chat_completion_module())
        await _complete_step(session, recorded.payload, signature.model, step, kwargs)
        return response
    if _stream_capture_enabled(session):
        response = await _forward_streaming_async(self, session, orig_create, args, kwargs, step)
    else:
        response = await orig_create(self, *args, **kwargs)
    _capture_live_span(
        session, kwargs=kwargs, response=response, signature_model=signature.model
    )
    await _complete_step(session, _response_to_raw(response, kwargs), signature.model, step, kwargs)
    return response


# ----------------------------------------------------------------------
# Live streaming capture (reasoning deltas)
# ----------------------------------------------------------------------
#: Env kill-switch: any non-empty value disables transparent streaming
#: capture so calls are forwarded exactly as the agent issued them.
_STREAM_CAPTURE_DISABLE_ENV = "AGENT_TIMETRAVEL_DISABLE_STREAM_CAPTURE"
#: Coalesce reasoning fragments into at most one delta event per interval,
#: bounding pressure on the channel's capacity-limited event queue.
_REASONING_DELTA_INTERVAL_S = 0.12
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _stream_capture_enabled(session: ReplaySession) -> bool:
    """Transparent streaming is on only while a UI is watching the run."""
    # pylint: disable=import-outside-toplevel
    import os
    # pylint: enable=import-outside-toplevel

    if os.environ.get(_STREAM_CAPTURE_DISABLE_ENV):
        return False
    channel = getattr(session, "approval", None)
    return channel is not None and getattr(channel, "emit_delta", None) is not None


def _thinking_so_far(content: str, reasoning: str) -> str:
    """Best-effort reasoning text from the accumulated stream buffers.

    Handles both provider conventions: separate ``reasoning_content``
    deltas and Gemma-style inline ``<think>…</think>`` markers — including a
    still-open ``<think>`` and markers split across chunks (a partial opener
    never matches, so tag fragments do not leak as reasoning text).
    """
    # pylint: disable=import-outside-toplevel
    import re
    # pylint: enable=import-outside-toplevel

    match = re.search(rf"<think>([\s\S]*?){_THINK_CLOSE}", content, flags=re.IGNORECASE)
    if match is not None:
        inline = match.group(1)
    else:
        start = content.lower().find(_THINK_OPEN)
        inline = content[start + len(_THINK_OPEN):] if start >= 0 else ""
    return reasoning + inline


async def _forward_streaming_async(
    self: Any,
    session: ReplaySession,
    orig_create: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    step: Any,
) -> Any:
    """Forward a live call as a token stream, then reassemble the response.

    The request is transparently upgraded to ``stream=True`` (plus
    ``include_usage``) so reasoning fragments can be published to the
    approval channel while the model works. Chunks are reassembled into the
    same non-streaming ``ChatCompletion`` a plain forward would have
    returned, keeping span capture, usage extraction, and the caller's own
    code unchanged.
    """
    # pylint: disable=import-outside-toplevel
    import time
    # pylint: enable=import-outside-toplevel

    stream_options = {**(kwargs.get("stream_options") or {}), "include_usage": True}
    stream_kwargs = {**kwargs, "stream": True, "stream_options": stream_options}

    emit_delta = getattr(session.approval, "emit_delta", None)
    cursor = getattr(step, "cursor", 0)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    meta: dict[str, Any] = {"id": None, "created": None, "model": None, "finish_reason": None}
    usage: dict[str, Any] | None = None
    emitted = 0
    last_flush = time.monotonic()

    def flush(force: bool = False) -> None:
        """Publish newly accumulated reasoning, coalesced by interval."""
        nonlocal emitted, last_flush
        if emit_delta is None:
            return
        now = time.monotonic()
        if not force and now - last_flush < _REASONING_DELTA_INTERVAL_S:
            return
        thinking = _thinking_so_far("".join(content_parts), "".join(reasoning_parts))
        if len(thinking) > emitted:
            emit_delta(cursor, thinking[emitted:])
            emitted = len(thinking)
        last_flush = now

    async for chunk in await orig_create(self, *args, **stream_kwargs):
        meta["id"] = getattr(chunk, "id", None) or meta["id"]
        meta["created"] = getattr(chunk, "created", None) or meta["created"]
        meta["model"] = getattr(chunk, "model", None) or meta["model"]
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = _to_jsonable(chunk_usage)
        choices = getattr(chunk, "choices", None) or []
        if choices:
            choice = choices[0]
            meta["finish_reason"] = (
                getattr(choice, "finish_reason", None) or meta["finish_reason"]
            )
            delta = getattr(choice, "delta", None)
            if delta is not None:
                piece = getattr(delta, "content", None)
                if isinstance(piece, str) and piece:
                    content_parts.append(piece)
                reasoning_piece = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                )
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
                for call in getattr(delta, "tool_calls", None) or []:
                    index = getattr(call, "index", 0) or 0
                    slot = tool_calls.setdefault(
                        index,
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    call_id = getattr(call, "id", None)
                    if call_id:
                        slot["id"] = call_id
                    function = getattr(call, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        if name:
                            slot["function"]["name"] += name
                        arguments = getattr(function, "arguments", None)
                        if isinstance(arguments, str) and arguments:
                            slot["function"]["arguments"] += arguments
        flush()
    flush(force=True)

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    reasoning = "".join(reasoning_parts)
    if reasoning.strip():
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    payload: dict[str, Any] = {
        "id": meta["id"] or "chatcmpl-timetravel-stream",
        "object": "chat.completion",
        "created": meta["created"] or 0,
        "model": meta["model"] or str(kwargs.get("model", "")),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": meta["finish_reason"] or "stop",
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return _materialise_chat_completion({"gen_ai.response": payload}, _chat_completion_module())


async def _complete_step(
    session: ReplaySession,
    payload: dict[str, Any],
    model: str,
    step: Any,
    request: dict[str, Any],
) -> None:
    """Emit a step_completed event with the response text (the verify loop).

    Extracts the assistant content from the raw payload and forwards it to
    :func:`agent_timetravel.stepping.complete_step` so the UI can show what the model
    returned before the developer chooses next/back/stop. A no-op when no
    channel is attached.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import complete_step
    # pylint: enable=import-outside-toplevel

    result = _extract_response_text(payload, model)
    await complete_step(session, step, result, usage=_extract_usage(payload, request, result))


def _complete_step_sync(
    session: ReplaySession,
    payload: dict[str, Any],
    model: str,
    step: Any,
    request: dict[str, Any],
) -> None:
    """Publish a sync response to a channel that supports post-call review."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import complete_step_sync
    # pylint: enable=import-outside-toplevel

    result = _extract_response_text(payload, model)
    complete_step_sync(
        session,
        step,
        result,
        usage=_extract_usage(payload, request, result),
    )


def _extract_usage(
    payload: dict[str, Any], request: dict[str, Any], response_text: str
) -> dict[str, int]:
    """Return provider usage, or a clearly-marked local fallback estimate.

    Local OpenAI-compatible servers commonly return a syntactically-valid
    ``usage`` object filled with zeroes. In that case account from the actual
    request and unmodified completion, including a ``<think>`` block.
    """
    response = payload.get("gen_ai.response")
    nested = response.get("usage") if isinstance(response, dict) else None
    usage = nested if isinstance(nested, dict) else {}

    def token_count(name: str) -> int | None:
        value = usage.get(name, payload.get(f"gen_ai.usage.{name}"))
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    input_tokens = token_count("prompt_tokens")
    output_tokens = token_count("completion_tokens")
    total_tokens = token_count("total_tokens")
    has_provider_usage = any(
        value not in (None, 0) for value in (input_tokens, output_tokens, total_tokens)
    )
    thinking_text, final_text = _split_thinking(response_text)
    if has_provider_usage:
        completion_tokens = output_tokens or 0
        thinking_tokens = min(_estimate_tokens(thinking_text), completion_tokens)
        return {
            "input_tokens": input_tokens or 0,
            "output_tokens": completion_tokens,
            "thinking_tokens": thinking_tokens,
            "final_tokens": max(0, completion_tokens - thinking_tokens),
            "total_tokens": (
                total_tokens
                if total_tokens is not None
                else (input_tokens or 0) + completion_tokens
            ),
            "estimated": False,
        }

    estimated_input = _estimate_tokens(_to_jsonable(request.get("messages") or []))
    estimated_thinking = _estimate_tokens(thinking_text)
    estimated_final = _estimate_tokens(final_text)
    estimated_output = estimated_thinking + estimated_final
    return {
        "input_tokens": estimated_input,
        "output_tokens": estimated_output,
        "thinking_tokens": estimated_thinking,
        "final_tokens": estimated_final,
        "total_tokens": estimated_input + estimated_output,
        "estimated": True,
    }


def _split_thinking(text: str) -> tuple[str, str]:
    """Separate explicit provider reasoning without removing it from accounting."""
    import re  # pylint: disable=import-outside-toplevel

    match = re.search(r"<think>([\s\S]*?)</think>\s*", text, flags=re.IGNORECASE)
    if match is None:
        return "", text
    return match.group(1), text[: match.start()] + text[match.end() :]


def _estimate_tokens(value: Any) -> int:
    """Conservative local fallback when the model server reports no usage.

    This deliberately avoids pretending to be Gemma's tokenizer. It uses the
    common four-characters-per-token approximation over the exact JSON prompt
    and raw completion so totals remain useful for local capacity planning.
    """
    import json  # pylint: disable=import-outside-toplevel

    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    return (len(text) + 3) // 4 if text else 0


def _extract_response_text(payload: dict[str, Any], model: str) -> str:
    """Pull the assistant's textual content out of a chat-completion payload.

    Handles three conventions:
    * GenAI semconv: ``gen_ai.response.choices[0].message.content``
    * OpenInference flat: ``llm.output_messages.0.message.content``
    * Older exporters: ``raw_response`` / ``response`` as a string or dict.
    """
    # 1. OpenInference flat keys (what OpenInference's OpenAI instrumentor emits).
    for key, val in payload.items():
        if (
            key.startswith("llm.output_messages.")
            and key.endswith(".message.content")
            and isinstance(val, str)
        ):
            return val
    # 2. GenAI semconv nested response.
    response = (
        payload.get("gen_ai.response")
        or payload.get("raw_response")
        or payload.get("response")
        or {}
    )
    if isinstance(response, str):
        return response
    choices = response.get("choices") if isinstance(response, dict) else None
    if choices and isinstance(choices, list):
        first = choices[0] if choices else {}
        message = first.get("message") if isinstance(first, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        reasoning = (
            message.get("reasoning_content") or message.get("reasoning")
            if isinstance(message, dict)
            else None
        )
        if isinstance(content, str):
            reasoning_text = reasoning if isinstance(reasoning, str) else ""
            has_separate_reasoning = (
                bool(reasoning_text.strip())
                and "<think>" not in content.lower()
            )
            if has_separate_reasoning:
                return f"<think>{reasoning_text.strip()}</think>\n{content}"
            return content
        if isinstance(reasoning, str) and reasoning.strip():
            return f"<think>{reasoning.strip()}</think>"
    return ""


def _build_llm_step(session: ReplaySession, kwargs: dict[str, Any]) -> Any:
    """Build the shared transport-friendly LLM step payload."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import Step, StepKind
    # pylint: enable=import-outside-toplevel

    messages = _to_jsonable(kwargs.get("messages") or [])
    tools_raw = kwargs.get("tools") or None
    params = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")}
    # Phase 3.1: lift the known sampling parameters into a structured sub-dict
    # so the UI can render them prominently (temperature, seed, max_tokens,
    # response_format, tool_choice). They remain in ``params`` too for
    # completeness; ``sampling`` is the curated projection the diff UI reads.
    sampling_keys = (
        "temperature",
        "seed",
        "max_tokens",
        "response_format",
        "tool_choice",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    )
    sampling = {
        k: _to_jsonable(params[k])
        for k in sampling_keys
        if k in params
    }
    step = Step(
        kind=StepKind.LLM,
        payload={
            "model": str(kwargs.get("model", "")),
            "messages": messages,
            "tools": _to_jsonable(tools_raw) if tools_raw is not None else None,
            "params": params,
            "sampling": sampling,
        },
        cursor=session.cursor,
    )
    return step


async def _step_async(
    session: ReplaySession,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Interactive stepping gate for the async Chat Completions path.

    Returns the (possibly edited) kwargs. Raises
    :class:`~agent_timetravel.stepping.SteppingStopped` on STOP. A no-op when no
    approval channel is attached.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind, SteppingStopped, gate_async
    # pylint: enable=import-outside-toplevel

    step = _build_llm_step(session, kwargs)
    decision = await gate_async(session, step)
    if decision is None:
        return kwargs, step
    if decision.kind is DecisionKind.STOP:
        raise SteppingStopped(step)
    if decision.kind is DecisionKind.EDIT:
        if decision.messages is not None:
            kwargs = {**kwargs, "messages": decision.messages}
        if decision.model is not None:
            kwargs = {**kwargs, "model": decision.model}
        if decision.params is not None:
            kwargs = {**kwargs, **decision.params}
    return kwargs, step


def _step_sync(
    session: ReplaySession,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Interactive stepping gate for the sync Chat Completions path."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind, SteppingStopped, gate_sync
    # pylint: enable=import-outside-toplevel

    step = _build_llm_step(session, kwargs)
    decision = gate_sync(session, step)
    if decision is None:
        return kwargs, step
    if decision.kind is DecisionKind.STOP:
        raise SteppingStopped(step)
    if decision.kind is DecisionKind.EDIT:
        if decision.messages is not None:
            kwargs = {**kwargs, "messages": decision.messages}
        if decision.model is not None:
            kwargs = {**kwargs, "model": decision.model}
        if decision.params is not None:
            kwargs = {**kwargs, **decision.params}
    return kwargs, step


def _chat_completion_module() -> Any:
    """Return the SDK module hosting ``ChatCompletion`` (typed or None)."""
    # pylint: disable=import-outside-toplevel
    try:
        from openai.types.chat import ChatCompletion as _cc
    except ImportError:
        return None
    return _cc
    # pylint: enable=import-outside-toplevel
