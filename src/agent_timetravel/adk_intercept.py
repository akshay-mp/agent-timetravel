"""Generic Google ADK workbench interceptor.

Patches every ``google.adk.models.BaseLlm`` subclass's
``generate_content_async`` (and every ``google.adk.tools.BaseTool`` subclass's
``run_async``) at class level for the duration of a workbench run, so **any**
ADK agent — including one that constructs its models and tools inside
callbacks at call time — is stepped, replayed, and captured without the
developer wrapping anything manually.

Unlike langchain's template-method ``invoke``, ADK's concrete models
(``Gemini``, ``GemmaLlm``, ``LiteLlm``, ``AnthropicLlm``, user subclasses…)
each override ``generate_content_async`` directly, so there is no single
base implementation to patch. The interceptor therefore:

1. walks ``BaseLlm.__subclasses__()`` recursively and wraps every class that
   defines its own ``generate_content_async`` (same walk for ``BaseTool`` /
   ``run_async``);
2. installs a temporary ``__init_subclass__`` hook on both bases so classes
   defined while the patch is active (late imports, test fakes) are wrapped
   too;
3. restores every wrapped method and the hook on exit — including methods
   installed via the hook.

During a :func:`agent_timetravel.replay.replay` context each intercepted
model call:

* pauses at the stepping gate (:func:`agent_timetravel.stepping.gate_async`)
  so the workbench can approve / edit / stop it;
* is served from a matching recorded span when one sits at the cursor
  (zero outbound traffic) — or forwarded live and captured in
  BRANCH / FULL_RERUN / INTERACTIVE modes;
* raises :class:`~agent_timetravel.replay.ReplayError` on a frozen
  divergence.

With no active replay session every call passes through unchanged. Models
already wrapped by :func:`agent_timetravel.adapters.adk.replay_llm` are
skipped (their ``generate_content_async`` owns the replay contract). Tool
calls reuse the shared TOOL dispatch from
:mod:`agent_timetravel.langgraph_intercept` so ADK tools get the same
mock / skip / reject / edit semantics as langchain tools.

The module never imports ``google.adk`` at load — only inside :func:`patch` —
matching the lazy-import contract of the other interceptors.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import TYPE_CHECKING, Any

from agent_timetravel.openai_intercept import (
    _build_llm_step,
    _to_jsonable,
    extract_signature,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from agent_timetravel.replay import RecordedResponse, ReplaySession
    from agent_timetravel.stepping import Decision, Step

__all__ = ["InterceptError", "patch"]


class InterceptError(RuntimeError):
    """Raised when the ADK interceptor cannot be installed or honoured."""


_PATCH_LOCK = Lock()
_PATCH_DEPTH = 0
_ORIGINALS: dict[tuple[type, str], Any] = {}
_ORIGINAL_INIT_SUBCLASS: dict[type, Any] = {}

#: True while a manual :func:`agent_timetravel.adapters.adk.replay_llm`
#: wrapper is forwarding to its inner model. The interceptor must stand down
#: for the inner call — the wrapper owns that call's replay contract, and
#: intercepting it would record a duplicate span. Scoped as a ContextVar so a
#: direct call on the same inner model from unrelated code is still stepped.
_WRAPPER_FORWARD: ContextVar[bool] = ContextVar("timetravel_adk_wrapper_forward", default=False)

#: Sampling fields lifted from ``LlmRequest.config`` (a ``google.genai``
#: ``GenerateContentConfig``) into the step's structured ``params`` so the UI
#: can render them like the OpenAI/LangGraph interceptors do.
_SAMPLING_CONFIG_ATTRS = (
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "seed",
    "stop_sequences",
)

#: ``decision.params`` key → ``GenerateContentConfig`` attribute. Only keys
#: present here (and on the config model) are applied on EDIT.
_PARAM_TO_CONFIG_ATTR = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_output_tokens",
    "max_output_tokens": "max_output_tokens",
    "seed": "seed",
    "stop_sequences": "stop_sequences",
}


# ----------------------------------------------------------------------
# Patch installation
# ----------------------------------------------------------------------
@contextmanager
def patch() -> Iterator[None]:
    """Monkey-patch ADK's models and tools for the block.

    Restores the original methods on exit even if the body raises. Installing
    a second time while one is already active is a no-op (nested workbench
    runs don't double-restore). Raises :class:`InterceptError` when
    ``google-adk`` is not installed.
    """
    # pylint: disable=import-outside-toplevel
    try:
        from google.adk.models import BaseLlm
        from google.adk.tools.base_tool import BaseTool as AdkBaseTool
    except ImportError as exc:  # pragma: no cover - exercised only without ADK
        raise InterceptError(
            "timetravel workbench ADK activation requires `google-adk`; "
            "install it with `pip install agent-timetravel[adk]`."
        ) from exc

    from agent_timetravel.replay import active_session
    # pylint: enable=import-outside-toplevel

    global _PATCH_DEPTH

    with _PATCH_LOCK:
        if _PATCH_DEPTH:
            _PATCH_DEPTH += 1
        else:
            try:
                _install_model_patch(BaseLlm, active_session)
                _install_tool_patch(AdkBaseTool, active_session)
            except BaseException:
                # BaseException: a KeyboardInterrupt between the two installs
                # must not leak the first patch with depth still at 0.
                _restore_all()
                raise
            _PATCH_DEPTH = 1

    try:
        yield
    finally:
        with _PATCH_LOCK:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0:
                _restore_all()


def _install_model_patch(base: type, active_session: Callable[[], Any]) -> None:
    """Wrap ``generate_content_async`` on every BaseLlm subclass that defines it."""

    def wrap(cls: type) -> None:
        descriptor = cls.__dict__.get("generate_content_async")
        if descriptor is None or getattr(descriptor, "__timetravel_patched__", False):
            return
        # Accept @staticmethod / @classmethod overrides: unwrap the plain
        # function for the forward, restore the original descriptor on exit.
        static = isinstance(descriptor, staticmethod)
        classmethod_orig = isinstance(descriptor, classmethod)
        original: Callable[..., Any] = getattr(descriptor, "__func__", descriptor)

        async def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
            session = active_session()
            if session is None or _is_replay_wrapper(self) or _WRAPPER_FORWARD.get():
                async for response in _call_original(
                    self, original, static, classmethod_orig, args, kwargs
                ):
                    yield response
                return
            async for response in _dispatch_llm_async(
                self, session, original, static, classmethod_orig, args, kwargs
            ):
                yield response

        patched.__timetravel_patched__ = True  # type: ignore[attr-defined]
        cls.generate_content_async = patched  # type: ignore[attr-defined]
        _ORIGINALS[(cls, "generate_content_async")] = descriptor

    def init_subclass(cls: type, **kwargs: Any) -> None:
        super(base, cls).__init_subclass__(**kwargs)  # type: ignore[arg-type]
        wrap(cls)

    _ORIGINAL_INIT_SUBCLASS[base] = base.__dict__.get("__init_subclass__")
    # The hook's parameter shape is checked by CPython at subclass creation,
    # not by mypy — hence the ignores on the assignment below.
    hook: Any = classmethod(init_subclass)  # type: ignore[arg-type]
    base.__init_subclass__ = hook  # type: ignore[method-assign]

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            wrap(sub)
            walk(sub)

    walk(base)


def _install_tool_patch(base: type, active_session: Callable[[], Any]) -> None:
    """Wrap ``run_async`` on every ADK BaseTool subclass that defines it."""

    def wrap(cls: type) -> None:
        descriptor = cls.__dict__.get("run_async")
        if descriptor is None or getattr(descriptor, "__timetravel_patched__", False):
            return
        static = isinstance(descriptor, staticmethod)
        classmethod_orig = isinstance(descriptor, classmethod)
        original: Callable[..., Any] = getattr(descriptor, "__func__", descriptor)

        async def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
            session = active_session()
            if session is None or _WRAPPER_FORWARD.get():
                return await _call_original(self, original, static, classmethod_orig, args, kwargs)
            return await _dispatch_adk_tool_async(
                self, session, original, static, classmethod_orig, args, kwargs
            )

        patched.__timetravel_patched__ = True  # type: ignore[attr-defined]
        cls.run_async = patched  # type: ignore[attr-defined]
        _ORIGINALS[(cls, "run_async")] = descriptor

    def init_subclass(cls: type, **kwargs: Any) -> None:
        super(base, cls).__init_subclass__(**kwargs)  # type: ignore[arg-type]
        wrap(cls)

    _ORIGINAL_INIT_SUBCLASS[base] = base.__dict__.get("__init_subclass__")
    hook: Any = classmethod(init_subclass)  # type: ignore[arg-type]
    base.__init_subclass__ = hook  # type: ignore[method-assign]

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            wrap(sub)
            walk(sub)

    walk(base)


def _restore_all() -> None:
    """Unwind every wrapped method and ``__init_subclass__`` hook."""
    for (cls, name), original in _ORIGINALS.items():
        setattr(cls, name, original)
    _ORIGINALS.clear()
    for base, original in _ORIGINAL_INIT_SUBCLASS.items():
        if original is None:
            delattr(base, "__init_subclass__")
        else:
            base.__init_subclass__ = original  # type: ignore[method-assign]
    _ORIGINAL_INIT_SUBCLASS.clear()


def _is_replay_wrapper(model: Any) -> bool:
    """True when the model is a manual ``adapters.adk.replay_llm`` wrapper."""
    return getattr(model, "_timetravel_wrapped", None) is not None


def _call_original(
    self: Any,
    original: Callable[..., Any],
    static: bool,
    classmethod_orig: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Await/iterate the unwrapped original, rebinding self/cls per descriptor.

    A plain-function override receives ``self``; a staticmethod override
    receives only the caller's args; a classmethod override receives the
    runtime class in its first slot.
    """
    if classmethod_orig:
        return original(type(self), *args, **kwargs)
    if static:
        return original(*args, **kwargs)
    return original(self, *args, **kwargs)


# ----------------------------------------------------------------------
# LLM path
# ----------------------------------------------------------------------
async def _dispatch_llm_async(
    model: Any,
    session: ReplaySession,
    original: Callable[..., Any],
    static: bool,
    classmethod_orig: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Gate → replay-or-forward for an ADK ``generate_content_async`` call.

    An async generator: yields the recorded materialisation on a fixture hit,
    or the live responses while capturing the final one.
    """
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import DecisionKind, SteppingStopped, gate_async
    # pylint: enable=import-outside-toplevel

    llm_request = args[0] if args else kwargs.get("llm_request")
    if llm_request is None or not hasattr(llm_request, "contents"):
        # Not a shape we can build a step from — forward unchanged.
        async for response in _call_original(
            model, original, static, classmethod_orig, args, kwargs
        ):
            yield response
        return

    call_kwargs = _call_kwargs(model, llm_request)
    step = _build_llm_step(session, call_kwargs)
    decision = await gate_async(session, step)
    if decision is not None and decision.kind is DecisionKind.STOP:
        raise SteppingStopped(step)
    if decision is not None and decision.kind is DecisionKind.EDIT:
        _apply_llm_edit(decision, llm_request)
        # The captured span and replay signature must describe the edited
        # call — rebuild the view from the mutated request.
        call_kwargs = _call_kwargs(model, llm_request)
        step = _build_llm_step(session, call_kwargs)

    signature = extract_signature(
        model=call_kwargs["model"],
        messages=call_kwargs["messages"],
        tools=call_kwargs["tools"],
    )
    recorded = session.respond_or_forward(signature)
    if recorded is not None:
        response = _materialise_response(recorded, str(call_kwargs["model"]))
        await _complete_step(session, step, _recorded_result_text(recorded))
        yield response
        return

    assert_not_frozen(session)
    span: Any = None
    # Capture BEFORE yielding: ADK's flow may never resume (or close) this
    # generator after a function-call response — it runs the requested tools
    # and abandons the generator — so post-loop capture would never run.
    # ``insert_span`` upserts on ``timetravel_id``, so a later chunk of a
    # streaming response refreshes the same span.
    async for response in _call_original(model, original, static, classmethod_orig, args, kwargs):
        final = response
        span = _capture_live_llm_span(
            session,
            model_name=str(call_kwargs["model"]),
            messages=call_kwargs["messages"],
            signature=signature,
            result=final,
            span=span,
        )
        await _complete_step(
            session,
            step,
            _response_result_text(final),
            usage=_step_usage(final),
        )
        yield response


def _call_kwargs(model: Any, llm_request: Any) -> dict[str, Any]:
    """Assemble the ``{model, messages, tools, **params}`` step-view kwargs."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.adapters.adk import _messages_from_adk
    # pylint: enable=import-outside-toplevel

    messages = _to_jsonable(_messages_from_adk(llm_request))
    tools = _tools_from_adk(llm_request)
    return {
        "model": _request_model_name(model, llm_request),
        "messages": messages,
        "tools": tools,
        **_config_params(llm_request),
    }


def _request_model_name(model: Any, llm_request: Any) -> str:
    return str(getattr(llm_request, "model", None) or getattr(model, "model", None) or "adk")


def _tools_from_adk(llm_request: Any) -> Any:
    config = getattr(llm_request, "config", None)
    tools = getattr(config, "tools", None) or None
    return _to_jsonable(tools) if tools is not None else None


def _config_params(llm_request: Any) -> dict[str, Any]:
    config = getattr(llm_request, "config", None)
    if config is None:
        return {}
    return {
        attr: _to_jsonable(getattr(config, attr))
        for attr in _SAMPLING_CONFIG_ATTRS
        if getattr(config, attr, None) is not None
    }


def _apply_llm_edit(decision: Decision, llm_request: Any) -> None:
    """Apply an EDIT decision to the request that will actually go out.

    Edited messages rebuild ``llm_request.contents``; the model override sets
    ``llm_request.model``; known sampling params are written onto the genai
    config (``max_tokens`` maps to its ADK name ``max_output_tokens``).
    Unknown params are ignored rather than corrupting the outbound call.
    """
    # pylint: disable=import-outside-toplevel
    from google.genai import types
    # pylint: enable=import-outside-toplevel

    if decision.messages is not None:
        contents = []
        for message in decision.messages:
            role = str(message.get("role", "user"))
            text = str(message.get("content", "") or "")
            adk_role = "model" if role in ("assistant", "model") else "user"
            contents.append(types.Content(role=adk_role, parts=[types.Part(text=text)]))
        llm_request.contents = contents
    if decision.model is not None:
        llm_request.model = decision.model
    if decision.params:
        config = getattr(llm_request, "config", None)
        if config is None:
            return
        fields = type(config).model_fields
        for key, value in decision.params.items():
            attr = _PARAM_TO_CONFIG_ATTR.get(key)
            if attr is not None and attr in fields:
                setattr(config, attr, value)


def _materialise_response(recorded: RecordedResponse, model_name: str) -> Any:
    """Rebuild an ADK ``LlmResponse`` from a recorded span payload.

    Understands the GenAI semconv nested response (what this interceptor
    captures) and OpenInference flat message keys. Assistant tool calls are
    restored as ``function_call`` parts so a frozen replay reproduces the
    agent's tool-call decisions.
    """
    # pylint: disable=import-outside-toplevel
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from agent_timetravel.adapters.adk import _llm_response_from_text
    # pylint: enable=import-outside-toplevel

    payload = recorded.payload or {}
    response = (
        payload.get("gen_ai.response")
        or payload.get("raw_response")
        or payload.get("response")
        or {}
    )
    message: dict[str, Any] = {}
    if isinstance(response, dict):
        choices = response.get("choices") or []
        first = choices[0] if isinstance(choices, list) and choices else None
        candidate = first.get("message") if isinstance(first, dict) else None
        if isinstance(candidate, dict):
            message = candidate
    if not message:
        for key, value in payload.items():
            if (
                key.startswith("llm.output_messages.")
                and key.endswith(".message.content")
                and isinstance(value, str)
            ):
                message = {"content": value}
                break

    content = message.get("content") or ""
    try:
        parts: list[Any] = [types.Part(text=content)] if content else []
        for call in message.get("tool_calls") or []:
            name, call_args = _parse_recorded_tool_call(call)
            if name is not None:
                parts.append(
                    types.Part(function_call=types.FunctionCall(name=name, args=call_args))
                )
        if not parts:
            parts = [types.Part(text="")]
        return LlmResponse(
            content=types.Content(role="model", parts=parts),
        )
    except (TypeError, ValueError):
        # Version drift in Content/Part signatures — degrade to the shared
        # text-only materialisation rather than failing the replay.
        return _llm_response_from_text(str(content), model=model_name)


def _parse_recorded_tool_call(call: Any) -> tuple[str | None, dict[str, Any]]:
    """Normalise a stored tool call (OpenAI wire or native form) to name/args."""
    # pylint: disable=import-outside-toplevel
    import json
    # pylint: enable=import-outside-toplevel

    if not isinstance(call, dict):
        return None, {}
    if isinstance(call.get("function"), dict):  # OpenAI wire form
        name = call["function"].get("name")
        arguments = call["function"].get("arguments")
        args: dict[str, Any] = (
            json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else {}
        )
    else:  # native form
        name = call.get("name")
        native_args = call.get("args")
        args = native_args if isinstance(native_args, dict) else {}
    if not isinstance(name, str):
        return None, {}
    return name, args


def _response_text(result: Any) -> str:
    """Best-effort text extraction from an ``LlmResponse``."""
    content = getattr(result, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.adapters.adk import _flatten_parts
    # pylint: enable=import-outside-toplevel

    return _flatten_parts(getattr(content, "parts", None) or [])


def _function_calls_of(result: Any) -> list[dict[str, Any]]:
    """Normalised function-call decisions (native storage form)."""
    content = getattr(result, "content", None)
    parts = getattr(content, "parts", None) or []
    calls: list[dict[str, Any]] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        name = getattr(function_call, "name", None)
        if not isinstance(name, str):
            continue
        calls.append(
            {
                "name": name,
                "args": getattr(function_call, "args", None) or {},
                "id": "",
                "type": "tool_call",
            }
        )
    return calls


def _usage_of(result: Any) -> dict[str, int]:
    """``{prompt, completion, total}`` from the genai usage metadata."""
    meta = getattr(result, "usage_metadata", None)
    if meta is None:
        return {}
    usage: dict[str, int] = {}
    for key, attr in (
        ("prompt", "prompt_token_count"),
        ("completion", "candidates_token_count"),
        ("total", "total_token_count"),
    ):
        value = getattr(meta, attr, None)
        if isinstance(value, int):
            usage[key] = value
    return usage


def _recorded_result_text(recorded: RecordedResponse) -> str:
    """Debugger result text for a replayed call (text + tool-call summary)."""
    payload = recorded.payload or {}
    response = (
        payload.get("gen_ai.response")
        or payload.get("raw_response")
        or payload.get("response")
        or {}
    )
    message: dict[str, Any] = {}
    if isinstance(response, dict):
        choices = response.get("choices") or []
        first = choices[0] if isinstance(choices, list) and choices else None
        candidate = first.get("message") if isinstance(first, dict) else None
        if isinstance(candidate, dict):
            message = candidate
    content = str(message.get("content") or "")
    calls = message.get("tool_calls") or []
    if content.strip() or not calls:
        return content
    # pylint: disable=import-outside-toplevel
    import json
    # pylint: enable=import-outside-toplevel

    lines = []
    for call in calls:
        name, call_args = _parse_recorded_tool_call(call)
        if name is not None:
            lines.append(f"→ {name}({json.dumps(call_args, default=str)})")
    return "\n".join(lines)


def _response_result_text(result: Any) -> str:
    """What the debugger shows as the step result for a live call.

    Text when present; otherwise a one-line summary of the function-call
    decisions, so a planning turn that emits only calls still previews.
    """
    content = _response_text(result)
    if content.strip():
        return content
    # pylint: disable=import-outside-toplevel
    import json
    # pylint: enable=import-outside-toplevel

    lines = [
        f"→ {call['name']}({json.dumps(call['args'], default=str)})"
        for call in _function_calls_of(result)
    ]
    return "\n".join(lines)


def _capture_live_llm_span(
    session: ReplaySession,
    *,
    model_name: str,
    messages: Any,
    signature: Any,
    result: Any,
    span: Any = None,
) -> Any:
    """Persist (or refresh) an LLM span for a live-forwarded ADK call.

    Pass the previously returned ``span`` when successive chunks of one call
    arrive: the same ``timetravel_id`` makes ``insert_span`` upsert, so a
    streaming response keeps one span whose payload tracks the latest chunk.
    """
    # pylint: disable=import-outside-toplevel
    from secrets import token_hex

    from agent_timetravel.enums import SpanKind, SpanStatus
    from agent_timetravel.models import Span
    # pylint: enable=import-outside-toplevel

    content = _response_text(result)
    usage = _usage_of(result)
    assistant: dict[str, Any] = {"role": "assistant", "content": content}
    calls = _function_calls_of(result)
    if calls:
        assistant["tool_calls"] = calls
    raw: dict[str, Any] = {
        "gen_ai.request.model": model_name,
        "gen_ai.response": {"choices": [{"message": assistant}]},
    }
    if usage:
        raw["gen_ai.response"]["usage"] = {
            "prompt_tokens": usage.get("prompt", 0),
            "completion_tokens": usage.get("completion", 0),
            "total_tokens": usage.get("total", 0),
        }
        raw["gen_ai.usage.prompt_tokens"] = usage.get("prompt", 0)
        raw["gen_ai.usage.completion_tokens"] = usage.get("completion", 0)
        raw["gen_ai.usage.total_tokens"] = usage.get("total", 0)
    if span is None:
        span = Span(
            trace_id=session.trace_id,
            span_id=token_hex(8),
            parent_span_id=None,
            name=f"adk.{model_name}",
            kind=SpanKind.LLM,
            status=SpanStatus.OK,
            model_name=model_name,
            prompt_tokens=usage.get("prompt"),
            completion_tokens=usage.get("completion"),
            total_tokens=usage.get("total"),
            messages_hash=signature.messages_hash,
            tools_hash=signature.tools_hash,
            raw_attributes=raw,
        )
    else:
        span.raw_attributes = raw
        span.total_tokens = usage.get("total") or span.total_tokens
    session.record_new(span)
    return span


async def _complete_step(
    session: ReplaySession,
    step: Step,
    result: str,
    usage: dict[str, int] | None = None,
) -> None:
    """Publish the completed step result to the approval channel."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.stepping import complete_step
    # pylint: enable=import-outside-toplevel

    await complete_step(session, step, result, usage=usage)


def _step_usage(result: Any) -> dict[str, int] | None:
    """Token usage in the ``complete_step`` event shape, or None when absent."""
    usage = _usage_of(result)
    if not usage:
        return None
    return {
        "input_tokens": usage.get("prompt", 0),
        "output_tokens": usage.get("completion", 0),
        "total_tokens": usage.get("total", 0),
    }


def assert_not_frozen(session: ReplaySession) -> None:
    """Raise :class:`ReplayError` if the active session is in FROZEN mode."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.adapters._common import assert_not_frozen as _guard
    # pylint: enable=import-outside-toplevel

    _guard(session)


# ----------------------------------------------------------------------
# Tool path
# ----------------------------------------------------------------------
async def _dispatch_adk_tool_async(
    tool: Any,
    session: ReplaySession,
    original: Callable[..., Any],
    static: bool,
    classmethod_orig: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Route an ADK ``BaseTool.run_async`` through the shared TOOL dispatch."""
    # pylint: disable=import-outside-toplevel
    from agent_timetravel.langgraph_intercept import _dispatch_tool_async
    # pylint: enable=import-outside-toplevel

    tool_input = kwargs.get("args", args[0] if args else None)

    async def forward(edited_input: Any, _edited_kwargs: dict[str, Any]) -> Any:
        # ADK's run_async is keyword-only (`*, args, tool_context`), so the
        # edited args replace the kwarg. If a foreign subclass ever took args
        # positionally, forward positionally instead of double-binding.
        call_kwargs = dict(kwargs)
        if args:
            call_kwargs.pop("args", None)
            positional = (edited_input, *args[1:])
        else:
            positional = args
            call_kwargs["args"] = edited_input
        return await _call_original(
            tool, original, static, classmethod_orig, positional, call_kwargs
        )

    return await _dispatch_tool_async(
        session,
        forward,
        tool_name=str(getattr(tool, "name", None) or type(tool).__name__),
        input=tool_input,
        kwargs={},
    )
