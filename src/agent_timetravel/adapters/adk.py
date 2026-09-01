"""Phase 6 — Google ADK replay adapter.

Plan §6 Phase 6 — extends the adapter-first replay strategy (first
shipped for LangGraph in Phase 3 Track 3B.1) to **Google ADK**.

ADK exposes :class:`google.adk.models.BaseLlm` (aliased as :class:`Llm`)
as the pluggable model slot. Subclasses implement the
``generate_content_async`` async generator and the agent loop calls it per
turn. This adapter provides :func:`replay_llm`, a factory that wraps a real
``BaseLlm`` so that during a :func:`timetravel.replay` context:

* Active + matching recorded LLM span ``<= cursor`` → yield the recorded
  :class:`google.adk.models.LlmResponse` (zero egress).
* Active + divergence in ``BRANCH`` / ``FULL_RERUN`` → forward to the
  wrapped model and capture the new span under the replay branch.
* Active + divergence in ``FROZEN`` → raise
  :class:`~timetravel.replay.ReplayError`.
* No active session → delegate to the wrapped model verbatim.

The factory pattern mirrors the LangGraph adapter: ``BaseLlm`` is lazily
imported inside the factory so ``agent-timetravel --version`` stays fast without
``google-adk`` installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr

from agent_timetravel.adapters._common import assert_not_frozen, build_live_span
from agent_timetravel.adk_intercept import (
    _WRAPPER_FORWARD,
    _canonical_messages_from_adk,
    _function_calls_of,
    _materialise_response,
    _response_text,
    _usage_of,
)
from agent_timetravel.openai_intercept import extract_signature

if TYPE_CHECKING:
    from google.adk.models import BaseLlm, LlmResponse

    from agent_timetravel.replay import ReplaySession

__all__ = ["AdapterError", "replay_llm"]


class AdapterError(RuntimeError):
    """Raised when the ADK adapter cannot satisfy a replay contract."""


def replay_llm(
    wrapped: BaseLlm,
    *,
    trace_id: str | None = None,
) -> BaseLlm:
    """Wrap a real ADK ``BaseLlm`` so replay consults the active session.

    Parameters
    ----------
    wrapped:
        The live ADK model to delegate to when no replay is active or when
        branch / full mode authorises a forward.
    trace_id:
        Optional explicit trace id. Defaults to the active session's
        ``trace_id`` (resolved per call) so one wrapper follows the
        session across forks.

    Returns
    -------
    BaseLlm
        A subclass instance whose ``generate_content_async`` routes through
        TimeTravel. The wrapped model is preserved as ``._timetravel_wrapped``.
    """
    try:
        BaseLlm, _LlmRequest, LlmResponse = _adk_types()
    except ImportError as exc:  # pragma: no cover - exercised only without ADK
        raise AdapterError(
            "agent_timetravel.adapters.adk requires `google-adk`; install it via "
            "`pip install agent-timetravel[adk]` or use the generic OpenAI "
            "monkey-patch (agent_timetravel.openai_intercept.patch)."
        ) from exc

    class _ReplayLlm(BaseLlm):  # type: ignore[misc]
        """Subclass created at factory-call time so imports resolve lazily."""

        _timetravel_wrapped: BaseLlm = PrivateAttr(default_factory=lambda: wrapped)
        _timetravel_trace_id: str | None = trace_id

        # ------------------------------------------------------------------
        # ADK contract
        # ------------------------------------------------------------------
        async def generate_content_async(
            self,
            llm_request: Any,
            stream: bool = False,
        ) -> Any:
            """Replay or forward ADK's async-generator model contract."""
            session = self._active_session()
            if session is None:
                async for response in self._forward_content(llm_request, stream):
                    yield response
                return

            signature = self._signature(llm_request)
            recorded = session.respond_or_forward(signature)
            if recorded is not None:
                yield self._materialise(recorded)
                return

            assert_not_frozen(session)
            captured_span: Any = None
            async for response in self._forward_content(llm_request, stream):
                captured_span = self._capture_live_span(
                    llm_request,
                    session,
                    signature,
                    response,
                    self._get_model_name(),
                    span=captured_span,
                )
                yield response

        @property
        def _llm_type(self) -> str:
            return f"timetravel-replay({getattr(self._timetravel_wrapped, 'model', 'adk')})"

        # ADK stores the literal model id on ``self.model``; the wrapped model
        # already populates it, and a no-arg init does nothing useful here so
        # we forward attribute access instead.
        def _get_model_name(self) -> str:
            return str(getattr(self._timetravel_wrapped, "model", "adk-replay"))

        async def generate_response_async(
            self,
            request: Any,
        ) -> Any:
            if not hasattr(self._timetravel_wrapped, "generate_response_async"):
                final: Any = None
                async for response in self.generate_content_async(request):
                    final = response
                return final or LlmResponse()
            response = await self._dispatch_async(request, self._active_session())
            return response or LlmResponse()

        async def _dispatch_async(
            self,
            request: Any,
            session: ReplaySession | None,
        ) -> Any:
            if session is None:
                return await self._forward_response_async(request)
            signature = self._signature(request)
            recorded = session.respond_or_forward(signature)
            if recorded is None:
                assert_not_frozen(session)
                result = await self._forward_response_async(request)
                self._capture_live_span(request, session, signature, result, self._get_model_name())
                return result
            return self._materialise(recorded)

        def generate_response(
            self,
            request: Any,
        ) -> Any:
            if not hasattr(self._timetravel_wrapped, "generate_response"):
                raise AdapterError(
                    "This ADK model exposes only generate_content_async; "
                    "use its async-generator API."
                )
            session = self._active_session()
            if session is None:
                return self._forward_response(request)
            signature = self._signature(request)
            recorded = session.respond_or_forward(signature)
            if recorded is None:
                assert_not_frozen(session)
                result = self._forward_response(request)
                self._capture_live_span(request, session, signature, result, self._get_model_name())
                return result
            return self._materialise(recorded)

        # ------------------------------------------------------------------
        # Forward helpers
        # ------------------------------------------------------------------
        async def _forward_content(self, llm_request: Any, stream: bool = False) -> Any:
            """Iterate the wrapped model with the interceptor stood down.

            The class-level interceptor (``agent_timetravel.adk_intercept``)
            also wraps the inner model; the ``_WRAPPER_FORWARD`` flag tells it
            this call is the wrapper's own forward, not a fresh interception
            point — otherwise one divergent call would be captured twice.
            """
            token = _WRAPPER_FORWARD.set(True)
            try:
                async for response in self._timetravel_wrapped.generate_content_async(
                    llm_request, stream=stream
                ):
                    yield response
            finally:
                _WRAPPER_FORWARD.reset(token)

        def _forward_response(self, request: Any) -> Any:
            token = _WRAPPER_FORWARD.set(True)
            try:
                return self._timetravel_wrapped.generate_response(request)
            finally:
                _WRAPPER_FORWARD.reset(token)

        async def _forward_response_async(self, request: Any) -> Any:
            token = _WRAPPER_FORWARD.set(True)
            try:
                return await self._timetravel_wrapped.generate_response_async(request)
            finally:
                _WRAPPER_FORWARD.reset(token)

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _signature(self, request: Any) -> Any:
            # ADK's LlmRequest keeps the agent's outbound messages on
            # ``.contents`` plus the model id on ``.model`` and live tools on
            # ``.config.tools``. We re-use the shared extraction from
            # openai_intercept because TimeTravel treats any (model, messages,
            # tools) triple as a playback key.
            messages = _canonical_messages_from_adk(request)
            return extract_signature(
                model=self._get_model_name(),
                messages=messages,
                tools=getattr(getattr(request, "config", None), "tools", None) or None,
            )

        def _materialise(self, recorded: Any) -> LlmResponse:
            return _materialise_response(recorded, self._get_model_name())

        def _capture_live_span(
            self,
            request: Any,
            session: ReplaySession,
            signature: Any,
            result: Any,
            model_name: str,
            span: Any = None,
        ) -> Any:
            content = _response_text(result)
            calls = _function_calls_of(result)
            usage = _usage_of(result)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                message["tool_calls"] = calls
            raw_extras: dict[str, Any] = {
                "gen_ai.response": {"choices": [{"message": message}]}
            }
            if usage:
                raw_extras["gen_ai.response"]["usage"] = {
                    "prompt_tokens": usage.get("prompt", 0),
                    "completion_tokens": usage.get("completion", 0),
                    "total_tokens": usage.get("total", 0),
                }
            previous_span = span
            span = build_live_span(
                session,
                model_name=model_name,
                messages=_canonical_messages_from_adk(request),
                content=content,
                raw_extras=raw_extras,
                # The interceptor hashes config.tools into its signature; a
                # span without the same tools_hash would never match a
                # tool-carrying call replayed through the workbench.
                tools_hash=getattr(signature, "tools_hash", None),
            )
            if previous_span is not None:
                span.timetravel_id = previous_span.timetravel_id
            if "prompt" in usage:
                span.prompt_tokens = usage["prompt"]
            if "completion" in usage:
                span.completion_tokens = usage["completion"]
            if "total" in usage:
                span.total_tokens = usage["total"]
            session.record_new(span)
            return span

        def _active_session(self) -> ReplaySession | None:
            # pylint: disable=import-outside-toplevel
            from agent_timetravel.replay import active_session
            # pylint: enable=import-outside-toplevel

            session = active_session()
            if session is None:
                return None
            if (
                self._timetravel_trace_id is not None
                and session.trace_id != self._timetravel_trace_id
            ):
                return None
            return session

    model_name = str(getattr(wrapped, "model", "adk-replay") or "adk-replay")
    try:
        instance = _ReplayLlm(model=model_name)
    except TypeError:  # pragma: no cover - legacy BaseLlm implementations
        instance = _ReplayLlm()
    return instance


# ----------------------------------------------------------------------
# ADK content-shape helpers
# ----------------------------------------------------------------------
def _messages_from_adk(request: Any) -> list[dict[str, Any]]:
    """Return the shared canonical message shape used by ADK replay paths."""
    return _canonical_messages_from_adk(request)


def _flatten_parts(parts: Any) -> str:
    """ADK parts may contain bare strings or Text parts — flatten loosely."""
    if isinstance(parts, str):
        return parts
    chunks: list[str] = []
    for part in parts or []:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict):
            chunks.append(str(part.get("text") or part.get("content") or ""))
        else:
            chunks.append(getattr(part, "text", "") or "")
    return "\n".join(c for c in chunks if c)


def _llm_response_from_text(content: str, *, model: str) -> Any:
    """Build an ADK ``LlmResponse`` from a recorded text payload.

    Re-imports lazily so the module imports cleanly without ADK installed.
    The exact constructor varies across ADK versions; we prefer the
    high-level pydantic class and fall back to a dict if unavailable.
    """
    # pylint: disable=import-outside-toplevel
    try:
        _, _, LlmResponse = _adk_types()
    except ImportError as exc:  # pragma: no cover - depends on ADK presence
        raise AdapterError(
            "agent_timetravel.adapters.adk requires `google-adk`; install it via "
            "`pip install agent-timetravel[adk]`."
        ) from exc
    # pylint: enable=import-outside-toplevel

    try:
        # ADK 0.2+ wraps text in a Content -> [Part(text=…)] block.
        # pylint: disable=import-outside-toplevel,no-name-in-module
        try:
            from google.genai.types import Content, Part
        except ImportError:  # pragma: no cover - pre-0.2 ADK without genai
            try:
                from google.adk.models import Content, Part
            except ImportError:
                from google.adk.models.llms import Content, Part
        # pylint: enable=import-outside-toplevel,no-name-in-module
        return LlmResponse(
            content=Content(role="model", parts=[Part(text=content)]),
        )
    # ADK shape drift across versions: Content/Part signatures vary, AND some
    # builds fall through to a bare-string form. Tolerate both rather than
    # chasing the upstream release cadence.
    except (ImportError, TypeError, ValueError):
        # Older/minimal builds: some ADK LlmResponse customs accept content
        # as a bare string. We don't use ``model`` here; it's preserved for
        # future use and to avoid a positioning-only-arg pitfall across
        # ADK versions.
        _ = model
        return LlmResponse(content=content)


def _llm_response_to_text(result: Any) -> str:
    """Best-effort text extraction from an ``LlmResponse``.

    ADK's content shape is ``Content(role=, parts=[Part(text=…)…])`` but
    older builds or test doubles may pass a bare string.
    """
    content = getattr(result, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    return _flatten_parts(parts)


def _adk_types() -> tuple[Any, Any, Any]:
    """Load ADK model types across the current and legacy module layouts."""
    # pylint: disable=import-outside-toplevel
    try:
        from google.adk.models import BaseLlm, LlmRequest, LlmResponse
    except ImportError:
        from google.adk.models.llms import BaseLlm, LlmResponse

        try:
            from google.adk.models.llms import LlmRequest
        except ImportError:  # pragma: no cover - legacy ADK shape
            LlmRequest = Any
    # pylint: enable=import-outside-toplevel
    return BaseLlm, LlmRequest, LlmResponse
