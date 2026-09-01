"""Google ADK agent demo — Gemma 4 on a local Unsloth server, TimeTravel-debugged.

A self-contained ADK project that shows the generic workbench activation
(``agent_timetravel.adk_intercept``) stepping through a real agent:

* ``OpenAICompatLlm`` — a custom ``google.adk.models.BaseLlm`` that bridges
  ADK to **any** OpenAI-compatible endpoint (Unsloth Studio, llama.cpp,
  vLLM, Ollama…) via the ``openai`` SDK. ADK's ``LiteLlm`` needs the
  ``google-adk[extensions]`` extra; this bridge needs only ``openai``,
  which TimeTravel already depends on. It converts ADK ``LlmRequest``
  contents/tools to Chat Completions and maps responses (text,
  ``function_call`` parts, usage) back to ``LlmResponse``.
* ``get_weather`` — a plain function ADK auto-wraps as a FunctionTool, so a
  run exercises tool calls as well as LLM turns.
* ``ask`` — a ``@timetravel.agent(framework="adk")`` entry point that runs
  the agent through ADK's ``Runner``. During a workbench run every
  ``generate_content_async`` and every tool ``run_async`` pauses at the
  stepping gate — no manual model wrapping.

Run headless::

    python examples/google_adk_demo/app.py "What is the weather in Tokyo?"

Run under the workbench::

    agent-timetravel dev examples.google_adk_demo.app:timetravel \\
        --db /tmp/adk-gemma-demo.db --no-open
    # → http://127.0.0.1:8484/ui  (start the "ask" agent from the browser)

Configuration (environment variables): ``OPENAI_API_KEY``,
``ADK_GEMMA_BASE_URL`` (default http://127.0.0.1:8888/v1), ``ADK_GEMMA_MODEL``
(default unsloth/gemma-4-12b-it-GGUF).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from openai import AsyncOpenAI

from agent_timetravel import TimeTravel, TimeTravelContext

# --- endpoint wiring -------------------------------------------------------
BASE_URL = os.environ.get("ADK_GEMMA_BASE_URL", "http://127.0.0.1:8888/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "local")
MODEL_ID = os.environ.get("ADK_GEMMA_MODEL", "unsloth/gemma-4-12b-it-GGUF")

timetravel = TimeTravel(title="Google ADK")


# --- ADK ⇆ OpenAI-compatible bridge ----------------------------------------
def _flatten_parts(parts: Any) -> str:
    chunks: list[str] = []
    for part in parts or []:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)
    return "\n".join(chunks)


def _system_prompt(llm_request: Any) -> str | None:
    config = getattr(llm_request, "config", None)
    instruction = getattr(config, "system_instruction", None)
    if instruction is None:
        return None
    if isinstance(instruction, str):
        return instruction
    return _flatten_parts(getattr(instruction, "parts", None)) or None


def _contents_to_messages(llm_request: Any, system: str | None) -> list[dict[str, Any]]:
    """Convert ADK ``LlmRequest.contents`` to Chat Completions messages.

    ``function_call`` parts become assistant ``tool_calls`` (with ids we
    remember by tool name) and the matching ``function_response`` parts
    become ``role=tool`` messages that reuse those ids — llama.cpp rejects
    tool messages whose ids don't line up.
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    pending_ids: dict[str, list[str]] = {}
    for content in getattr(llm_request, "contents", None) or []:
        role = getattr(content, "role", None) or "user"
        text = _flatten_parts(getattr(content, "parts", None))
        calls: list[Any] = []
        responses: list[Any] = []
        for part in getattr(content, "parts", None) or []:
            function_call = getattr(part, "function_call", None)
            function_response = getattr(part, "function_response", None)
            if function_call is not None:
                calls.append(function_call)
            if function_response is not None:
                responses.append(function_response)

        if calls:
            oai_calls = []
            for index, call in enumerate(calls):
                call_id = f"call_{index}_{call.name}"
                pending_ids.setdefault(call.name, []).append(call_id)
                oai_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(dict(call.args or {})),
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": oai_calls,
                }
            )
        elif responses:
            for response in responses:
                ids = pending_ids.get(response.name)
                call_id = ids.pop(0) if ids else f"call_{response.name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(dict(response.response or {})),
                    }
                )
        elif text:
            messages.append(
                {
                    "role": "assistant" if role in ("model", "assistant") else "user",
                    "content": text,
                }
            )
    return messages


def _normalize_schema_types(value: Any) -> Any:
    """Convert ADK schema type enums to lowercase OpenAI JSON Schema strings."""
    if isinstance(value, dict):
        return {
            key: (
                _normalize_schema_type(item)
                if key == "type"
                else _normalize_schema_types(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_schema_types(item) for item in value]
    return value


def _normalize_schema_type(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_schema_type(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_schema_type(item) for item in value)
    raw_value = getattr(value, "value", value)
    return raw_value.lower() if isinstance(raw_value, str) else value


def _declarations_to_tools(llm_request: Any) -> list[dict[str, Any]] | None:
    """Convert ADK ``config.tools`` function declarations to OpenAI ``tools``."""
    config = getattr(llm_request, "config", None)
    tools: list[dict[str, Any]] = []
    for tool in getattr(config, "tools", None) or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            schema = getattr(declaration, "parameters_json_schema", None)
            if schema is None:
                parameters = getattr(declaration, "parameters", None)
                schema = (
                    parameters.model_dump(exclude_none=True)
                    if parameters is not None
                    else {"type": "object", "properties": {}}
                )
            schema = _normalize_schema_types(schema)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": schema,
                    },
                }
            )
    return tools or None


class OpenAICompatLlm(BaseLlm):
    """ADK model backed by any OpenAI-compatible Chat Completions endpoint."""

    base_url: str = BASE_URL
    api_key: str = API_KEY
    temperature: float = 0.3
    max_output_tokens: int = 2048
    client: Any = None

    def model_post_init(self, _context: Any) -> None:
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        """One non-streaming Chat Completions turn, mapped to ``LlmResponse``."""
        if stream:  # SSE is out of scope for the demo; serve one shot.
            raise ValueError("OpenAICompatLlm does not implement streaming")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=_contents_to_messages(llm_request, _system_prompt(llm_request)),
            tools=_declarations_to_tools(llm_request),
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        message = response.choices[0].message

        parts: list[Any] = []
        if message.content:
            parts.append(types.Part(text=message.content))
        for tool_call in message.tool_calls or []:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except ValueError:
                args = {}
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(name=tool_call.function.name, args=args)
                )
            )
        if not parts:
            parts = [types.Part(text="")]

        usage = response.usage
        usage_metadata = (
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=usage.prompt_tokens,
                candidates_token_count=usage.completion_tokens,
                total_token_count=usage.total_tokens,
            )
            if usage
            else None
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=usage_metadata,
            model_version=response.model or self.model,
        )


# --- the agent -------------------------------------------------------------
def get_weather(city: str) -> dict[str, Any]:
    """Toy tool: deterministic weather so demo runs are reproducible."""
    conditions = {
        "tokyo": {"condition": "sunny", "celsius": 26},
        "london": {"condition": "rainy", "celsius": 14},
    }
    return {**conditions.get(city.lower(), {"condition": "cloudy", "celsius": 19}), "city": city}


def _build_agent() -> LlmAgent:
    return LlmAgent(
        name="weather_agent",
        model=OpenAICompatLlm(model=MODEL_ID),
        instruction=(
            "You are a weather assistant. Use the get_weather tool for any "
            "city weather question, then answer in one short sentence."
        ),
        tools=[get_weather],
    )


async def _run(question: str) -> str:
    runner = InMemoryRunner(agent=_build_agent(), app_name="google_adk_demo")
    session = await runner.session_service.create_session(app_name="google_adk_demo", user_id="dev")
    final = ""
    async for event in runner.run_async(
        user_id="dev",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
    ):
        if event.content and event.content.parts:
            text = _flatten_parts(event.content.parts)
            if text:
                final = text
    return final or "(no text)"


@timetravel.agent(
    framework="adk",
    description="ADK weather agent (Gemma 4 via Unsloth) with a get_weather tool",
)
async def ask(question: str, context: TimeTravelContext | None = None) -> str:
    """Run the ADK agent for one user question.

    Direct calls run ADK untouched. Workbench runs auto-activate the ADK
    interceptor: every ``generate_content_async`` and tool ``run_async``
    pauses at the stepping gate for approve / edit / mock.
    """
    return await _run(question)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the weather in Tokyo right now?"
    print(f"[google_adk_demo] {MODEL_ID} @ {BASE_URL}", file=sys.stderr)
    print(f"[google_adk_demo] question: {question}", file=sys.stderr)
    print(asyncio.run(ask(question)))
