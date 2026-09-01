"""Decorator-first agent definitions and framework capability metadata.

The decorator is deliberately a thin registration layer. Calling a decorated
function directly still calls the user's function; the workbench is the only
caller that validates inputs, injects :class:`TimeTravelContext`, and installs
replay/interception state.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import types
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    Secret,
    SecretBytes,
    SecretStr,
    TypeAdapter,
    create_model,
)

FrameworkName = str
FRAMEWORKS = (
    "auto",
    "openai",
    "langgraph",
    "crewai",
    "pydantic_ai",
    "adk",
    "smolagents",
)


@dataclass(frozen=True)
class TimeTravelContext:
    """Execution context injected into an agent during a workbench run."""

    session: Any
    trace_id: str
    branch_id: str

    @property
    def cursor(self) -> int:
        """Current replay cursor."""
        return int(self.session.cursor)


@dataclass(frozen=True)
class FrameworkPlugin:
    """Description of one supported framework integration."""

    name: str
    modules: tuple[str, ...]
    capabilities: Mapping[str, bool]
    adapter: str | None = None
    workbench_supported: bool = True
    unsupported_reason: str | None = None

    def available(self) -> bool:
        return any(importlib.util.find_spec(module) is not None for module in self.modules)


_PLUGINS: dict[str, FrameworkPlugin] = {
    "openai": FrameworkPlugin(
        "openai",
        ("openai",),
        {"interactive_llm": True, "native_tool_calls": False, "python_tools": True},
    ),
    "langgraph": FrameworkPlugin(
        "langgraph",
        ("langgraph", "langchain_core"),
        {"interactive_llm": True, "native_tool_calls": True, "python_tools": True},
        "agent_timetravel.adapters.langgraph.replay_chat_model",
    ),
    "crewai": FrameworkPlugin(
        "crewai",
        ("crewai",),
        {"interactive_llm": False, "native_tool_calls": False, "python_tools": True},
        "agent_timetravel.adapters.crewai.replay_llm",
        workbench_supported=False,
        unsupported_reason=(
            "generic CrewAI activation is not wired; wrap the target LLM with "
            "agent_timetravel.adapters.crewai.replay_llm"
        ),
    ),
    "pydantic_ai": FrameworkPlugin(
        "pydantic_ai",
        ("pydantic_ai",),
        {"interactive_llm": False, "native_tool_calls": False, "python_tools": True},
        "agent_timetravel.adapters.pydantic_ai.replay_model",
        workbench_supported=False,
        unsupported_reason=(
            "generic PydanticAI activation is not wired; wrap the target model with "
            "agent_timetravel.adapters.pydantic_ai.replay_model"
        ),
    ),
    "adk": FrameworkPlugin(
        "adk",
        ("google.adk",),
        {"interactive_llm": True, "native_tool_calls": True, "python_tools": True},
    ),
    "smolagents": FrameworkPlugin(
        "smolagents",
        ("smolagents",),
        {"interactive_llm": False, "native_tool_calls": False, "python_tools": True},
        "agent_timetravel.adapters.smolagents.replay_model",
        workbench_supported=False,
        unsupported_reason=(
            "generic SmolAgents activation is not wired; wrap the target model with "
            "agent_timetravel.adapters.smolagents.replay_model"
        ),
    ),
}


def framework_plugins() -> dict[str, FrameworkPlugin]:
    """Return a copy of the framework plugin registry."""
    return dict(_PLUGINS)


def _is_context_annotation(annotation: object) -> bool:
    if annotation is TimeTravelContext:
        return True
    origin = get_origin(annotation)
    return origin in (types.UnionType, getattr(__import__("typing"), "Union", object)) and (
        TimeTravelContext in get_args(annotation)
    )


def mask_secrets(value: Any) -> Any:  # noqa: ANN401
    """Return structured JSON-safe data with Pydantic secret values redacted."""
    # Pydantic's JSON-mode serializer handles models, dataclasses, containers,
    # enums, dates, UUIDs, and nested SecretStr values without flattening the
    # result to a string. Unknown objects raise a useful serialization error
    # before the SQLite layer can fall back to ``default=str``.
    return TypeAdapter(Any).dump_python(value, mode="json")


def _resolve_framework(requested: str, target: object, func: Callable[..., Any]) -> str:
    if requested not in FRAMEWORKS:
        valid = ", ".join(FRAMEWORKS)
        raise ValueError(f"unsupported framework {requested!r}; expected one of: {valid}")
    if requested != "auto":
        return requested

    candidates: list[str] = []

    def consider(value: object) -> None:
        module = getattr(value, "__module__", "")
        text = f"{module} {type(value).__module__} {type(value).__name__}".lower()
        for name, plugin in _PLUGINS.items():
            if any(root.lower() in text for root in plugin.modules) and name not in candidates:
                candidates.append(name)

    # Explicit target is authoritative for auto detection.
    if target is not None:
        consider(target)
        if candidates:
            return candidates[0]

    # Then inspect closure cells and module globals referenced by the function.
    closure = inspect.getclosurevars(func)
    for value in (*closure.nonlocals.values(), *closure.globals.values()):
        consider(value)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "framework='auto' is ambiguous; set framework explicitly to one of: "
            + ", ".join(sorted(candidates))
        )
    if _PLUGINS["openai"].available():
        return "openai"
    raise ValueError(
        "framework='auto' could not detect a supported framework. Set framework="
        " explicitly or install openai for the compatible fallback."
    )


def _build_input_model(func: Callable[..., Any]) -> tuple[type[BaseModel], bool]:
    hints = get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}
    context_injected = False
    for parameter in inspect.signature(func).parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError("agent_timetravel.agent functions cannot use *args or **kwargs")
        annotation = hints.get(parameter.name, parameter.annotation)
        if _is_context_annotation(annotation):
            context_injected = True
            continue
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = parameter.default if parameter.default is not inspect.Parameter.empty else ...
        fields[parameter.name] = (annotation, default)
    model = create_model(
        f"{func.__name__.title().replace('_', '')}Inputs",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return model, context_injected


@dataclass
class AgentDefinition:
    """Registered metadata and invocation contract for one decorated function."""

    ref: str
    name: str
    func: Callable[..., Any]
    framework: str
    description: str
    tags: tuple[str, ...]
    target: object | None
    input_model: type[BaseModel]
    context_injected: bool
    owner: TimeTravel

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    @property
    def output_schema(self) -> dict[str, Any]:
        hints = get_type_hints(self.func, include_extras=True)
        output = hints.get("return", Any)
        # TypeAdapter produces the same JSON schema semantics for primitives,
        # models, and containers without requiring a synthetic result model.
        from pydantic import TypeAdapter

        return TypeAdapter(output).json_schema()

    @property
    def capabilities(self) -> dict[str, bool]:
        plugin = _PLUGINS[self.framework]
        return dict(plugin.capabilities)

    @property
    def available(self) -> bool:
        plugin = _PLUGINS[self.framework]
        return plugin.workbench_supported and plugin.available()

    @property
    def availability_reason(self) -> str | None:
        plugin = _PLUGINS[self.framework]
        if plugin.unsupported_reason is not None:
            return plugin.unsupported_reason
        if not plugin.available():
            return f"install the optional {self.framework} dependency to enable this integration"
        return None

    @staticmethod
    def _contains_secret(annotation: object) -> bool:
        if annotation in (Secret, SecretStr, SecretBytes) or get_origin(annotation) is Secret:
            return True
        if any(AgentDefinition._contains_secret(item) for item in get_args(annotation)):
            return True
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return any(
                AgentDefinition._contains_secret(field.annotation)
                for field in annotation.model_fields.values()
            )
        return False

    @property
    def secret_input_fields(self) -> tuple[str, ...]:
        """Top-level input fields that contain Pydantic secret inputs."""
        return tuple(
            name
            for name, field in self.input_model.model_fields.items()
            if self._contains_secret(field.annotation)
        )

    @property
    def has_secret_inputs(self) -> bool:
        return bool(self.secret_input_fields)

    def validate_inputs(self, inputs: object) -> BaseModel:
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be a JSON object")
        return self.input_model.model_validate(dict(inputs))

    async def _invoke_model(self, session: Any, model: BaseModel) -> Any:  # noqa: ANN401
        kwargs = model.model_dump(mode="python")
        if self.context_injected:
            context = TimeTravelContext(
                session=session,
                trace_id=session.trace_id,
                branch_id=str(session.branch_id),
            )
            parameter = next(
                name
                for name, annotation in get_type_hints(self.func, include_extras=True).items()
                if name != "return" and _is_context_annotation(annotation)
            )
            kwargs[parameter] = context
        patch_context: AbstractContextManager[None] = nullcontext()
        if self.framework == "openai":
            from agent_timetravel.openai_intercept import patch

            patch_context = patch()
        elif self.framework == "langgraph":
            from agent_timetravel.langgraph_intercept import patch

            patch_context = patch()
        elif self.framework == "adk":
            from agent_timetravel.adk_intercept import patch

            patch_context = patch()
        with patch_context:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**kwargs)
            return await asyncio.to_thread(self.func, **kwargs)

    async def invoke(self, session: Any, inputs: object) -> Any:  # noqa: ANN401
        """Validate once, then invoke on the appropriate execution path."""
        return await self._invoke_model(session, self.validate_inputs(inputs))

    def compile_runner(self, inputs: object) -> Callable[[Any], Awaitable[Any]]:
        """Compile validated workbench inputs into the existing runner shape."""
        validated = inputs if isinstance(inputs, self.input_model) else self.validate_inputs(inputs)

        async def runner(session: Any) -> Any:  # noqa: ANN401
            return await self._invoke_model(session, validated)

        return runner


class TimeTravel:
    """Own a collection of decorator-registered agents for one application."""

    def __init__(self, title: str = "TimeTravel") -> None:
        self.title = title
        self._agents: dict[str, AgentDefinition] = {}

    @property
    def agents(self) -> Mapping[str, AgentDefinition]:
        return dict(self._agents)

    @property
    def registry(self) -> Mapping[str, AgentDefinition]:
        """Alias for integrations that call the owned collection a registry."""
        return self.agents

    def agent(
        self,
        name: str | None = None,
        *,
        framework: str = "auto",
        target: object | None = None,
        description: str = "",
        tags: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorate and register a sync or async agent function."""
        if framework not in FRAMEWORKS:
            valid = ", ".join(FRAMEWORKS)
            raise ValueError(f"unsupported framework {framework!r}; expected one of: {valid}")

        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            ref = name or func.__name__
            if ref in self._agents:
                raise ValueError(f"duplicate agent name/ref {ref!r} on this TimeTravel instance")
            resolved = _resolve_framework(framework, target, func)
            input_model, context_injected = _build_input_model(func)
            definition = AgentDefinition(
                ref=ref,
                name=ref,
                func=func,
                framework=resolved,
                description=description,
                tags=tuple(tags),
                target=target,
                input_model=input_model,
                context_injected=context_injected,
                owner=self,
            )
            self._agents[ref] = definition
            func.__timetravel_agent__ = definition  # type: ignore[attr-defined]
            return func

        return decorate

    def get(self, ref: str) -> AgentDefinition | None:
        return self._agents.get(ref)

    def __iter__(self) -> Iterator[AgentDefinition]:
        return iter(self._agents.values())


__all__ = [
    "FRAMEWORKS",
    "AgentDefinition",
    "FrameworkPlugin",
    "TimeTravel",
    "TimeTravelContext",
    "framework_plugins",
    "mask_secrets",
]
