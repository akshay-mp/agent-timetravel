"""Agent Timetravel — time-travel debugging for AI agents.

OTel-in / replay-out: consume standard OpenTelemetry/OpenInference traces,
then timetravel an agent to any span, branch it live, and diff the timelines.
"""

from __future__ import annotations

from agent_timetravel.agents import AgentDefinition, TimeTravel, TimeTravelContext
from agent_timetravel.checkpoint import checkpoint
from agent_timetravel.tool_intercept import tool

__version__ = "0.2.1"
timetravel = TimeTravel()
__all__ = [
    "AgentDefinition",
    "TimeTravel",
    "TimeTravelContext",
    "__version__",
    "checkpoint",
    "timetravel",
    "tool",
]


# Lazy re-export of the public Phase 5.5 eval surface. We use __getattr__
# rather than a top-level import so ``import agent_timetravel`` doesn't eagerly pull
# in asyncio / dataclasses machinery for callers only using Phase 1-4.
def __getattr__(name: str) -> object:
    """Lazy re-export for the Phase 5.5 eval harness public surface."""
    if name in {
        "evaluate",
        "EvalScenario",
        "EvalSuite",
        "EvalSuiteResult",
        "ScenarioResult",
        "EvaluatorOutcome",
        "validate_suite",
    }:
        # pylint: disable=import-outside-toplevel
        import agent_timetravel.evaluate as _eval
        # pylint: enable=import-outside-toplevel

        return getattr(_eval, name)
    raise AttributeError(f"module 'agent_timetravel' has no attribute {name!r}")
