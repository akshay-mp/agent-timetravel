"""deepagents deep-research graph under TimeTravel — the modern integration.

This is the smallest full integration for a foreign LangGraph project:

* dependency: ``agent-timetravel[langgraph]`` (see requirements.txt),
* this file: import the graph's own prompts/tools, register it with the
  ``@<registry>.agent(framework="langgraph", ...)`` decorator,
* launch: ``agent-timetravel app:main`` — the workbench UI opens in your browser
  with every LLM and tool call in the graph gated in the step-by-step
  debugger (including calls made inside subagents). No model wrapping, no
  instrumentation setup.

Model resolution (from ``.env`` — see app README):

* ``OPENAI_BASE_URL`` + ``TIMETRAVEL_MODEL`` → local OpenAI-compatible
  server (e.g. Unsloth/vLLM), ``TIMETRAVEL_TEMPERATURE`` (default 0.3),
  ``OPENAI_API_KEY`` (default "local").
* Otherwise → Anthropic via ``ANTHROPIC_API_KEY``.

Run from this directory::

    agent-timetravel app:main
"""

import os
from datetime import datetime
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from agent_timetravel import TimeTravel

load_dotenv()

try:
    from research_agent.prompts import (
        RESEARCH_WORKFLOW_INSTRUCTIONS,
        RESEARCHER_INSTRUCTIONS,
        SUBAGENT_DELEGATION_INSTRUCTIONS,
    )
    from research_agent.tools import tavily_search, think_tool
except Exception as exc:
    raise SystemExit(
        f"could not import research_agent ({exc}).\n"
        "Clone the deepagents example, copy .env.example to .env, and set "
        "TAVILY_API_KEY plus either the local-model vars or ANTHROPIC_API_KEY."
    ) from exc

INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=3,
        max_researcher_iterations=3,
    )
)

research_sub_agent = {
    "name": "research-agent",
    "description": (
        "Delegate research to the sub-agent researcher. "
        "Only give this researcher one topic at a time."
    ),
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(
        date=datetime.now().strftime("%Y-%m-%d")
    ),
    "tools": [tavily_search, think_tool],
}


def build_model() -> Any:
    """Env-configured chat model; local server first, Anthropic fallback."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("TIMETRAVEL_MODEL")
    if base_url and model_name:
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "local"),
            temperature=float(os.environ.get("TIMETRAVEL_TEMPERATURE", "0.3")),
        )
    return init_chat_model(
        model="anthropic:claude-sonnet-4-5-20250929", temperature=0.0
    )


agent_graph = create_deep_agent(
    model=build_model(),
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)

main = TimeTravel(title="Deep Research")


@main.agent(
    name="deep_research",
    framework="langgraph",
    target=agent_graph,
    description="Type your research question — TimeTravel builds the graph input.",
)
async def run(query: str, config: dict | None = None) -> Any:
    """Run the graph; TimeTravel intercepts every LLM and tool call inside."""
    return await agent_graph.ainvoke(
        {"messages": [{"role": "user", "content": query}]}, config or None
    )
