# Per-Framework Replay Adapters

Replay is the debug-mode counterpart to capture. Where the
OpenInference instrumentation *records* every LLM call, the TimeTravel replay
adapters **serve recorded responses back to your agent** so you can
re-run a trace from any span without paying for live model calls.

Capture is always-on (production-safe, zero TimeTravel code in the hot path).
Replay is opt-in (debug only, wraps your model with a context manager).

## The common shape

Every adapter follows the same usage pattern:

```python
from agent_timetravel.adapters.<framework> import replay_<model>

# Wrap your existing model once:
real_model = MyFrameworkModel(...)
replay_model = replay_<model>(real_model, trace_id="…")

# Then run your agent inside the replay ctxmgr:
from agent_timetravel.replay import replay

with replay(trace_id="…", branch_at=3, mode="branch"):
    agent.run()  # calls ≤ cursor served from fixtures; 4+ go live
```

The wrapped model is what your agent receives. Calls whose
`messages_hash` matches a recorded span at index ≤ cursor return the
recorded response — **zero HTTP egress**. Calls beyond the cursor are
forwarded live to the underlying `real_model` and the new response is
captured under a new branch id.

## Adapter matrix

| Framework | Adapter factory | Wraps | Import extra |
|---|---|---|---|
| **LangGraph** | `agent_timetravel.adapters.langgraph.replay_chat_model` | `BaseChatModel._generate` (sync + async) | `pip install agent-timetravel[langgraph]` |
| **Google ADK** | `agent_timetravel.adapters.adk.replay_llm` | `BaseLlm.generate_content_async` | `pip install agent-timetravel[adk]` |
| **CrewAI** | `agent_timetravel.adapters.crewai.replay_llm` | `BaseLLM.call[_async]`, `get_response[_async]` | `pip install crewai` |
| **PydanticAI** | `agent_timetravel.adapters.pydantic_ai.replay_model` | `Model.request[_stream]` | `pip install agent-timetravel[pydantic-ai]` |
| **SmolAgents** | `agent_timetravel.adapters.smolagents.replay_model` | `Model.__call__`, `generate`, `astream` | `pip install agent-timetravel[smolagents]` |
| **Generic OpenAI** | `agent_timetravel.replay` ctxmgr (monkey-patch fallback) | `openai.resources.chat.completions.Completions.create` (sync + async + streaming) | None — always available |

The **generic OpenAI** path is the fallback when no framework-specific
adapter exists. It's the only path that uses monkey-patching; the
per-framework adapters subclass the framework's own LLM base class and
are the recommended path for any supported framework.

## Per-framework notes

### LangGraph  *(pattern origin — Phase 3)*

> **Workbench runs no longer need this wrapper.** `agent-timetravel dev` /
> `agent-timetravel app:main` auto-activates LangGraph interception (every
> `BaseChatModel` / `BaseTool` invoke, stepped and captured). The factory
> below remains for replay contexts you drive yourself from Python.

```python
from agent_timetravel.adapters.langgraph import replay_chat_model

real_chat_model = ChatOpenAI(model="gpt-4o-mini")
replay_chat = replay_chat_model(real_chat_model, trace_id="…")

# Use replay_chat anywhere you'd have used real_chat_model:
graph = build_graph(llm=replay_chat)
with replay(trace_id="…", branch_at=2, mode="branch"):
    graph.invoke(initial_state)
```

### ADK  *(Phase 6)*

> **Workbench runs no longer need this wrapper.** `agent-timetravel dev` /
> `agent-timetravel app:main` auto-activates ADK interception (every
> `BaseLlm.generate_content_async` and ADK `BaseTool.run_async` is stepped
> and captured). The factory below remains for replay contexts you drive
> yourself from Python.

Install the supported Google ADK 1.x range alongside TimeTravel's adapter extra:

```bash
pip install "google-adk>=1.28.1,<2" agent-timetravel[adk]
```

```python
from agent_timetravel.adapters.adk import replay_llm
from google.adk.models import BaseLlm

real_llm: BaseLlm = load_your_adk_model()
replay_wrapped = replay_llm(real_llm, trace_id="…")
# Pass replay_wrapped to your ADK agent's `model=…` slot.
# ADK invokes replay_wrapped.generate_content_async(request, stream=False).
```

Decorate the workbench entry point with `@timetravel.agent(framework="adk")`
(or rely on `framework="auto"` detection) and the interceptor patches every
concrete `BaseLlm` subclass for the run — models constructed inside
callbacks are covered too, and ADK tools get the same mock / skip / reject
/ edit semantics as langchain tools.

### CrewAI  *(Phase 6)*

```python
from agent_timetravel.adapters.crewai import replay_llm
from crewai.llms.base_llm import BaseLLM

real_crewai_llm: BaseLLM = load_your_crewai_llm()
replay_wrapped = replay_llm(real_crewai_llm, trace_id="…")
# Pass replay_wrapped as the `llm=` arg to your Crew.
```

### PydanticAI  *(Phase 6)*

```python
from agent_timetravel.adapters.pydantic_ai import replay_model
from pydantic_ai.models import Model

real_model: Model = load_your_pydantic_ai_model()
replay_wrapped = replay_model(real_model, trace_id="…")
# Pass replay_wrapped to your `Agent(model=…)` constructor.
```

### SmolAgents  *(Phase 6)*

```python
from agent_timetravel.adapters.smolagents import replay_model
from smolagents.models import Model

real_smol: Model = load_your_smolagents_model()
replay_wrapped = replay_model(real_smol, trace_id="…")
# Pass replay_wrapped to your `CodeAgent(model=…)` or similar.
```

### Generic OpenAI  *(fallback — Phase 3)*

When no framework-specific adapter exists:

```python
from agent_timetravel.replay import replay

with replay(trace_id="…", branch_at=2, mode="branch"):
    # Any call to openai.ChatCompletion.create (sync/async/streaming)
    # is intercepted: ≤ cursor → cached, > cursor → live + captured.
    response = client.chat.completions.create(...)
```

This path monkey-patches the OpenAI SDK at ctxmgr entry and removes the
patch at exit. The patch is **not** thread-safe; use the per-framework
adapters for any multi-threaded agent.

## When to use what

| If your framework… | Use |
|---|---|
| Is in the adapter matrix above | The framework-specific adapter. Always preferred. |
| Uses the OpenAI SDK directly (no framework) | The generic `agent_timetravel.replay()` ctxmgr. |
| Uses a non-OpenAI LLM client (Anthropic, Cohere, custom HTTP) | The generic ctxmgr won't apply. Either write a thin adapter (see `docs/phases/phase-6.md` §6.1 for the pattern) or use the eval harness (`docs/phases/phase-5.5.md`) to score variants without replay. |

## Troubleshooting

**`AdapterError: install the framework dependency`** — you called the
adapter factory without the framework installed. Install the dependency
listed in the adapter matrix, or switch to the generic OpenAI ctxmgr if
your framework isn't supported.

**Frozen replay diverged** — the agent's `messages_hash` doesn't match
any recorded span at or before the cursor. Most common cause: the
agent's prompt template changed since the trace was recorded. Either
branch from an earlier cursor (before the template change) or re-capture
the trace.

**"Zero egress" assertion fails** — during frozen replay, the agent made
an outbound call that didn't match any recorded span. This is exactly
what frozen mode is supposed to prevent — investigate the divergent call
in the diff UI; it usually surfaces a non-deterministic prompt source
(timestamps, UUIDs, random shuffles).
