# TimeTravel

### Time-travel debugging for AI agents — an OTel-in / replay-out engine

> TimeTravel an agent to any span, change a prompt, and re-run **live** from there —
> branching a new timeline you can diff against the original.
> Consumes standard OpenTelemetry / OpenInference traces. No cloud, no API keys,
> no persistent production proxy, no data leaving the machine.

---

## What it does

A developer running an agent on `qwen3:32b` via Ollama captures a run with any
OpenInference/OTel instrumentor, timetravels to span 4, edits the system prompt,
branches the execution forward live against the same local model, and sees a
side-by-side diff of what changed — all offline, in under a minute of setup.

## Architecture (the key insight)

```
Capture = PASSIVE   ->  an OTel span only exists *after* a call completes.
                        OpenTelemetry + OpenInference solve this. We ingest.
Replay  = ACTIVE    ->  to timetravel we *inject* the cached response during a
                        re-run. That is runtime patching, not observability.
```

So TimeTravel does **not** need its own capture proxy. It needs:

1. A **local OTLP receiver** that stores traces into SQLite (production path,
   zero agent-side lock-in).
2. A **decorator-first workbench** that invokes registered agents with typed
   inputs and a `TimeTravelContext` only during a debug session.
3. An **opt-in replay-time LLM-client wrapper** (`timetravel.replay()`) — *only*
   active during a debug session, never in production.

## Status

| Phase | What | Status |
|---|---|---|
| P0 | Foundation + OTel-shaped data model | ✅ Done (`docs/phases/phase-0.md`) |
| P1 | OTLP ingestion + receiver + storage | ✅ Done (`docs/phases/phase-1.md`) |
| P2 | Read-only timeline UI | ✅ Done (`docs/phases/phase-2.md`) |
| P3 | Replay engine + interceptor (the moat) | ✅ Done (`docs/phases/phase-3.md`) |
| P4 | State checkpointing | ✅ Done (`docs/phases/phase-4.md`) |
| P5 | Branching & diff UI | ✅ Done (`docs/phases/phase-5.md`) |
| P5.5 | Batch parallel eval harness | ✅ Done (`docs/phases/phase-5.5.md`) |
| P6 | Per-framework replay adapters | ✅ Done (`docs/phases/phase-6.md`) |
| P7 | Local-model enrichment | ✅ Done (`docs/phases/phase-7.md`) |
| P8 | Polish, packaging, distribution | ✅ Done (`docs/phases/phase-8.md`) |
| P9 | Interactive step-through debugging | ✅ Done ([docs/phases/phase-9.md](docs/phases/phase-9.md)) |

## Quick start

```bash
pip install agent-timetravel
# The wheel includes the built timeline UI; no separate frontend build is needed.
agent-timetravel serve --port 4318 --db ./timetravel.db
# Point your OTel/OpenInference-instrumented agent at:
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
# Run a trace, then inspect it in the browser:
agent-timetravel ui --port 8484 --db ./timetravel.db
# → http://127.0.0.1:8484/ui
```

### Decorator-first agents

The current workbench entry point is a `TimeTravel` object with typed agent inputs:

```python
from agent_timetravel import TimeTravelContext, timetravel

@timetravel.agent(description="Answer a question")
async def answer(question: str, context: TimeTravelContext | None = None) -> str:
    return question
```

Run `agent-timetravel dev app:timetravel` to expose the agent list and interactive
sessions at the local UI. Direct calls to `answer(...)` remain ordinary
pass-through calls; `TimeTravelContext` is injected only for workbench runs.
For a custom title or separate registry, use:

```python
from agent_timetravel import TimeTravel, TimeTravelContext

timetravel = TimeTravel(title="Research")
```

Existing names such as `debugger` remain supported.

During a workbench run, TimeTravel auto-activates interception for the
frameworks below — no manual model wrapping required:

* **OpenAI** — official OpenAI Python SDK Chat Completions calls
  (`chat.completions.create`, sync and async), including when the SDK is
  configured for an OpenAI-compatible endpoint.
* **LangGraph / langchain** — every `BaseChatModel` / `BaseTool`
  `invoke`/`ainvoke` (which `bind_tools` bindings and ToolNode calls route
  through), so graphs that construct models inside their nodes are stepped,
  replayed, and captured unchanged. This includes graphs built with
  deepagents, LangGraph's prebuilt `create_react_agent`, and similar
  frameworks on top of `langchain-core`.
* **Google ADK** — every `BaseLlm.generate_content_async` (Gemini,
  Gemma, LiteLlm, Anthropic, and user-defined subclasses alike) and every ADK
  `BaseTool.run_async`, so agents built on `google-adk` are stepped,
  replayed, and captured without wrapping their models, including models
  constructed inside callbacks.

Other framework replay adapters (CrewAI, PydanticAI, SmolAgents)
remain explicit: use the factories in
[`docs/replay-adapters.md`](docs/replay-adapters.md) when needed. Generic
decorator auto-activation for those frameworks is currently unavailable, and
the workbench reports the actionable adapter/wrapper instead of claiming the
framework is installed.

### Run any LangGraph app

A foreign LangGraph project needs no TimeTravel-specific code. Install
`agent-timetravel[langgraph]` alongside the app, then point the CLI at the
exported graph:

```bash
pip install agent-timetravel[langgraph]   # or: pip install -e /path/to/timetravel[langgraph]
agent-timetravel app:main                          # ≡ agent-timetravel dev app:main
```

`app:main` may be a `timetravel.TimeTravel` registry, a compiled LangGraph graph /
langchain runnable (wrapped into a one-agent registry automatically), or a
plain callable. The workbench opens in your browser
(`--no-open` to suppress) with the graph registered as an interactive agent:
start it from the form, and every LLM and tool call pauses in the
step-by-step debugger.

### Replay a recorded trace

```bash
# Read-only inspection (prints cursor + branch info):
agent-timetravel replay <trace_id> --mode frozen --db ./timetravel.db

# Branch from span index 4 and go live from there:
agent-timetravel replay <trace_id> --branch-at 4 --mode branch --db ./timetravel.db
```

### From Python (the load-bearing integration point)

```python
from agent_timetravel.replay import replay
from agent_timetravel.storage import TraceStore

store = TraceStore("~/.agent-timetravel/timetravel.db")

# Frozen replay — zero outbound calls, deterministic:
with replay(store, trace_id="<trace>", mode="frozen"):
    agent.run()  # every LLM call served from the recorded spans

# Branch from span 4, then go live:
with replay(store, trace_id="<trace>", branch_at=4, mode="branch"):
    agent.run()  # spans 0-3 from recording, span 4+ calls your live model
```

### Wrap a framework model (Phase 6 adapters)

One import + one wrapper call per agent — no upstream framework changes:

```python
# Google ADK (manual factory — workbench runs on framework="adk" intercept
# automatically, so this is only needed for replay contexts you drive yourself)
from agent_timetravel.adapters.adk import replay_llm
agent = Agent(model=replay_llm(real_adk_llm))

# CrewAI
from agent_timetravel.adapters.crewai import replay_llm
crew.llm = replay_llm(real_crewai_llm)

# PydanticAI
from agent_timetravel.adapters.pydantic_ai import replay_model
agent = Agent(model=replay_model(real_model))

# HuggingFace SmolAgents
from agent_timetravel.adapters.smolagents import replay_model
agent.model = replay_model(real_smol_model)

# LangGraph (Phase 3 — adapter pattern origin)
from agent_timetravel.adapters.langgraph import replay_chat_model
graph.compiled = replay_chat_model(real_chat_model)
```

Install the optional extras as needed:

```bash
pip install agent-timetravel[adk]              # one framework
pip install agent-timetravel[adk,pydantic-ai] # several TimeTravel-managed frameworks
pip install crewai                             # CrewAI adapter dependency
pip install agent-timetravel[adapters]         # all four
```

Without an extra installed, the corresponding factory raises
`agent_timetravel.adapters.<fw>.AdapterError` with an actionable install hint at call
time. `agent-timetravel --version` and `import agent_timetravel.adapters.<fw>` both succeed
without any framework installed.

### Eval a replay candidate against a baseline (Phase 5.5)

```bash
agent-timetravel eval suite.yaml --db ./timetravel.db --suite-name my-suite
# exit 0 = PASS, 1 = FAIL, 2 = ERROR/validation
```

### Run the live decorator-first workbench

The verified local demo uses the decorator-first registry and a seeded
OpenAI-compatible Gemma/Unsloth endpoint. Start the backend with
[`examples/start_deep_research_stepping.py`](examples/start_deep_research_stepping.py),
run Vite on port 5174, and open the workbench on port 8484. The complete
commands and acceptance walkthrough are in
[`docs/interactive-workbench-testing.md`](docs/interactive-workbench-testing.md).

The current walkthrough verifies fresh sessions, automatic `PROCEED`
advancement, substantive-call pauses after the final response, separate
thinking, token/cost/latency/context panels, checkpoints, saved-step
navigation, no-call timetravel/forward, continue-from-checkpoint, and the
available edit, variant, assertion, review, and regression flows. It does not
claim that every planned recording or framework-integration feature is
implemented or demonstrated.

The older `register_runner`/`POST /api/v1/sessions` surface remains available
as an advanced escape hatch for custom runners and explicit framework wiring;
it is not the primary decorator-first usage path.

## Development

```bash
# from agent_timetravel/
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# full quality gate (run before commit)
ruff check src/agent_timetravel tests
pylint src/agent_timetravel/
mypy --strict src/agent_timetravel
python -m pytest tests --no-cov -q

# per-phase security scan (ruff S-rules + bandit, deepsec if available)
python scripts/security_scan.py --phase <N>

# frontend dev server (Vite 6 — must run from web/; --host avoids IPv6 trap)
cd web && pnpm dev   # or:  node_modules/.bin/vite --host 127.0.0.1
```

Latest verified full suite: **527 passed, 13 skipped, 49 deselected, and 3
warnings**. Frontend TypeScript typecheck and production build passed, and
`git diff --check` passed. See
[`docs/interactive-workbench-testing.md`](docs/interactive-workbench-testing.md)
for the live workbench checks.

## Layout

```
timetravel/
  src/agent_timetravel/          Python package
    adapters/          Phase 6 — per-framework replay wrappers (adk, crewai,
                       pydantic_ai, smolagents, langgraph) + shared _common.py
    receiver.py        OTLP/HTTP ingest (Phase 1)
    replay.py          Frozen / branch / full replay engine (Phase 3)
    checkpoint.py      State snapshot/restore (Phase 4)
    diff.py            Trace diff (Phase 5)
    eval_api.py        Suite runner + baseline diff (Phase 5.5)
    cli.py             Click-based CLI: serve / ui / replay / eval / version
  tests/               pytest suites (latest full suite: 527 passed)
  web/                 React + Vite + TypeScript timeline UI (P2)
  docs/
    phases/            Per-phase: QA, security, dev-handoff, design
    diagrams/          Architecture + sequence (.mmd) for each phase
  scripts/
    dev_seed_serve.py  Local dev harness
    security_scan.py   ruff S + bandit per-phase vulnerability scan
  .deepsec/            Vulnerability scan reports (when deepsec available)
  pyproject.toml       Strict ruff + pylint + mypy config; optional extras
```

See `docs/README.md` for a navigable index of all phase docs and diagrams.
For wheel builds, installation requirements, and release publishing, see
[`docs/packaging-release.md`](docs/packaging-release.md).

## Out of scope (v1)

- A bespoke capture proxy / capture decorator SDK (OTel + OpenInference solve
  this; Phase 6 adds a thin adapter factory for replay, not capture).
- Cloud / multi-user / team / sync (a trace is a local SQLite file).
- MCP security sandbox, model routing, mobile/remote access.

See `plan.md` (parent dir) for the full phased plan.
