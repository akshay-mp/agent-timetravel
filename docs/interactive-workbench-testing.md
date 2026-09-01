# Interactive Workbench Verification

This guide covers the current decorator-first local debugger at `/ui/`. It is
separate from [`e2e-ui-testing.md`](e2e-ui-testing.md), which covers the
original read-only trace timeline.

## Start the live demo

The verified demo uses a local OpenAI-compatible Gemma/Unsloth server at
`127.0.0.1:8888/v1`. Keep the API key in the shell environment or the local
repository `.env` file; do not commit it, paste it into a command transcript,
or record it in screenshots or video. The example loads `.env` when present.

Set the endpoint variables in the environment or `.env` (use the model ID
reported by the server):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8888/v1
export OPENAI_API_KEY="${LOCAL_OPENAI_API_KEY:?set this in the environment}"
export TIMETRAVEL_MODEL=unsloth/gemma-4-12b-it-GGUF
```

`LOCAL_OPENAI_API_KEY` is only a shell variable name; keep its value out of
this file and all recording surfaces.

Check model-server health before starting the workbench:

```bash
curl -fsS http://127.0.0.1:8888/v1/models \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-local}"
```

The response should list the Gemma/Unsloth model. From the repository root,
start the seeded interactive backend:

```bash
./.venv/bin/python examples/start_deep_research_stepping.py \
  --db /tmp/timetravel-demo.db \
  --host 127.0.0.1 \
  --port 8484
```

Start the Vite UI in another terminal if it is not already running:

```bash
cd web
npm run dev -- --host 127.0.0.1 --port 5174
```

Open <http://127.0.0.1:5174/ui/> and click **Start Agent**.

The backend serves the API and workbench on
<http://127.0.0.1:8484/ui/>. Vite on port 5174 is the development UI used
for this live verification. The seeded example imports
`from agent_timetravel import TimeTravel, TimeTravelContext`, creates
`timetravel = TimeTravel(title="Deep Research")`, and registers typed input with
`@timetravel.agent`. The explicit example command above is convenient for this
custom-titled demo. Existing names such as `debugger` remain supported.

## Interception boundary

The decorator is generic registration plus typed input validation. During a
workbench invocation, official OpenAI Python SDK Chat Completions calls
(`chat.completions.create`, sync and async) are intercepted, including when
that SDK is configured for an OpenAI-compatible endpoint. LangGraph /
langchain apps get the same auto-activation: every `BaseChatModel` and
`BaseTool` `invoke`/`ainvoke` inside the run is stepped, replayed, and
captured — `agent-timetravel app:main` accepts a bare compiled graph as the launch
target. Google ADK agents get the same generic activation: every
`BaseLlm.generate_content_async` and ADK `BaseTool.run_async` is stepped and
captured (see `examples/google_adk_demo/` for a live Gemma-on-Unsloth demo).
Other framework replay adapters (CrewAI, PydanticAI, SmolAgents) remain
explicit; the capability response reports the actionable adapter/wrapper
instead of assuming an optional framework package is installed. See
[`replay-adapters.md`](replay-adapters.md) and [`wiring.md`](wiring.md).

## Acceptance Walkthrough

1. **Start a fresh session.** Start the seeded agent and verify that a new
   session has no prior execution path or saved state.
2. **Review an LLM call.** A substantive call should pause after its final
   response is ready. The UI shows a collapsed **Thinking** section separate
   from the **Final response**, plus token, cost, latency, and context panels.
   **Next Step** controls appear only after the response is ready.
3. **Continue the workflow.** Approve the response and confirm the next
   intercepted call appears in the execution path. An exact `PROCEED` output
   auto-advances to the next substantive step.
4. **Inspect a saved step.** Click any completed execution-path item. The
   saved response should open without another model or tool call.
5. **TimeTravel and continue.** Use **Step Back / TimeTravel**, move forward through
   saved history, and verify timetravel/forward performs no model or tool call.
   Use **Continue from here** to create a successor run only when execution
   is actually resumed. Checkpoint events should remain visible at the saved
   boundaries.
6. **Edit a prompt.** Choose **Edit Prompt & Run**, change the messages or
   model, and run the variant. The prompt-version list should retain the
   baseline and edited variant, including parameters, reasoning, final text,
   usage, pricing, assertions, and review state.
7. **Compare variants.** Select two completed variants from the same
   checkpoint and open the comparison matrix. Verify prompt and response
   diffs, token/cost/latency deltas, assertion results, and review verdicts.
   Reasoning must be separate from the displayed final response.
8. **Verify pricing.** Set an output price such as `2.5` dollars per million
   tokens. Step and session totals should update without changing token counts.
   Set all prices to `0` for a local model.
9. **Verify browser refresh.** Refresh while an LLM step is paused after its
   response. The same step, response, thinking section, usage, cost, and
   review state should return without another LLM call. The server retains the
   in-flight `paused -> dispatching -> step_completed` snapshot for the new
   SSE connection.
10. **Verify tool safety.** Pause at a tool and test **Run Tool**, **Mock**,
   **Skip**, and **Reject**. Mock, skip, and reject must not invoke the live
   tool. Use the integration suite to verify exactly-once behavior.

The edit, variant, assertion, review, and regression flows are verified where
they are surfaced by the current UI. This walkthrough documents current
behavior; it is not a claim that every planned recording feature exists.

### Synchronous OpenAI Calls

Synchronous OpenAI calls can be stepped through when they run in a worker
thread, for example with `await asyncio.to_thread(sync_agent_call)`. A sync
call cannot wait for browser approval while it is executing on the same
asyncio event loop that serves the SSE connection; TimeTravel fails fast with a
clear error in that case. Prefer the async OpenAI client in an async runner,
or move the synchronous call to a worker thread. Sync and async responses both
publish the response usage used by the token and cost panels.

## Automated Checks

Run the full verified quality checks:

```bash
./.venv/bin/pytest -q tests --no-cov
cd web && npm run typecheck && npm run build
cd .. && git diff --check
```

Latest verified result: **527 passed, 13 skipped, 49 deselected, and 3
warnings**; frontend typecheck/build passed; `git diff --check` passed. The
count does not imply that optional framework packages are installed.

The reconnect contract is pinned by
`TestSSEApprovalChannel.test_replay_snapshot_preserves_llm_review_after_refresh`.
The browser walkthrough is retained as a manual check because the repository
does not currently ship a browser-test runner dependency.

## Troubleshooting

- Use one active workbench tab during a stepping run. Multiple tabs sharing
  one browser profile can reconnect to the same session and compete for SSE
  decisions.
- If the UI shows `0 steps` after a backend restart, start a new session.
  Live sessions are process-local; durable prompt versions and saved cases
  remain in SQLite, but an in-memory runner cannot survive process shutdown.
- A local endpoint that does not report usage shows estimated token counts and
  cost. The workbench labels these as local estimates.
