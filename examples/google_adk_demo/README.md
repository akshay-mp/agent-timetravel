# Google ADK demo — Gemma 4 on Unsloth, stepped through TimeTravel

A minimal [Google ADK](https://google.github.io/adk-docs/) agent project that
runs a **tool-calling agent on Gemma 4 served locally by Unsloth** and lets
TimeTravel's step debugger drive every LLM and tool call from the browser.

No manual model wrapping: the workbench auto-activates the ADK interceptor
(`agent_timetravel.adk_intercept`) for the `framework="adk"` agent below.

## What's inside

- `app.py` — the whole project:
  - `OpenAICompatLlm` — a custom ADK `BaseLlm` bridging to any
    OpenAI-compatible endpoint (Unsloth Studio, llama.cpp, vLLM, Ollama)
    through the `openai` SDK. No `google-adk[extensions]` needed.
  - `get_weather` — a plain function ADK auto-wraps as a tool, so runs
    include a real tool-call round trip.
  - `ask` — the `@timetravel.agent(framework="adk")` entry point the
    workbench registers.

## Prerequisites

- `pip install agent-timetravel[adk]` (ADK + TimeTravel) in the environment.
- Unsloth Studio (or llama.cpp) serving Gemma, e.g. the model id
  `unsloth/gemma-4-12b-it-GGUF` on `http://127.0.0.1:8888/v1`.

## Run headless (no UI)

```bash
python examples/google_adk_demo/app.py "What is the weather in Tokyo?"
```

## Run under the workbench

From the repository root (so `examples.google_adk_demo` imports):

```bash
agent-timetravel dev examples.google_adk_demo.app:timetravel \
  --db /tmp/adk-gemma-demo.db --no-open
```

Open <http://127.0.0.1:8484/ui/>, pick the **ask** agent, start a session
with `{"question": "What is the weather in Tokyo?"}`. Each LLM turn and the
`get_weather` tool call pause at the gate — approve, edit the prompt, mock
the tool, or branch the run.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ADK_GEMMA_BASE_URL` | `http://127.0.0.1:8888/v1` | OpenAI-compatible endpoint |
| `ADK_GEMMA_MODEL` | `unsloth/gemma-4-12b-it-GGUF` | model id as reported by the server |
| `OPENAI_API_KEY` | `local` | Unsloth Studio requires the key from your `.env` |
