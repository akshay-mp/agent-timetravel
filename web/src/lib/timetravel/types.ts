/**
 * TimeTravel × Deep Research — core types
 *
 * Mirrors the data model from akshay-mp/TimeTravel:
 *   - A Trace is a recorded run of an agent (here: the Deep Research flow).
 *   - A Trace contains an ordered list of Spans. Each Span captures one LLM call
 *     (system prompt, user input, assistant output, model, latency, tokens).
 *   - A Trace has a branchId. The first run is branchId === "main".
 *   - A Branch is just a Trace whose parentBranchId points at another Trace and
 *     whose branchAtSpanIndex says where the fork happened.
 *   - Span.source === "cached"  → served from the recorded parent (FROZEN replay,
 *     zero outbound LLM calls, zero tokens, zero cost — the headline TimeTravel benefit).
 *   - Span.source === "live"    → a real LLM call was made this run (BRANCH mode,
 *     the divergent tail starting at the branch point).
 */

export type SpanKind =
  | "clarify_with_user"
  | "write_research_brief"
  | "supervisor_think"
  | "conduct_research"
  | "research_complete"
  | "final_report";

export interface Span {
  id: string;
  /** 0-indexed position inside the trace. */
  index: number;
  /** Human-readable name shown in the timeline. */
  name: string;
  kind: SpanKind;
  type: "llm";
  model: string;
  /** System prompt for this LLM call. Editable when branching. */
  systemPrompt: string;
  /** User-side input for this LLM call (rebuilt from prior span outputs). */
  userInput: string;
  /** Assistant response. */
  output: string;
  /**
   * The model's chain-of-thought / `<think>` reasoning for this call.
   * Captured from `reasoning_content` (llama-server / Qwen3) while thinking is
   * enabled. Empty for servers that don't emit reasoning, and for cached spans
   * that inherited no reasoning from their parent.
   */
  reasoning?: string;
  latencyMs: number;
  tokensIn: number;
  tokensOut: number;
  /**
   * "live" = a real LLM call was made for this span during this trace.
   * "cached" = served verbatim from the parent trace's recording (FROZEN replay).
   */
  source: "live" | "cached";
}

export interface Trace {
  id: string;
  /** "main" for the original run; a short id for branches. */
  branchId: string;
  /** Parent branch id, or null if this is the root. */
  parentBranchId: string | null;
  /** Where in the parent this branch forked (null for root). */
  branchAtSpanIndex: number | null;
  /** The original user query. */
  query: string;
  /** Optional label the user gave the branch. */
  label: string;
  /** Optional note describing what prompt change the branch tests. */
  note: string;
  spans: Span[];
  createdAt: number;
}

/** Pair of spans at the same index from two branches, with a word-level diff. */
export interface SpanDiffPair {
  index: number;
  kind: SpanKind;
  name: string;
  left: Span | null;
  right: Span | null;
  /** True if left.output !== right.output (or one side missing). */
  diverged: boolean;
  /** Word-level diff of the two outputs (and of the system prompt). */
  outputDiff: DiffToken[];
  systemPromptDiff: DiffToken[];
}

export type DiffToken =
  | { type: "equal"; value: string }
  | { type: "add"; value: string }
  | { type: "remove"; value: string };

export interface BranchDiff {
  leftBranchId: string;
  rightBranchId: string;
  /** First index where the two branches diverge. */
  firstDivergenceIndex: number | null;
  pairs: SpanDiffPair[];
}

// ---------------------------------------------------------------------------
// Live-run streaming types
//
// A run (capture or branch) is streamed as NDJSON: one JSON object per line.
// The store reconstructs a LiveRun from these events and the ThinkingPanel
// renders it as the model works. On `trace_end` the finished Trace is
// committed via addTrace() and the live view is dismissed.
// ---------------------------------------------------------------------------

/** One span as it is being generated (shown in the ThinkingPanel). */
export interface LiveSpan {
  index: number;
  name: string;
  kind: SpanKind;
  /** Accumulated reasoning so far (may still be streaming in). */
  reasoning: string;
  /** Accumulated answer so far (may still be streaming in). */
  output: string;
  /** Where this span is in its lifecycle. */
  status: "thinking" | "answering" | "done";
  startedAt: number;
  endedAt: number | null;
}

/** A run in progress, surfaced in the ThinkingPanel before it's committed. */
export interface LiveRun {
  query: string;
  /** Whether this is a fresh capture ("run") or a divergence ("branch"). */
  kind: "run" | "branch";
  spans: LiveSpan[];
  /** Index of the span currently generating, or null when finished. */
  currentIndex: number | null;
  status: "running" | "done" | "error";
  error: string | null;
  startedAt: number;
}

/**
 * Wire format for the streaming run/branch endpoints (one JSON object per
 * line). Events arrive in order; the client appends to the matching LiveSpan.
 */
export type StreamEvent =
  | { type: "span_start"; index: number; name: string; kind: SpanKind }
  | { type: "reasoning_delta"; index: number; chunk: string }
  | { type: "content_delta"; index: number; chunk: string }
  | { type: "span_end"; index: number; span: Span }
  | { type: "trace_end"; trace: Trace }
  | { type: "error"; message: string };

// ---------------------------------------------------------------------------
// Interactive stepping sessions (Phase 9)
//
// A stepping session runs an agent server-side under mode=INTERACTIVE; the
// agent pauses at every LLM/tool call and surfaces the pending step to the
// browser via an SSE stream. The browser posts a Decision to resume it.
// These types mirror the Python view models in stepping_api.py and the
// Step/Decision dataclasses in stepping.py.
// ---------------------------------------------------------------------------

/** The payload of a paused LLM or tool step, mirroring stepping.Step.payload. */
export interface StepPayload {
  model?: string;
  messages?: unknown[];
  tools?: unknown[];
  params?: Record<string, unknown>;
  /** For tool steps only. */
  name?: string;
  args?: unknown[];
  kwargs?: Record<string, unknown>;
  /** Optional server/tool status marker used by the workbench graph. */
  status?: string;
}

/** Token accounting reported by the provider for a completed LLM call. */
export interface StepUsage {
  input_tokens: number;
  cached_input_tokens?: number;
  /** Completion tokens, including hidden/provider reasoning when present. */
  output_tokens: number;
  thinking_tokens: number;
  final_tokens: number;
  total_tokens: number;
  /** True when TimeTravel calculated a local approximation. */
  estimated: boolean;
}

/** Local pricing for one model/provider profile, in US dollars per million tokens. */
export interface PricingProfile {
  name: string;
  inputPerMillion: number;
  cachedInputPerMillion: number;
  outputPerMillion: number;
  thinkingPerMillion: number;
}

/** Reusable expected-output checks attached to a reviewed step or variant. */
export interface OutputAssertions {
  requiredText: string[];
  forbiddenText: string[];
  requireJson: boolean;
  requireCitations: boolean;
  maxTokens: number | null;
  maxCostUsd: number | null;
}

export interface AssertionResult {
  passed: boolean;
  failures: string[];
  evaluatedAt: number;
}

export interface LiveCheckpoint {
  name: string;
  label: string;
  cursor: number;
  keys: string[];
}

/** A local rule that stops automatic execution before an intercepted call. */
export interface BreakpointRule {
  id: string;
  label: string;
  type: "tool_name" | "model_name" | "message_contains" | "token_limit";
  value: string;
  enabled: boolean;
}

/** One SSE event from GET /api/v1/sessions/{id}/stream (one data: <json> per event). */
export type StepEvent =
  | { type: "paused"; cursor: number; kind: string; step: StepPayload; pause_reason?: "pause_after_current" | "breakpoint" | null }
  | { type: "dispatching"; cursor: number; decision: string }
  | { type: "reasoning_delta"; cursor: number; chunk: string }
  | { type: "resumed"; decision: string }
  | { type: "step_completed"; cursor: number; kind: string; result: string; usage?: StepUsage }
  | { type: "checkpoint"; name: string; label: string; cursor: number; keys: string[] }
  | { type: "done"; reason?: string; cursor?: number }
  | { type: "errored"; message: string };

/** The step currently awaiting a Decision, held in the store. */
export interface PausedStep {
  cursor: number;
  /** "llm" | "tool" | "mcp" — drives the kind tint in the UI. */
  kind: string;
  payload: StepPayload;
  /** Date.now() when the pause arrived — drives the live elapsed timer. */
  pausedAt: number;
  /** Date.now() when the model response arrived; freezes LLM duration. */
  completedAt: number | null;
  /** The model's response text, once the step has executed (verify loop). */
  result: string | null;
  /** Provider-emitted reasoning extracted from an explicit <think> block. */
  reasoning: string | null;
  /** Provider token accounting, when the response includes it. */
  usage: StepUsage | null;
  /** queued -> model call not started; running -> in flight; completed -> reviewable. */
  phase: "queued" | "running" | "completed";
  /** A prior captured step restored locally by the inspection timetravel control. */
  restored?: boolean;
  reviewNote?: string;
  reviewVerdict?: "accepted" | "rejected" | null;
  breakpointLabel?: string;
  assertions?: OutputAssertions;
  assertionResult?: AssertionResult;
}

/** A consumed step in the history rail. */
export interface StepHistoryEntry {
  cursor: number;
  kind: string;
  /** "approve" | "edit" | "stop" | "step_once" | "reject" | "run_until_breakpoint". */
  decision: string;
  payload?: StepPayload;
  /** What the model returned for this step (the verify-loop result). */
  result?: string | null;
  reasoning?: string | null;
  usage?: StepUsage | null;
  /** Provider/tool execution duration; excludes time spent waiting for review. */
  latencyMs?: number;
  resolvedAt: number;
  reviewNote?: string;
  reviewVerdict?: "accepted" | "rejected" | null;
  assertions?: OutputAssertions;
  assertionResult?: AssertionResult;
}

/** Immutable record of a prompt experiment initiated from an executed step. */
export interface PromptVersion {
  id: string;
  cursor: number;
  createdAt: number;
  baseMessages: unknown[];
  messages: unknown[];
  baseModel: string;
  model: string;
  status: "running" | "completed";
  result: string | null;
  usage: StepUsage | null;
  /** Exact provider kwargs sent for this call, including sampling controls. */
  parameters?: Record<string, unknown>;
  branchId?: string;
  parentVersionId?: string | null;
  authorNote?: string;
  pricing?: PricingProfile;
  assertions?: OutputAssertions;
  assertionResult?: AssertionResult;
  reviewVerdict?: "accepted" | "rejected" | null;
  reviewNote?: string;
  evaluatorNames?: string[];
  evaluatorResults?: Record<string, { passed: boolean; detail?: string }>;
  reasoning?: string | null;
  latencyMs?: number;
  completedAt?: number;
  /** 1-based creation ordinal among variants of the same step; derived at render time. */
  ordinal?: number;
}

/** One stepping session's live state, parallel to LiveRun. */
export interface LiveSession {
  sessionId: string;
  traceId: string;
  branchId: string;
  runnerRef: string;
  agentRef?: string;
  inputPayload?: unknown;
  resultPayload?: unknown;
  status: "running" | "paused" | "done" | "errored";
  error: string | null;
  pausedStep: PausedStep | null;
  history: StepHistoryEntry[];
  /** Captured steps available for local-only forward navigation after timetravel. */
  savedFuture: StepHistoryEntry[];
  promptVersions: PromptVersion[];
  checkpoints: LiveCheckpoint[];
  startedAt: number;
}

/** A local, portable snapshot shown in the saved-session library. */
export interface SavedSessionCase {
  id: string;
  createdAt: string;
  traceId: string;
  runnerRef: string;
  steps: StepHistoryEntry[];
  promptVersions: PromptVersion[];
  checkpoints?: LiveCheckpoint[];
  /** Pricing active when this local regression case was captured. */
  pricing?: PricingProfile;
  summary?: {
    accepted: number;
    rejected: number;
    totalTokens: number;
    totalLatencyMs: number;
  };
  regression?: {
    passed: boolean;
    checkedAt: string;
    total: number;
    failures: string[];
  };
}

/** POST body for POST /api/v1/sessions. */
export interface StartSessionBody {
  trace_id: string;
  runner_ref: string;
  mode?: string;
  branch_at?: number | null;
  label?: string;
}

/** Response from POST /api/v1/sessions. */
export interface StartSessionResponse {
  session_id: string;
  trace_id: string;
  branch_id: string;
  status: string;
}

/** POST body for POST /api/v1/sessions/{id}/decide. */
export interface DecisionBody {
  kind: "approve" | "edit" | "stop" | "step_once" | "mock" | "skip" | "reject" | "run_until_breakpoint";
  messages?: unknown[];
  params?: Record<string, unknown>;
  args?: unknown[];
  kwargs?: Record<string, unknown>;
  model?: string;
  mock_result?: unknown;
  /** For reject: human-readable reason surfaced back to the agent. */
  reason?: string;
}

/** Server-owned run-control intent. PATCH /api/v1/sessions/{id}/run-control. */
export interface RunControlIntent {
  pause_after_current: boolean;
  run_until_breakpoint: boolean;
  breakpoints: BreakpointRule[];
}
