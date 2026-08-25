"use client";

/**
 * Client-side helper for the interactive stepping server (Phase 9).
 *
 * Sibling to streamTimeTravel, but for the bidirectional SSE + decision protocol.
 * The Python stepping server emits SSE events (paused/resumed/done/errored)
 * over GET /api/v1/sessions/{id}/stream; the browser POSTs decisions to
 * /api/v1/sessions/{id}/decide on a SEPARATE connection while the stream
 * stays open. EventSource + fetch multiplex naturally in a real browser.
 *
 * All endpoints are relative ("/api/v1/sessions*") and proxied to the Python
 * backend by next.config.ts rewrites — no CORS, no env-var URL in the browser.
 */

import { api } from "@/api";
import type { AgentSessionRequest } from "@/types";
import { splitReasoning, useTimeTravelStore } from "./store";
import type {
  BreakpointRule,
  DecisionBody,
  PausedStep,
  RunControlIntent,
  StartSessionBody,
  StartSessionResponse,
  StepEvent,
  PromptVersion,
  StepUsage,
  OutputAssertions,
  AssertionResult,
  PricingProfile,
  SavedSessionCase,
  StepHistoryEntry,
} from "./types";

/**
 * Start a stepping session: POST /api/v1/sessions, seed the store's
 * liveSession, then open the SSE stream. Returns the session id.
 *
 * The SSE stream is opened as a side effect (streamSessionDecisions) so the
 * caller can `await startSession(...)` and immediately have events flowing.
 */
export async function startSession(body: StartSessionBody): Promise<string> {
  const {
    startLiveSession,
    failSession,
  } = useTimeTravelStore.getState();

  const res = await fetch("/api/v1/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const message = (err as { detail?: string }).detail ?? `HTTP ${res.status}`;
    failSession(message);
    throw new Error(message);
  }
  const data = (await res.json()) as StartSessionResponse;
  _runUntilNextLlm = false;
  startLiveSession(
    data.session_id,
    data.trace_id,
    data.branch_id,
    body.runner_ref,
  );
  window.localStorage.setItem("timetravel-active-session", data.session_id);
  void hydrateExperimentRecords(data.trace_id);
  void hydrateRunControl(data.session_id);

  // Open the SSE stream. Not awaited — it runs until the session terminates.
  void streamSessionDecisions(data.session_id);
  return data.session_id;
}

/** Start a decorated agent through its agent-scoped session endpoint. */
export async function startAgentSession(agentRef: string, body: AgentSessionRequest): Promise<string> {
  const { startLiveSession, failSession } = useTimeTravelStore.getState();
  let data: StartSessionResponse;
  try {
    data = await api.startAgentSession(agentRef, body);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    failSession(message);
    throw error;
  }
  _runUntilNextLlm = false;
  startLiveSession(
    data.session_id,
    data.trace_id,
    data.branch_id,
    agentRef,
    false,
    { agentRef, inputPayload: body.inputs },
  );
  window.localStorage.setItem("timetravel-active-session", data.session_id);
  void hydrateExperimentRecords(data.trace_id);
  void hydrateRunControl(data.session_id);
  void streamSessionDecisions(data.session_id);
  return data.session_id;
}

/** Reconnect to a server-owned session after a browser refresh. */
export async function resumeSession(sessionId: string): Promise<string> {
  if (_resumeInFlight) return _resumeInFlight;
  _resumeInFlight = resumeSessionOnce(sessionId);
  try {
    return await _resumeInFlight;
  } finally {
    _resumeInFlight = null;
  }
}

let _resumeInFlight: Promise<string> | null = null;

async function resumeSessionOnce(sessionId: string): Promise<string> {
  const session = await api.getSession(sessionId);
  useTimeTravelStore.getState().startLiveSession(
    session.session_id,
    session.trace_id,
    session.branch_id,
    session.runner_ref,
    false,
    {
      agentRef: session.agent_ref ?? undefined,
      inputPayload: session.input_payload,
      resultPayload: session.result_payload,
    },
  );
  window.localStorage.setItem("timetravel-active-session", session.session_id);
  await hydrateExperimentRecords(session.trace_id);
  void hydrateRunControl(session.session_id);
  void streamSessionDecisions(session.session_id);
  return session.session_id;
}

/** Persist a prompt variant; HTTP failures must remain visible to the caller. */
export async function persistPromptVersion(traceId: string, version: PromptVersion): Promise<void> {
  const response = await fetch(`/api/v1/traces/${encodeURIComponent(traceId)}/prompt-versions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      version_id: version.id,
      cursor_index: version.cursor,
      base_messages: version.baseMessages,
      messages: version.messages,
      base_model: version.baseModel,
      model: version.model,
      branch_id: version.branchId ?? "",
      parent_version_id: version.parentVersionId ?? null,
      parameters: version.parameters ?? {},
      author_note: version.authorNote ?? "",
      assertions: version.assertions ?? {},
      evaluator_names: version.evaluatorNames ?? [],
      created_at: new Date(version.createdAt).toISOString(),
      updated_at: new Date(version.createdAt).toISOString(),
    }),
  });
  if (!response.ok) throw new Error(`prompt version persistence failed: HTTP ${response.status}`);
}

/** Persist a completed result snapshot, including assertion/review deltas. */
export async function persistPromptVersionResult(
  versionId: string,
  version: PromptVersion,
): Promise<void> {
  const response = await fetch(`/api/v1/prompt-versions/${encodeURIComponent(versionId)}/result`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      result: version.result ?? "",
      usage: version.usage ?? {},
      latency_ms: version.latencyMs,
      completed_at: version.completedAt ? new Date(version.completedAt).toISOString() : "",
      reasoning: version.reasoning,
      pricing: version.pricing ?? {},
      assertion_result: version.assertionResult ?? {},
      review_verdict: version.reviewVerdict ?? null,
      review_note: version.reviewNote ?? null,
      evaluator_results: version.evaluatorResults ?? {},
    }),
  });
  if (!response.ok) throw new Error(`prompt version result persistence failed: HTTP ${response.status}`);
}

/** Persist a local pricing snapshot; callers may retain localStorage offline. */
export async function persistPricingProfile(profile: PricingProfile): Promise<void> {
  const now = new Date().toISOString();
  const slug = profile.name.toLowerCase().replace(/[^a-z0-9]+/g, "-") || "model";
  const response = await fetch("/api/v1/pricing-profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile_id: `local-${slug}-${Date.now()}`,
      name: profile.name,
      input_per_million: profile.inputPerMillion,
      cached_input_per_million: profile.cachedInputPerMillion,
      output_per_million: profile.outputPerMillion,
      thinking_per_million: profile.thinkingPerMillion,
      effective_at: now,
      created_at: now,
      updated_at: now,
    }),
  });
  if (!response.ok) throw new Error(`pricing profile persistence failed: HTTP ${response.status}`);
}

/** Load durable pricing profiles when the backend is available. */
export async function loadPricingProfiles(): Promise<PricingProfile[]> {
  const response = await fetch("/api/v1/pricing-profiles");
  if (!response.ok) throw new Error(`pricing profile load failed: HTTP ${response.status}`);
  const records = (await response.json()) as Array<Record<string, unknown>>;
  return records.map((record) => ({
    name: String(record.name ?? "Local model"),
    inputPerMillion: Number(record.input_per_million ?? 0),
    cachedInputPerMillion: Number(record.cached_input_per_million ?? 0),
    outputPerMillion: Number(record.output_per_million ?? 0),
    thinkingPerMillion: Number(record.thinking_per_million ?? 0),
  }));
}

/** Save a regression snapshot and its captured assertions on the server. */
export async function persistRegressionCase(record: SavedSessionCase): Promise<void> {
  const expected = {
    captured_step_count: record.steps.length,
    captured_steps: record.steps.map((step) => ({
      cursor: step.cursor,
      kind: step.kind,
      decision: step.decision,
      payload: step.payload ?? {},
      result: step.result ?? "",
      reasoning: step.reasoning ?? null,
      usage: step.usage ?? {},
      latency_ms: step.latencyMs ?? 0,
      assertions: step.assertions ?? {},
      assertion_result: step.assertionResult ?? {},
      review_verdict: step.reviewVerdict ?? null,
      review_note: step.reviewNote ?? null,
    })),
    prompt_versions: record.promptVersions,
    checkpoints: record.checkpoints ?? [],
    pricing: record.pricing ?? {},
    summary: record.summary ?? {},
  };
  const response = await fetch("/api/v1/regression-cases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: record.id,
      name: `${record.runnerRef} · ${record.createdAt}`,
      seed_trace_id: record.traceId,
      expected,
      created_at: record.createdAt,
    }),
  });
  if (!response.ok) throw new Error(`regression persistence failed: HTTP ${response.status}`);
}

/** Run a persisted case using frozen output/checks; no live call is made. */
export async function runPersistedRegressionCase(caseId: string): Promise<{ passed: boolean; detail?: string }> {
  const response = await fetch(`/api/v1/regression-cases/${encodeURIComponent(caseId)}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  if (!response.ok) throw new Error(`regression run failed: HTTP ${response.status}`);
  return (await response.json()) as { passed: boolean; detail?: string };
}

/** Load server-owned regression snapshots, if the local backend is online. */
export async function loadRegressionCases(): Promise<SavedSessionCase[]> {
  const response = await fetch("/api/v1/regression-cases");
  if (!response.ok) throw new Error(`regression load failed: HTTP ${response.status}`);
  const records = (await response.json()) as Array<Record<string, unknown>>;
  return records.flatMap((record) => {
    const expected = record.expected && typeof record.expected === "object"
      ? record.expected as Record<string, unknown>
      : {};
    const captured = Array.isArray(expected.captured_steps) ? expected.captured_steps : [];
    const steps = captured.flatMap((item) => {
      if (!item || typeof item !== "object") return [];
      const entry = item as Record<string, unknown>;
      return [{
        cursor: Number(entry.cursor ?? 0),
        kind: typeof entry.kind === "string" ? entry.kind : "llm",
        decision: typeof entry.decision === "string" ? entry.decision : "saved",
        payload: entry.payload && typeof entry.payload === "object" ? entry.payload as StepHistoryEntry["payload"] : {},
        result: typeof entry.result === "string" ? entry.result : null,
        reasoning: typeof entry.reasoning === "string" ? entry.reasoning : null,
        usage: entry.usage && typeof entry.usage === "object" ? entry.usage as StepUsage : null,
        latencyMs: Number(entry.latency_ms ?? 0),
        assertions: entry.assertions && typeof entry.assertions === "object" ? entry.assertions as OutputAssertions : undefined,
        assertionResult: entry.assertion_result && typeof entry.assertion_result === "object" ? entry.assertion_result as AssertionResult : undefined,
        reviewVerdict: entry.review_verdict === "accepted" || entry.review_verdict === "rejected" ? entry.review_verdict : null,
        reviewNote: typeof entry.review_note === "string" ? entry.review_note : undefined,
        resolvedAt: Date.now(),
      } satisfies SavedSessionCase["steps"][number]];
    });
    return [{
      id: String(record.case_id ?? ""),
      createdAt: String(record.created_at ?? ""),
      traceId: String(record.seed_trace_id ?? ""),
      runnerRef: String(record.name ?? "server regression"),
      steps,
      promptVersions: Array.isArray(expected.prompt_versions) ? expected.prompt_versions as PromptVersion[] : [],
      checkpoints: Array.isArray(expected.checkpoints) ? expected.checkpoints as SavedSessionCase["checkpoints"] : [],
      pricing: expected.pricing && typeof expected.pricing === "object" ? expected.pricing as PricingProfile : undefined,
      summary: expected.summary && typeof expected.summary === "object" ? expected.summary as SavedSessionCase["summary"] : undefined,
    }];
  });
}

async function hydrateExperimentRecords(traceId: string): Promise<void> {
  try {
    const [versionsResponse, reviewsResponse] = await Promise.all([
      fetch(`/api/v1/traces/${encodeURIComponent(traceId)}/prompt-versions`),
      fetch(`/api/v1/traces/${encodeURIComponent(traceId)}/reviews`),
    ]);
    if (versionsResponse.ok) {
      const records = (await versionsResponse.json()) as Array<Record<string, unknown>>;
      useTimeTravelStore.getState().hydratePromptVersions(records.map(promptVersionFromApi));
    }
    if (reviewsResponse.ok) {
      const records = (await reviewsResponse.json()) as Array<Record<string, unknown>>;
      useTimeTravelStore.getState().hydrateStepReviews(records.map((record) => ({
        cursor: Number(record.cursor_index ?? 0),
        note: typeof record.review_note === "string" ? record.review_note : undefined,
        verdict: record.review_verdict === "accepted" || record.review_verdict === "rejected" ? record.review_verdict : null,
        assertions: record.assertions && typeof record.assertions === "object" ? record.assertions as OutputAssertions : undefined,
        assertionResult: record.assertion_result && typeof record.assertion_result === "object" ? record.assertion_result as AssertionResult : undefined,
      })));
    }
  } catch {
    // An offline browser keeps its in-memory/session cache; the next reconnect retries.
  }
}

function promptVersionFromApi(record: Record<string, unknown>): PromptVersion {
  const usage = record.usage;
  return {
    id: String(record.version_id ?? ""),
    cursor: Number(record.cursor_index ?? 0),
    createdAt: Date.parse(String(record.created_at ?? "")) || Date.now(),
    baseMessages: Array.isArray(record.base_messages) ? record.base_messages : [],
    messages: Array.isArray(record.messages) ? record.messages : [],
    baseModel: String(record.base_model ?? ""),
    model: String(record.model ?? ""),
    status: record.status === "completed" ? "completed" : "running",
    result: typeof record.result === "string" ? record.result : null,
    usage: usage && typeof usage === "object" ? usage as StepUsage : null,
    parameters: record.parameters && typeof record.parameters === "object" ? record.parameters as Record<string, unknown> : {},
    branchId: typeof record.branch_id === "string" ? record.branch_id : undefined,
    parentVersionId: typeof record.parent_version_id === "string" ? record.parent_version_id : null,
    authorNote: typeof record.author_note === "string" ? record.author_note : undefined,
    assertions: record.assertions && typeof record.assertions === "object" ? record.assertions as PromptVersion["assertions"] : undefined,
    evaluatorNames: Array.isArray(record.evaluator_names) ? record.evaluator_names.filter((name): name is string => typeof name === "string") : [],
    assertionResult: record.assertion_result && typeof record.assertion_result === "object" ? record.assertion_result as PromptVersion["assertionResult"] : undefined,
    reviewVerdict: record.review_verdict === "accepted" || record.review_verdict === "rejected" ? record.review_verdict : null,
    reviewNote: typeof record.review_note === "string" ? record.review_note : undefined,
    evaluatorResults: record.evaluator_results && typeof record.evaluator_results === "object" ? record.evaluator_results as PromptVersion["evaluatorResults"] : undefined,
    reasoning: typeof record.reasoning === "string" ? record.reasoning : null,
    pricing: record.pricing && typeof record.pricing === "object" ? record.pricing as PromptVersion["pricing"] : undefined,
    latencyMs: typeof record.latency_ms === "number" ? record.latency_ms : undefined,
    completedAt: typeof record.completed_at === "string" ? Date.parse(record.completed_at) : undefined,
  };
}

/**
 * End a paused run and fork a new interactive session from ``branchAt``.
 *
 * A timetravel is intentionally a new branch, not an attempt to mutate history:
 * the source trace stays available for comparison and the replacement run
 * reuses its recorded prefix.
 */
export async function restartSessionFrom(
  sessionId: string,
  branchAt: number,
  label: string,
): Promise<string> {
  const { startLiveSession, failSession } = useTimeTravelStore.getState();
  const source = useTimeTravelStore.getState().liveSession;

  // A paused runner holds an approval gate open. Stop it before creating the
  // successor so a timetravel never leaves a second agent waiting in the back.
  if (source?.status === "paused" || source?.status === "running") {
    await postDecision(sessionId, { kind: "stop" });
  }
  closeSessionStream();

  const res = await fetch(`/api/v1/sessions/${sessionId}/restart-from`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ branch_at: branchAt, label }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const message = (err as { detail?: string }).detail ?? `HTTP ${res.status}`;
    failSession(message);
    throw new Error(message);
  }

  const data = (await res.json()) as StartSessionResponse;
  _runUntilNextLlm = false;
  startLiveSession(
    data.session_id,
    data.trace_id,
    data.branch_id,
    source?.runnerRef ?? "agent",
  );
  void hydrateRunControl(data.session_id);
  void streamSessionDecisions(data.session_id);
  return data.session_id;
}

/**
 * Open the SSE event stream for a session and dispatch events into the store.
 *
 * Returns a close function (also stashed on the store-free module global so
 * a future "cancel" can tear it down). The stream stays open across decision
 * POSTs — that's the whole point of stepping.
 */
let _activeStream: EventSource | null = null;
let _pendingPromptEdit: { cursor: number; messages: unknown[]; model: string } | null = null;
let _runUntilNextLlm = false;

function matchingBreakpoint(
  rules: BreakpointRule[],
  step: Extract<StepEvent, { type: "paused" }>,
): string | null {
  const messages = JSON.stringify(step.step.messages ?? "").toLowerCase();
  const model = (step.step.model ?? "").toLowerCase();
  const toolName = (step.step.name ?? "").toLowerCase();
  const maxTokens = Number(step.step.params?.max_tokens ?? 0);

  for (const rule of rules) {
    if (!rule.enabled || !rule.value.trim()) continue;
    const value = rule.value.trim().toLowerCase();
    const matched = (
      (rule.type === "tool_name" && toolName.includes(value))
      || (rule.type === "model_name" && model.includes(value))
      || (rule.type === "message_contains" && messages.includes(value))
      || (rule.type === "token_limit" && Number.isFinite(Number(value)) && maxTokens >= Number(value))
    );
    if (matched) return rule.label || `${rule.type}: ${rule.value}`;
  }
  return null;
}

/** Approve the current call and keep stepping through tools until the next LLM gate. */
export async function continueUntilNextLlm(sessionId: string): Promise<void> {
  _runUntilNextLlm = true;
  await postDecision(sessionId, { kind: "approve" });
}

/** Re-run one edited LLM step from a new branch at its restored cursor. */
export async function rerunEditedStep(
  sessionId: string,
  runnerRef: string,
  cursor: number,
  messages: unknown[],
  model: string,
  onBranchCreated?: (response: StartSessionResponse) => Promise<void>,
): Promise<string> {
  const source = useTimeTravelStore.getState().liveSession;
  closeSessionStream();
  if (source && !source.pausedStep?.restored && (source.status === "paused" || source.status === "running")) {
    await postDecision(sessionId, { kind: "stop" });
  }
  const res = await fetch(`/api/v1/sessions/${sessionId}/restart-from`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ branch_at: cursor, label: `Prompt version at step ${cursor + 1}` }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  const data = (await res.json()) as StartSessionResponse;
  _runUntilNextLlm = false;
  _pendingPromptEdit = { cursor, messages, model };
  try {
    await onBranchCreated?.(data);
  } catch (error) {
    _pendingPromptEdit = null;
    await stopDetachedSession(data.session_id);
    throw error;
  }
  useTimeTravelStore.getState().startLiveSession(data.session_id, data.trace_id, data.branch_id, runnerRef, true);
  void streamSessionDecisions(data.session_id);
  return data.session_id;
}

/** Continue a locally restored timeline from its saved cursor. */
export async function continueFromSavedState(
  sessionId: string,
  runnerRef: string,
  throughCursor: number,
): Promise<string> {
  const { startLiveSession, failSession } = useTimeTravelStore.getState();
  closeSessionStream();
  const res = await fetch(`/api/v1/sessions/${sessionId}/restart-from`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Include the restored completed span in the frozen prefix. The
    // successor starts live after it, so continuation never re-executes the
    // saved step through the interactive UI.
    body: JSON.stringify({ branch_at: throughCursor + 1, label: "Continue from saved step" }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const message = (err as { detail?: string }).detail ?? `HTTP ${res.status}`;
    failSession(message);
    throw new Error(message);
  }
  const data = (await res.json()) as StartSessionResponse;
  _runUntilNextLlm = false;
  _pendingPromptEdit = null;
  startLiveSession(data.session_id, data.trace_id, data.branch_id, runnerRef);
  void streamSessionDecisions(data.session_id);
  return data.session_id;
}

export function streamSessionDecisions(sessionId: string): void {
  const {
    pauseAtStep,
    markStepDispatching,
    appendStepReasoning,
    completeStep,
    completePromptVersion,
    addCheckpoint,
    resumeAfterStep,
    finishSession,
    failSession,
  } = useTimeTravelStore.getState();

  // Close any prior stream (defensive — shouldn't happen in normal flow).
  if (_activeStream) {
    _activeStream.close();
    _activeStream = null;
  }

  const es = new EventSource(`/api/v1/sessions/${sessionId}/stream`);
  _activeStream = es;

  es.onmessage = async (e: MessageEvent<string>) => {
    let evt: StepEvent;
    try {
      evt = JSON.parse(e.data) as StepEvent;
    } catch {
      // Malformed event — skip; the next well-formed one re-syncs the UI.
      return;
    }
    // A late event from a session that was just rewound must not overwrite
    // the successor session's state.
    if (useTimeTravelStore.getState().liveSession?.sessionId !== sessionId) return;
    switch (evt.type) {
      case "paused": {
        const session = useTimeTravelStore.getState().liveSession;
        const breakpointLabel = evt.pause_reason === "breakpoint"
          ? matchingBreakpoint(useTimeTravelStore.getState().breakpoints, evt) ?? "Server breakpoint"
          : matchingBreakpoint(useTimeTravelStore.getState().breakpoints, evt);
        const step: PausedStep = {
          cursor: evt.cursor,
          kind: evt.kind,
          payload: evt.step ?? {},
          pausedAt: Date.now(),
          completedAt: null,
          result: null,
          reasoning: null,
          usage: null,
          phase: "queued",
          breakpointLabel: breakpointLabel ?? undefined,
        };
        pauseAtStep(step);
        // The server is authoritative for persisted controls. This guard
        // must run before all local auto-approve paths below.
        if (evt.pause_reason) break;
        // LLM calls start immediately so their completed response can be
        // reviewed. Tool/MCP calls stay at the gate: their arguments may have
        // side effects, so the developer explicitly runs or edits them.
        if (breakpointLabel) {
          break;
        }
        if (_pendingPromptEdit && evt.cursor === _pendingPromptEdit.cursor) {
          const edit = _pendingPromptEdit;
          _pendingPromptEdit = null;
          void postDecision(sessionId, { kind: "edit", messages: edit.messages, model: edit.model }).catch(() => undefined);
        } else if (_runUntilNextLlm && (evt.kind === "tool" || evt.kind === "mcp")) {
          void postDecision(sessionId, { kind: "approve" }).catch(() => undefined);
        } else if (_runUntilNextLlm && evt.kind === "llm") {
          _runUntilNextLlm = false;
        } else if (session && evt.kind !== "tool" && evt.kind !== "mcp") {
          void postDecision(sessionId, { kind: "approve" }).catch(() => undefined);
        }
        break;
      }
      case "dispatching":
        markStepDispatching(evt.cursor);
        break;
      case "reasoning_delta":
        appendStepReasoning(evt.cursor, evt.chunk);
        break;
      case "step_completed": {
        // The model has returned; attach the result to the current paused
        // step so the UI can render it in the verify panel.
        completeStep(evt.cursor, evt.result, evt.usage);
        const pendingVersion = useTimeTravelStore.getState().liveSession?.promptVersions
          .slice().reverse().find((version) => version.cursor === evt.cursor && version.status === "running");
        const parsedResult = splitReasoning(evt.result);
        const assertionResult = pendingVersion?.assertions
          ? evaluateAssertions(pendingVersion.assertions, parsedResult.response, evt.usage, pendingVersion.pricing)
          : undefined;
        completePromptVersion(evt.cursor, evt.result, evt.usage, assertionResult);
        if (pendingVersion) {
          const completedVersion = useTimeTravelStore.getState().liveSession?.promptVersions.find((version) => version.id === pendingVersion.id);
          if (completedVersion) {
            if (completedVersion.evaluatorNames?.length) {
              await rerunRegisteredEvaluators(completedVersion);
            }
            const finalizedVersion = useTimeTravelStore.getState().liveSession?.promptVersions.find((version) => version.id === pendingVersion.id);
            if (finalizedVersion) await persistPromptVersionResult(finalizedVersion.id, finalizedVersion);
          }
        }
        // Control-only outputs are useful trace evidence but do not need a
        // human review gate. Preserve them in history and continue directly
        // to the next substantive agent step.
        if (_runUntilNextLlm && (evt.kind === "tool" || evt.kind === "mcp")) {
          void postDecision(sessionId, { kind: "approve" }).catch(() => undefined);
        } else if (shouldAutoContinue(evt.result)) {
          void postDecision(sessionId, { kind: "approve" }).catch(() => undefined);
        }
        break;
      }
      case "checkpoint":
        addCheckpoint({ name: evt.name, label: evt.label, cursor: evt.cursor, keys: evt.keys });
        break;
      case "resumed":
        resumeAfterStep(evt.decision);
        break;
      case "done":
        finishSession();
        void hydrateSessionResult(sessionId);
        es.close();
        _activeStream = null;
        break;
      case "errored":
        failSession(evt.message);
        es.close();
        _activeStream = null;
        break;
    }
  };

  // EventSource fires onerror on every reconnect attempt. We only treat it
  // as a hard failure if the session is already terminal — otherwise the
  // auto-reconnect is desirable (the server replays state on reconnect).
  es.onerror = () => {
    const { liveSession } = useTimeTravelStore.getState();
    if (liveSession && (liveSession.status === "done" || liveSession.status === "errored")) {
      es.close();
      _activeStream = null;
    }
  };
}

/** Stop a branch whose setup failed before it became the active UI session. */
async function stopDetachedSession(sessionId: string): Promise<void> {
  try {
    await fetch(`/api/v1/sessions/${sessionId}/decide`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "stop" }),
    });
  } catch {
    // The original persistence error is the actionable failure for the caller.
  }
}

async function rerunRegisteredEvaluators(version: PromptVersion): Promise<void> {
  const entries = await Promise.all((version.evaluatorNames ?? []).map(async (name) => {
    try {
      const response = await fetch("/api/v1/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, result: version.result ?? "", context: { cursor: version.cursor } }),
      });
      const body = await response.json().catch(() => ({})) as { passed?: boolean; detail?: string };
      return [name, { passed: response.ok && body.passed === true, detail: body.detail }] as const;
    } catch (error) {
      return [name, { passed: false, detail: error instanceof Error ? error.message : String(error) }] as const;
    }
  }));
  const evaluatorResults = Object.fromEntries(entries);
  useTimeTravelStore.getState().updatePromptVersion(version.id, { evaluatorResults });
}

function evaluateAssertions(
  assertions: OutputAssertions,
  output: string,
  usage?: StepUsage,
  pricing?: PricingProfile,
): AssertionResult {
  const failures: string[] = [];
  if (assertions.requireJson) {
    try { JSON.parse(output); } catch { failures.push("Response is not valid JSON."); }
  }
  for (const value of assertions.requiredText) if (!output.toLowerCase().includes(value.toLowerCase())) failures.push(`Missing required text: ${value}`);
  for (const value of assertions.forbiddenText) if (output.toLowerCase().includes(value.toLowerCase())) failures.push(`Contains forbidden text: ${value}`);
  if (assertions.requireCitations && !/\[[^\]]+\]/.test(output)) failures.push("No bracketed citation found.");
  if (assertions.maxTokens !== null && (usage?.total_tokens ?? 0) > assertions.maxTokens) failures.push(`Token limit exceeded: ${usage?.total_tokens ?? 0}/${assertions.maxTokens}.`);
  if (assertions.maxCostUsd !== null && pricing && usageCost(usage, pricing) > assertions.maxCostUsd) {
    failures.push(`Cost limit exceeded: $${usageCost(usage, pricing).toFixed(6)}/$${assertions.maxCostUsd.toFixed(6)}.`);
  }
  return { passed: failures.length === 0, failures, evaluatedAt: Date.now() };
}

function usageCost(usage: StepUsage | undefined, pricing: PricingProfile): number {
  if (!usage) return 0;
  const cachedInput = usage.cached_input_tokens ?? 0;
  const uncachedInput = Math.max(0, usage.input_tokens - cachedInput);
  return ((uncachedInput * pricing.inputPerMillion)
    + (cachedInput * pricing.cachedInputPerMillion)
    + (usage.final_tokens * pricing.outputPerMillion)
    + (usage.thinking_tokens * pricing.thinkingPerMillion)) / 1_000_000;
}

function shouldAutoContinue(rawResult: string): boolean {
  const { response } = splitReasoning(rawResult);
  return response.trim().toUpperCase() === "PROCEED";
}

/** Close the active SSE stream, if any. Called on unmount / view switch. */
export function closeSessionStream(): void {
  if (_activeStream) {
    _activeStream.close();
    _activeStream = null;
  }
}

/** Stop the paused runner before restoring a saved step for local inspection. */
export async function stopSessionForInspection(sessionId: string): Promise<void> {
  closeSessionStream();
  await postDecision(sessionId, { kind: "stop" });
}

/**
 * PATCH the server-owned run-control intent for the session.
 *
 * Persists the intent server-side so a page refresh or SSE reconnect does
 * not silently reset "pause after current" / "run until breakpoint". The
 * runner picks it up at the next gate without an extra round-trip.
 */
export async function patchRunControl(
  sessionId: string,
  intent: { pause_after_current: boolean; run_until_breakpoint: boolean; breakpoints?: BreakpointRule[] },
): Promise<void> {
  const res = await fetch(`/api/v1/sessions/${sessionId}/run-control`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(intent),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
}

/** Read and apply the server-owned run-control intent after a reconnect. */
export async function getRunControl(sessionId: string): Promise<RunControlIntent> {
  const res = await fetch(`/api/v1/sessions/${sessionId}/run-control`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  const intent = (await res.json()) as RunControlIntent;
  const rules: BreakpointRule[] = (intent.breakpoints ?? []).filter(
    (rule): rule is BreakpointRule => Boolean(
      rule && typeof rule === "object" &&
      typeof (rule as BreakpointRule).type === "string" &&
      typeof (rule as BreakpointRule).value === "string",
    ),
  );
  useTimeTravelStore.getState().setBreakpoints(rules);
  return { ...intent, breakpoints: rules };
}

async function hydrateRunControl(sessionId: string): Promise<void> {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      await getRunControl(sessionId);
      return;
    } catch {
      await new Promise((resolve) => window.setTimeout(resolve, 50 * (attempt + 1)));
    }
  }
}

async function hydrateSessionResult(sessionId: string): Promise<void> {
  try {
    const session = await api.getSession(sessionId);
    if (session.result_payload !== undefined) {
      useTimeTravelStore.getState().setSessionResult(session.result_payload);
    }
  } catch {
    // The completion event remains useful even if the optional result fetch is unavailable.
  }
}

/**
 * POST a Decision to resume the blocked agent. The SSE stream stays open —
 * this is a separate fetch on a separate connection.
 *
 * Returns the server's acknowledgement; the actual step advancement arrives
 * asynchronously via the stream (a "resumed" event followed by the next
 * "paused" or "done").
 */
export async function postDecision(
  sessionId: string,
  body: DecisionBody,
): Promise<void> {
  const { failSession } = useTimeTravelStore.getState();
  const res = await fetch(`/api/v1/sessions/${sessionId}/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const message = (err as { detail?: string }).detail ?? `HTTP ${res.status}`;
    // A 400 here is usually a validation error (bad decision shape) — surface
    // it via the session's error field so the StepPanel can show it inline
    // without tearing down the still-open stream.
    failSession(message);
    throw new Error(message);
  }
}
