/**
 * Client-side session state for the TimeTravel × Deep Research demo.
 *
 * The store owns:
 *   - traces:      every Trace the user has captured or branched, keyed by branchId.
 *   - rootBranchId: branchId of the original run (always "main" once captured).
 *   - selectedBranchId: which branch is shown in the timeline + span detail.
 *   - cursor:      index of the currently-selected span in the selected branch.
 *                  "Step Down" decrements this; "Step Up" increments it.
 *   - mode:        "inspect" (FROZEN — just reading the recording) or
 *                  "branch"  (editing the system prompt to fork a new branch).
 *   - draftSystemPrompt: the edited prompt being staged before running a branch.
 *   - isRunning:   true while an API call is in flight.
 *   - diff:        optional side-by-side diff between two branches.
 */

"use client";

import { create } from "zustand";
import type {
  BranchDiff,
  LiveRun,
  LiveSession,
  PausedStep,
  LiveCheckpoint,
  StepHistoryEntry,
  PromptVersion,
  StepUsage,
  OutputAssertions,
  AssertionResult,
  SavedSessionCase,
  BreakpointRule,
  SpanKind,
  Trace,
} from "./types";
import { diffBranches } from "./diff";

type StepReview = {
  reviewNote?: string;
  reviewVerdict?: "accepted" | "rejected" | null;
  assertions?: OutputAssertions;
  assertionResult?: AssertionResult;
};

type StepAssertions = {
  assertions?: OutputAssertions;
  assertionResult?: AssertionResult;
};

const STEP_REVIEWS_STORAGE_KEY = "timetravel-step-reviews";
const STEP_ASSERTIONS_STORAGE_KEY = "timetravel-step-assertions";
const BREAKPOINTS_STORAGE_KEY = "timetravel-breakpoints";
const ACTIVE_SESSION_STORAGE_KEY = "timetravel-active-session";

function clearActiveSession(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
  } catch {
    // Local persistence is optional; clearing the in-memory session still works.
  }
}

function savedStepReview(traceId: string, cursor: number): StepReview {
  if (typeof window === "undefined") return {};
  try {
    const reviews = JSON.parse(window.localStorage.getItem(STEP_REVIEWS_STORAGE_KEY) ?? "{}") as Record<string, StepReview>;
    return reviews[`${traceId}:${cursor}`] ?? {};
  } catch {
    return {};
  }
}

function persistStepReview(traceId: string, cursor: number, review: StepReview): void {
  if (typeof window === "undefined") return;
  try {
    const reviews = JSON.parse(window.localStorage.getItem(STEP_REVIEWS_STORAGE_KEY) ?? "{}") as Record<string, StepReview>;
    reviews[`${traceId}:${cursor}`] = review;
    window.localStorage.setItem(STEP_REVIEWS_STORAGE_KEY, JSON.stringify(reviews));
  } catch {
    // Local persistence is optional; the active session still retains reviews.
  }
}

function savedStepAssertions(traceId: string, cursor: number): StepAssertions {
  if (typeof window === "undefined") return {};
  try {
    const records = JSON.parse(window.localStorage.getItem(STEP_ASSERTIONS_STORAGE_KEY) ?? "{}") as Record<string, StepAssertions>;
    return records[`${traceId}:${cursor}`] ?? {};
  } catch {
    return {};
  }
}

function persistStepAssertions(traceId: string, cursor: number, record: StepAssertions): void {
  if (typeof window === "undefined") return;
  try {
    const records = JSON.parse(window.localStorage.getItem(STEP_ASSERTIONS_STORAGE_KEY) ?? "{}") as Record<string, StepAssertions>;
    records[`${traceId}:${cursor}`] = record;
    window.localStorage.setItem(STEP_ASSERTIONS_STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Persistence is best-effort; checks remain attached in memory.
  }
}

function readBreakpoints(): BreakpointRule[] {
  if (typeof window === "undefined") return [];
  try {
    const rules = JSON.parse(window.localStorage.getItem(BREAKPOINTS_STORAGE_KEY) ?? "[]") as unknown;
    return Array.isArray(rules) ? rules.filter((rule): rule is BreakpointRule => Boolean(rule && typeof rule === "object" && "id" in rule && "type" in rule && "value" in rule)) : [];
  } catch {
    return [];
  }
}

function persistBreakpoints(rules: BreakpointRule[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(BREAKPOINTS_STORAGE_KEY, JSON.stringify(rules));
}

export type UIMode = "inspect" | "branch";

/**
 * Top-level view discriminator (separate from UIMode). The "demo" view is the
 * bundled-agent trace/branch experience; "session" is the Phase 9 interactive
 * stepping view driven by the Python stepping server.
 */
export type UIView = "demo" | "session";

interface TimeTravelState {
  traces: Record<string, Trace>;
  rootBranchId: string | null;
  selectedBranchId: string | null;
  cursor: number;
  mode: UIMode;
  draftSystemPrompt: string;
  draftLabel: string;
  draftNote: string;
  isRunning: boolean;
  runError: string | null;
  diff: BranchDiff | null;
  diffLeftBranchId: string | null;
  diffRightBranchId: string | null;
  lastEvent: string | null;
  /**
   * A run in progress, shown in the ThinkingPanel. null when no run is
   * streaming. Set by startLiveRun, mutated by the delta actions as
   * StreamEvents arrive, and cleared by finishLiveRun once the Trace commits.
   */
  liveRun: LiveRun | null;

  /**
   * Top-level view: "demo" (bundled agent) or "session" (stepping server).
   * Defaults to "demo" so the existing experience is unchanged.
   */
  uiView: UIView;
  /**
   * A stepping session in progress, parallel to liveRun. null when no session
   * is active. Mutated by the session-client as SSE events arrive.
   */
  liveSession: LiveSession | null;
  breakpoints: BreakpointRule[];

  // actions
  setRunning: (v: boolean) => void;
  setRunError: (e: string | null) => void;
  addTrace: (t: Trace) => void;
  selectBranch: (branchId: string) => void;
  setCursor: (i: number) => void;
  stepDown: () => void;
  stepUp: () => void;
  enterBranchMode: () => void;
  exitBranchMode: () => void;
  setDraftSystemPrompt: (s: string) => void;
  setDraftLabel: (s: string) => void;
  setDraftNote: (s: string) => void;
  setDiff: (
    leftBranchId: string | null,
    rightBranchId: string | null,
  ) => void;
  reset: () => void;
  setLastEvent: (s: string | null) => void;

  // live-run actions
  startLiveRun: (query: string, kind: "run" | "branch") => void;
  beginSpan: (index: number, name: string, kind: SpanKind) => void;
  appendReasoning: (index: number, chunk: string) => void;
  appendOutput: (index: number, chunk: string) => void;
  finishSpan: (index: number) => void;
  failLiveRun: (message: string) => void;
  /** Commit the finished trace and dismiss the live view. */
  finishLiveRun: (trace: Trace) => void;
  clearLiveRun: () => void;

  // session (stepping) actions — additive, do not touch liveRun/traces/mode
  setUIView: (v: UIView) => void;
  startLiveSession: (
    sessionId: string,
    traceId: string,
    branchId: string,
    runnerRef: string,
    preservePromptVersions?: boolean,
    metadata?: { agentRef?: string; inputPayload?: unknown; resultPayload?: unknown },
  ) => void;
  setSessionResult: (result: unknown) => void;
  pauseAtStep: (step: PausedStep) => void;
  markStepDispatching: (cursor: number) => void;
  appendStepReasoning: (cursor: number, chunk: string) => void;
  /** Attach the model's response text to the current paused step (verify loop). */
  completeStep: (cursor: number, result: string, usage?: StepUsage) => void;
  addCheckpoint: (checkpoint: LiveCheckpoint) => void;
  addPromptVersion: (version: PromptVersion) => void;
  updatePromptVersion: (versionId: string, update: Partial<PromptVersion>) => void;
  completePromptVersion: (cursor: number, result: string, usage?: StepUsage, assertionResult?: AssertionResult) => void;
  setStepReview: (note: string, verdict: "accepted" | "rejected") => void;
  setStepAssertions: (assertions: OutputAssertions, result: AssertionResult) => void;
  hydratePromptVersions: (versions: PromptVersion[]) => void;
  hydrateStepReviews: (reviews: Array<{ cursor: number; note?: string; verdict?: "accepted" | "rejected" | null; assertions?: OutputAssertions; assertionResult?: AssertionResult }>) => void;
  /** Restore the immediately previous captured step without invoking the agent. */
  restorePreviousStep: () => boolean;
  /** Restore the next captured step from the local timetravel stack. */
  restoreNextStep: () => boolean;
  resumeAfterStep: (decision: string) => void;
  finishSession: () => void;
  failSession: (message: string) => void;
  clearLiveSession: () => void;
  openSavedSessionCase: (record: SavedSessionCase) => void;
  addBreakpoint: (rule: Omit<BreakpointRule, "id">) => void;
  removeBreakpoint: (id: string) => void;
  setBreakpoints: (rules: BreakpointRule[]) => void;
}

export const useTimeTravelStore = create<TimeTravelState>((set, get) => ({
  traces: {},
  rootBranchId: null,
  selectedBranchId: null,
  cursor: 0,
  mode: "inspect",
  draftSystemPrompt: "",
  draftLabel: "",
  draftNote: "",
  isRunning: false,
  runError: null,
  diff: null,
  diffLeftBranchId: null,
  diffRightBranchId: null,
  lastEvent: null,
  liveRun: null,
  uiView: "demo",
  liveSession: null,
  breakpoints: readBreakpoints(),

  setRunning: (v) => set({ isRunning: v }),
  setRunError: (e) => set({ runError: e }),
  addTrace: (t) =>
    set((s) => {
      const traces = { ...s.traces, [t.branchId]: t };
      const rootBranchId = s.rootBranchId ?? t.branchId;
      return {
        traces,
        rootBranchId,
        selectedBranchId: t.branchId,
        cursor: 0,
        mode: "inspect",
        draftSystemPrompt: "",
        draftLabel: "",
        draftNote: "",
        runError: null,
        diff: null,
        diffLeftBranchId: null,
        diffRightBranchId: null,
        lastEvent: t.parentBranchId
          ? `Branch “${t.label}” captured — ${t.spans.filter((x) => x.source === "cached").length} spans served from cache, ${t.spans.filter((x) => x.source === "live").length} live LLM calls.`
          : `Original trace captured — ${t.spans.length} live LLM spans.`,
      };
    }),

  selectBranch: (branchId) =>
    set((s) => {
      const t = s.traces[branchId];
      if (!t) return s;
      const clamped = Math.min(s.cursor, t.spans.length - 1);
      return {
        selectedBranchId: branchId,
        cursor: Math.max(0, clamped),
        mode: "inspect",
        draftSystemPrompt: "",
        draftLabel: "",
        draftNote: "",
      };
    }),

  setCursor: (i) => set({ cursor: i }),

  stepDown: () =>
    set((s) => ({
      cursor: Math.max(0, s.cursor - 1),
      mode: "inspect",
      lastEvent: `Stepped down to span #${Math.max(0, s.cursor - 1) + 1}.`,
    })),

  stepUp: () => {
    const s = get();
    const t = s.traces[s.selectedBranchId!];
    if (!t) return;
    const max = t.spans.length - 1;
    set({
      cursor: Math.min(max, s.cursor + 1),
      mode: "inspect",
      lastEvent: `Stepped up to span #${Math.min(max, s.cursor + 1) + 1}.`,
    });
  },

  enterBranchMode: () =>
    set((s) => {
      const t = s.traces[s.selectedBranchId!];
      if (!t) return s;
      const span = t.spans[s.cursor];
      return {
        mode: "branch",
        draftSystemPrompt: span.systemPrompt,
        draftLabel: `Branch @ #${s.cursor + 1}`,
        draftNote: "",
      };
    }),

  exitBranchMode: () =>
    set({
      mode: "inspect",
      draftSystemPrompt: "",
      draftLabel: "",
      draftNote: "",
    }),

  setDraftSystemPrompt: (s2) => set({ draftSystemPrompt: s2 }),
  setDraftLabel: (s2) => set({ draftLabel: s2 }),
  setDraftNote: (s2) => set({ draftNote: s2 }),

  setDiff: (leftBranchId, rightBranchId) =>
    set((s) => {
      if (!leftBranchId || !rightBranchId) {
        return {
          diff: null,
          diffLeftBranchId: null,
          diffRightBranchId: null,
        };
      }
      const left = s.traces[leftBranchId];
      const right = s.traces[rightBranchId];
      if (!left || !right) {
        return {
          diff: null,
          diffLeftBranchId: null,
          diffRightBranchId: null,
        };
      }
      const diff = diffBranches(left, right);
      return {
        diff,
        diffLeftBranchId: leftBranchId,
        diffRightBranchId: rightBranchId,
      };
    }),

  reset: () =>
    set({
      traces: {},
      rootBranchId: null,
      selectedBranchId: null,
      cursor: 0,
      mode: "inspect",
      draftSystemPrompt: "",
      draftLabel: "",
      draftNote: "",
      isRunning: false,
      runError: null,
      diff: null,
      diffLeftBranchId: null,
      diffRightBranchId: null,
      lastEvent: "Session reset.",
      liveRun: null,
    }),

  setLastEvent: (s2) => set({ lastEvent: s2 }),

  // --- live-run actions -----------------------------------------------------
  // These rebuild the liveRun object each call so Zustand sees a new ref and
  // the ThinkingPanel re-renders. Deltas arrive frequently but each is a small
  // string append, so the cost is negligible for an 8-span demo.
  startLiveRun: (query, kind) =>
    set({
      isRunning: true,
      runError: null,
      liveRun: {
        query,
        kind,
        spans: [],
        currentIndex: null,
        status: "running",
        error: null,
        startedAt: Date.now(),
      },
    }),

  beginSpan: (index, name, kind) =>
    set((s) => {
      if (!s.liveRun) return s;
      const spans = [...s.liveRun.spans];
      spans[index] = {
        index,
        name,
        kind,
        reasoning: "",
        output: "",
        status: "thinking",
        startedAt: Date.now(),
        endedAt: null,
      };
      return { liveRun: { ...s.liveRun, spans, currentIndex: index } };
    }),

  appendReasoning: (index, chunk) =>
    set((s) => {
      if (!s.liveRun || !s.liveRun.spans[index]) return s;
      const span = s.liveRun.spans[index];
      const spans = [...s.liveRun.spans];
      spans[index] = { ...span, reasoning: span.reasoning + chunk };
      return { liveRun: { ...s.liveRun, spans } };
    }),

  appendOutput: (index, chunk) =>
    set((s) => {
      if (!s.liveRun || !s.liveRun.spans[index]) return s;
      const span = s.liveRun.spans[index];
      const spans = [...s.liveRun.spans];
      spans[index] = {
        ...span,
        output: span.output + chunk,
        status: "answering",
      };
      return { liveRun: { ...s.liveRun, spans } };
    }),

  finishSpan: (index) =>
    set((s) => {
      if (!s.liveRun || !s.liveRun.spans[index]) return s;
      const span = s.liveRun.spans[index];
      const spans = [...s.liveRun.spans];
      spans[index] = { ...span, status: "done", endedAt: Date.now() };
      return { liveRun: { ...s.liveRun, spans } };
    }),

  failLiveRun: (message) =>
    set((s) => ({
      isRunning: false,
      runError: message,
      liveRun: s.liveRun
        ? { ...s.liveRun, status: "error", error: message }
        : null,
    })),

  finishLiveRun: (trace) =>
    set((s) => {
      const traces = { ...s.traces, [trace.branchId]: trace };
      const rootBranchId = s.rootBranchId ?? trace.branchId;
      return {
        traces,
        rootBranchId,
        selectedBranchId: trace.branchId,
        cursor: 0,
        mode: "inspect",
        draftSystemPrompt: "",
        draftLabel: "",
        draftNote: "",
        runError: null,
        diff: null,
        diffLeftBranchId: null,
        diffRightBranchId: null,
        isRunning: false,
        liveRun: null,
        lastEvent: trace.parentBranchId
          ? `Branch “${trace.label}” captured — ${trace.spans.filter((x) => x.source === "cached").length} spans served from cache, ${trace.spans.filter((x) => x.source === "live").length} live LLM calls.`
          : `Original trace captured — ${trace.spans.length} live LLM spans.`,
      };
    }),

  clearLiveRun: () => set({ liveRun: null, isRunning: false }),

  // --- session (stepping) actions ----------------------------------------
  // All additive: none of these touch liveRun, traces, mode, or cursor.
  // They mutate only liveSession, the parallel state object for the Phase 9
  // interactive stepping view. session-client.ts drives these as SSE events
  // arrive from the Python stepping server.
  setUIView: (v) => set({ uiView: v }),

  startLiveSession: (sessionId, traceId, branchId, runnerRef, preservePromptVersions = false, metadata) =>
    set((state) => ({
      liveSession: {
        sessionId,
        traceId,
        branchId,
        runnerRef,
        ...metadata,
        status: "running",
        error: null,
        pausedStep: null,
        history: [],
        savedFuture: [],
        promptVersions: preservePromptVersions ? state.liveSession?.promptVersions ?? [] : [],
        checkpoints: [],
        startedAt: Date.now(),
      },
    })),

  setSessionResult: (result) => set((s) => s.liveSession ? {
    liveSession: { ...s.liveSession, resultPayload: result },
  } : s),

  pauseAtStep: (step) =>
    set((s) => {
      if (!s.liveSession) return s;
      const review = savedStepReview(s.liveSession.traceId, step.cursor);
      const assertions = savedStepAssertions(s.liveSession.traceId, step.cursor);
      return {
        liveSession: {
          ...s.liveSession,
          status: "paused",
          pausedStep: { ...step, ...review, ...assertions },
        },
      };
    }),

  markStepDispatching: (cursor) =>
    set((s) => {
      if (!s.liveSession?.pausedStep) return s;
      if (s.liveSession.pausedStep.cursor !== cursor) return s;
      return {
        liveSession: {
          ...s.liveSession,
          status: "running",
          pausedStep: { ...s.liveSession.pausedStep, phase: "running" },
        },
      };
    }),

  appendStepReasoning: (cursor, chunk) =>
    set((s) => {
      const step = s.liveSession?.pausedStep;
      if (!step || step.cursor !== cursor) return s;
      if (step.phase === "completed") return s;
      return {
        liveSession: {
          ...s.liveSession!,
          pausedStep: { ...step, reasoning: (step.reasoning ?? "") + chunk },
        },
      };
    }),

  completeStep: (cursor, result, usage) =>
    set((s) => {
      if (!s.liveSession || !s.liveSession.pausedStep) return s;
      if (s.liveSession.pausedStep.cursor !== cursor) return s;
      const { reasoning, response } = splitReasoning(result);
      return {
        liveSession: {
          ...s.liveSession,
          status: "paused",
          pausedStep: {
            ...s.liveSession.pausedStep,
            result: response,
            reasoning: reasoning ?? s.liveSession.pausedStep.reasoning,
            usage: usage ?? null,
            phase: "completed",
            completedAt: Date.now(),
          },
        },
      };
    }),

  addCheckpoint: (checkpoint) =>
    set((s) => {
      if (!s.liveSession) return s;
      if (s.liveSession.checkpoints.some((item) => item.name === checkpoint.name)) return s;
      return { liveSession: { ...s.liveSession, checkpoints: [...s.liveSession.checkpoints, checkpoint] } };
    }),

  addPromptVersion: (version) => set((s) => s.liveSession ? {
    liveSession: { ...s.liveSession, promptVersions: [...s.liveSession.promptVersions, version] },
  } : s),

  updatePromptVersion: (versionId, update) => set((s) => s.liveSession ? {
    liveSession: {
      ...s.liveSession,
      promptVersions: s.liveSession.promptVersions.map((version) => (
        version.id === versionId ? { ...version, ...update } : version
      )),
    },
  } : s),

  completePromptVersion: (cursor, result, usage, assertionResult) => set((s) => {
    if (!s.liveSession) return s;
    const index = [...s.liveSession.promptVersions].map((item) => item.cursor).lastIndexOf(cursor);
    if (index < 0) return s;
    const parsed = splitReasoning(result);
    return {
      liveSession: {
        ...s.liveSession,
        promptVersions: s.liveSession.promptVersions.map((item, itemIndex) => itemIndex === index
          ? { ...item, status: "completed", result: parsed.response, reasoning: parsed.reasoning ?? item.reasoning, usage: usage ?? null, assertionResult: assertionResult ?? item.assertionResult, completedAt: Date.now() }
          : item),
      },
    };
  }),

  setStepReview: (reviewNote, reviewVerdict) => set((s) => {
    if (!s.liveSession?.pausedStep) return s;
    const pausedStep = { ...s.liveSession.pausedStep, reviewNote, reviewVerdict };
    persistStepReview(s.liveSession.traceId, pausedStep.cursor, { reviewNote, reviewVerdict });
    return { liveSession: { ...s.liveSession, pausedStep } };
  }),

  setStepAssertions: (assertions, assertionResult) => set((s) => {
    if (!s.liveSession?.pausedStep) return s;
    const pausedStep = { ...s.liveSession.pausedStep, assertions, assertionResult };
    persistStepAssertions(s.liveSession.traceId, pausedStep.cursor, { assertions, assertionResult });
    return { liveSession: { ...s.liveSession, pausedStep } };
  }),

  hydratePromptVersions: (versions) => set((s) => {
    if (!s.liveSession) return s;
    const byId = new Map(s.liveSession.promptVersions.map((version) => [version.id, version]));
    for (const version of versions) {
      byId.set(version.id, { ...byId.get(version.id), ...version });
    }
    return {
      liveSession: {
        ...s.liveSession,
        promptVersions: [...byId.values()].sort((a, b) => a.createdAt - b.createdAt),
      },
    };
  }),

  hydrateStepReviews: (reviews) => set((s) => {
    if (!s.liveSession) return s;
    const reviewByCursor = new Map(reviews.map((review) => [review.cursor, review]));
    const apply = <T extends { cursor: number; reviewNote?: string; reviewVerdict?: "accepted" | "rejected" | null; assertions?: OutputAssertions; assertionResult?: AssertionResult }>(step: T): T => {
      const review = reviewByCursor.get(step.cursor);
      return review ? { ...step, reviewNote: review.note, reviewVerdict: review.verdict, assertions: review.assertions, assertionResult: review.assertionResult } : step;
    };
    const pausedStep = s.liveSession.pausedStep ? apply(s.liveSession.pausedStep) : null;
    return {
      liveSession: {
        ...s.liveSession,
        pausedStep,
        history: s.liveSession.history.map(apply),
      },
    };
  }),

  restorePreviousStep: () => {
    const session = get().liveSession;
    const current = session?.pausedStep;
    const previous = session?.history.at(-1);
    if (!session || !current || !previous) return false;

    const restored = historyEntryToSavedStep(previous);
    set({
      liveSession: {
        ...session,
        status: "paused",
        pausedStep: restored,
        history: session.history.slice(0, -1),
        savedFuture: [pausedStepToHistoryEntry(current), ...session.savedFuture],
      },
    });
    return true;
  },

  restoreNextStep: () => {
    const session = get().liveSession;
    const current = session?.pausedStep;
    const next = session?.savedFuture[0];
    if (!session || !current || !next) return false;
    set({
      liveSession: {
        ...session,
        status: "paused",
        pausedStep: historyEntryToSavedStep(next),
        history: [...session.history, pausedStepToHistoryEntry(current)],
        savedFuture: session.savedFuture.slice(1),
      },
    });
    return true;
  },

  resumeAfterStep: (decision) =>
    set((s) => {
      if (!s.liveSession || !s.liveSession.pausedStep) return s;
      const paused = s.liveSession.pausedStep;
      const entry = {
        cursor: paused.cursor,
        kind: paused.kind,
        decision,
        payload: paused.payload,
        result: paused.result,
        reasoning: paused.reasoning,
        usage: paused.usage,
        latencyMs: paused.completedAt === null ? 0 : Math.max(0, paused.completedAt - paused.pausedAt),
        reviewNote: paused.reviewNote,
        reviewVerdict: paused.reviewVerdict,
        assertions: paused.assertions,
        assertionResult: paused.assertionResult,
        resolvedAt: Date.now(),
      };
      return {
        liveSession: {
          ...s.liveSession,
          status: "running",
          pausedStep: null,
          history: [...s.liveSession.history, entry],
        },
      };
    }),

  finishSession: () =>
    set((s) => ({
      liveSession: s.liveSession
        ? { ...s.liveSession, status: "done", pausedStep: null }
        : null,
    })),

  failSession: (message) =>
    set((s) => ({
      liveSession: s.liveSession
        ? { ...s.liveSession, status: "errored", error: message, pausedStep: null }
        : null,
    })),

  clearLiveSession: () => {
    clearActiveSession();
    set({ liveSession: null });
  },

  openSavedSessionCase: (record) => set({
    uiView: "session",
    liveSession: {
      sessionId: `saved-${record.id}`,
      traceId: record.traceId,
      branchId: "saved",
      runnerRef: record.runnerRef,
      status: "done",
      error: null,
      pausedStep: null,
      history: record.steps,
      savedFuture: [],
      promptVersions: record.promptVersions,
      checkpoints: record.checkpoints ?? [],
      startedAt: new Date(record.createdAt).getTime(),
    },
  }),

  addBreakpoint: (rule) => set((state) => {
    const breakpoints = [...state.breakpoints, { ...rule, id: `breakpoint-${Date.now()}-${Math.random().toString(36).slice(2, 7)}` }];
    persistBreakpoints(breakpoints);
    return { breakpoints };
  }),

  removeBreakpoint: (id) => set((state) => {
    const breakpoints = state.breakpoints.filter((rule) => rule.id !== id);
    persistBreakpoints(breakpoints);
    return { breakpoints };
  }),

  setBreakpoints: (rules) => {
    persistBreakpoints(rules);
    set({ breakpoints: rules });
  },
}));

export function splitReasoning(
  result: string,
): { reasoning: string | null; response: string } {
  const match = result.match(/<think>([\s\S]*?)<\/think>\s*/i);
  if (!match) return { reasoning: null, response: result.trim() };
  return {
    reasoning: match[1].trim() || null,
    response: result.replace(match[0], "").trim(),
  };
}

function historyEntryToSavedStep(entry: StepHistoryEntry): PausedStep {
  return {
    cursor: entry.cursor,
    kind: entry.kind,
    payload: entry.payload ?? {},
    pausedAt: entry.resolvedAt,
    completedAt: entry.resolvedAt,
    result: entry.result ?? null,
    reasoning: entry.reasoning ?? null,
    usage: entry.usage ?? null,
    phase: "completed",
    restored: true,
  };
}

function pausedStepToHistoryEntry(step: PausedStep): StepHistoryEntry {
  return {
    cursor: step.cursor,
    kind: step.kind,
    decision: "saved",
    payload: step.payload,
    result: step.result,
    reasoning: step.reasoning,
    usage: step.usage,
    resolvedAt: step.completedAt ?? Date.now(),
  };
}
