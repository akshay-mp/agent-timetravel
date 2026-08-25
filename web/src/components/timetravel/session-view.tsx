"use client";

/**
 * SessionView — the interactive stepping layout.
 *
 * A resizable two-panel layout mirroring page.tsx's inspect view:
 *   - Left (28%): a step-history rail — one node + card per consumed step,
 *     with the currently-paused step highlighted.
 *   - Right (72%): the StepPanel when paused, or a status state (waiting /
 *     done / errored) otherwise.
 *
 * A progress header strip across the top shows status icon + step count +
 * live elapsed timer + a Progress bar — adapted from thinking-panel.tsx.
 */

import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ResizablePanel,
  ResizableHandle,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2,
  PauseCircle,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Bot,
  ListTree,
  Pencil,
  Wrench,
} from "lucide-react";
import { splitReasoning, useTimeTravelStore } from "@/lib/timetravel/store";
import {
  loadPricingProfiles,
  loadRegressionCases,
  persistPricingProfile,
  persistRegressionCase,
  restartSessionFrom,
  runPersistedRegressionCase,
} from "@/lib/timetravel/session-client";
import { StepPanel } from "./step-panel";
import type { BreakpointRule, LiveSession, PausedStep, PricingProfile, PromptVersion, SavedSessionCase, StepHistoryEntry, StepUsage } from "@/lib/timetravel/types";
import { wordDiff } from "@/lib/timetravel/diff";

const SAVED_SESSIONS_STORAGE_KEY = "timetravel-regression-cases";
const PRICING_PROFILE_STORAGE_KEY = "timetravel-pricing-profile";
const MAX_COMPARISON_VARIANTS = 10;
const DEFAULT_PRICING: PricingProfile = {
  name: "Local model",
  inputPerMillion: 0,
  cachedInputPerMillion: 0,
  outputPerMillion: 0,
  thinkingPerMillion: 0,
};

function readPricingProfile(): PricingProfile {
  if (typeof window === "undefined") return DEFAULT_PRICING;
  try {
    const stored = JSON.parse(window.localStorage.getItem(PRICING_PROFILE_STORAGE_KEY) ?? "{}") as Partial<PricingProfile>;
    return {
      ...DEFAULT_PRICING,
      ...stored,
      inputPerMillion: Math.max(0, Number(stored.inputPerMillion) || 0),
      cachedInputPerMillion: Math.max(0, Number(stored.cachedInputPerMillion) || 0),
      outputPerMillion: Math.max(0, Number(stored.outputPerMillion) || 0),
      thinkingPerMillion: Math.max(0, Number(stored.thinkingPerMillion) || 0),
    };
  } catch {
    return DEFAULT_PRICING;
  }
}

function kindTint(kind: string): string {
  switch (kind) {
    case "llm": return "bg-sky-500";
    case "tool": return "bg-emerald-500";
    case "mcp": return "bg-violet-500";
    default: return "bg-muted-foreground";
  }
}

function displayAgentResult(payload: unknown): string {
  if (typeof payload !== "string") return JSON.stringify(payload, null, 2) ?? "";
  const parsed = splitReasoning(payload);
  if (parsed.reasoning !== null) return parsed.response || "(No final response returned)";
  return parsed.response || payload;
}

function decisionBadge(decision: string, reviewVerdict?: "accepted" | "rejected" | null): { label: string; className: string } {
  if (reviewVerdict === "accepted") return { label: "accepted", className: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" };
  if (reviewVerdict === "rejected") return { label: "rejected", className: "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300" };
  switch (decision) {
    case "approve": return { label: "approved", className: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" };
    case "edit": return { label: "edited", className: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300" };
    case "stop": return { label: "stopped", className: "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300" };
    case "step_once": return { label: "stepped", className: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300" };
    case "reject": return { label: "rejected by dev", className: "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300" };
    case "skip": return { label: "skipped", className: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300" };
    case "run_until_breakpoint": return { label: "ran to bp", className: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300" };
    default: return { label: decision, className: "bg-muted text-muted-foreground" };
  }
}

function completedSiblingGroups(versions: PromptVersion[]): Array<{ cursor: number; versions: PromptVersion[] }> {
  const groups = new Map<number, PromptVersion[]>();
  for (const version of versions) {
    if (version.status !== "completed") continue;
    const group = groups.get(version.cursor) ?? [];
    group.push(version);
    groups.set(version.cursor, group);
  }
  return [...groups.entries()]
    .filter(([, group]) => group.length >= 2)
    .sort(([left], [right]) => left - right)
    .map(([cursor, group]) => ({ cursor, versions: group }));
}


export function SessionView() {
  const { liveSession, clearLiveSession, breakpoints, addBreakpoint, removeBreakpoint } = useTimeTravelStore();
  const [pricing, setPricing] = useState<PricingProfile>(readPricingProfile);
  const [pricingHydrated, setPricingHydrated] = useState(false);
  const [pricingDirty, setPricingDirty] = useState(false);
  const pricingDirtyRef = useRef(false);
  const initialPricingRef = useRef(pricing);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number | null>(null);
  const [comparisonIds, setComparisonIds] = useState<string[] | null>(null);
  const [comparisonCursor, setComparisonCursor] = useState<number | null>(null);
  const [matrixCursor, setMatrixCursor] = useState<number | null>(null);
  const [regressionSaved, setRegressionSaved] = useState(false);
  const [checkpointRestarting, setCheckpointRestarting] = useState<string | null>(null);
  const [checkpointPayloads, setCheckpointPayloads] = useState<Record<string, Record<string, unknown>>>({});
  const [checkpointExpanded, setCheckpointExpanded] = useState<Record<string, boolean>>({});
  const [breakpointType, setBreakpointType] = useState<BreakpointRule["type"]>("tool_name");
  const [breakpointValue, setBreakpointValue] = useState("");

  useEffect(() => {
    if (!pricingHydrated || !pricingDirty) return;
    window.localStorage.setItem(PRICING_PROFILE_STORAGE_KEY, JSON.stringify(pricing));
    void persistPricingProfile(pricing).catch(() => {
      // Keep the local cache as the offline fallback.
    });
  }, [pricing, pricingDirty, pricingHydrated]);

  useEffect(() => {
    void loadPricingProfiles()
      .then((profiles) => {
        const hasLocalRates = Object.values(initialPricingRef.current).some(
          (rate) => typeof rate === "number" && rate > 0,
        );
        const serverProfile = profiles.find((profile) => Object.values(profile).some(
          (rate) => typeof rate === "number" && rate > 0,
        ));
        // Do not replace durable local rates with a stale zero-rate default
        // profile left by an earlier first-mount write.
        if (!pricingDirtyRef.current && profiles[0] && (!hasLocalRates || serverProfile)) {
          setPricing(serverProfile ?? profiles[0]);
        }
        setPricingHydrated(true);
      })
      .catch(() => {
        // The local profile is used until the backend becomes available.
        setPricingHydrated(true);
      });
  }, []);

  if (!liveSession) {
    return <SavedSessionLibrary />;
  }

  const stepCount = liveSession.history.length + (liveSession.pausedStep ? 1 : 0);
  const isPaused = liveSession.status === "paused";
  const isDone = liveSession.status === "done";
  const isErrored = liveSession.status === "errored";
  const usage = sessionUsage(liveSession.history, liveSession.pausedStep?.usage);
  const estimatedCost = usageCost(usage, pricing);
  const reviewedSteps = [...liveSession.history, ...(liveSession.pausedStep ? [liveSession.pausedStep] : [])];
  const acceptedCount = reviewedSteps.filter((step) => step.reviewVerdict === "accepted").length;
  const rejectedCount = reviewedSteps.filter((step) => step.reviewVerdict === "rejected").length;
  const latency = sessionLatency(liveSession.history);
  const saveRegressionCase = async () => {
    const record: SavedSessionCase = {
      id: `timetravel-regression-${Date.now()}`,
      createdAt: new Date().toISOString(),
      traceId: liveSession.traceId,
      runnerRef: liveSession.runnerRef,
      steps: liveSession.history,
      promptVersions: liveSession.promptVersions,
      checkpoints: liveSession.checkpoints,
      pricing,
      summary: { accepted: acceptedCount, rejected: rejectedCount, totalTokens: usage.total_tokens, totalLatencyMs: latency.totalMs },
    };
    const existing = readSavedSessions();
    localStorage.setItem("timetravel-regression-cases", JSON.stringify([...existing, record]));
    try {
      await persistRegressionCase(record);
    } catch {
      // Local storage remains the offline fallback; backend sync can be retried later.
    }
    setRegressionSaved(true);
  };
  const restartFromCheckpoint = async (checkpoint: { name: string; label: string; cursor: number }) => {
    setCheckpointRestarting(checkpoint.name);
    try {
      await restartSessionFrom(
        liveSession.sessionId,
        checkpoint.cursor,
        `Restart from ${checkpoint.label}`,
      );
      setSelectedHistoryIndex(null);
    } finally {
      setCheckpointRestarting(null);
    }
  };
  const fetchCheckpointPayload = async (branchId: string, name: string): Promise<void> => {
    if (checkpointPayloads[name]) return; // already cached
    try {
      const res = await fetch(`/api/v1/branches/${branchId}/checkpoints/${encodeURIComponent(name)}`);
      if (res.ok) {
        const data = (await res.json()) as { payload?: Record<string, unknown> };
        setCheckpointPayloads((prev) => ({ ...prev, [name]: data.payload ?? {} }));
      }
    } catch {
      // silently ignore — keys are still shown from the SSE event
    }
  };
  const createBreakpoint = () => {
    const value = breakpointValue.trim();
    if (!value) return;
    addBreakpoint({
      type: breakpointType,
      value,
      label: `${breakpointType.replace(/_/g, " ")}: ${value}`,
      enabled: true,
    });
    setBreakpointValue("");
  };
  const exportBundle = () => {
    const bundle = redactBundle({
      format: "timetravel-bundle/v1",
      exportedAt: new Date().toISOString(),
      session: {
        traceId: liveSession.traceId,
        branchId: liveSession.branchId,
        runnerRef: liveSession.runnerRef,
        status: liveSession.status,
        steps: liveSession.history,
        promptVersions: liveSession.promptVersions,
        checkpoints: liveSession.checkpoints,
      },
      pricing,
      reproducibility: reproducibilityMetadata(liveSession),
    });
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `timetravel-${liveSession.traceId.slice(0, 8)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };
  const selectedEntry = selectedHistoryIndex === null
    ? null
    : liveSession.history[selectedHistoryIndex] ?? null;
  const inspectedStep = selectedEntry ? historyEntryToStep(selectedEntry) : null;
  const displayedStep = inspectedStep ?? liveSession.pausedStep;
  const displayedStepNumber = selectedEntry ? selectedHistoryIndex! + 1 : stepCount;
  const comparisonVersions = comparisonIds === null
    ? []
    : comparisonIds.map((id) => liveSession.promptVersions.find((version) => version.id === id)).filter((version): version is PromptVersion => Boolean(version));
  const siblingGroups = completedSiblingGroups(liveSession.promptVersions);
  const matrixVersions = matrixCursor === null
    ? []
    : liveSession.promptVersions.filter((version) => version.cursor === matrixCursor && version.status === "completed");
  const selectedSiblingGroup = siblingGroups.find((group) => group.cursor === comparisonCursor);
  const selectedComparisonIds = comparisonIds ?? [];
  const setComparisonSlot = (slot: 0 | 1, id: string) => {
    const next = [selectedComparisonIds[0] ?? "", selectedComparisonIds[1] ?? ""];
    next[slot] = id;
    if (next[0] === next[1]) next[1 - slot] = "";
    const selected = next.filter(Boolean);
    setComparisonIds(selected.length > 0 ? selected : null);
  };

  return (
    <div className="h-full overflow-hidden bg-[#080d17] p-3 sm:p-5">
      <div className="mx-auto flex h-full max-w-[1640px] flex-col overflow-hidden rounded-lg border border-slate-700/70 bg-[#0c1320] shadow-2xl shadow-black/30">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-700/60 bg-[#101927] px-4 py-3 sm:px-5">
          <div className="flex items-center gap-2.5">
            <span className="flex size-8 items-center justify-center rounded-md border border-cyan-400/25 bg-cyan-400/10 text-cyan-300">
              <Bot className="size-4" />
            </span>
            <div>
              <p className="text-sm font-semibold text-slate-100">Interactive workbench</p>
              <p className="text-xs text-slate-400">{liveSession.runnerRef} · local development session</p>
            </div>
          </div>
          <div className="ml-auto flex flex-wrap items-center justify-end gap-3 text-xs text-slate-400">
            <span className="hidden items-center gap-1.5 sm:flex"><ListTree className="size-3.5 text-cyan-300" /> {stepCount} steps observed</span>
            {liveSession.checkpoints.length > 0 && <span>{liveSession.checkpoints.length} checkpoints</span>}
            {liveSession.promptVersions.length > 0 && <span>{liveSession.promptVersions.length} prompt version{liveSession.promptVersions.length === 1 ? "" : "s"}</span>}
            <span>{acceptedCount} accepted · {rejectedCount} rejected</span>
            <span>{formatDuration(latency.totalMs)} LLM/tool time</span>
            <span className="font-mono text-slate-300">{formatTokens(usage.total_tokens)} tokens · {formatCost(estimatedCost)}</span>
            <label className="flex items-center gap-1.5 text-slate-400" htmlFor="input-token-rate">
              In $/1M
              <Input
                id="input-token-rate"
                type="number"
                min="0"
                step="0.01"
                value={pricing.inputPerMillion}
                onChange={(event) => {
                  pricingDirtyRef.current = true;
                  setPricingDirty(true);
                  setPricing((current) => ({ ...current, inputPerMillion: Math.max(0, Number(event.target.value) || 0) }));
                }}
                className="h-7 w-16 border-slate-600 bg-slate-950/60 px-2 font-mono text-xs text-slate-100"
                aria-label="Input cost per one million tokens in US dollars"
              />
            </label>
            <label className="flex items-center gap-1.5 text-slate-400" htmlFor="cached-token-rate">Cache $/1M<Input id="cached-token-rate" type="number" min="0" step="0.01" value={pricing.cachedInputPerMillion} onChange={(event) => { pricingDirtyRef.current = true; setPricingDirty(true); setPricing((current) => ({ ...current, cachedInputPerMillion: Math.max(0, Number(event.target.value) || 0) })); }} className="h-7 w-16 border-slate-600 bg-slate-950/60 px-2 font-mono text-xs text-slate-100" aria-label="Cached input cost per one million tokens in US dollars" /></label>
            <label className="flex items-center gap-1.5 text-slate-400" htmlFor="output-token-rate">Out $/1M<Input id="output-token-rate" type="number" min="0" step="0.01" value={pricing.outputPerMillion} onChange={(event) => { pricingDirtyRef.current = true; setPricingDirty(true); setPricing((current) => ({ ...current, outputPerMillion: Math.max(0, Number(event.target.value) || 0) })); }} className="h-7 w-16 border-slate-600 bg-slate-950/60 px-2 font-mono text-xs text-slate-100" aria-label="Output cost per one million tokens in US dollars" /></label>
            <label className="flex items-center gap-1.5 text-slate-400" htmlFor="thinking-token-rate">Think $/1M<Input id="thinking-token-rate" type="number" min="0" step="0.01" value={pricing.thinkingPerMillion} onChange={(event) => { pricingDirtyRef.current = true; setPricingDirty(true); setPricing((current) => ({ ...current, thinkingPerMillion: Math.max(0, Number(event.target.value) || 0) })); }} className="h-7 w-16 border-slate-600 bg-slate-950/60 px-2 font-mono text-xs text-slate-100" aria-label="Thinking cost per one million tokens in US dollars" /></label>
          </div>
        </div>
        <ResizablePanelGroup direction="horizontal" className="min-h-0 flex-1">
      {/* Left: step-history rail */}
      <ResizablePanel defaultSize={29} minSize={22} maxSize={40} className="min-h-0">
        <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[#0a111d]">
          <div className="border-b border-slate-700/60 px-4 py-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">Execution path</p>
            <p className="mt-1 text-xs text-slate-500">Each call is paused for review.</p>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <div className="p-4">
              {/* Consumed steps */}
              {liveSession.history.map((h, index) => {
                const badge = decisionBadge(h.decision, h.reviewVerdict);
                return (
                  <div key={`${h.cursor}-${index}`} className="group relative mb-3 pl-6">
                    {/* rail line */}
                    <span className="absolute left-[7px] top-0 h-[calc(100%+12px)] w-px bg-slate-700/80" />
                    {/* node */}
                    <span className={`absolute left-1 top-2 size-3 rounded-full ring-4 ring-[#0a111d] ${kindTint(h.kind)}`} />
                    <button
                      type="button"
                      onClick={() => setSelectedHistoryIndex(index)}
                      className={`w-full rounded-md border p-3 text-left text-xs transition-colors ${selectedHistoryIndex === index ? "border-cyan-300/70 bg-cyan-400/[0.08]" : "border-slate-700/80 bg-slate-900/70 group-hover:border-slate-500/70"}`}
                      title={`Open executed step ${index + 1}`}
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-slate-400">#{index + 1}</span>
                        <Badge variant="secondary" className={`text-[10px] ${badge.className}`}>
                          <CheckCircle2 className="mr-1 size-2.5" /> {badge.label}
                        </Badge>
                      </div>
                      <p className="mt-1.5 line-clamp-2 leading-relaxed text-slate-400">
                        {previewEntry(h)}
                      </p>
                    </button>
                  </div>
                );
              })}

              {/* Currently paused step (highlighted) */}
              {liveSession.pausedStep && (
                <div className="group relative mb-3 pl-6">
                  <span className="absolute left-[7px] top-0 h-full w-px bg-slate-700/80" />
                  <span className="absolute left-[2px] top-1 size-4 animate-pulse rounded-full border-2 border-cyan-300 bg-cyan-400/30 ring-4 ring-cyan-400/10" />
                  <button
                    type="button"
                    onClick={() => setSelectedHistoryIndex(null)}
                    className={`w-full rounded-md border p-3 text-left text-xs shadow-[0_0_24px_rgba(34,211,238,0.12)] ${selectedEntry ? "border-slate-600 bg-slate-900/70" : "border-cyan-300/70 bg-cyan-400/[0.08]"}`}
                    title="Open current step"
                  >
                    <div className="flex items-center gap-1.5 font-medium">
                      <PauseCircle className="size-3.5 text-cyan-300" />
                      <span className="font-mono text-cyan-50">#{stepCount}</span>
                      <span className="text-cyan-200/80">awaiting review</span>
                    </div>
                    <p className="mt-1.5 line-clamp-2 leading-relaxed text-slate-300">
                      {previewEntry(liveSession.pausedStep)}
                    </p>
                  </button>
                </div>
              )}

              {liveSession.history.length === 0 && !liveSession.pausedStep && (
                <p className="px-1 py-4 text-xs text-muted-foreground">
                  Waiting for the first intercepted call…
                </p>
              )}
              {liveSession.checkpoints.length > 0 && (
                <div className="mt-5 border-t border-slate-700/60 pt-4">
                  <p className="px-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">State checkpoints</p>
                  {liveSession.checkpoints.map((checkpoint) => {
                    const isExpanded = checkpointExpanded[checkpoint.name] ?? false;
                    const payload = checkpointPayloads[checkpoint.name];
                    return (
                      <div key={checkpoint.name} className="mt-2">
                        <div className="rounded-md border border-violet-400/20 bg-violet-500/[0.05] px-3 py-2 text-xs">
                          <p className="font-medium text-violet-200">{checkpoint.label}</p>
                          <p className="mt-1 font-mono text-[10px] text-slate-400">{checkpoint.keys.join(" · ")}</p>
                          <div className="mt-2 flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={() => void restartFromCheckpoint(checkpoint)}
                              disabled={checkpointRestarting !== null}
                              className="rounded border border-violet-300/30 bg-violet-300/10 px-2 py-0.5 text-[10px] text-violet-100 hover:bg-violet-300/20 disabled:opacity-60"
                            >
                              {checkpointRestarting === checkpoint.name ? "Starting..." : "Restart from here"}
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                const next = !isExpanded;
                                setCheckpointExpanded((prev) => ({ ...prev, [checkpoint.name]: next }));
                                if (next && !payload && liveSession.branchId) {
                                  void fetchCheckpointPayload(liveSession.branchId, checkpoint.name);
                                }
                              }}
                              className="rounded border border-slate-600/60 bg-slate-800/40 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-700/60"
                            >
                              {isExpanded ? "Hide payload" : "Show payload"}
                            </button>
                          </div>
                          {isExpanded && (
                            <div className="mt-2 space-y-1">
                              {payload ? (
                                Object.entries(redactBundle(payload) as Record<string, unknown>).map(([k, v]) => (
                                  <div key={k} className="flex gap-2 text-[10px]">
                                    <span className="min-w-0 shrink-0 font-mono text-slate-400">{k}:</span>
                                    <pre className="min-w-0 overflow-x-auto font-mono text-slate-200">{typeof v === "string" ? v : JSON.stringify(v, null, 2)}</pre>
                                  </div>
                                ))
                              ) : (
                                <p className="text-[10px] text-slate-500">Loading…</p>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              <div className="mt-5 border-t border-slate-700/60 pt-4">
                <p className="px-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Conditional breakpoints</p>
                <div className="mt-2 grid gap-2">
                  <select value={breakpointType} onChange={(event) => setBreakpointType(event.target.value as BreakpointRule["type"])} className="h-8 rounded-md border border-slate-600 bg-slate-950/60 px-2 text-xs text-slate-200">
                    <option value="tool_name">Tool name</option>
                    <option value="model_name">Model name</option>
                    <option value="message_contains">Message contains</option>
                    <option value="token_limit">Max tokens at least</option>
                  </select>
                  <div className="flex gap-2">
                    <Input value={breakpointValue} onChange={(event) => setBreakpointValue(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") createBreakpoint(); }} placeholder="Rule value" className="h-8 border-slate-600 bg-slate-950/60 text-xs" />
                    <button type="button" onClick={createBreakpoint} className="rounded-md border border-violet-300/30 bg-violet-300/10 px-2 text-xs text-violet-100 hover:bg-violet-300/20">Add</button>
                  </div>
                  {breakpoints.map((rule) => (
                    <div key={rule.id} className="flex items-center gap-2 rounded-md border border-violet-400/20 bg-violet-500/[0.05] px-2 py-1.5 text-[11px] text-violet-100">
                      <span className="min-w-0 flex-1 truncate">{rule.label}</span>
                      <button type="button" onClick={() => removeBreakpoint(rule.id)} className="text-violet-300 hover:text-white">Remove</button>
                    </div>
                  ))}
                </div>
              </div>
              {liveSession.promptVersions.length > 0 && (
                <div className="mt-5 border-t border-slate-700/60 pt-4">
                  <p className="px-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Prompt versions</p>
                  {liveSession.promptVersions.map((version) => (
                    <div key={version.id} className="mt-2 rounded-md border border-amber-400/20 bg-amber-500/[0.05] px-3 py-2 text-xs">
                      <div className="flex items-center gap-2">
                        <p className="min-w-0 flex-1 truncate font-medium text-amber-200">Step {version.cursor + 1} · {version.status}</p>
                        {version.usage ? (
                          <span className="shrink-0 text-[10px] text-slate-500">{version.usage.total_tokens.toLocaleString()} tokens</span>
                        ) : null}
                      </div>
                      <p className="mt-1 line-clamp-2 text-slate-400">{JSON.stringify(version.messages)}</p>
                      {version.status === "completed" ? (
                        <div className="mt-2 border-t border-amber-400/10 pt-2">
                          <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Response</p>
                          <p className="mt-1 line-clamp-3 whitespace-pre-wrap text-slate-300">{version.result || "(No response captured.)"}</p>
                        </div>
                      ) : (
                        <p className="mt-2 border-t border-amber-400/10 pt-2 text-[10px] italic text-slate-500">Response will appear here when this variant completes.</p>
                      )}
                    </div>
                  ))}
                  {siblingGroups.length > 0 && (
                    <div className="mt-3 rounded-md border border-cyan-300/20 bg-cyan-300/[0.04] p-3">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-cyan-200">Compare sibling variants</p>
                      <p className="mt-1 text-[11px] text-slate-400">Choose two completed variants created at the same step.</p>
                      <select
                        aria-label="Comparison step"
                        value={comparisonCursor ?? ""}
                        onChange={(event) => {
                          const cursor = event.target.value === "" ? null : Number(event.target.value);
                          setComparisonCursor(cursor);
                          setComparisonIds(null);
                        }}
                        className="mt-2 h-8 w-full rounded-md border border-slate-600 bg-slate-950/70 px-2 text-xs text-slate-200"
                      >
                        <option value="">Choose a step</option>
                        {siblingGroups.map((group) => (
                          <option key={group.cursor} value={group.cursor}>
                            Step {group.cursor + 1} · {group.versions.length} completed variants
                          </option>
                        ))}
                      </select>
                      {selectedSiblingGroup && (
                        <div className="mt-2 grid gap-2 sm:grid-cols-2">
                          <select
                            aria-label="Left prompt variant"
                            value={selectedComparisonIds[0] ?? ""}
                            onChange={(event) => setComparisonSlot(0, event.target.value)}
                            className="h-8 min-w-0 rounded-md border border-slate-600 bg-slate-950/70 px-2 text-xs text-slate-200"
                          >
                            <option value="">Choose left variant</option>
                            {selectedSiblingGroup.versions.map((version, index) => (
                              <option key={version.id} value={version.id}>Variant {index + 1} · {version.model || "default model"}</option>
                            ))}
                          </select>
                          <select
                            aria-label="Right prompt variant"
                            value={selectedComparisonIds[1] ?? ""}
                            onChange={(event) => setComparisonSlot(1, event.target.value)}
                            className="h-8 min-w-0 rounded-md border border-slate-600 bg-slate-950/70 px-2 text-xs text-slate-200"
                          >
                            <option value="">Choose right variant</option>
                            {selectedSiblingGroup.versions.map((version, index) => (
                              <option key={version.id} value={version.id}>Variant {index + 1} · {version.model || "default model"}</option>
                            ))}
                          </select>
                        </div>
                      )}
                      {comparisonIds && <button type="button" onClick={() => setComparisonIds(null)} className="mt-2 w-full rounded-md border border-slate-600 px-3 py-2 text-xs text-slate-300">Clear pair selection</button>}
                      {selectedSiblingGroup && <button type="button" onClick={() => setMatrixCursor(selectedSiblingGroup.cursor)} className="mt-2 w-full rounded-md border border-cyan-300/30 bg-cyan-300/10 px-3 py-2 text-xs font-medium text-cyan-100 hover:bg-cyan-300/20">Open comparison matrix (max 10)</button>}
                    </div>
                  )}
                </div>
              )}
              <ExecutionGraph
                history={liveSession.history}
                pausedStep={liveSession.pausedStep}
                checkpoints={liveSession.checkpoints}
                sessionError={liveSession.status === "errored" ? liveSession.error : null}
                onSelectHistory={setSelectedHistoryIndex}
              />
            </div>
          </ScrollArea>
        </div>
      </ResizablePanel>

      <ResizableHandle withHandle />

      {/* Right: progress header + StepPanel / status */}
      <ResizablePanel defaultSize={71} minSize={40} className="min-h-0">
        <div className="flex h-full min-h-0 flex-col overflow-hidden">
          {/* Progress header */}
          <div className="flex items-center gap-2 border-b border-slate-700/60 bg-[#0f1826] px-5 py-3">
            {liveSession.status === "running" && <Loader2 className="size-4 animate-spin text-violet-500" />}
            {isPaused && <PauseCircle className="size-4 text-amber-500" />}
            {isDone && <Sparkles className="size-4 text-emerald-500" />}
            {isErrored && <AlertCircle className="size-4 text-destructive" />}
            <span className="text-sm font-semibold text-slate-100">
              {isDone ? "Session complete" : isErrored ? "Session errored" : isPaused ? "Paused" : "Running"}
            </span>
            <span className="text-xs text-slate-400">
              Step {stepCount}{liveSession.pausedStep ? ` · paused at #${stepCount}` : ""}
            </span>
            <Progress
              value={isDone ? 100 : (liveSession.history.length / Math.max(stepCount, 1)) * 100}
              className="ml-2 h-1.5 w-28 bg-slate-800"
            />
          </div>

          {/* Body */}
          <div className="min-h-0 flex-1 overflow-hidden">
            {matrixVersions.length >= 2 ? (
              <PromptVersionMatrix versions={matrixVersions} pricing={pricing} onClose={() => setMatrixCursor(null)} />
            ) : comparisonVersions.length === 2 ? (
              <PromptVersionComparison versions={comparisonVersions} pricing={pricing} onClose={() => setComparisonIds(null)} />
            ) : displayedStep ? (
              <StepPanel
                sessionId={liveSession.sessionId}
                step={displayedStep}
                stepNumber={displayedStepNumber}
                pricing={pricing}
                canTimeTravel={liveSession.history.length > 0}
                canStepForward={liveSession.savedFuture.length > 0}
                readOnly={selectedEntry !== null}
                onReturnToCurrent={liveSession.pausedStep ? () => setSelectedHistoryIndex(null) : undefined}
              />
            ) : isDone ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
                <Sparkles className="size-8 text-emerald-500" />
                <p className="text-sm">
                  Session complete — {liveSession.history.length} step{liveSession.history.length === 1 ? "" : "s"}.
                </p>
                <p className="text-xs text-muted-foreground">
                  Start another session from the top bar.
                </p>
                {liveSession.resultPayload !== undefined && (
                  <div className="w-full max-w-2xl rounded-md border border-emerald-500/25 bg-emerald-500/[0.06] p-4 text-left">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <p className="text-sm font-medium text-emerald-200">Agent Result</p>
                      <span className="text-[10px] uppercase tracking-wide text-emerald-300/70">completed</span>
                    </div>
                    <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-slate-950/50 p-3 font-mono text-xs text-slate-200">
                      {displayAgentResult(liveSession.resultPayload)}
                    </pre>
                  </div>
                )}
                <div className="rounded-md border border-slate-700 bg-slate-950/40 px-4 py-3 text-xs text-slate-300">Evaluation summary · {acceptedCount} accepted · {rejectedCount} rejected · {formatTokens(usage.total_tokens)} tokens · {formatCost(estimatedCost)} · {formatDuration(latency.totalMs)} execution <span className="text-slate-500">(LLM {formatDuration(latency.llmMs)} · tools {formatDuration(latency.toolMs)})</span></div>
                <ReproducibilitySummary session={liveSession} />
                <div className="flex flex-wrap justify-center gap-2"><button type="button" onClick={() => void saveRegressionCase()} className="rounded-md bg-emerald-500 px-4 py-2 text-sm font-medium text-emerald-950 hover:bg-emerald-400">Save regression case</button><button type="button" onClick={exportBundle} className="rounded-md border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-medium text-cyan-100 hover:bg-cyan-300/20">Export redacted bundle</button><button type="button" onClick={clearLiveSession} className="rounded-md border border-slate-600 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-100 hover:bg-slate-800">Saved sessions</button></div>
                {regressionSaved && <p className="text-xs text-emerald-300">Saved locally as a reusable regression case.</p>}
              </div>
            ) : isErrored ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
                <AlertCircle className="size-8 text-destructive" />
                <p className="text-sm font-medium text-destructive">Session errored</p>
                <pre className="max-w-md overflow-x-auto rounded-md bg-muted/40 p-3 text-left font-mono text-xs">
                  {liveSession.error}
                </pre>
              </div>
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
                <Loader2 className="size-8 animate-spin text-violet-500" />
                <p className="text-sm text-muted-foreground">
                  Running — waiting for the agent to reach a step…
                </p>
                <p className="text-xs text-muted-foreground">
                  runner: <code className="font-mono">{liveSession.runnerRef}</code>
                </p>
              </div>
            )}
          </div>
        </div>
      </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}

function sessionUsage(
  history: Array<{ usage?: StepUsage | null }>,
  pendingUsage: StepUsage | null | undefined,
): StepUsage {
  return [...history.map((step) => step.usage), pendingUsage].reduce<StepUsage>(
    (total, usage) => ({
      input_tokens: total.input_tokens + (usage?.input_tokens ?? 0),
      cached_input_tokens: (total.cached_input_tokens ?? 0) + (usage?.cached_input_tokens ?? 0),
      output_tokens: total.output_tokens + (usage?.output_tokens ?? 0),
      thinking_tokens: total.thinking_tokens + (usage?.thinking_tokens ?? 0),
      final_tokens: total.final_tokens + (usage?.final_tokens ?? 0),
      total_tokens: total.total_tokens + (usage?.total_tokens ?? 0),
      estimated: total.estimated || (usage?.estimated ?? false),
    }),
    { input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, thinking_tokens: 0, final_tokens: 0, total_tokens: 0, estimated: false },
  );
}

function usageCost(usage: StepUsage, pricing: PricingProfile): number {
  const cachedInput = usage.cached_input_tokens ?? 0;
  const uncachedInput = Math.max(0, usage.input_tokens - cachedInput);
  return ((uncachedInput * pricing.inputPerMillion)
    + (cachedInput * pricing.cachedInputPerMillion)
    + (usage.final_tokens * pricing.outputPerMillion)
    + (usage.thinking_tokens * pricing.thinkingPerMillion)) / 1_000_000;
}

function emptyUsage(): StepUsage {
  return { input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, thinking_tokens: 0, final_tokens: 0, total_tokens: 0, estimated: false };
}

function completedLatency(version: PromptVersion): number {
  return version.completedAt ? Math.max(0, version.completedAt - version.createdAt) : 0;
}

function sessionLatency(history: StepHistoryEntry[]): { totalMs: number; llmMs: number; toolMs: number } {
  return history.reduce(
    (total, step) => {
      const duration = step.latencyMs ?? 0;
      total.totalMs += duration;
      if (step.kind === "llm") total.llmMs += duration;
      else total.toolMs += duration;
      return total;
    },
    { totalMs: 0, llmMs: 0, toolMs: 0 },
  );
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) return `${durationMs}ms`;
  return `${(durationMs / 1000).toFixed(durationMs < 10_000 ? 1 : 0)}s`;
}

function reproducibilityMetadata(session: { history: StepHistoryEntry[]; checkpoints: Array<{ name: string }>; promptVersions: PromptVersion[]; startedAt: number }) {
  const models = [...new Set(session.history.map((step) => step.payload?.model).filter((model): model is string => Boolean(model)))];
  const toolSchemas = session.history.reduce((count, step) => count + (step.payload?.tools?.length ?? 0), 0);
  const toolSchemaHashes = [...new Set(session.history.flatMap((step) => (step.payload?.tools ?? []).map((tool) => stableHash(JSON.stringify(tool)))))];
  const parameterSets = [...new Set(session.history.map((step) => JSON.stringify(step.payload?.params ?? {})))].filter((value) => value !== "{}").length;
  const seeds = [...new Set(session.history.map((step) => step.payload?.params?.seed).filter((seed): seed is string | number => typeof seed === "string" || typeof seed === "number"))];
  return {
    captureStartedAt: new Date(session.startedAt).toISOString(),
    models,
    checkpoints: session.checkpoints.map((checkpoint) => checkpoint.name),
    parameterSets,
    toolSchemas,
    toolSchemaHashes,
    seeds,
    environment: typeof navigator === "undefined" ? "local client" : navigator.userAgent,
    client: "Agent Timetravel local debugger",
    queueTime: "unavailable",
    timeToFirstToken: "unavailable",
    generationTime: "unavailable",
    providerRevision: "unavailable",
  };
}

function ReproducibilitySummary({ session }: { session: LiveSession }) {
  const metadata = reproducibilityMetadata(session);
  return (
    <div className="max-w-2xl rounded-md border border-slate-700 bg-slate-950/40 px-4 py-3 text-left text-xs text-slate-300">
      <p className="font-medium text-slate-100">Reproducibility</p>
      <p className="mt-1">{metadata.models.join(", ") || "No model metadata reported"} · {session.checkpoints.length} checkpoints · {session.promptVersions.length} prompt variants · {metadata.toolSchemas} tool schema{metadata.toolSchemas === 1 ? "" : "s"}</p>
      <p className="mt-1 text-slate-500">Parameters {metadata.parameterSets || "none"} · Seeds {metadata.seeds.join(", ") || "not reported"} · Provider revision unavailable · Queue/TTFT/generation unavailable from current provider events.</p>
      {metadata.toolSchemaHashes.length > 0 && <p className="mt-1 truncate font-mono text-[10px] text-slate-500">Tool hashes {metadata.toolSchemaHashes.join(" ")}</p>}
    </div>
  );
}

function stableHash(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function mergeSavedCases(localRecords: SavedSessionCase[], backendRecords: SavedSessionCase[]): SavedSessionCase[] {
  const byId = new Map<string, SavedSessionCase>();
  for (const record of localRecords) byId.set(record.id, record);
  for (const record of backendRecords) byId.set(record.id, { ...byId.get(record.id), ...record });
  return [...byId.values()].sort((left, right) => Date.parse(right.createdAt) - Date.parse(left.createdAt));
}

function readSavedSessions(): SavedSessionCase[] {
  if (typeof window === "undefined") return [];
  try {
    const records = JSON.parse(window.localStorage.getItem(SAVED_SESSIONS_STORAGE_KEY) ?? "[]") as unknown[];
    return records.filter((record): record is SavedSessionCase => Boolean(record && typeof record === "object" && "id" in record && "steps" in record));
  } catch {
    return [];
  }
}

function evaluateSavedCase(record: SavedSessionCase): NonNullable<SavedSessionCase["regression"]> {
  const failures: string[] = [];
  let total = 0;
  for (const step of record.steps) {
    const assertions = step.assertions;
    if (!assertions) continue;
    total += 1;
    const output = step.result ?? "";
    const label = `Step ${step.cursor + 1}`;
    if (assertions.requireJson) {
      try { JSON.parse(output); } catch { failures.push(`${label}: response is not valid JSON.`); }
    }
    for (const required of assertions.requiredText) {
      if (!output.toLowerCase().includes(required.toLowerCase())) failures.push(`${label}: missing “${required}”.`);
    }
    for (const forbidden of assertions.forbiddenText) {
      if (output.toLowerCase().includes(forbidden.toLowerCase())) failures.push(`${label}: contains forbidden “${forbidden}”.`);
    }
    if (assertions.requireCitations && !/\[[^\]]+\]/.test(output)) failures.push(`${label}: missing citation.`);
    if (assertions.maxTokens !== null && (step.usage?.total_tokens ?? 0) > assertions.maxTokens) failures.push(`${label}: token budget exceeded.`);
    if (assertions.maxCostUsd !== null && usageCost(step.usage ?? emptyUsage(), record.pricing ?? DEFAULT_PRICING) > assertions.maxCostUsd) {
      failures.push(`${label}: cost budget exceeded.`);
    }
  }
  return { passed: failures.length === 0, checkedAt: new Date().toISOString(), total, failures };
}

function ExecutionGraph({
  history,
  pausedStep,
  checkpoints,
  sessionError,
  onSelectHistory,
}: {
  history: StepHistoryEntry[];
  pausedStep: PausedStep | null;
  checkpoints: Array<{ cursor: number; label: string }>;
  sessionError: string | null;
  onSelectHistory: (index: number) => void;
}) {
  const [filter, setFilter] = useState<"all" | "llm" | "tool" | "error" | "edited" | "checkpoint">("all");
  const nodes = [...history.map((step, index) => ({ step, index, current: false })), ...(pausedStep ? [{ step: pausedStep, index: history.length, current: true }] : [])];
  const decorated = nodes.map((node) => {
    const checkpoint = checkpoints.find((item) => item.cursor === node.step.cursor);
    const decision = "decision" in node.step ? node.step.decision : "";
    const edited = decision === "edit";
    const error = node.step.payload?.status === "error";
    return { ...node, checkpoint, edited, error };
  });
  const visible = decorated.filter((node) => {
    if (filter === "all") return true;
    if (filter === "checkpoint") return Boolean(node.checkpoint);
    if (filter === "edited") return node.edited;
    if (filter === "error") return node.error;
    return node.step.kind === filter;
  });
  const errorNode = sessionError
    ? { index: history.length + (pausedStep ? 1 : 0), message: sessionError }
    : null;
  const visibleErrorNode = errorNode && (filter === "all" || filter === "error");
  if (nodes.length === 0 && !errorNode) return null;
  const filters: Array<{ id: typeof filter; label: string }> = [
    { id: "all", label: "All" },
    { id: "llm", label: "LLM" },
    { id: "tool", label: "Tool" },
    { id: "checkpoint", label: "Checkpoint" },
    { id: "edited", label: "Edited" },
    { id: "error", label: "Error" },
  ];
  return (
    <div className="mt-5 border-t border-slate-700/60 pt-4">
      <div className="flex items-center gap-2 px-1">
        <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Run graph</p>
        <span className="ml-auto text-[10px] text-slate-500">{visible.length}/{nodes.length}</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5 px-1">
        {filters.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setFilter(item.id)}
            className={`rounded border px-2 py-1 text-[10px] ${filter === item.id ? "border-cyan-300/50 bg-cyan-300/10 text-cyan-100" : "border-slate-700 bg-slate-900/60 text-slate-400 hover:border-slate-500"}`}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="mt-3 space-y-1.5 px-1">
        {visible.map(({ step, index, current, checkpoint, edited, error }) => {
          return (
            <div key={`${step.cursor}-${index}`}>
              <button type="button" onClick={() => !current && onSelectHistory(index)} className={`flex w-full items-center gap-2 rounded-md border px-2.5 py-2 text-left text-[11px] ${current ? "border-cyan-300/50 bg-cyan-400/[0.08] text-cyan-100" : "border-slate-700 bg-slate-900/70 text-slate-300 hover:border-slate-500"}`}>
                <span className={`size-2 rounded-full ${kindTint(step.kind)}`} />
                <span className="font-mono text-slate-500">{index + 1}</span>
                {step.kind === "tool" && <Wrench className="size-3 text-emerald-300" />}
                {edited && <Pencil className="size-3 text-violet-300" />}
                {error && <AlertCircle className="size-3 text-rose-300" />}
                <span className="truncate">{step.kind === "tool" ? step.payload?.name ?? "tool call" : step.payload?.model ?? "LLM call"}</span>
                {current && <span className="ml-auto text-cyan-300">current</span>}
              </button>
              {checkpoint && <p className="ml-3 border-l border-violet-400/40 py-1 pl-3 text-[10px] text-violet-200">checkpoint · {checkpoint.label}</p>}
            </div>
          );
        })}
        {visibleErrorNode && (
          <div className="rounded-md border border-rose-400/30 bg-rose-500/[0.06] px-2.5 py-2 text-[11px] text-rose-100">
            <div className="flex items-center gap-2">
              <AlertCircle className="size-3 text-rose-300" />
              <span className="font-mono text-rose-300">{errorNode.index + 1}</span>
              <span className="truncate">Session error</span>
            </div>
            <p className="mt-1 line-clamp-2 text-rose-200/80">{errorNode.message}</p>
          </div>
        )}
        {visible.length === 0 && !visibleErrorNode && <p className="rounded-md border border-slate-800 bg-slate-950/40 px-3 py-2 text-[11px] text-slate-500">No graph nodes match this filter.</p>}
      </div>
    </div>
  );
}

function SavedSessionLibrary() {
  const [records, setRecords] = useState<SavedSessionCase[]>([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<SavedSessionCase | null>(null);
  const [backendAvailable, setBackendAvailable] = useState(true);
  const [busyCaseId, setBusyCaseId] = useState<string | null>(null);
  const openSavedSessionCase = useTimeTravelStore((state) => state.openSavedSessionCase);

  useEffect(() => {
    const localRecords = readSavedSessions();
    setRecords(localRecords);
    void loadRegressionCases()
      .then((backendRecords) => {
        const merged = mergeSavedCases(localRecords, backendRecords);
        setRecords(merged);
        window.localStorage.setItem(SAVED_SESSIONS_STORAGE_KEY, JSON.stringify(merged));
        setBackendAvailable(true);
      })
      .catch(() => setBackendAvailable(false));
  }, []);
  const visibleRecords = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return records;
    return records.filter((record) => [record.traceId, record.runnerRef, record.createdAt].join(" ").toLowerCase().includes(needle));
  }, [query, records]);
  const removeRecord = async (id: string) => {
    setBusyCaseId(id);
    let canRemoveLocal = true;
    try {
      const response = await fetch(`/api/v1/regression-cases/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!response.ok) {
        canRemoveLocal = response.status === 404;
        if (!canRemoveLocal) setBackendAvailable(true);
      }
    } catch {
      setBackendAvailable(false);
    } finally {
      setBusyCaseId(null);
    }
    if (!canRemoveLocal) return;
    const next = records.filter((record) => record.id !== id);
    window.localStorage.setItem(SAVED_SESSIONS_STORAGE_KEY, JSON.stringify(next));
    setRecords(next);
    if (selected?.id === id) setSelected(null);
  };
  const runSavedChecks = async (record: SavedSessionCase) => {
    setBusyCaseId(record.id);
    let regression: NonNullable<SavedSessionCase["regression"]>;
    try {
      const result = await runPersistedRegressionCase(record.id);
      regression = {
        passed: result.passed,
        checkedAt: new Date().toISOString(),
        total: 1,
        failures: result.passed ? [] : [result.detail ?? "Regression failed."],
      };
      setBackendAvailable(true);
    } catch {
      regression = evaluateSavedCase(record);
      setBackendAvailable(false);
    } finally {
      setBusyCaseId(null);
    }
    const next = records.map((candidate) => candidate.id === record.id ? { ...candidate, regression } : candidate);
    window.localStorage.setItem(SAVED_SESSIONS_STORAGE_KEY, JSON.stringify(next));
    setRecords(next);
    setSelected({ ...record, regression });
  };
  const importBundle = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as { session?: Partial<SavedSessionCase>; pricing?: PricingProfile };
      const session = parsed.session;
      if (!session?.traceId || !Array.isArray(session.steps)) throw new Error("Invalid Agent Timetravel bundle");
      const imported: SavedSessionCase = {
        id: `timetravel-import-${Date.now()}`,
        createdAt: new Date().toISOString(),
        traceId: session.traceId,
        runnerRef: session.runnerRef ?? "imported bundle",
        steps: session.steps,
        promptVersions: session.promptVersions ?? [],
        checkpoints: session.checkpoints ?? [],
        pricing: parsed.pricing,
        summary: session.summary,
      };
      const next = [...records, imported];
      window.localStorage.setItem(SAVED_SESSIONS_STORAGE_KEY, JSON.stringify(next));
      setRecords(next);
      setSelected(imported);
    } catch {
      event.currentTarget.value = "";
    }
  };

  return (
    <div className="h-full overflow-auto bg-[#080d17] p-5">
      <div className="mx-auto max-w-5xl">
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-700/60 pb-4">
          <div><p className="text-base font-semibold text-slate-100">Saved sessions</p><p className="mt-1 text-xs text-slate-400">Backend regression cases with local snapshot fallback. {backendAvailable ? "Backend sync active." : "Using local fallback."}</p></div>
          <label className="ml-auto cursor-pointer rounded-md border border-cyan-300/30 bg-cyan-300/10 px-3 py-2 text-xs font-medium text-cyan-100 hover:bg-cyan-300/20">Import bundle<input type="file" accept="application/json,.json" className="hidden" onChange={importBundle} /></label>
          <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search trace, runner, or date" className="h-8 w-64 border-slate-600 bg-slate-950/60 text-xs" />
        </div>
        {visibleRecords.length === 0 ? <p className="py-12 text-center text-sm text-slate-500">No saved regression cases yet. Complete a session, then save it from the summary.</p> : <div className="mt-4 grid gap-3 md:grid-cols-2">{visibleRecords.map((record) => <button key={record.id} type="button" onClick={() => setSelected(record)} className={`rounded-md border p-4 text-left text-xs ${selected?.id === record.id ? "border-cyan-300/60 bg-cyan-400/[0.08]" : "border-slate-700 bg-slate-900/60 hover:border-slate-500"}`}><p className="font-mono text-slate-200">{record.traceId.slice(0, 12)}</p><p className="mt-1 text-slate-400">{record.runnerRef} · {new Date(record.createdAt).toLocaleString()}</p><p className="mt-3 text-slate-300">{record.steps.length} steps · {record.summary?.accepted ?? 0} accepted · {record.summary?.rejected ?? 0} rejected</p></button>)}</div>}
        {selected && <div className="mt-5 rounded-md border border-slate-700 bg-slate-950/50 p-4 text-xs text-slate-300"><div className="flex items-center gap-2"><p className="font-medium text-slate-100">Saved run details</p><button type="button" disabled={busyCaseId === selected.id} onClick={() => void removeRecord(selected.id)} className="ml-auto text-rose-300 hover:text-rose-200 disabled:opacity-50">Delete</button></div><p className="mt-2">{selected.steps.length} steps · {formatTokens(selected.summary?.totalTokens ?? 0)} tokens · {formatDuration(selected.summary?.totalLatencyMs ?? 0)} execution</p><p className="mt-1 text-slate-400">{selected.checkpoints?.length ?? 0} checkpoints · {selected.promptVersions.length} prompt variants</p><div className="mt-4 flex flex-wrap gap-2"><button type="button" disabled={busyCaseId === selected.id} onClick={() => void runSavedChecks(selected)} className="rounded-md bg-emerald-500 px-3 py-2 font-medium text-emerald-950 hover:bg-emerald-400 disabled:opacity-60">{busyCaseId === selected.id ? "Running..." : "Run frozen checks"}</button><button type="button" onClick={() => openSavedSessionCase(selected)} className="rounded-md border border-cyan-300/30 bg-cyan-300/10 px-3 py-2 font-medium text-cyan-100 hover:bg-cyan-300/20">Open saved trace</button></div>{selected.regression && <p className={`mt-3 rounded-md px-3 py-2 ${selected.regression.passed ? "bg-emerald-500/10 text-emerald-300" : "bg-rose-500/10 text-rose-200"}`}>{selected.regression.passed ? `Frozen verification passed at ${new Date(selected.regression.checkedAt).toLocaleString()}.` : `${selected.regression.failures.length || 1} check failed: ${selected.regression.failures.join(" ")}`}</p>}</div>}
      </div>
    </div>
  );
}

function redactBundle(value: unknown, key = ""): unknown {
  // Usage metrics such as input_tokens are safe and essential to exported traces.
  const normalizedKey = key.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  const metricKey = /^(?:cached_input|completion|final|input|output|prompt|thinking|total)_tokens$/.test(normalizedKey)
    || normalizedKey === "token_count" || normalizedKey === "tokens";
  if (!metricKey && /^(?:api[_-]?key|x[_-]?api[_-]?key|authorization|secret|password|access[_-]?token|refresh[_-]?token|bearer|bearer[_-]?token|client[_-]?secret|session[_-]?token|token|email|phone|ssn|social[_-]?security(?:[_-]?number)?)$/i.test(key)) return "[REDACTED]";
  if (typeof value === "string") {
    return value
      .replace(/\bsk-[A-Za-z0-9_-]+\b/g, "[REDACTED]")
      .replace(/(?:api[_-]?key|x-api-key)\s*[:=]\s*["']?[A-Za-z0-9._~+/=-]+/gi, "api_key=[REDACTED]")
      .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, "Bearer [REDACTED]")
      .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[REDACTED_EMAIL]")
      .replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  }
  if (Array.isArray(value)) return value.map((item) => redactBundle(item));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([entryKey, entryValue]) => [entryKey, redactBundle(entryValue, entryKey)]));
  }
  return value;
}

function formatTokens(tokens: number): string {
  return new Intl.NumberFormat("en-US").format(tokens);
}

function formatCost(cost: number): string {
  return cost === 0 ? "$0.00" : `$${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)}`;
}

/** One-line preview of a step's payload or result for the rail cards. */
function previewEntry(entry: { payload?: unknown; result?: string | null }): string {
  // Prefer the model's response (the verify-loop result) over the input payload.
  if (entry.result && entry.result.trim()) {
    return entry.result.slice(0, 120);
  }
  const payload = entry.payload;
  if (!payload || typeof payload !== "object") return "";
  const p = payload as { model?: string; messages?: unknown[]; name?: string };
  if (p.name) return `tool: ${p.name}`;
  if (p.messages && Array.isArray(p.messages) && p.messages.length > 0) {
    const last = p.messages[p.messages.length - 1] as { content?: unknown };
    const c = typeof last?.content === "string" ? last.content : "";
    return c.slice(0, 120) || `${p.messages.length} message(s)`;
  }
  return p.model ?? "";
}

function historyEntryToStep(entry: StepHistoryEntry): PausedStep {
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
  };
}

function PromptVersionComparison({ versions, pricing, onClose }: { versions: PromptVersion[]; pricing: PricingProfile; onClose: () => void }) {
  const baseline = versions[0];
  const variant = versions[1];
  if (!baseline || !variant) return null;
  const tokenDelta = (variant.usage?.total_tokens ?? 0) - (baseline.usage?.total_tokens ?? 0);
  const baselineCost = usageCost(baseline.usage ?? emptyUsage(), baseline.pricing ?? pricing);
  const variantCost = usageCost(variant.usage ?? emptyUsage(), variant.pricing ?? pricing);
  const costDelta = variantCost - baselineCost;
  const latencyDelta = (variant.latencyMs ?? completedLatency(variant)) - (baseline.latencyMs ?? completedLatency(baseline));
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-5">
      <div className="mb-4 flex items-center gap-3">
        <div><p className="text-base font-semibold">Prompt Variant Comparison</p><p className="text-xs text-slate-400">Step {baseline.cursor + 1} · {tokenDelta >= 0 ? "+" : ""}{formatTokens(tokenDelta)} tokens · {costDelta >= 0 ? "+" : ""}{formatCost(costDelta)} · {latencyDelta >= 0 ? "+" : ""}{formatDuration(latencyDelta)}</p></div>
        <button type="button" onClick={onClose} className="ml-auto text-xs text-cyan-200 hover:text-white">Back to step</button>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-2 gap-4">
        {[baseline, variant].map((version, index) => (
          <div key={version.id} className="flex min-h-0 flex-col overflow-hidden rounded-md border border-slate-700 bg-slate-950/40">
            <div className="border-b border-slate-700 px-4 py-3 text-xs font-semibold text-slate-200">{index === 0 ? "Original" : "Variant"} · {version.model || "default model"} · {formatTokens(version.usage?.total_tokens ?? 0)} tokens · {formatCost(usageCost(version.usage ?? emptyUsage(), version.pricing ?? pricing))}</div>
            <div className="border-b border-slate-800 px-4 py-3 text-xs text-slate-400"><p className="font-medium uppercase tracking-wide text-slate-500">Parameter snapshot</p><pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-slate-300">{JSON.stringify(version.parameters ?? {}, null, 2)}</pre></div>
            <div className="border-b border-slate-800 px-4 py-3 text-xs"><p className="font-medium uppercase tracking-wide text-slate-500">Prompt messages</p><DiffText left={JSON.stringify(baseline.messages, null, 2)} right={JSON.stringify(variant.messages, null, 2)} side={index === 0 ? "left" : "right"} /></div>
            <div className="min-h-0 flex-1 overflow-auto p-4 font-mono text-xs leading-relaxed"><p className="mb-2 font-sans font-medium uppercase tracking-wide text-slate-500">Response</p><DiffText left={baseline.result ?? ""} right={variant.result ?? ""} side={index === 0 ? "left" : "right"} /></div>
            <div className="space-y-1 border-t border-slate-800 px-4 py-2 text-xs"><p className={version.assertionResult?.passed ? "text-emerald-300" : "text-rose-300"}>Assertions: {version.assertionResult ? (version.assertionResult.passed ? "passed" : version.assertionResult.failures.join(" ")) : "not run"}</p><p className="text-slate-300">Review: {version.reviewVerdict ?? "unreviewed"}{version.reviewNote ? ` · ${version.reviewNote}` : ""}</p></div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DiffText({ left, right, side }: { left: string; right: string; side: "left" | "right" }) {
  const tokens = wordDiff(left, right).filter((token) => token.type === "equal" || (side === "left" ? token.type === "remove" : token.type === "add"));
  return <pre className="mt-2 whitespace-pre-wrap break-words">{tokens.map((token, index) => <span key={`${token.type}-${index}`} className={token.type === "add" ? "rounded bg-emerald-400/20 text-emerald-200" : token.type === "remove" ? "rounded bg-rose-400/20 text-rose-200 line-through" : "text-slate-300"}>{token.value}</span>)}</pre>;
}

function PromptVersionMatrix({ versions, pricing, onClose }: { versions: PromptVersion[]; pricing: PricingProfile; onClose: () => void }) {
  const capped = versions.slice(0, MAX_COMPARISON_VARIANTS);
  const baseline = capped[0];
  if (!baseline) return null;
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-5">
      <div className="mb-4 flex items-center gap-3"><div><p className="text-base font-semibold">Prompt comparison matrix</p><p className="text-xs text-slate-400">Step {baseline.cursor + 1} · showing {capped.length} of {versions.length} completed variants · capped at {MAX_COMPARISON_VARIANTS}</p></div><button type="button" onClick={onClose} className="ml-auto text-xs text-cyan-200 hover:text-white">Back to step</button></div>
      <div className="min-h-0 flex-1 overflow-auto rounded-md border border-slate-700"><table className="w-full min-w-[680px] text-left text-xs"><thead><tr className="border-b border-slate-700 bg-slate-900/70"><th className="px-3 py-2 text-slate-500">Metric</th>{capped.map((version, index) => <th key={version.id} className="px-3 py-2 text-slate-200">{index === 0 ? "Original" : `Variant ${index}`}</th>)}</tr></thead><tbody>{["model", "parameters", "tokens", "cost", "latency", "assertions", "evaluators", "review"].map((metric) => <tr key={metric} className="border-b border-slate-800"><th className="px-3 py-2 font-medium text-slate-400">{metric}</th>{capped.map((version) => <td key={version.id} className="max-w-56 px-3 py-2 align-top font-mono text-slate-300">{metric === "model" ? version.model || "default" : metric === "parameters" ? JSON.stringify(version.parameters ?? {}) : metric === "tokens" ? formatTokens(version.usage?.total_tokens ?? 0) : metric === "cost" ? formatCost(usageCost(version.usage ?? emptyUsage(), version.pricing ?? pricing)) : metric === "latency" ? formatDuration(version.latencyMs ?? completedLatency(version)) : metric === "assertions" ? (version.assertionResult?.passed ? "passed" : version.assertionResult ? "failed" : "not run") : metric === "evaluators" ? Object.entries(version.evaluatorResults ?? {}).map(([name, result]) => `${name}: ${result.passed ? "passed" : "failed"}`).join(" · ") || "not run" : version.reviewVerdict ?? "unreviewed"}</td>)}</tr>)}</tbody></table></div>
    </div>
  );
}
