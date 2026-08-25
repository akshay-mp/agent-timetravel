"use client";

/**
 * StepPanel — the verify-then-navigate step viewer for interactive stepping.
 *
 * The stepping loop is: pause BEFORE the call → developer approves → call
 * executes → result surfaces HERE → developer verifies and chooses
 * Next / Step back / Stop. This component renders both the pending call
 * (messages/model/params) and, once the model responds, the result text.
 *
 * Decision hierarchy (shadcn button variants):
 *   - default (solid)  → Next step (approve + continue)
 *   - outline          → Edit (toggles edit mode) / Apply edit & continue
 *   - outline          → Step back (restart-from a prior cursor)
 *   - destructive      → Stop (inline AlertDialog confirm)
 */

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  PauseCircle,
  PlayCircle,
  Pencil,
  Square,
  Wrench,
  MessageSquare,
  SlidersHorizontal,
  Sparkles,
  CornerUpLeft,
  Brain,
  ChevronDown,
  Loader2,
} from "lucide-react";
import type { BreakpointRule, OutputAssertions, PausedStep, PricingProfile, PromptVersion } from "@/lib/timetravel/types";
import { continueFromSavedState, continueUntilNextLlm, patchRunControl, persistPromptVersion, persistPromptVersionResult, postDecision, restartSessionFrom, rerunEditedStep, stopSessionForInspection } from "@/lib/timetravel/session-client";
import { useTimeTravelStore } from "@/lib/timetravel/store";

interface StepPanelProps {
  sessionId: string;
  step: PausedStep;
  /** Ordinal within this workbench session, distinct from replay cursor. */
  stepNumber: number;
  pricing: PricingProfile;
  canTimeTravel: boolean;
  canStepForward: boolean;
  readOnly?: boolean;
  onReturnToCurrent?: () => void;
}

function kindMeta(kind: string): { label: string; className: string } {
  switch (kind) {
    case "llm":
      return { label: "LLM", className: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300" };
    case "tool":
      return { label: "Tool", className: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" };
    case "mcp":
      return { label: "MCP", className: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300" };
    default:
      return { label: kind, className: "bg-muted text-muted-foreground" };
  }
}

function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

export function StepPanel({ sessionId, step, stepNumber, pricing, canTimeTravel, canStepForward, readOnly = false, onReturnToCurrent }: StepPanelProps) {
  const [editing, setEditing] = useState(false);
  const [editedMessages, setEditedMessages] = useState("");
  const [editedModel, setEditedModel] = useState("");
  const [editedArgs, setEditedArgs] = useState("");
  const [editedKwargs, setEditedKwargs] = useState("");
  const [mockResult, setMockResult] = useState("{}");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const [requiredText, setRequiredText] = useState("");
  const [forbiddenText, setForbiddenText] = useState("");
  const [maxTokens, setMaxTokens] = useState("");
  const [maxCostUsd, setMaxCostUsd] = useState("");
  const [requireJson, setRequireJson] = useState(false);
  const [requireCitations, setRequireCitations] = useState(false);
  const [assertionResults, setAssertionResults] = useState<string[] | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const [rejectReason, setRejectReason] = useState("");
  const [evaluatorNames, setEvaluatorNames] = useState<string[]>([]);
  const [selectedEvaluator, setSelectedEvaluator] = useState("");
  const [evaluatorResult, setEvaluatorResult] = useState<string | null>(null);
  const liveReasoningRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    const msgs = step.payload.messages;
    setEditedMessages(msgs ? JSON.stringify(msgs, null, 2) : "");
    setEditedModel(step.payload.model ?? "");
    setEditedArgs(JSON.stringify(step.payload.args ?? [], null, 2));
    setEditedKwargs(JSON.stringify(step.payload.kwargs ?? {}, null, 2));
    setEditing(false);
    setError(null);
    setThinkingOpen(false);
    setRejectReason("");
    setRequiredText(step.assertions?.requiredText.join(", ") ?? "");
    setForbiddenText(step.assertions?.forbiddenText.join(", ") ?? "");
    setMaxTokens(step.assertions?.maxTokens?.toString() ?? "");
    setMaxCostUsd(step.assertions?.maxCostUsd?.toString() ?? "");
    setRequireJson(step.assertions?.requireJson ?? false);
    setRequireCitations(step.assertions?.requireCitations ?? false);
    setAssertionResults(step.assertionResult ? (step.assertionResult.passed ? ["All configured checks passed."] : step.assertionResult.failures) : null);
    setEvaluatorResult(null);
  }, [step]);

  const meta = kindMeta(step.kind);
  const isTool = step.kind === "tool" || step.kind === "mcp";
  const isExecuting = step.phase === "running" || (!isTool && step.phase === "queued");
  const isCompleted = step.phase === "completed";
  const context = !isTool ? contextBreakdown(step.payload) : null;
  const modelDuration = step.completedAt === null
    ? null
    : fmtElapsed(step.completedAt - step.pausedAt);

  useEffect(() => {
    if (!isCompleted || isTool) return;
    void fetch("/api/v1/evaluators")
      .then((response) => response.ok ? response.json() as Promise<unknown> : [])
      .then((value) => setEvaluatorNames(Array.isArray(value) ? value.filter((name): name is string => typeof name === "string") : []))
      .catch(() => setEvaluatorNames([]));
  }, [isCompleted, isTool]);

  // Keep the live reasoning stream pinned to its newest line, and keep the
  // completed thinking visible (auto-expanded) so reviewing the reasoning
  // behind a finished step never needs a hunt for the toggle.
  useEffect(() => {
    const el = liveReasoningRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [step.reasoning]);
  useEffect(() => {
    if (isCompleted && step.reasoning) setThinkingOpen(true);
  }, [isCompleted, step.reasoning]);

  const decide = async (kind: "approve" | "edit" | "stop" | "step_once" | "mock" | "skip" | "reject" | "run_until_breakpoint", body?: Record<string, unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await postDecision(sessionId, { kind, ...body });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRejectTool = async () => {
    setBusy(true);
    setError(null);
    try {
      const reason = rejectReason.trim() || undefined;
      await postDecision(sessionId, { kind: "reject", ...(reason ? { reason } : {}) });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRunUntilBreakpoint = async () => {
    setBusy(true);
    setError(null);
    try {
      // Persist intent server-side so a page refresh keeps the run-control flag.
      const breakpoints = useTimeTravelStore.getState().breakpoints;
      await patchRunControl(sessionId, { pause_after_current: false, run_until_breakpoint: true, breakpoints });
      // Then approve the current step — the server auto-approves until a breakpoint fires.
      await postDecision(sessionId, { kind: "run_until_breakpoint" });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const armPauseAfterCurrent = async () => {
    const breakpoints: BreakpointRule[] = useTimeTravelStore.getState().breakpoints;
    await patchRunControl(sessionId, { pause_after_current: true, run_until_breakpoint: false, breakpoints });
  };

  const handlePauseAfterCurrent = async () => {
    setBusy(true);
    setError(null);
    try {
      await armPauseAfterCurrent();
      await postDecision(sessionId, { kind: "approve" });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleArmPauseAfterCurrent = async () => {
    setBusy(true);
    setError(null);
    try {
      await armPauseAfterCurrent();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleMockTool = async () => {
    let value: unknown = mockResult;
    try {
      value = JSON.parse(mockResult);
    } catch {
      // Plain text is a valid mock result too.
    }
    await decide("mock", { mock_result: value });
  };

  const handleRetryTool = async () => {
    setBusy(true);
    setError(null);
    try {
      await restartSessionFrom(sessionId, step.cursor, `Retry tool at step ${step.cursor + 1}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleApplyEdit = async () => {
    if (isTool) {
      try {
        const args = JSON.parse(editedArgs) as unknown;
        const kwargs = JSON.parse(editedKwargs) as unknown;
        if (!Array.isArray(args) || kwargs === null || typeof kwargs !== "object" || Array.isArray(kwargs)) {
          throw new Error("arguments must be an array and keyword arguments must be an object");
        }
        void decide("edit", { args, kwargs });
        setEditing(false);
        return;
      } catch (e) {
        setError(`invalid tool JSON: ${e instanceof Error ? e.message : String(e)}`);
        return;
      }
    }
    let messages: unknown[] | undefined;
    if (editedMessages.trim()) {
      try {
        messages = JSON.parse(editedMessages) as unknown[];
      } catch (e) {
        setError(`invalid messages JSON: ${e instanceof Error ? e.message : String(e)}`);
        return;
      }
    }
    const revisedMessages = messages ?? step.payload.messages ?? [];
    const revisedModel = editedModel.trim() || step.payload.model || "";
    if (isCompleted) {
      setBusy(true);
      try {
        const runnerRef = useTimeTravelStore.getState().liveSession?.runnerRef ?? "agent";
        const traceId = useTimeTravelStore.getState().liveSession?.traceId ?? "";
        const now = Date.now();
        const baseline: PromptVersion = {
          id: `baseline-${Date.now()}-${step.cursor}`,
          cursor: step.cursor,
          createdAt: now,
          baseMessages: step.payload.messages ?? [],
          messages: step.payload.messages ?? [],
          baseModel: step.payload.model ?? "",
          model: step.payload.model ?? "",
          status: "completed",
          result: step.result,
          usage: step.usage,
          parameters: parameterSnapshot(step.payload),
          branchId: useTimeTravelStore.getState().liveSession?.branchId,
          pricing,
          assertions: step.assertions,
          evaluatorNames: selectedEvaluator ? [selectedEvaluator] : [],
          assertionResult: step.assertionResult,
          reviewVerdict: step.reviewVerdict,
          reviewNote: step.reviewNote,
        };
        await persistPromptVersion(traceId, baseline);
        await persistPromptVersionResult(baseline.id, baseline);
        useTimeTravelStore.getState().addPromptVersion(baseline);
        await rerunEditedStep(sessionId, runnerRef, step.cursor, revisedMessages, revisedModel, async (branch) => {
          const variant: PromptVersion = {
            id: `${Date.now()}-${step.cursor}`,
            cursor: step.cursor,
            createdAt: Date.now(),
            baseMessages: step.payload.messages ?? [],
            messages: revisedMessages,
            baseModel: step.payload.model ?? "",
            model: revisedModel,
            status: "running",
            result: null,
            usage: null,
            parameters: parameterSnapshot(step.payload),
            branchId: branch.branch_id,
            parentVersionId: baseline.id,
            pricing,
            assertions: step.assertions,
            evaluatorNames: selectedEvaluator ? [selectedEvaluator] : [],
            reviewVerdict: step.reviewVerdict,
            reviewNote: step.reviewNote,
          };
          await persistPromptVersion(traceId, variant);
          useTimeTravelStore.getState().addPromptVersion(variant);
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    } else {
      await decide("edit", { messages: revisedMessages, model: revisedModel });
    }
    setEditing(false);
  };

  const handleTimeTravel = async () => {
    setBusy(true);
    setError(null);
    try {
      if (!step.restored) await stopSessionForInspection(sessionId);
      if (!useTimeTravelStore.getState().restorePreviousStep()) {
        throw new Error("No earlier captured step is available to restore.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleStepForward = () => {
    if (!useTimeTravelStore.getState().restoreNextStep()) {
      setError("No later captured step is available to restore.");
    }
  };

  const runAssertions = () => {
    const output = step.result ?? "";
    const failures: string[] = [];
    const required = requiredText.split(",").map((value) => value.trim()).filter(Boolean);
    const forbidden = forbiddenText.split(",").map((value) => value.trim()).filter(Boolean);
    if (requireJson) {
      try { JSON.parse(output); } catch { failures.push("Response is not valid JSON."); }
    }
    required.filter((value) => !output.toLowerCase().includes(value.toLowerCase())).forEach((value) => failures.push(`Missing required text: ${value}`));
    forbidden.filter((value) => output.toLowerCase().includes(value.toLowerCase())).forEach((value) => failures.push(`Contains forbidden text: ${value}`));
    if (requireCitations && !/\[[^\]]+\]/.test(output)) failures.push("No bracketed citation found.");
    const limit = Number(maxTokens);
    const costLimit = Number(maxCostUsd);
    if (maxTokens && Number.isFinite(limit) && (step.usage?.total_tokens ?? 0) > limit) failures.push(`Token limit exceeded: ${step.usage?.total_tokens ?? 0}/${limit}.`);
    const currentCost = usageCost(step.usage, pricing);
    if (maxCostUsd && Number.isFinite(costLimit) && currentCost > costLimit) failures.push(`Cost limit exceeded: ${formatCost(currentCost)}/${formatCost(costLimit)}.`);
    const assertions: OutputAssertions = {
      requiredText: required,
      forbiddenText: forbidden,
      requireJson,
      requireCitations,
      maxTokens: maxTokens && Number.isFinite(limit) ? limit : null,
      maxCostUsd: maxCostUsd && Number.isFinite(costLimit) ? costLimit : null,
    };
    const result = { passed: failures.length === 0, failures, evaluatedAt: Date.now() };
    useTimeTravelStore.getState().setStepAssertions(assertions, result);
    const traceId = useTimeTravelStore.getState().liveSession?.traceId;
    if (traceId) {
      void fetch(`/api/v1/traces/${encodeURIComponent(traceId)}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, cursor_index: step.cursor, review_note: step.reviewNote ?? null, review_verdict: step.reviewVerdict ?? null, assertions, assertion_result: result, updated_at: new Date().toISOString() }),
      });
    }
    const versions = useTimeTravelStore.getState().liveSession?.promptVersions
      .filter((version) => version.cursor === step.cursor) ?? [];
    for (const version of versions) {
      const updated = { ...version, assertions, assertionResult: result };
      useTimeTravelStore.getState().updatePromptVersion(version.id, { assertions, assertionResult: result });
      if (traceId) {
        void persistPromptVersion(traceId, updated);
        if (updated.status === "completed") void persistPromptVersionResult(updated.id, updated);
      }
    }
    setAssertionResults(failures.length ? failures : ["All configured checks passed."]);
  };

  const saveReview = (verdict: "accepted" | "rejected") => {
    useTimeTravelStore.getState().setStepReview(reviewNote, verdict);
    const traceId = useTimeTravelStore.getState().liveSession?.traceId;
    const versions = useTimeTravelStore.getState().liveSession?.promptVersions
      .filter((version) => version.cursor === step.cursor) ?? [];
    for (const version of versions) {
      const updated = { ...version, reviewNote, reviewVerdict: verdict };
      useTimeTravelStore.getState().updatePromptVersion(version.id, { reviewNote, reviewVerdict: verdict });
      if (traceId && updated.status === "completed") void persistPromptVersionResult(updated.id, updated);
    }
    if (traceId) {
      void fetch(`/api/v1/traces/${encodeURIComponent(traceId)}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, cursor_index: step.cursor, review_note: reviewNote, review_verdict: verdict, assertions: step.assertions ?? {}, assertion_result: step.assertionResult ?? {}, updated_at: new Date().toISOString() }),
      });
    }
  };

  const runCustomEvaluator = async () => {
    if (!selectedEvaluator) return;
    try {
      const response = await fetch("/api/v1/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selectedEvaluator, result: step.result ?? "", context: { cursor: step.cursor } }),
      });
      const body = await response.json() as { passed?: boolean; detail?: string };
      setEvaluatorResult(`${body.passed ? "passed" : "failed"}${body.detail ? ` · ${body.detail}` : ""}`);
      const traceId = useTimeTravelStore.getState().liveSession?.traceId;
      const versions = useTimeTravelStore.getState().liveSession?.promptVersions
        .filter((version) => version.cursor === step.cursor) ?? [];
      for (const version of versions) {
        const evaluatorNames = Array.from(new Set([...(version.evaluatorNames ?? []), selectedEvaluator]));
        const evaluatorResults = {
          ...(version.evaluatorResults ?? {}),
          [selectedEvaluator]: { passed: response.ok && body.passed === true, detail: body.detail },
        };
        const updated = { ...version, evaluatorNames, evaluatorResults };
        useTimeTravelStore.getState().updatePromptVersion(version.id, { evaluatorNames, evaluatorResults });
        if (traceId) {
          void persistPromptVersion(traceId, updated);
          if (updated.status === "completed") void persistPromptVersionResult(updated.id, updated);
        }
      }
    } catch (e) {
      setEvaluatorResult(`error · ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleContinue = async () => {
    setBusy(true);
    setError(null);
    try {
      const session = useTimeTravelStore.getState().liveSession;
      await continueFromSavedState(sessionId, session?.runnerRef ?? "agent", step.cursor);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-white/10 bg-white/[0.02] px-5 py-4">
        {isExecuting ? (
          <Loader2 className="size-5 animate-spin text-violet-300" />
        ) : isCompleted ? (
          <Sparkles className="size-5 text-emerald-500" />
        ) : (
          <PauseCircle className="size-5 text-amber-500" />
        )}
        <span className="text-base font-semibold">
          {isExecuting
            ? `Step #${stepNumber} · ${isTool ? "Running tool" : "Thinking"}`
            : isCompleted
              ? `Step #${stepNumber} result`
              : `Preparing step #${stepNumber}`}
        </span>
        <Badge variant="secondary" className={meta.className}>{meta.label}</Badge>
        {modelDuration && (
          <span className="ml-auto font-mono text-xs text-muted-foreground">
            {isTool ? "Tool time" : "LLM time"} {modelDuration}
          </span>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 space-y-4 overflow-y-auto p-5">
        {error && (
          <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        {/* The model's RESPONSE — shown once the step has executed (verify loop) */}
        {isCompleted && !editing && (
          <>
            {step.reasoning && (
              <Card className="border-violet-400/20 bg-violet-500/[0.04] shadow-none">
                <Collapsible open={thinkingOpen} onOpenChange={setThinkingOpen}>
                  <CollapsibleTrigger className="flex w-full items-center gap-2 px-4 py-3 text-left">
                    <Brain className="size-4 text-violet-300" />
                    <span className="text-xs font-medium uppercase tracking-wide text-violet-200">Thinking</span>
                    <span className="ml-auto flex items-center gap-1 text-xs italic text-slate-400">
                      Provider reasoning
                      <ChevronDown className={`size-3.5 transition-transform ${thinkingOpen ? "rotate-180" : ""}`} />
                    </span>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent className="pt-0">
                      <pre className="max-h-60 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-violet-400/10 bg-black/20 p-3 text-xs italic leading-relaxed text-slate-300">
                        {step.reasoning}
                      </pre>
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
            )}
            <Card className="border-emerald-400/25 bg-emerald-500/[0.06] shadow-none">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-emerald-700 dark:text-emerald-400">
                  <Sparkles className="size-3.5" /> {isTool ? "Tool result" : "Final response"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="whitespace-pre-wrap break-words rounded-md border border-white/5 bg-black/20 p-3 text-xs leading-relaxed">
                  {step.result || "(No final response returned.)"}
                </pre>
                {!isTool && <TokenSummary usage={step.usage} pricing={pricing} />}
              </CardContent>
            </Card>
          </>
        )}

        {isExecuting && (
          <Card className="border-violet-400/20 bg-violet-500/[0.05] shadow-none">
            <CardContent className="py-4">
              <div className="flex items-center gap-3 text-sm text-slate-300">
                <span className="relative flex size-8 shrink-0 items-center justify-center rounded-md bg-violet-400/10">
                  <Brain className="size-4 text-violet-300" />
                  <span className="absolute -right-1 -top-1 size-2 animate-pulse rounded-full bg-violet-300" />
                </span>
                <div>
                  <p className="font-medium text-slate-100">{isTool ? "Tool is running" : "Gemma is thinking"}</p>
                  <p className="mt-0.5 text-xs text-slate-400">
                    {!isTool && step.reasoning
                      ? "Reasoning streams in live — the final response will follow automatically."
                      : isTool
                        ? "Its result will appear here automatically."
                        : "The final response will appear here automatically."}
                  </p>
                </div>
              </div>
              {!isTool && step.reasoning && (
                <pre
                  ref={liveReasoningRef}
                  className="mt-3 max-h-56 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-violet-400/10 bg-black/20 p-3 text-xs italic leading-relaxed text-slate-300"
                >
                  {step.reasoning}
                  <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-violet-300 align-middle" />
                </pre>
              )}
            </CardContent>
          </Card>
        )}

        {/* The pending call — messages/model/params (shown when editing or no result yet) */}
        {(!isCompleted || editing) && step.payload.model && (
          <Card className="shadow-none">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                <MessageSquare className="size-3.5" /> Model
              </CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="whitespace-pre-wrap break-all bg-muted/40 p-2 font-mono text-xs">
                {step.payload.model}
              </pre>
            </CardContent>
          </Card>
        )}

        {!isTool && !editing && context && (
          <Card className="border-sky-400/20 bg-sky-500/[0.035] shadow-none">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-sky-200">
                <SlidersHorizontal className="size-3.5" /> Context inspector
                <span className="ml-auto font-mono normal-case text-slate-400">
                  ~{formatTokens(context.total)} prompt tokens
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-2 sm:grid-cols-2">
              {context.parts.map((part) => (
                <div key={part.label} className="flex items-center justify-between gap-3 rounded-md border border-sky-300/10 bg-black/10 px-3 py-2 text-xs">
                  <span className="text-slate-300">{part.label}</span>
                  <span className="font-mono text-sky-200">~{formatTokens(part.tokens)}</span>
                </div>
              ))}
              <p className="sm:col-span-2 text-xs text-slate-500">
                Local estimate from the exact request payload; use provider usage when your endpoint reports it.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Tool input (read-only outside edit mode) */}
        {isTool && !editing && (
          <Card className="shadow-none">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                <Wrench className="size-3.5" /> {step.payload.name ?? "Tool call"}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <ToolJson label="Arguments" value={step.payload.args ?? []} />
              <ToolJson label="Keyword arguments" value={step.payload.kwargs ?? {}} />
            </CardContent>
          </Card>
        )}

        {/* Edit mode */}
        {editing && (
          <Card className="shadow-none">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {isTool ? "Edit tool inputs" : step.restored ? "Edit prompt for a new branch" : "Edit messages (JSON)"}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {step.restored && !isTool && (
                <p className="text-xs text-cyan-200">
                  The original restored result stays preserved. Applying this edit runs the changed prompt on a new divergent branch.
                </p>
              )}
              {isTool ? (
                <>
                  <JsonEditor label="Arguments (JSON array)" value={editedArgs} onChange={setEditedArgs} />
                  <JsonEditor label="Keyword arguments (JSON object)" value={editedKwargs} onChange={setEditedKwargs} />
                </>
              ) : (
                <>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">Model override</label>
                <Input
                  value={editedModel}
                  onChange={(e) => setEditedModel(e.target.value)}
                  placeholder="(leave unchanged)"
                  className="h-8 font-mono text-xs"
                />
              </div>
              <Textarea
                value={editedMessages}
                onChange={(e) => setEditedMessages(e.target.value)}
                className="min-h-[200px] font-mono text-xs"
                spellCheck={false}
              />
                </>
              )}
            </CardContent>
          </Card>
        )}

        {/* Messages (read-only, when not editing and no result or has result) */}
        {!editing && step.payload.messages && step.payload.messages.length > 0 && (
          <Card className="shadow-none">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                <MessageSquare className="size-3.5" /> Messages sent
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {step.payload.messages.map((m, i) => (
                <MessageRow key={i} message={m} tokenEstimate={estimateTokens(m)} />
              ))}
            </CardContent>
          </Card>
        )}

        {/* Params */}
        {!editing && step.payload.params && Object.keys(step.payload.params).length > 0 && (
          <Card className="shadow-none">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                <SlidersHorizontal className="size-3.5" /> Params
              </CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="overflow-x-auto bg-muted/40 p-2 font-mono text-xs">
                {JSON.stringify(step.payload.params, null, 2)}
              </pre>
            </CardContent>
          </Card>
        )}

        {isCompleted && !readOnly && (
          <Card className="shadow-none">
            <CardHeader className="pb-2"><CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Expected output checks</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-xs">
              <Input value={requiredText} onChange={(event) => setRequiredText(event.target.value)} placeholder="Required text, comma separated" className="h-8 text-xs" />
              <Input value={forbiddenText} onChange={(event) => setForbiddenText(event.target.value)} placeholder="Forbidden text, comma separated" className="h-8 text-xs" />
              <div className="flex flex-wrap gap-4 text-muted-foreground">
                <label className="flex items-center gap-2"><input type="checkbox" checked={requireJson} onChange={(event) => setRequireJson(event.target.checked)} /> Valid JSON</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={requireCitations} onChange={(event) => setRequireCitations(event.target.checked)} /> Citations</label>
                <label className="flex items-center gap-2">Max tokens <Input value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} type="number" min="0" className="h-7 w-20 text-xs" /></label>
                <label className="flex items-center gap-2">Max cost $ <Input value={maxCostUsd} onChange={(event) => setMaxCostUsd(event.target.value)} type="number" min="0" step="0.0001" className="h-7 w-24 text-xs" /></label>
              </div>
              <Button size="sm" variant="outline" onClick={runAssertions}>Run checks</Button>
              {assertionResults && <div className={`rounded-md px-3 py-2 ${assertionResults[0] === "All configured checks passed." ? "bg-emerald-500/10 text-emerald-300" : "bg-rose-500/10 text-rose-200"}`}>{assertionResults.map((result) => <p key={result}>{result}</p>)}</div>}
            </CardContent>
          </Card>
        )}

        {isCompleted && !readOnly && !isTool && evaluatorNames.length > 0 && (
          <Card className="shadow-none">
            <CardHeader className="pb-2"><CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Registered evaluators</CardTitle></CardHeader>
            <CardContent className="space-y-2">
              <div className="flex gap-2">
                <select value={selectedEvaluator} onChange={(event) => setSelectedEvaluator(event.target.value)} className="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 text-xs">
                  <option value="">Choose a registered evaluator</option>
                  {evaluatorNames.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
                <Button size="sm" variant="outline" onClick={() => void runCustomEvaluator()} disabled={!selectedEvaluator}>Run</Button>
              </div>
              {evaluatorResult && <p className="rounded-md bg-muted/40 px-3 py-2 text-xs text-slate-300">{selectedEvaluator}: {evaluatorResult}</p>}
            </CardContent>
          </Card>
        )}

        {isCompleted && !readOnly && (
          <Card className="shadow-none">
            <CardHeader className="pb-2"><CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Review decision</CardTitle></CardHeader>
            <CardContent className="space-y-2"><Textarea value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} placeholder="Why is this response accepted or rejected?" className="min-h-20 text-xs" /><div className="flex gap-2"><Button size="sm" onClick={() => saveReview("accepted")} className="bg-emerald-500 text-emerald-950 hover:bg-emerald-400">Accept</Button><Button size="sm" variant="outline" onClick={() => saveReview("rejected")}>Reject</Button></div>{step.reviewVerdict && <p className="text-xs text-slate-400">Marked {step.reviewVerdict}.</p>}</CardContent>
          </Card>
        )}

        {/* Tools */}
        {!editing && step.payload.tools && step.payload.tools.length > 0 && (
          <Card className="shadow-none">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                <Wrench className="size-3.5" /> Tools ({step.payload.tools.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {step.payload.tools.map((t, i) => (
                <pre key={i} className="overflow-x-auto bg-muted/40 p-2 font-mono text-xs">
                  {JSON.stringify(t, null, 2)}
                </pre>
              ))}
            </CardContent>
          </Card>
        )}

        {!editing && isTool && step.phase === "queued" && (
          <Card className="border-amber-400/20 bg-amber-500/[0.04] shadow-none">
            <CardHeader className="pb-2"><CardTitle className="text-xs font-medium uppercase tracking-wide text-amber-200">Safe tool controls</CardTitle></CardHeader>
            <CardContent className="space-y-2">
              <Textarea value={mockResult} onChange={(event) => setMockResult(event.target.value)} className="min-h-20 font-mono text-xs" placeholder='{"items":[]}' />
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="outline" onClick={() => void handleMockTool()} disabled={busy}>Use mock result</Button>
                <Button size="sm" variant="outline" onClick={() => void decide("skip")} disabled={busy}>Skip tool</Button>
              </div>
              <div className="mt-2 space-y-1">
                <label className="text-xs text-muted-foreground">Reject reason (optional — returned to agent)</label>
                <Textarea
                  id="reject-reason-input"
                  value={rejectReason}
                  onChange={(e) => setRejectReason(e.target.value)}
                  className="min-h-16 text-xs font-mono"
                  placeholder="e.g. This action is unsafe; choose a read-only alternative."
                />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void handleRejectTool()}
                  disabled={busy}
                  className="border-rose-400/40 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20"
                  title="Return a structured rejection result to the agent without calling the live tool"
                >
                  Reject tool call
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Decision footer — reframed for the verify loop */}
      <div className="flex flex-wrap items-center gap-2 border-t border-white/10 bg-black/10 px-5 py-4">
        {readOnly ? (
          <>
            <span className="text-xs text-cyan-200">Viewing captured step data. No model or tool call will be made.</span>
            {onReturnToCurrent && (
              <Button size="sm" variant="outline" onClick={onReturnToCurrent} className="ml-auto">
                <CornerUpLeft className="mr-1.5 size-4" /> Back to current step
              </Button>
            )}
          </>
        ) : step.restored && !editing ? (
          <>
            <span className="text-xs text-cyan-200">Saved step restored locally. The original result is preserved; branching runs only new work.</span>
            {canTimeTravel && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => void handleTimeTravel()}
                disabled={busy}
                className="ml-auto border-sky-400/40 bg-sky-500/15 text-sky-200 hover:bg-sky-500/25 hover:text-white"
              >
                <CornerUpLeft className="mr-1.5 size-4" /> Step Back
              </Button>
            )}
            {isCompleted && !isTool && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => setEditing(true)}
                disabled={busy}
                className="border-violet-400/40 bg-violet-500/20 text-violet-100 hover:bg-violet-500/30"
                title="Edit the restored prompt and run it on a new branch"
              >
                <Pencil className="mr-1.5 size-4" /> Edit Prompt &amp; Branch
              </Button>
            )}
            {canStepForward && (
              <Button
                size="sm"
                onClick={handleStepForward}
                disabled={busy}
                className={canTimeTravel ? "bg-emerald-500 text-emerald-950 hover:bg-emerald-400" : "ml-auto bg-emerald-500 text-emerald-950 hover:bg-emerald-400"}
              >
                <PlayCircle className="mr-1.5 size-4" /> Step Forward
              </Button>
            )}
            {!canStepForward && (
              <Button
                size="sm"
                onClick={() => void handleContinue()}
                disabled={busy}
                className="ml-auto bg-emerald-500 text-emerald-950 hover:bg-emerald-400"
                title="Resume from this saved step; only new work can call the agent"
              >
                <PlayCircle className="mr-1.5 size-4" /> Continue From Here
              </Button>
            )}
          </>
        ) : isTool && step.phase === "queued" && !editing ? (
          <>
            <span className="text-xs text-slate-400">Review the arguments before this tool can run.</span>
            <Button size="sm" onClick={() => void decide("approve")} disabled={busy} className="ml-auto bg-emerald-500 text-emerald-950 hover:bg-emerald-400">
              <PlayCircle className="mr-1.5 size-4" /> Run Tool
            </Button>
            <Button size="sm" variant="outline" onClick={() => void continueUntilNextLlm(sessionId)} disabled={busy} title="Run this tool and any following tools until the next LLM call">
              <PlayCircle className="mr-1.5 size-4" /> Step over tool
            </Button>
            <Button size="sm" variant="outline" onClick={() => void handlePauseAfterCurrent()} disabled={busy} title="Run this tool, then pause before the next intercepted call">
              <PauseCircle className="mr-1.5 size-4" /> Run once, then pause
            </Button>
            <Button size="sm" variant="outline" onClick={() => void handleRunUntilBreakpoint()} disabled={busy} title="Auto-approve all steps until a breakpoint fires">
              <PlayCircle className="mr-1.5 size-4" /> Run to breakpoint
            </Button>
            <Button size="sm" variant="outline" onClick={() => setEditing(true)} disabled={busy}>
              <Pencil className="mr-1.5 size-4" /> Edit arguments
            </Button>
            <Button size="sm" onClick={() => void decide("stop")} disabled={busy} className="bg-rose-500/90 text-white hover:bg-rose-500">
              <Square className="mr-1.5 size-4 fill-current" /> Stop
            </Button>
          </>
        ) : isExecuting ? (
          <>
            <span className="flex items-center gap-2 text-xs text-slate-400">
              <Loader2 className="size-3.5 animate-spin text-violet-300" />
              Waiting for the model response
            </span>
            <Button
              size="sm"
              onClick={() => void decide("stop")}
              disabled={busy}
              className="ml-auto bg-rose-500/90 text-white hover:bg-rose-500"
            >
              <Square className="size-4 fill-current" /> Stop Execution
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => void handleArmPauseAfterCurrent()}
              disabled={busy}
              title="Let this in-flight call finish, then pause before the next intercepted call"
            >
              <PauseCircle className="mr-1.5 size-4" /> Pause after this call
            </Button>
          </>
        ) : !editing && isCompleted ? (
          <>
            {/* 1. Green: Next Step (Approve) */}
            <Button
              size="sm"
              onClick={() => void decide("approve")}
              disabled={busy}
              className="min-w-40 bg-emerald-500 text-emerald-950 hover:bg-emerald-400 font-semibold shadow-sm"
              title="Accept this response and run the next agent step"
            >
              <PlayCircle className="mr-1.5 size-4" /> Next Step
            </Button>

            <Button
              size="sm"
              variant="outline"
              onClick={() => void continueUntilNextLlm(sessionId)}
              disabled={busy}
              title="Approve this response and pause at the next LLM call"
            >
              <PlayCircle className="mr-1.5 size-4" /> Continue to next LLM
            </Button>

            <Button
              size="sm"
              variant="outline"
              onClick={() => void handlePauseAfterCurrent()}
              disabled={busy}
              title="Approve this response, then pause before the next intercepted call"
            >
              <PauseCircle className="mr-1.5 size-4" /> Next, then pause
            </Button>

            {isTool && (
              <Button size="sm" variant="outline" onClick={() => void handleRetryTool()} disabled={busy} title="Start a new branch and run this tool again">
                <PlayCircle className="mr-1.5 size-4" /> Retry tool
              </Button>
            )}

            {/* 2. Blue: Step Back / Agent Timetravel */}
            <Button
              size="sm"
              variant="outline"
              onClick={() => void handleTimeTravel()}
              disabled={busy || !canTimeTravel}
              className="border-sky-400/40 bg-sky-500/15 text-sky-200 hover:bg-sky-500/25 hover:text-white font-medium shadow-sm"
              title="Restore the previous captured step without rerunning the agent"
            >
              <CornerUpLeft className="mr-1.5 size-4" /> Step Back / Agent Timetravel
            </Button>

            {!isTool && (
              <Button
                size="sm"
                onClick={() => setEditing(true)}
                disabled={busy}
                className="border border-violet-400/40 bg-violet-500/20 text-violet-100 hover:bg-violet-500/30 font-medium shadow-sm"
                title="Edit system prompt or user query"
              >
                <Pencil className="mr-1.5 size-4" /> Edit Prompt &amp; Run
              </Button>
            )}

            {/* 4. Red: Stop Execution */}
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  size="sm"
                  disabled={busy}
                  className="ml-auto bg-rose-500/90 hover:bg-rose-500 text-white font-medium shadow-sm"
                >
                  <Square className="mr-1.5 size-4 fill-current" /> Stop Execution
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Stop the agent run?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This terminates the session. The captured spans under this
                    branch are preserved, but the agent will not continue.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => void decide("stop")}
                    className="bg-destructive text-white hover:bg-destructive/90"
                  >
                    Stop run
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </>
        ) : (
          <>
            <Button size="sm" onClick={() => void handleApplyEdit()} disabled={busy}>
              <PlayCircle className="mr-1 size-4" /> {step.restored ? "Apply edit & run new branch" : "Apply edit & continue"}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setEditing(false)} disabled={busy}>
              <CornerUpLeft className="mr-1 size-4" /> Cancel
            </Button>
          </>
        )}
      </div>
    </div>
  );
}

function TokenSummary({
  usage,
  pricing,
}: {
  usage: PausedStep["usage"];
  pricing: PricingProfile;
}) {
  if (!usage) {
    return <p className="mt-3 text-xs text-slate-400">Token usage was not reported by this provider.</p>;
  }
  const cost = usageCost(usage, pricing);
  return (
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-emerald-300/10 pt-3 font-mono text-xs text-slate-300">
      <span>Input {formatTokens(usage.input_tokens)}</span>
      <span>Thinking {formatTokens(usage.thinking_tokens)}</span>
      <span>Final {formatTokens(usage.final_tokens)}</span>
      <span>Output {formatTokens(usage.output_tokens)}</span>
      <span>Total {formatTokens(usage.total_tokens)}</span>
      <span className="text-emerald-300">Cost {formatCost(cost)}</span>
      {usage.estimated && <span className="text-amber-300">Estimated locally</span>}
    </div>
  );
}

function usageCost(usage: PausedStep["usage"], pricing: PricingProfile): number {
  if (!usage) return 0;
  const cachedInput = usage.cached_input_tokens ?? 0;
  const uncachedInput = Math.max(0, usage.input_tokens - cachedInput);
  return ((uncachedInput * pricing.inputPerMillion)
    + (cachedInput * pricing.cachedInputPerMillion)
    + (usage.final_tokens * pricing.outputPerMillion)
    + (usage.thinking_tokens * pricing.thinkingPerMillion)) / 1_000_000;
}

function formatTokens(tokens: number): string {
  return new Intl.NumberFormat("en-US").format(tokens);
}

function formatCost(cost: number): string {
  return cost === 0 ? "$0.00" : `$${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)}`;
}

function ToolJson({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">{label}</p>
      <pre className="overflow-x-auto rounded-md bg-muted/40 p-2 font-mono text-xs">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function JsonEditor({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-muted-foreground">{label}</label>
      <Textarea value={value} onChange={(event) => onChange(event.target.value)} className="min-h-28 font-mono text-xs" spellCheck={false} />
    </div>
  );
}

function MessageRow({ message, tokenEstimate }: { message: unknown; tokenEstimate: number }) {
  if (typeof message === "string") {
    return <pre className="whitespace-pre-wrap bg-muted/40 p-2 font-mono text-xs">{message}</pre>;
  }
  if (message !== null && typeof message === "object") {
    const m = message as { role?: unknown; content?: unknown };
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          {typeof m.role === "string" && (
            <Badge variant="outline" className="text-[10px] uppercase">{m.role}</Badge>
          )}
          <span className="font-mono text-[10px] text-slate-500">~{formatTokens(tokenEstimate)} tokens</span>
        </div>
        <pre className="whitespace-pre-wrap bg-muted/40 p-2 font-mono text-xs">
          {typeof m.content === "string"
            ? m.content
            : JSON.stringify(m.content ?? message, null, 2)}
        </pre>
      </div>
    );
  }
  return <pre className="whitespace-pre-wrap bg-muted/40 p-2 font-mono text-xs">{String(message)}</pre>;
}

type ContextPart = { label: string; tokens: number };

function contextBreakdown(payload: PausedStep["payload"]): { parts: ContextPart[]; total: number } {
  const buckets = new Map<string, number>();
  for (const message of payload.messages ?? []) {
    const role = messageRole(message);
    const serialized = JSON.stringify(message).toLowerCase();
    const label = role === "system"
      ? "System instructions"
      : role === "user"
        ? "User input"
        : role === "assistant"
          ? "Previous agent output"
          : role === "tool"
            ? "Tool outputs"
            : serialized.includes("retrieval") || serialized.includes("rag") || serialized.includes("memory")
              ? "Memory / retrieved context"
            : "Other messages";
    buckets.set(label, (buckets.get(label) ?? 0) + estimateTokens(message));
  }
  if (payload.tools?.length) buckets.set("Tool definitions", estimateTokens(payload.tools));
  if (payload.params && Object.keys(payload.params).length) buckets.set("Model parameters", estimateTokens(payload.params));
  const parts = Array.from(buckets, ([label, tokens]) => ({ label, tokens }));
  return { parts, total: parts.reduce((total, part) => total + part.tokens, 0) };
}

/** Keep the exact provider kwargs together so a variant can be reproduced. */
function parameterSnapshot(payload: PausedStep["payload"]): Record<string, unknown> {
  return payload.params ? JSON.parse(JSON.stringify(payload.params)) as Record<string, unknown> : {};
}

function messageRole(message: unknown): string {
  if (message !== null && typeof message === "object") {
    const role = (message as { role?: unknown }).role;
    if (typeof role === "string") return role;
  }
  return "other";
}

function estimateTokens(value: unknown): number {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text ? Math.ceil(text.length / 4) : 0;
}
