import assert from "node:assert/strict";
import test from "node:test";
import { useTimeTravelStore } from "./store";

const pausedStep = {
  cursor: 1,
  kind: "llm",
  payload: {},
  pausedAt: 0,
  completedAt: null,
  result: null,
  reasoning: null,
  usage: null,
  phase: "queued" as const,
};

function startSession(): void {
  useTimeTravelStore.getState().startLiveSession("session", "trace", "branch", "runner");
  useTimeTravelStore.getState().pauseAtStep(pausedStep);
}

test.afterEach(() => {
  useTimeTravelStore.getState().clearLiveSession();
});

test("completeStep preserves streamed reasoning when the final result has none", () => {
  startSession();
  useTimeTravelStore.getState().appendStepReasoning(1, "provider reasoning");

  useTimeTravelStore.getState().completeStep(1, "tool result");

  assert.equal(useTimeTravelStore.getState().liveSession?.pausedStep?.reasoning, "provider reasoning");
  assert.equal(useTimeTravelStore.getState().liveSession?.pausedStep?.result, "tool result");
});

test("explicit reasoning in the final result replaces streamed reasoning", () => {
  startSession();
  useTimeTravelStore.getState().appendStepReasoning(1, "streamed reasoning");

  useTimeTravelStore.getState().completeStep(1, "<think>final reasoning</think>answer");

  assert.equal(useTimeTravelStore.getState().liveSession?.pausedStep?.reasoning, "final reasoning");
  assert.equal(useTimeTravelStore.getState().liveSession?.pausedStep?.result, "answer");
});

test("completePromptVersion preserves existing reasoning without a final reasoning block", () => {
  startSession();
  useTimeTravelStore.getState().addPromptVersion({
    id: "version",
    cursor: 1,
    createdAt: 0,
    baseMessages: [],
    messages: [],
    baseModel: "model",
    model: "model",
    status: "running",
    result: null,
    usage: null,
    reasoning: "streamed reasoning",
  });

  useTimeTravelStore.getState().completePromptVersion(1, "answer");

  assert.equal(useTimeTravelStore.getState().liveSession?.promptVersions[0]?.reasoning, "streamed reasoning");
});
