import {afterEach, beforeEach, describe, expect, it, vi} from "vitest";

import type {LabEvent, LabRunRequest} from "../src/lab/protocol";
import {PROTOCOL_VERSION} from "../src/lab/protocol";
import {
  acquireRuntimeReference,
  cancelRun,
  runOnRuntime
} from "../src/lab/runtimeClient";
import type {RuntimeCallbacks} from "../src/lab/runtimeClient";

/**
 * A stand-in for the DOM `Worker` the client constructs.
 *
 * The real class is not available under jsdom in a form worth exercising --
 * this is the whole reason runtimeClient's own state machine has no coverage
 * today, only the end-to-end Playwright suite. Each instance records what was
 * posted to it and lets the test dispatch `message`/`error` back, standing in
 * for the browser delivering an event from `lab.worker.ts`.
 */
class MockWorker {
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessage: ((message: MessageEvent<unknown>) => void) | null = null;
  posted: unknown[] = [];
  terminated = false;

  postMessage(payload: unknown): void {
    this.posted.push(payload);
  }

  terminate(): void {
    this.terminated = true;
  }

  emit(event: LabEvent): void {
    this.onmessage?.(new MessageEvent("message", {data: event}));
  }

  fail(message: string, filename = "", lineno = 0): void {
    this.onerror?.(new ErrorEvent("error", {filename, lineno, message}));
  }
}

let instances: MockWorker[] = [];

function requestFor(runId: string): LabRunRequest {
  return {
    protocolVersion: PROTOCOL_VERSION,
    type: "run",
    runId,
    problem: {
      scores: [[0], [1]],
      weights: [1, 1],
      nBins: 2,
      solver: "d_exchange",
      seed: 1
    }
  };
}

function callbacks(): RuntimeCallbacks & {
  errors: string[];
  messages: LabEvent[];
  reused: number;
  started: number;
} {
  const record = {
    errors: [] as string[],
    messages: [] as LabEvent[],
    reused: 0,
    started: 0
  };
  return Object.assign(record, {
    onMessage: (event: LabEvent) => record.messages.push(event),
    onReuse: () => {
      record.reused += 1;
    },
    onStart: () => {
      record.started += 1;
    },
    onWorkerError: (message: string) => record.errors.push(message)
  });
}

beforeEach(() => {
  instances = [];
  vi.stubGlobal(
    "Worker",
    vi.fn(function ConstructMockWorker() {
      const instance = new MockWorker();
      instances.push(instance);
      return instance;
    })
  );
});

afterEach(() => {
  // Every test starts from zero references and no worker: releasing whatever
  // this test acquired keeps the module-level singleton clean for the next.
  cancelRun();
  vi.unstubAllGlobals();
});

describe("runtimeClient", () => {
  it("constructs no Worker on import or on acquiring a reference", () => {
    const release = acquireRuntimeReference();
    expect(instances).toHaveLength(0);
    release();
  });

  it("constructs exactly one Worker on the first run request", () => {
    const release = acquireRuntimeReference();
    runOnRuntime(requestFor("run-1"), callbacks());
    expect(instances).toHaveLength(1);
    release();
  });

  it("does not construct a second Worker for a second acquirer", () => {
    const releaseA = acquireRuntimeReference();
    const releaseB = acquireRuntimeReference();
    runOnRuntime(requestFor("run-1"), callbacks());
    expect(instances).toHaveLength(1);
    releaseA();
    releaseB();
  });

  it("does not terminate while a reference remains, and does terminate the last one", () => {
    const releaseA = acquireRuntimeReference();
    const releaseB = acquireRuntimeReference();
    runOnRuntime(requestFor("run-1"), callbacks());
    const [instance] = instances;
    if (instance === undefined) throw new Error("expected a worker to have been constructed");

    releaseA();
    expect(instance.terminated).toBe(false);

    releaseB();
    expect(instance.terminated).toBe(true);
  });

  it("ignores an event whose runId does not match the current run", () => {
    const release = acquireRuntimeReference();
    const first = callbacks();
    runOnRuntime(requestFor("run-1"), first);
    const [instance] = instances;
    if (instance === undefined) throw new Error("expected a worker to have been constructed");

    // Superseded by a second run before the first ever reports back.
    const second = callbacks();
    runOnRuntime(requestFor("run-2"), second);

    instance.emit({protocolVersion: PROTOCOL_VERSION, runId: "run-1", type: "result", progress: 1, result: {
      centers: [[0]], execution: "numpy/float64/cpu", labels: [0, 1], objective: 0, retention: 1
    }});
    expect(first.messages).toHaveLength(0);
    expect(second.messages).toHaveLength(0);

    instance.emit({protocolVersion: PROTOCOL_VERSION, runId: "run-2", type: "result", progress: 1, result: {
      centers: [[0]], execution: "numpy/float64/cpu", labels: [0, 1], objective: 0, retention: 1
    }});
    expect(second.messages).toHaveLength(1);
    release();
  });

  it("supersedes the first run with the second and reuses the same Worker instance", () => {
    const release = acquireRuntimeReference();
    const first = callbacks();
    runOnRuntime(requestFor("run-1"), first);
    expect(first.started).toBe(1);
    expect(instances).toHaveLength(1);

    const second = callbacks();
    runOnRuntime(requestFor("run-2"), second);
    expect(second.reused).toBe(1);
    expect(second.started).toBe(0);
    expect(instances).toHaveLength(1);

    const [instance] = instances;
    if (instance === undefined) throw new Error("expected a worker to have been constructed");
    expect(instance.posted).toHaveLength(2);
    release();
  });

  it("terminates on cancel, and a subsequent run constructs a fresh Worker", () => {
    const release = acquireRuntimeReference();
    runOnRuntime(requestFor("run-1"), callbacks());
    expect(instances).toHaveLength(1);

    expect(cancelRun()).toBe(true);
    expect(instances[0]?.terminated).toBe(true);
    // Nothing left to cancel a second time.
    expect(cancelRun()).toBe(false);

    runOnRuntime(requestFor("run-2"), callbacks());
    expect(instances).toHaveLength(2);
    expect(instances[1]).not.toBe(instances[0]);
    release();
  });

  it("terminates the worker and reports the formatted message when the worker itself fails", () => {
    const release = acquireRuntimeReference();
    const handlers = callbacks();
    runOnRuntime(requestFor("run-1"), handlers);
    const [instance] = instances;
    if (instance === undefined) throw new Error("expected a worker to have been constructed");

    instance.fail("boom", "lab.worker.js", 1);
    expect(instance.terminated).toBe(true);
    expect(handlers.errors).toEqual([
      "The local runtime could not start. boom (lab.worker.js:1) The verified fixture remains available."
    ]);

    // The next run starts a fresh worker rather than reusing the failed one.
    runOnRuntime(requestFor("run-2"), callbacks());
    expect(instances).toHaveLength(2);
    release();
  });

  it("terminates on an in-protocol error event from the worker", () => {
    const release = acquireRuntimeReference();
    const handlers = callbacks();
    runOnRuntime(requestFor("run-1"), handlers);
    const [instance] = instances;
    if (instance === undefined) throw new Error("expected a worker to have been constructed");

    instance.emit({protocolVersion: PROTOCOL_VERSION, runId: "run-1", type: "error", message: "solver diverged"});
    expect(handlers.messages).toHaveLength(1);
    expect(instance.terminated).toBe(true);
    release();
  });
});
