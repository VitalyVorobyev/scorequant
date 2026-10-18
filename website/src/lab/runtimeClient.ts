import {isLabEvent} from "./protocol";
import type {LabEvent, LabRunRequest} from "./protocol";

/**
 * The single browser worker for the whole tab.
 *
 * Two live-demo components mounted on the same page would otherwise each hold
 * their own Pyodide heap -- roughly 15 MB apiece, and an out-of-memory risk on
 * a phone. This module is the one place `new Worker(...)` may appear: every
 * `useLabRunner` instance is a subscriber to the runtime this module owns,
 * never a second owner of one.
 *
 * The module must be safe to import during server-side rendering, where
 * `Worker` does not exist. Nothing at module scope touches it -- the worker is
 * constructed lazily, only inside `runOnRuntime`, and only once a run is
 * actually requested.
 */

/** Callbacks a caller of `runOnRuntime` supplies for the run it is issuing. */
export interface RuntimeCallbacks {
  /** A message from the worker for this run, already filtered by `runId`. */
  onMessage: (event: LabEvent) => void;
  /** The runtime is reusing an already-running worker. */
  onReuse: () => void;
  /** The runtime is starting a fresh worker. */
  onStart: () => void;
  /** The worker itself failed (module evaluation, not the run protocol). */
  onWorkerError: (message: string) => void;
}

let worker: Worker | null = null;
let currentRunId: string | null = null;
let activeCallbacks: RuntimeCallbacks | null = null;
let refCount = 0;

/** Tear down the shared worker, if one exists. Safe to call when it does not. */
function terminate(): void {
  worker?.terminate();
  worker = null;
  currentRunId = null;
  activeCallbacks = null;
}

/**
 * Register this tab's interest in the shared runtime.
 *
 * Call once per mounted consumer (a `useLabRunner` instance); call the
 * returned function once on unmount. The worker, once started, is terminated
 * when the last reference is released -- this reproduces today's
 * unmount-terminates behaviour for the single-consumer case while making a
 * second concurrent consumer safe: the worker outlives whichever one of them
 * unmounts first.
 */
export function acquireRuntimeReference(): () => void {
  refCount += 1;
  let released = false;
  return (): void => {
    if (released) return;
    released = true;
    refCount = Math.max(0, refCount - 1);
    if (refCount === 0) terminate();
  };
}

/**
 * Cancel the in-flight run, if the shared worker exists.
 *
 * Returns whether a worker was actually terminated, so a caller with no
 * worker of its own knowledge (a fresh `useLabRunner` that never ran anything)
 * can no-op exactly as the pre-singleton `cancel()` did.
 */
export function cancelRun(): boolean {
  if (worker === null) return false;
  terminate();
  return true;
}

/**
 * Run `request` on the shared worker, creating it if none exists yet.
 *
 * A request while a worker already exists reuses it and supersedes whatever
 * run was in flight: `currentRunId` moves to this request's id, so any event
 * still arriving for the superseded run is dropped by the `runId` check in
 * the worker's `onmessage` below, exactly as it always was when that check
 * lived inline in the hook.
 */
export function runOnRuntime(request: LabRunRequest, callbacks: RuntimeCallbacks): void {
  currentRunId = request.runId;
  activeCallbacks = callbacks;
  const existing = worker;
  if (existing !== null) {
    callbacks.onReuse();
    existing.postMessage(request);
    return;
  }
  callbacks.onStart();
  // `name` is what gives the emitted chunk a stable name, which
  // docusaurus.config.ts needs in order to target it. See the note there.
  const created = new Worker(new URL("./lab.worker.ts", import.meta.url), {
    name: "lab-worker",
    type: "module"
  });
  worker = created;
  created.onmessage = (message: MessageEvent<unknown>): void => {
    if (!isLabEvent(message.data) || message.data.runId !== currentRunId) return;
    const event: LabEvent = message.data;
    activeCallbacks?.onMessage(event);
    // The worker is deliberately kept alive on every other event: it holds the
    // warmed Pyodide runtime, and re-running with a different bin budget is
    // the normal interaction. An "error" event is the exception.
    if (event.type === "error") terminate();
  };
  // A worker that fails while its module is evaluated reports through
  // onerror rather than through the protocol, so this is the only place the
  // cause is ever visible. Discarding it leaves "could not start" as the
  // whole diagnosis, which is indistinguishable from a missing runtime asset,
  // a bad MIME type and a syntax error the bundler let through.
  created.onerror = (event: ErrorEvent): void => {
    const where = event.filename === "" ? "" : ` (${event.filename}:${String(event.lineno)})`;
    const cause = event.message === "" ? "" : ` ${event.message}${where}`;
    const message = `The local runtime could not start.${cause} The verified fixture remains available.`;
    activeCallbacks?.onWorkerError(message);
    terminate();
  };
  created.postMessage(request);
}
