import {useCallback, useEffect, useState} from "react";

import {createRunRequest, validateProblem} from "./protocol";
import type {LabProblem, LabResult} from "./protocol";
import {acquireRuntimeReference, cancelRun, runOnRuntime} from "./runtimeClient";

export type LabState = "idle" | "loading" | "running" | "complete" | "error" | "cancelled";

export interface LabRunner {
  cancel: () => void;
  error: string | null;
  progress: number;
  result: LabResult | null;
  run: (problem: LabProblem) => void;
  stage: string;
  state: LabState;
  /** Whether a warmed runtime is held, so the next run skips the cold start. */
  warm: boolean;
}

export function useLabRunner(): LabRunner {
  const [state, setState] = useState<LabState>("idle");
  const [stage, setStage] = useState("Ready to run");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<LabResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warm, setWarm] = useState(false);

  // The shared runtime is a tab-wide singleton owned by runtimeClient.ts; this
  // instance only registers its interest in it on mount and releases that
  // interest on unmount. The worker, once started, is terminated once the
  // last such reference is released -- for the single-consumer case (this
  // hook mounted once, on /lab) that reproduces today's unmount-terminates
  // behaviour exactly.
  useEffect(() => acquireRuntimeReference(), []);

  const cancel = useCallback((): void => {
    if (!cancelRun()) return;
    setWarm(false);
    setState("cancelled");
    setStage("Run cancelled; the isolated worker was terminated");
    setProgress(0);
  }, []);

  const run = useCallback((problem: LabProblem): void => {
    const validationError = validateProblem(problem);
    if (validationError !== null) {
      setError(validationError);
      setState("error");
      return;
    }
    setResult(null);
    setError(null);
    setProgress(0.02);
    const request = createRunRequest(problem);
    setState("loading");
    runOnRuntime(request, {
      onReuse: () => setStage("Reusing the warm browser runtime"),
      onStart: () => setStage("Starting isolated browser worker"),
      onMessage: (event) => {
        if (event.stage !== undefined) setStage(event.stage);
        if (event.progress !== undefined) setProgress(event.progress);
        if (event.type === "ready" || event.type === "progress") setState(event.type === "ready" || (event.progress ?? 0) >= .65 ? "running" : "loading");
        if (event.type === "result" && event.result !== undefined) {
          setResult(event.result);
          setProgress(1);
          setState("complete");
          // The worker is deliberately kept alive: it holds the warmed Pyodide
          // runtime, and re-running with a different bin budget is the normal
          // interaction. Cancelling still terminates it.
          setWarm(true);
        }
        if (event.type === "error") {
          setError(event.message ?? "The browser runtime failed without a diagnostic.");
          setState("error");
          setWarm(false);
        }
      },
      onWorkerError: (message) => {
        setError(message);
        setState("error");
      }
    });
  }, []);

  return {cancel, error, progress, result, run, stage, state, warm};
}
