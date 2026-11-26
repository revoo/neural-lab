// authored-by: Claude Opus 5.5
/**
 * The learner's live run, shared by the lesson's widgets: the training widget writes checkpoints,
 * the "open it up" widgets can then read the learner's own model instead of the recorded one.
 */
import { useSyncExternalStore } from 'react';
import type { RunSpec } from '../../lib/compute/gptRun';
import type { LiveCheckpoint } from './tinyGpt';

export type LiveRun = { spec: RunSpec; label: string; tier: string; checkpoints: LiveCheckpoint[]; done: boolean };

let run: LiveRun | null = null;
const listeners = new Set<() => void>();

export const runStore = {
  get: () => run,
  set(next: LiveRun | null) {
    run = next;
    listeners.forEach((l) => l());
  },
  subscribe(l: () => void) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
};

export const useLiveRun = () => useSyncExternalStore(runStore.subscribe, runStore.get, runStore.get);
