// authored-by: Claude Opus 5.5
/**
 * V2-WP3: picks the compute tier. In a browser with a working WebGPU adapter, heavy widgets use
 * `GpuMlpEngine`; everywhere else (Node, Vitest, headless CI without WebGPU, older browsers) they
 * use `CpuMlpEngine` on the main thread. The learner can force the CPU to compare the two live;
 * that choice is stored in localStorage under `nl-compute-v1`.
 */
import { useEffect, useState, useSyncExternalStore } from 'react';
import { CpuMlpEngine } from './cpuBackend';
import { getGpu, GpuMlpEngine } from './webgpuBackend';
import type { MlpEngine, MlpSpec } from './types';

export type ComputePref = 'auto' | 'cpu';
const KEY = 'nl-compute-v1';

function load(): ComputePref {
  try {
    return localStorage.getItem(KEY) === 'cpu' ? 'cpu' : 'auto';
  } catch {
    return 'auto';
  }
}
let pref: ComputePref = typeof localStorage === 'undefined' ? 'auto' : load();
const listeners = new Set<() => void>();
export const computePref = {
  get: () => pref,
  set(next: ComputePref) {
    pref = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* ignore */
    }
    listeners.forEach((l) => l());
  },
  subscribe(l: () => void) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
};
export const useComputePref = () => useSyncExternalStore(computePref.subscribe, computePref.get, computePref.get);

export type GpuStatus = { state: 'checking' } | { state: 'none' } | { state: 'ready'; name: string };

/** Whether this browser has a usable WebGPU adapter (checked once, shared by every widget). */
export function useGpuStatus(): GpuStatus {
  const [s, setS] = useState<GpuStatus>({ state: 'checking' });
  useEffect(() => {
    let live = true;
    getGpu().then((g) => live && setS(g ? { state: 'ready', name: g.name } : { state: 'none' }));
    return () => {
      live = false;
    };
  }, []);
  return s;
}

/** The engine a widget should use right now: WebGPU when available and not overridden, else the CPU. */
export async function createMlpEngine(spec: MlpSpec, p: ComputePref = computePref.get()): Promise<MlpEngine> {
  if (p === 'auto') {
    const g = await getGpu();
    if (g) return new GpuMlpEngine(spec, g.device);
  }
  return new CpuMlpEngine(spec);
}

/** Which tier `createMlpEngine` would pick, as a label for the status chip. */
export function activeKind(status: GpuStatus, p: ComputePref): 'webgpu' | 'cpu' | 'checking' {
  if (status.state === 'checking') return 'checking';
  return status.state === 'ready' && p === 'auto' ? 'webgpu' : 'cpu';
}
