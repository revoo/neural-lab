// authored-by: Gemini Antigravity
/**
 * Global equation reading mode, persisted in localStorage under `nl-math-mode`.
 * - `'full'` (default): show both the plain-English word equation and symbolic TeX.
 * - `'words-first'`: show the plain-English word equation and legend, collapsing
 *   symbolic TeX behind a 1-click "Show symbols" button whenever `words` is provided.
 */
import { useSyncExternalStore } from 'react';

export type MathMode = 'full' | 'words-first';

const KEY = 'nl-math-mode';

function load(): MathMode {
  try {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(KEY) : null;
    if (raw === 'words-first' || raw === 'full') return raw;
  } catch {
    /* storage unavailable: default to full */
  }
  return 'full';
}

let current: MathMode = load();
const listeners = new Set<() => void>();

export function getMathMode(): MathMode {
  return current;
}

export function setMathMode(next: MathMode): void {
  if (next === current) return;
  current = next;
  try {
    if (typeof localStorage !== 'undefined') localStorage.setItem(KEY, next);
  } catch {
    /* ignore storage write failures */
  }
  listeners.forEach((l) => l());
}

export function subscribeMathMode(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useMathMode(): {
  mathMode: MathMode;
  setMathMode: (next: MathMode) => void;
  toggleMathMode: () => void;
} {
  const mathMode = useSyncExternalStore(subscribeMathMode, getMathMode, getMathMode);
  return {
    mathMode,
    setMathMode,
    toggleMathMode: () => setMathMode(mathMode === 'full' ? 'words-first' : 'full'),
  };
}
