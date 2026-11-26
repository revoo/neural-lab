// authored-by: Gemini Antigravity
/**
 * High-dimensional saddle-point geometry:
 * 1. Exact combinatorics for random critical points whose d principal curvatures are
 *    independently positive or negative with probability 1/2:
 *      P(local min) = (1/2)^d,  P(local max) = (1/2)^d,  P(saddle) = 1 - 2*(1/2)^d (d >= 2).
 * 2. Live 2D saddle descent on f(x, y) = 0.5 * (a*x^2 - b*y^2), demonstrating both the
 *    deceptive plateau-then-escape near the ridge and strict trapping dead-on the ridge.
 */

export const SADDLE_A = 1.0;
export const SADDLE_B = 0.6;
export const SADDLE_LR = 0.18;
export const SADDLE_STEPS = 42;

export type SaddlePresetId = 'near' | 'dead' | 'off';

export const SADDLE_PRESETS: Record<SaddlePresetId, { label: string; x0: number; y0: number }> = {
  near: { label: 'Near ridge (1.2, 0.02)', x0: 1.2, y0: 0.02 },
  dead: { label: 'Dead on ridge (1.2, 0.0)', x0: 1.2, y0: 0.0 },
  off: { label: 'Off ridge (1.2, 0.30)', x0: 1.2, y0: 0.3 },
};

export type SaddleStep = {
  step: number;
  x: number;
  y: number;
  loss: number;
  gradNorm: number;
};

export type CriticalKind = 'local-min' | 'local-max' | 'saddle';

export function saddleLoss(x: number, y: number, a = SADDLE_A, b = SADDLE_B): number {
  return 0.5 * (a * x * x - b * y * y);
}

export function saddleGrad(x: number, y: number, a = SADDLE_A, b = SADDLE_B): [number, number] {
  return [a * x, -b * y];
}

export function runSaddleDescent(
  x0: number,
  y0: number,
  lr = SADDLE_LR,
  steps = SADDLE_STEPS,
  a = SADDLE_A,
  b = SADDLE_B,
): SaddleStep[] {
  const path: SaddleStep[] = [];
  let x = x0;
  let y = y0;
  for (let step = 0; step <= steps; step++) {
    const [gx, gy] = saddleGrad(x, y, a, b);
    path.push({
      step,
      x,
      y,
      loss: saddleLoss(x, y, a, b),
      gradNorm: Math.hypot(gx, gy),
    });
    x -= lr * gx;
    y -= lr * gy;
  }
  return path;
}

/** Exact probability that all d independent ±1 curvatures are positive (true local minimum). */
export function minProbability(d: number): number {
  return Math.pow(0.5, d);
}

/** Exact probability that a critical point has at least one + and one - curvature (a saddle). */
export function saddleProbability(d: number): number {
  if (d <= 1) return 0;
  return 1 - 2 * Math.pow(0.5, d);
}

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sample d principal curvatures with independent 50/50 signs and classify the critical point. */
export function sampleCurvatures(d: number, seed: number): {
  curvatures: number[];
  upCount: number;
  downCount: number;
  kind: CriticalKind;
} {
  const rng = mulberry32(seed);
  const curvatures: number[] = [];
  let upCount = 0;
  let downCount = 0;
  for (let i = 0; i < d; i++) {
    const sign = rng() < 0.5 ? 1 : -1;
    const mag = 0.25 + rng() * 0.95;
    curvatures.push(sign * mag);
    if (sign > 0) upCount++;
    else downCount++;
  }
  const kind: CriticalKind = upCount === d ? 'local-min' : downCount === d ? 'local-max' : 'saddle';
  return { curvatures, upCount, downCount, kind };
}

/** Monte Carlo cross-check of critical-point fractions over `trials` independent samples. */
export function monteCarloCriticalPoints(d: number, trials: number, seed = 42): {
  minFrac: number;
  maxFrac: number;
  saddleFrac: number;
} {
  const rng = mulberry32(seed);
  let mins = 0;
  let maxs = 0;
  let saddles = 0;
  for (let t = 0; t < trials; t++) {
    let pos = 0;
    for (let i = 0; i < d; i++) {
      if (rng() < 0.5) pos++;
    }
    if (pos === d) mins++;
    else if (pos === 0) maxs++;
    else saddles++;
  }
  return {
    minFrac: mins / trials,
    maxFrac: maxs / trials,
    saddleFrac: saddles / trials,
  };
}
