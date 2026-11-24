/**
 * Optimizer races on 2D landscapes. The optimizers themselves are the shared, tested ones in
 * src/lib/optimizers.ts; this file supplies the landscapes, runs, step-size tuning and the golf par.
 */
import { createOptimizer, type OptimizerName } from '../../lib/optimizers';

export type Landscape = {
  id: string;
  label: string;
  story: string;
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => [number, number];
  domain: [number, number, number, number]; // x0, x1, y0, y1
  start: [number, number];
  /** Loss below which the runner has "arrived". */
  target: number;
  /** Heatmap colour range and transform. */
  range: [number, number];
  minimum: [number, number][];
};

// A long, narrow valley tilted at 30°: steep across (curvature 1), gentle along (curvature 0.02).
const C = Math.cos(Math.PI / 6);
const S = Math.sin(Math.PI / 6);
const along = (x: number, y: number) => C * x + S * y;
const across = (x: number, y: number) => -S * x + C * y;

export const LANDSCAPES: Landscape[] = [
  {
    id: 'valley',
    label: 'Narrow valley',
    story: 'Steep walls, a gently sloping floor — the classic shape that makes one step size serve two masters.',
    f: (x, y) => 0.01 * along(x, y) ** 2 + 0.5 * across(x, y) ** 2,
    grad: (x, y) => {
      const u = along(x, y);
      const v = across(x, y);
      // d/dx and d/dy of 0.01u² + 0.5v²
      return [0.02 * u * C + v * -S, 0.02 * u * S + v * C];
    },
    domain: [-10, 10, -8, 8],
    start: [-8, 1],
    target: 0.01,
    range: [0, 30],
    minimum: [[0, 0]],
  },
  {
    id: 'plateau',
    label: 'Plateau',
    story: 'A bowl surrounded by a vast, almost flat plain. Out on the plain the slope is tiny; near the bowl it is steep.',
    // 1 − exp(−(0.5x² + 2y²)): minimum 0 at the origin, almost flat (≈ 1) far away.
    f: (x, y) => 1 - Math.exp(-(0.5 * x * x + 2 * y * y)),
    grad: (x, y) => {
      const e = Math.exp(-(0.5 * x * x + 2 * y * y));
      return [x * e, 4 * y * e];
    },
    domain: [-4, 4, -3, 3],
    start: [-3, 1.2],
    target: 0.01,
    range: [0, 1],
    minimum: [[0, 0]],
  },
  {
    id: 'banana',
    label: 'Banana',
    story: 'A curved, flat-bottomed valley (Rosenbrock’s function). Finding the valley is easy; following its bend to the minimum is not.',
    // (1 − x)² + 20(y − x²)²
    f: (x, y) => (1 - x) ** 2 + 20 * (y - x * x) ** 2,
    grad: (x, y) => [-2 * (1 - x) - 80 * x * (y - x * x), 40 * (y - x * x)],
    domain: [-2, 2, -1, 3],
    start: [-1.5, 2.5],
    target: 0.01,
    range: [0, 60],
    minimum: [[1, 1]],
  },
];

export type Config = { name: OptimizerName; lr: number; beta1?: number; beta2?: number };
export type Run = { path: [number, number][]; losses: number[]; reachedAt: number | null; diverged: boolean };

export function runOptimizer(L: Landscape, cfg: Config, steps: number): Run {
  const opt = createOptimizer({ name: cfg.name, lr: cfg.lr, beta1: cfg.beta1, beta2: cfg.beta2 });
  const p = [[L.start[0], L.start[1]]];
  const path: [number, number][] = [[p[0][0], p[0][1]]];
  const losses = [L.f(p[0][0], p[0][1])];
  let reachedAt: number | null = losses[0] < L.target ? 0 : null;
  for (let s = 1; s <= steps; s++) {
    const g = L.grad(p[0][0], p[0][1]);
    opt.step(p, [g]);
    const [x, y] = p[0];
    if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) return { path, losses, reachedAt, diverged: true };
    path.push([x, y]);
    const l = L.f(x, y);
    losses.push(l);
    if (reachedAt === null && l < L.target) reachedAt = s;
  }
  return { path, losses, reachedAt, diverged: false };
}

/** Steps to reach the target (Infinity if never within the cap). */
export function stepsToTarget(L: Landscape, cfg: Config, cap = 1000): number {
  const opt = createOptimizer({ name: cfg.name, lr: cfg.lr, beta1: cfg.beta1, beta2: cfg.beta2 });
  const p = [[L.start[0], L.start[1]]];
  for (let s = 1; s <= cap; s++) {
    opt.step(p, [L.grad(p[0][0], p[0][1])]);
    const [x, y] = p[0];
    if (!Number.isFinite(x) || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) return Infinity;
    if (L.f(x, y) < L.target) return s;
  }
  return Infinity;
}

/** Log-spaced step sizes to search. */
export const LR_GRID = Array.from({ length: 65 }, (_, k) => Math.round(10 ** (-3 + k * 0.0625) * 1e5) / 1e5); // 0.001 … 10

export const DEFAULT_BETAS: Record<'momentum' | 'rmsprop' | 'adam', { beta1?: number; beta2?: number }> = {
  momentum: { beta1: 0.9 },
  rmsprop: { beta2: 0.9 },
  adam: { beta1: 0.9, beta2: 0.999 },
};

/** The best step size for one optimizer (its usual β settings) on one landscape, by grid search. */
export function tuneLR(L: Landscape, name: 'sgd' | 'momentum' | 'rmsprop' | 'adam', cap = 1000): { lr: number; steps: number } {
  let best = { lr: LR_GRID[0], steps: Infinity };
  const betas = name === 'sgd' ? {} : DEFAULT_BETAS[name];
  for (const lr of LR_GRID) {
    const s = stepsToTarget(L, { name, lr, ...betas }, cap);
    if (s < best.steps) best = { lr, steps: s };
  }
  return best;
}

export const BETA1_GRID = [0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98];
export const BETA2_GRID = [0.9, 0.99, 0.999];

/** Golf par: the fewest steps any optimizer reaches the target in, searching step size and β values. */
export function golfPar(L: Landscape, cap = 1000): { steps: number; cfg: Config } {
  let best: { steps: number; cfg: Config } = { steps: Infinity, cfg: { name: 'sgd', lr: 0.01 } };
  const consider = (cfg: Config) => {
    const s = stepsToTarget(L, cfg, Math.min(cap, best.steps === Infinity ? cap : best.steps));
    if (s < best.steps) best = { steps: s, cfg };
  };
  for (const lr of LR_GRID) {
    consider({ name: 'sgd', lr });
    for (const b1 of BETA1_GRID) consider({ name: 'momentum', lr, beta1: b1 });
    for (const b2 of BETA2_GRID) consider({ name: 'rmsprop', lr, beta2: b2 });
    for (const b1 of BETA1_GRID) for (const b2 of BETA2_GRID) consider({ name: 'adam', lr, beta1: b1, beta2: b2 });
  }
  return best;
}
