/** Simulation core for the gradient-descent lesson: 1D curves, a 2D valley, and "par" search. */

export type Curve1D = { label: string; f: (w: number) => number; df: (w: number) => number; domain: [number, number]; y: [number, number]; start: number };

export const CURVES: Record<string, Curve1D> = {
  bowl: { label: 'Smooth bowl', f: (w) => (w - 3) ** 2 + 1, df: (w) => 2 * (w - 3), domain: [-2, 8], y: [0, 14], start: -1 },
  bumpy: {
    label: 'Bumpy landscape',
    f: (w) => 0.12 * (w - 3) ** 2 + 0.6 * Math.sin(2.2 * w) + 1.5,
    df: (w) => 0.24 * (w - 3) + 1.32 * Math.cos(2.2 * w),
    domain: [-2, 8],
    y: [0, 5],
    start: -1.5,
  },
};

/** Classify plain gradient descent on the bowl (w−3)²+1: each step multiplies the distance to the minimum by (1 − 2·lr). */
export function bowlRegime(lr: number): 'slow' | 'smooth' | 'oscillating' | 'bouncing' | 'diverging' {
  const r = 1 - 2 * lr;
  if (Math.abs(r) > 1 + 1e-9) return 'diverging';
  if (Math.abs(Math.abs(r) - 1) < 1e-9) return lr < 0.25 ? 'slow' : 'bouncing';
  if (r < 0) return 'oscillating';
  return lr < 0.05 ? 'slow' : 'smooth';
}

// ---- 2D valley: long and narrow, steep across, gentle along.
export const valley = (x: number, y: number) => 0.02 * x * x + 0.5 * y * y;
export const valleyGrad = (x: number, y: number): [number, number] => [0.04 * x, y];
export const VALLEY_TARGET = 0.05;

export type RunResult = { path: [number, number][]; steps: number; reached: boolean; diverged: boolean };

/** Run gradient descent (optionally with momentum) on the valley until the loss drops below target. */
export function runValley(start: [number, number], lr: number, momentum = 0, maxSteps = 400): RunResult {
  let [x, y] = start;
  let vx = 0;
  let vy = 0;
  const path: [number, number][] = [[x, y]];
  for (let s = 1; s <= maxSteps; s++) {
    const [gx, gy] = valleyGrad(x, y);
    vx = momentum * vx + gx;
    vy = momentum * vy + gy;
    x -= lr * vx;
    y -= lr * vy;
    path.push([x, y]);
    if (!Number.isFinite(x) || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) return { path, steps: s, reached: false, diverged: true };
    if (valley(x, y) < VALLEY_TARGET) return { path, steps: s, reached: true, diverged: false };
  }
  return { path, steps: maxSteps, reached: false, diverged: false };
}

/** Best achievable step count by searching learning rates (and momentum values, if allowed). */
export function par(start: [number, number], withMomentum: boolean) {
  let best = { steps: Infinity, lr: 0, momentum: 0 };
  const betas = withMomentum ? [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95] : [0];
  for (const m of betas)
    for (let lr = 0.02; lr <= 2.2; lr += 0.02) {
      const r = runValley(start, lr, m);
      if (r.reached && r.steps < best.steps) best = { steps: r.steps, lr: Math.round(lr * 100) / 100, momentum: m };
    }
  return best;
}

/** Location of the lowest point of a curve on its domain (dense scan). */
export function globalMin(c: Curve1D): number {
  let best = c.domain[0];
  for (let w = c.domain[0]; w <= c.domain[1]; w += 0.001) if (c.f(w) < c.f(best)) best = w;
  return best;
}
