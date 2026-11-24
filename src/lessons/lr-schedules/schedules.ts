/**
 * Learning-rate schedules on a noisy problem. The problem is the same one as the SGD lesson
 * (duplicated here so lessons stay independent): fit ŷ = w·x + b to 1,000 noisy deliveries with
 * squared error, using small mini-batches so every step is noisy.
 */
import { createRng } from '../../lib/rng';

export type Data = { x: Float64Array; y: Float64Array };

export function makeData(n = 1000, seed = 7): Data {
  const r = createRng(seed);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = r.uniform(-1, 1);
    y[i] = 30 + 12 * x[i] + r.normal(0, 5);
  }
  return { x, y };
}

export function stats(d: Data) {
  const n = d.x.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sx += d.x[i];
    sy += d.y[i];
    sxx += d.x[i] ** 2;
    sxy += d.x[i] * d.y[i];
    syy += d.y[i] ** 2;
  }
  return { ex: sx / n, ey: sy / n, exx: sxx / n, exy: sxy / n, eyy: syy / n };
}
export type Stats = ReturnType<typeof stats>;
export const lossAt = (s: Stats, w: number, b: number) => w * w * s.exx + b * b + s.eyy + 2 * w * b * s.ex - 2 * w * s.exy - 2 * b * s.ey;
export function bestFit(s: Stats) {
  const w = (s.exy - s.ex * s.ey) / (s.exx - s.ex * s.ex);
  const b = s.ey - w * s.ex;
  return { w, b, loss: lossAt(s, w, b) };
}

// ---- Schedules: the step size as a function of the step number t ∈ [0, T).

export type ScheduleName = 'constant' | 'step' | 'cosine' | 'warmup-cosine' | 'custom';
export type Schedule = (t: number, T: number) => number;

export function makeSchedule(name: ScheduleName, peak: number, knots?: number[]): Schedule {
  switch (name) {
    case 'constant':
      return () => peak;
    case 'step':
      // Divide by 10 at 50% and again at 75% of training.
      return (t, T) => peak * (t < 0.5 * T ? 1 : t < 0.75 * T ? 0.1 : 0.01);
    case 'cosine':
      return (t, T) => peak * 0.5 * (1 + Math.cos((Math.PI * t) / T));
    case 'warmup-cosine': {
      return (t, T) => {
        const w = Math.max(1, Math.round(0.1 * T));
        if (t < w) return (peak * (t + 1)) / w;
        return peak * 0.5 * (1 + Math.cos((Math.PI * (t - w)) / (T - w)));
      };
    }
    case 'custom': {
      // Piecewise-linear in log space through evenly spaced knots (values are step sizes).
      const k = knots ?? [peak, peak];
      return (t, T) => {
        const u = (t / Math.max(1, T - 1)) * (k.length - 1);
        const i = Math.min(k.length - 2, Math.floor(u));
        const f = u - i;
        return Math.exp(Math.log(k[i]) * (1 - f) + Math.log(k[i + 1]) * f);
      };
    }
  }
}

/** Mini-batch SGD with the given schedule. Returns the loss after every step (recorded every `every` steps). */
export function train(d: Data, s: Stats, schedule: Schedule, T: number, B: number, seed: number, start: [number, number] = [0, 10], every = 1) {
  const r = createRng(seed);
  const n = d.x.length;
  let [w, b] = start;
  const losses: number[] = [lossAt(s, w, b)];
  const path: [number, number][] = [[w, b]];
  for (let t = 0; t < T; t++) {
    let gw = 0;
    let gb = 0;
    for (let k = 0; k < B; k++) {
      const i = r.int(n);
      const e = w * d.x[i] + b - d.y[i];
      gw += 2 * e * d.x[i];
      gb += 2 * e;
    }
    const lr = schedule(t, T);
    w -= (lr * gw) / B;
    b -= (lr * gb) / B;
    if (!Number.isFinite(w) || Math.abs(w) > 1e6) {
      losses.push(Infinity);
      break;
    }
    if ((t + 1) % every === 0 || t === T - 1) {
      losses.push(lossAt(s, w, b));
      path.push([w, b]);
    }
  }
  return { losses, path, final: losses[losses.length - 1] };
}

/** Average final excess loss (above the best possible) over several seeds. */
export function meanFinalExcess(d: Data, s: Stats, schedule: Schedule, T: number, B: number, seeds: number[]) {
  const best = bestFit(s).loss;
  let tot = 0;
  for (const seed of seeds) {
    const f = train(d, s, schedule, T, B, seed).final;
    if (!Number.isFinite(f)) return Infinity;
    tot += f - best;
  }
  return tot / seeds.length;
}

/** Best peak step size for a schedule family, by grid search. */
export function bestPeak(d: Data, s: Stats, name: Exclude<ScheduleName, 'custom'>, T: number, B: number, seeds: number[], grid: number[]) {
  let best = { peak: grid[0], excess: Infinity };
  for (const peak of grid) {
    const e = meanFinalExcess(d, s, makeSchedule(name, peak), T, B, seeds);
    if (e < best.excess) best = { peak, excess: e };
  }
  return best;
}

/** Knot values (log10 step size) tried by the designer search. */
export const KNOT_GRID = [-0.8, -1, -1.2, -1.4, -1.6, -1.8, -2, -2.3, -2.6, -3];

/**
 * Coarse search over non-increasing five-knot schedules (log10 step sizes from KNOT_GRID; the fourth
 * knot is at most −2 and the last is −2.6 or −3). Not the true optimum over all schedules — a
 * reference point found by brute force, reported as such.
 */
export function searchKnots(d: Data, s: Stats, T: number, B: number, seeds: number[]) {
  let best = { excess: Infinity, knots: [] as number[], tried: 0 };
  for (const a of KNOT_GRID)
    for (const b of KNOT_GRID) {
      if (b > a) continue;
      for (const c of KNOT_GRID) {
        if (c > b) continue;
        for (const e of [-2.6, -3]) {
          const k = [a, b, c, Math.min(c, -2), e];
          const ex = meanFinalExcess(d, s, makeSchedule('custom', 0, k.map((v) => 10 ** v)), T, B, seeds);
          best.tried++;
          if (ex < best.excess) best = { excess: ex, knots: k, tried: best.tried };
        }
      }
    }
  return best;
}
