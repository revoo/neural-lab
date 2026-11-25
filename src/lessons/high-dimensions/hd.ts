// authored-by: Claude Opus 5.5
// High-dimensional geometry, computed: shell volumes, ball-in-cube, the exact distribution of the cosine between
// random directions, distance concentration (Monte Carlo with fixed seeds), and the ±1 superposition lab.
import { createRng } from '../../lib/rng';

// ------------------------------------------------------------------ volume

/** Fraction of a d-dimensional ball's volume within the outer shell of relative thickness ε: 1 − (1 − ε)^d. */
export const shellFraction = (d: number, eps: number) => 1 - (1 - eps) ** d;

/** ln Γ(x) for x > 0 (Lanczos, |rel. error| < 1e-13 over the range we use). */
export function lnGamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lnGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** log10 of the fraction of the cube [−1, 1]^d filled by the unit ball inside it: π^(d/2) / Γ(d/2 + 1) / 2^d. */
export const log10BallInCube = (d: number) => ((d / 2) * Math.log(Math.PI) - lnGamma(d / 2 + 1) - d * Math.log(2)) / Math.LN10;
export const ballInCube = (d: number) => 10 ** log10BallInCube(d);

/** Distance from the centre of the cube [−1, 1]^d to a corner. */
export const cornerDistance = (d: number) => Math.sqrt(d);

// ------------------------------------------------------------------ random directions

/**
 * The cosine between two independent random directions in d dimensions has density ∝ (1 − t²)^((d − 3)/2) on [−1, 1],
 * mean 0 and variance exactly 1/d. Returned normalised (d ≥ 2).
 */
export function cosineDensity(d: number, t: number): number {
  if (Math.abs(t) >= 1) return d === 2 ? Infinity : 0;
  const logNorm = lnGamma(d / 2) - 0.5 * Math.log(Math.PI) - lnGamma((d - 1) / 2);
  return Math.exp(logNorm + ((d - 3) / 2) * Math.log(1 - t * t));
}

/** P(|cos| ≥ c) for two random directions, by Simpson integration of the exact density (d ≥ 3). */
export function probAbsCosAtLeast(d: number, c: number, steps = 20000): number {
  if (c <= 0) return 1;
  if (c >= 1) return 0;
  const h = (1 - c) / steps;
  let s = cosineDensity(d, c) + cosineDensity(d, 1 - 1e-15);
  for (let i = 1; i < steps; i++) s += (i % 2 ? 4 : 2) * cosineDensity(d, c + i * h);
  return 2 * (s * h) / 3;
}

/** Cosines between `pairs` random Gaussian vector pairs in d dimensions (seeded). */
export function sampleCosines(d: number, pairs: number, seed: number): number[] {
  const rng = createRng(seed);
  const out: number[] = [];
  for (let k = 0; k < pairs; k++) {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < d; i++) {
      const a = rng.normal();
      const b = rng.normal();
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    out.push(dot / Math.sqrt(na * nb));
  }
  return out;
}

// ------------------------------------------------------------------ distance concentration

/**
 * N random points and one random query, uniform in the unit cube [0, 1]^d. Returns every distance from the query and
 * the relative contrast (farthest − nearest) / nearest.
 */
export function distancesFromQuery(d: number, N: number, seed: number) {
  const rng = createRng(seed);
  const q = Array.from({ length: d }, () => rng.next());
  const dists: number[] = [];
  for (let k = 0; k < N; k++) {
    let s = 0;
    for (let i = 0; i < d; i++) {
      const diff = rng.next() - q[i];
      s += diff * diff;
    }
    dists.push(Math.sqrt(s));
  }
  const lo = Math.min(...dists);
  const hi = Math.max(...dists);
  return { dists, nearest: lo, farthest: hi, contrast: (hi - lo) / lo };
}

export const CONTRAST_DIMS = [1, 2, 3, 10, 100, 1000] as const;
export const CONTRAST_POINTS = 1000;
export const CONTRAST_SEEDS = 20;

/** Mean relative contrast over fixed seeds (and the seed-to-seed range). */
export function meanContrast(d: number, N = CONTRAST_POINTS, seeds = CONTRAST_SEEDS) {
  const cs: number[] = [];
  for (let s = 1; s <= seeds; s++) cs.push(distancesFromQuery(d, N, s * 101 + d).contrast);
  return { mean: cs.reduce((a, b) => a + b, 0) / cs.length, min: Math.min(...cs), max: Math.max(...cs) };
}

// ------------------------------------------------------------------ the lab: packing ±1 concept vectors

export type PackBrief = {
  id: string;
  title: string;
  story: string;
  d: number;
  /** Largest allowed |dot product| between any two vectors (|cos| ≤ maxDot / d). */
  maxDot: number;
  goal: number;
  /** True when the goal can't be met: the learner should declare it impossible. */
  impossible?: boolean;
};

export const PACK_BRIEFS: PackBrief[] = [
  {
    id: 'cube',
    title: 'Brief 1 · Four arrows in 3D',
    story: 'In three dimensions, place 4 arrows (each entry +1 or −1) so that every pair is far from parallel: |cos| at most 1/3.',
    d: 3,
    maxDot: 1,
    goal: 4,
  },
  {
    id: 'perp4',
    title: 'Brief 2 · Exactly perpendicular',
    story: 'In four dimensions, place 4 arrows of ±1s that are all exactly perpendicular to each other: every dot product 0.',
    d: 4,
    maxDot: 0,
    goal: 4,
  },
  {
    id: 'double4',
    title: 'Brief 3 · Twice as many as dimensions',
    story: 'Still four dimensions, but now a little overlap is allowed: |cos| at most 0.5. Fit 8 arrows — twice as many as there are dimensions.',
    d: 4,
    maxDot: 2,
    goal: 8,
  },
  {
    id: 'perp6',
    title: 'Brief 4 · Three perpendicular arrows in 6D',
    story: 'Six dimensions, ±1 entries, all exactly perpendicular. Surely 3 is easy? Fit 3 — or declare it impossible.',
    d: 6,
    maxDot: 0,
    goal: 3,
    impossible: true,
  },
  {
    id: 'pack6',
    title: 'Brief 5 · Superposition',
    story: 'Six dimensions, |cos| at most 1/3 for every pair. Fit 10 concept vectors — more concepts than dimensions.',
    d: 6,
    maxDot: 2,
    goal: 10,
  },
];

export const dotInt = (a: readonly number[], b: readonly number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

export function checkPacking(b: PackBrief, rows: readonly (readonly number[])[]) {
  const n = rows.length;
  const bad: [number, number][] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.abs(dotInt(rows[i], rows[j])) > b.maxDot) bad.push([i, j]);
  return { count: n, bad, pass: !b.impossible && bad.length === 0 && n >= b.goal };
}

/** Every ±1 vector of length d whose first entry is +1 (one per line through the origin). */
export function signLines(d: number): number[][] {
  const out: number[][] = [];
  for (let m = 0; m < 1 << (d - 1); m++) {
    const v = [1];
    for (let k = 0; k < d - 1; k++) v.push(m & (1 << k) ? -1 : 1);
    out.push(v);
  }
  return out;
}

/** Exhaustive maximum: the largest set of ±1 vectors with every pairwise |dot| ≤ maxDot (branch and bound). */
export function maxPacking(d: number, maxDot: number): number[][] {
  const L = signLines(d);
  const adj = L.map((a) => L.map((b) => a !== b && Math.abs(dotInt(a, b)) <= maxDot));
  let best: number[] = [];
  const rec = (cur: number[], cand: number[]) => {
    if (cur.length + cand.length <= best.length) return;
    if (!cand.length) {
      best = cur.slice();
      return;
    }
    for (let i = 0; i < cand.length; i++) {
      if (cur.length + cand.length - i <= best.length) return;
      const v = cand[i];
      rec([...cur, v], cand.slice(i + 1).filter((u) => adj[v][u]));
    }
  };
  rec([], [...L.keys()]);
  return best.map((i) => L[i]);
}
