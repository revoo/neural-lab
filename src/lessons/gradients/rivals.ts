// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for gradients: three ways to walk
 * downhill on the same tilted bowl (the narrow valley from the explorer,
 * turned by 30° so that neither knob points along it), with the same budget
 * of model runs, then the same race with 100 knobs. Each method gets its own best step size, found by
 * searching a grid; the random method is averaged over 20 fixed seeds.
 */
import type { Rival } from '../../components/RivalTable';
import { createRng } from '../../lib/rng';

const TH = (30 * Math.PI) / 180;
const C = Math.cos(TH);
const S = Math.sin(TH);
/** The narrow valley 0.2u² + 2v², turned by 30°. Its lowest point is 0, at the origin. */
export const bowl = (x: number, y: number) => {
  const u = C * x + S * y;
  const v = -S * x + C * y;
  return 0.2 * u * u + 2 * v * v;
};
export const grad = (x: number, y: number): [number, number] => {
  const u = C * x + S * y;
  const v = -S * x + C * y;
  return [0.4 * u * C - 4 * v * S, 0.4 * u * S + 4 * v * C];
};
export const START: [number, number] = [-3, 2];
/**
 * Budget: 60 runs of the model. One gradient by backpropagation costs about 3 runs, so
 * 20 moves. One slope measured by nudging costs 2 runs, so one knob at a time gets 30 moves.
 * A random try costs 1 run, so 60 tries.
 */
export const RUNS = 60;

export function gradientDescent(lr: number): number {
  let [x, y] = START;
  for (let k = 0; k < RUNS / 3; k++) {
    const [gx, gy] = grad(x, y);
    x -= lr * gx;
    y -= lr * gy;
  }
  return bowl(x, y);
}

export function coordinateDescent(lr: number): number {
  let [x, y] = START;
  for (let k = 0; k < RUNS / 2; k++) {
    const g = grad(x, y);
    if (k % 2 === 0) x -= lr * g[0];
    else y -= lr * g[1];
  }
  return bowl(x, y);
}

/** Try a random direction of length `step`; keep the move only if the loss goes down. */
export function randomDirections(step: number, seed: number): number {
  const rng = createRng(seed);
  let [x, y] = START;
  let f = bowl(x, y);
  for (let k = 0; k < RUNS; k++) {
    const a = rng.uniform(0, 2 * Math.PI);
    const nx = x + step * Math.cos(a);
    const ny = y + step * Math.sin(a);
    const nf = bowl(nx, ny);
    if (nf < f) [x, y, f] = [nx, ny, nf];
  }
  return f;
}
export const SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);
export const randomAverage = (step: number) => SEEDS.reduce((s, sd) => s + randomDirections(step, sd), 0) / SEEDS.length;

const grid = (lo: number, hi: number, d: number) => Array.from({ length: Math.round((hi - lo) / d) + 1 }, (_, i) => +(lo + i * d).toFixed(4));
/** The step-size grids each method's best setting is searched over. */
export const LR_GRID = grid(0.01, 1.5, 0.01);
export const STEP_GRID = grid(0.05, 3, 0.05);

function best(values: number[], f: (v: number) => number) {
  let b = { setting: values[0], loss: Infinity };
  for (const v of values) {
    const l = f(v);
    if (l < b.loss) b = { setting: v, loss: l };
  }
  return b;
}

export function race() {
  return {
    start: bowl(...START),
    gradient: best(LR_GRID, gradientDescent),
    coordinate: best(LR_GRID, coordinateDescent),
    random: best(STEP_GRID, randomAverage),
  };
}

/* ---------- The same race with 100 knobs ---------- */

/**
 * A tilted bowl with 100 knobs: ½ xᵀAx with A = Q diag(λ) Qᵀ, the curvatures λ spread
 * evenly from 0.4 to 4 (the explorer's narrow valley, 0.2u² + 2v², has 0.4 and 4), and
 * Q a random rotation, so no knob lines up with the valley. Five seeded bowls.
 */
export const N = 100;
export const BOWL_SEEDS = [1, 2, 3, 4, 5];
/** Same budget as the two-knob race: RUNS. */

type Bowl = { A: number[][]; x0: number[] };
export function makeBowl(seed: number, n = N): Bowl {
  const r = createRng(seed);
  const Q: number[][] = [];
  for (let i = 0; i < n; i++) {
    let v = Array.from({ length: n }, () => r.normal());
    for (const u of Q) {
      const d = v.reduce((t, x, k) => t + x * u[k], 0);
      v = v.map((x, k) => x - d * u[k]);
    }
    const len = Math.hypot(...v);
    Q.push(v.map((x) => x / len));
  }
  const lam = Array.from({ length: n }, (_, i) => 0.4 + (3.6 * i) / (n - 1));
  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => Q.reduce((t, q, k) => t + lam[k] * q[i] * q[j], 0)));
  return { A, x0: Array.from({ length: n }, () => r.normal()) };
}
const loss = (A: number[][], x: number[]) => 0.5 * x.reduce((t, xi, i) => t + xi * A[i].reduce((u, a, j) => u + a * x[j], 0), 0);
const gradN = (A: number[][], x: number[]) => A.map((row) => row.reduce((u, a, j) => u + a * x[j], 0));
const partial = (A: number[][], x: number[], i: number) => A[i].reduce((u, a, j) => u + a * x[j], 0);

let bowls: Bowl[] | null = null;
const getBowls = () => (bowls ??= BOWL_SEEDS.map((sd) => makeBowl(sd)));
/** Average fraction of the starting loss left after the budget, over the five bowls. */
const left = (run: (b: Bowl, i: number) => number) => getBowls().reduce((t, b, i) => t + run(b, i) / loss(b.A, b.x0), 0) / BOWL_SEEDS.length;

export const bigGradient = (lr: number) =>
  left(({ A, x0 }) => {
    let x = x0.slice();
    for (let k = 0; k < RUNS / 3; k++) {
      const g = gradN(A, x);
      x = x.map((v, i) => v - lr * g[i]);
    }
    return loss(A, x);
  });
export const bigCoordinate = (lr: number) =>
  left(({ A, x0 }) => {
    const x = x0.slice();
    for (let k = 0; k < RUNS / 2; k++) x[k % N] -= lr * partial(A, x, k % N);
    return loss(A, x);
  });
export const bigRandom = (step: number) =>
  left(({ A, x0 }, bi) => {
    const r = createRng(100 + bi);
    let x = x0.slice();
    let fx = loss(A, x);
    for (let k = 0; k < RUNS; k++) {
      const d = Array.from({ length: N }, () => r.normal());
      const len = Math.hypot(...d);
      const y = x.map((v, i) => v + (step * d[i]) / len);
      const fy = loss(A, y);
      if (fy < fx) [x, fx] = [y, fy];
    }
    return fx;
  });

/** Best settings found by searching LR_GRID (gradient, one knob at a time) and STEP_GRID (random); re-verified in rivals.test.ts. */
export const BIG_BEST = { gradient: 0.43, coordinate: 0.45, random: 1.1 };

export function bigRace() {
  return {
    gradient: bigGradient(BIG_BEST.gradient),
    coordinate: bigCoordinate(BIG_BEST.coordinate),
    random: bigRandom(BIG_BEST.random),
  };
}

/** Average |cos| between a random direction and the downhill direction in n dimensions: Γ(n/2) / (√π Γ((n+1)/2)). */
export function randomAlignment(n: number): number {
  // log-gamma via Stirling series, accurate for the sizes used here
  const lg = (z: number): number => {
    if (z < 7) return lg(z + 1) - Math.log(z);
    return (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI) + 1 / (12 * z) - 1 / (360 * z ** 3);
  };
  return Math.exp(lg(n / 2) - lg((n + 1) / 2)) / Math.sqrt(Math.PI);
}

const sci = (v: number) => (v < 0.001 ? v.toExponential(1).replace('e-', ' × 10⁻').replace(/⁻(\d+)/, (_, d: string) => '⁻' + [...d].map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('')) : v.toFixed(3));
const pct = (v: number) => (v < 0.001 ? `${(v * 100).toPrecision(1)}%` : `${Math.round(v * 100)}%`);

export function rivals(): Rival[] {
  const r = race();
  const b = bigRace();
  return [
    {
      name: 'One knob at a time',
      result: `Turn x, then y, then x again, each by its own measured slope (${RUNS / 2} moves). On the two-knob bowl, with its best step size, it goes from ${r.start.toFixed(1)} to ${sci(r.coordinate.loss)}: it wins! With 100 knobs it has turned only ${RUNS / 2} of them when the budget runs out, and ${pct(b.coordinate)} of the loss is left.`,
      verdicts: ['pass', 'pass', 'fail'],
    },
    {
      name: 'Random directions',
      result: `Try a random direction and keep it if the loss drops (${RUNS} tries, best step length, ${SEEDS.length} seeds). Two knobs: down to ${sci(r.random.loss)}. 100 knobs: ${pct(b.random)} of the loss left. A random direction in two dimensions is fairly well lined up with downhill (average alignment ${randomAlignment(2).toFixed(2)}); with 100 knobs it is ${randomAlignment(100).toFixed(2)}, and with a million, ${randomAlignment(1e6).toFixed(4)}.`,
      verdicts: ['pass', 'fail', 'fail'],
    },
    {
      name: 'The gradient',
      result: `Move every knob at once, each by its own slope (${RUNS / 3} moves, since each gradient costs about three runs). Two knobs: down to ${sci(r.gradient.loss)}, beaten by one knob at a time. 100 knobs: ${pct(b.gradient)} of the loss left, because every move uses every knob’s slope. For a small step no direction goes downhill faster, by the dot-product argument in this lesson.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
