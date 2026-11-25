// authored-by: Claude Opus 5.5
/**
 * The perceptron learning rule, the datasets used by the lesson, and the
 * exhaustive searches behind the Mark I lab.
 *
 * Weights are [w1, w2, b]. The unit fires (outputs 1) when w1·x1 + w2·x2 + b ≥ 0,
 * the same step convention as the neuron lesson. On a mistake the rule adds
 * step × (right answer − output) × [x1, x2, 1] to the weights.
 */

export type Pt = { x: [number, number]; y: 0 | 1; name: string };
export type W = [number, number, number];

const P = (x1: number, x2: number, y: 0 | 1, name: string): Pt => ({ x: [x1, x2], y, name });

/** The four corners of the unit square, named for the inputs they stand for. */
const corners = (f: (a: number, b: number) => 0 | 1): Pt[] => [
  P(0, 0, f(0, 0), '(0, 0)'),
  P(0, 1, f(0, 1), '(0, 1)'),
  P(1, 0, f(1, 0), '(1, 0)'),
  P(1, 1, f(1, 1), '(1, 1)'),
];
export const OR = corners((a, b) => (a || b ? 1 : 0));
export const AND = corners((a, b) => (a && b ? 1 : 0));
export const XOR = corners((a, b) => (a !== b ? 1 : 0));

/** A separable cloud for the stepper: ten points either side of the line x1 + 2·x2 = 1.4 (fixed, hand-placed). */
export const CLOUD: Pt[] = [
  P(0.1, 0.1, 0, 'A'),
  P(0.5, 0.15, 0, 'B'),
  P(0.9, 0.05, 0, 'C'),
  P(0.2, 0.45, 0, 'D'),
  P(-0.2, 0.6, 0, 'E'),
  P(0.3, 0.8, 1, 'F'),
  P(0.9, 0.6, 1, 'G'),
  P(1.2, 0.3, 1, 'H'),
  P(0.6, 1.1, 1, 'I'),
  P(1.3, 0.9, 1, 'J'),
];

export const out = (w: W, p: Pt): 0 | 1 => (w[0] * p.x[0] + w[1] * p.x[1] + w[2] >= 0 ? 1 : 0);
export const score = (w: W, p: Pt) => w[0] * p.x[0] + w[1] * p.x[1] + w[2];

/** One step of the rule on one example. Returns the new weights and whether it was a mistake. */
export function update(w: W, p: Pt, eta: number): { w: W; mistake: boolean; said: 0 | 1 } {
  const said = out(w, p);
  if (said === p.y) return { w, mistake: false, said };
  const d = eta * (p.y - said);
  return { w: [w[0] + d * p.x[0], w[1] + d * p.x[1], w[2] + d], mistake: true, said };
}

export type RunEvent = { i: number; mistake: boolean; said: 0 | 1; before: W; after: W };
export type RunResult = { converged: boolean; mistakes: number; looks: number; w: W; events: RunEvent[] };

/**
 * Show the examples in `order` (indices into `pts`), cycling, until a full pass
 * makes no mistake (converged) or `maxMistakes` updates have happened.
 */
export function run(pts: Pt[], order: number[], w0: W, eta: number, maxMistakes = 200, keepEvents = false): RunResult {
  let w: W = [...w0];
  let mistakes = 0;
  let looks = 0;
  let clean = 0;
  const events: RunEvent[] = [];
  if (order.length === 0) return { converged: true, mistakes, looks, w, events };
  for (let k = 0; ; k++) {
    const i = order[k % order.length];
    const r = update(w, pts[i], eta);
    looks++;
    if (keepEvents) events.push({ i, mistake: r.mistake, said: r.said, before: w, after: r.w });
    w = r.w;
    if (r.mistake) {
      mistakes++;
      clean = 0;
      if (mistakes >= maxMistakes) return { converged: false, mistakes, looks, w, events };
    } else if (++clean === order.length) {
      return { converged: true, mistakes, looks, w, events };
    }
  }
}

export const allCorrect = (w: W, pts: Pt[]) => pts.every((p) => out(w, p) === p.y);

/** All orderings of [0, …, n − 1]. */
export function permutations(n: number): number[][] {
  if (n === 0) return [[]];
  const res: number[][] = [];
  for (const rest of permutations(n - 1)) for (let at = 0; at <= rest.length; at++) res.push([...rest.slice(0, at), n - 1, ...rest.slice(at)]);
  return res;
}

// ---- Novikoff's bound ----------------------------------------------------

/** Largest distance from the origin of a point written as (x1, x2, 1). */
export const radius = (pts: Pt[]) => Math.max(...pts.map((p) => Math.hypot(p.x[0], p.x[1], 1)));

/** The margin of a direction u (any length) on points written as (x1, x2, 1): the smallest signed distance. */
export function marginOf(u: W, pts: Pt[]): number {
  const n = Math.hypot(...u);
  return Math.min(...pts.map((p) => ((p.y ? 1 : -1) * (u[0] * p.x[0] + u[1] * p.x[1] + u[2])) / n));
}

/**
 * The widest margin found by searching directions on the sphere: a 1° grid,
 * then repeated local refinement around the best direction. Any margin found
 * is ≤ the true best, so (R / found)² is ≥ the true bound (a safe bound).
 */
export function bestMargin(pts: Pt[]): { gamma: number; u: W } {
  const dir = (th: number, ph: number): W => [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
  let best = { gamma: -Infinity, th: 0, ph: 0 };
  for (let a = 0; a <= 180; a += 1)
    for (let b = 0; b < 360; b += 1) {
      const th = (a * Math.PI) / 180;
      const ph = (b * Math.PI) / 180;
      const g = marginOf(dir(th, ph), pts);
      if (g > best.gamma) best = { gamma: g, th, ph };
    }
  for (let step = Math.PI / 180; step > 1e-7; step /= 2) {
    let improved = true;
    while (improved) {
      improved = false;
      for (const [dt, dp] of [[step, 0], [-step, 0], [0, step], [0, -step], [step, step], [-step, -step], [step, -step], [-step, step]]) {
        const g = marginOf(dir(best.th + dt, best.ph + dp), pts);
        if (g > best.gamma) {
          best = { gamma: g, th: best.th + dt, ph: best.ph + dp };
          improved = true;
        }
      }
    }
  }
  return { gamma: best.gamma, u: dir(best.th, best.ph) };
}

/** Novikoff: starting from zero weights, the perceptron makes at most (R/γ)² mistakes on separable data. */
export function novikoffBound(pts: Pt[]): number {
  const { gamma } = bestMargin(pts);
  if (gamma <= 0) return Infinity;
  return (radius(pts) / gamma) ** 2;
}

// ---- The Mark I lab ------------------------------------------------------

export const ETAS = [0.1, 0.5, 1] as const;
export const XOR_CAP = 50;
export const RUN_CAP = 200;

/** Brief 1's knobs start where the last experiment left them: computing AND. */
export const OR_START: W = [1, 1, -1.5];
/** Brief 2's knobs start at a fixed "factory setting" the learner can't change. */
export const AND_START: W = [-1, 1, -0.5];

/** Brief 3: the cloud plus one point, K, just across the line from H. Zero start, step 1, examples in listed order. */
export const THIN: Pt[] = [...CLOUD, P(1.25, 0.05, 0, 'K')];
export const THIN_TARGET = 30;

/** Exhaustive search for brief 2: every ordering of the four AND examples × every step size, from AND_START. */
export function andSearch() {
  const rows = permutations(4).flatMap((order) => ETAS.map((eta) => ({ order, eta, r: run(AND, order, AND_START, eta, RUN_CAP) })));
  const conv = rows.filter((x) => x.r.converged);
  const min = Math.min(...conv.map((x) => x.r.mistakes));
  return { total: rows.length, converged: conv.length, min, best: conv.filter((x) => x.r.mistakes === min), worst: Math.max(...conv.map((x) => x.r.mistakes)) };
}

/** Brief 3: mistakes with each single point removed (listed order, zero start, step 1). */
export function thinRun(removed: number | null) {
  const pts = removed === null ? THIN : THIN.filter((_, i) => i !== removed);
  return { pts, r: run(pts, pts.map((_, i) => i), [0, 0, 0], 1, 1000) };
}
export function thinSearch() {
  const each = THIN.map((p, i) => ({ name: p.name, i, mistakes: thinRun(i).r.mistakes }));
  const min = Math.min(...each.map((e) => e.mistakes));
  return { full: thinRun(null).r.mistakes, each, min, passing: each.filter((e) => e.mistakes <= THIN_TARGET), best: each.filter((e) => e.mistakes === min) };
}
