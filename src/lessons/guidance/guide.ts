// authored-by: Claude Opus 5.5
/**
 * Guidance and conditioning, exactly. Data are 2D mixtures of round blobs, each blob labelled with a
 * class. Mixtures of Gaussians stay mixtures of Gaussians under noise, so both the unconditional
 * best guess E[x₀ | x_t] and the conditional one E[x₀ | x_t, class] are exact. Classifier-free
 * guidance mixes them: guess = unconditional + w × (conditional − unconditional).
 * (Self-contained: a few helpers are duplicated from the diffusion lesson on purpose.)
 */
import { createRng, type Rng } from '../../lib/rng';

export type Pt = [number, number];
export type Labelled = { name: string; w: number[]; m: Pt[]; s: number[]; cls: number[]; classNames: string[] };

/** Two "tribes" of three blobs each. The innermost blobs of each tribe sit close to the other tribe. */
export const TRIBES: Labelled = {
  name: 'tribes',
  w: [1, 1, 1, 1, 1, 1].map((v) => v / 6),
  m: [
    [-1.7, 0.35],
    [-1.0, -0.55],
    [-0.3, 0.4],
    [0.3, -0.4],
    [1.0, 0.55],
    [1.7, -0.35],
  ],
  s: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
  cls: [0, 0, 0, 1, 1, 1],
  classNames: ['blue', 'orange'],
};

export function sampleData(mix: Labelled, n: number, rng: Rng, cls: number | null = null): Pt[] {
  const idx = mix.w.map((_, k) => k).filter((k) => cls === null || mix.cls[k] === cls);
  const tot = idx.reduce((s, k) => s + mix.w[k], 0);
  const out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    let u = rng.next() * tot;
    let j = 0;
    while (j < idx.length - 1 && u >= mix.w[idx[j]]) u -= mix.w[idx[j++]];
    const k = idx[j];
    out.push([mix.m[k][0] + mix.s[k] * rng.normal(), mix.m[k][1] + mix.s[k] * rng.normal()]);
  }
  return out;
}

/** Best guess of the clean point; restricted to one class's blobs when cls is given. */
export function guess(mix: Labelled, a: number, x: Pt, cls: number | null): { x0: Pt; resp: number[] } {
  const K = mix.w.length;
  const ra = Math.sqrt(a);
  const logs = new Array(K).fill(-Infinity);
  let mx = -Infinity;
  for (let k = 0; k < K; k++) {
    if (cls !== null && mix.cls[k] !== cls) continue;
    const v = a * mix.s[k] ** 2 + 1 - a;
    const d2 = (x[0] - ra * mix.m[k][0]) ** 2 + (x[1] - ra * mix.m[k][1]) ** 2;
    logs[k] = Math.log(mix.w[k]) - Math.log(v) - d2 / (2 * v);
    mx = Math.max(mx, logs[k]);
  }
  let z = 0;
  const resp = logs.map((l) => {
    const e = l === -Infinity ? 0 : Math.exp(l - mx);
    z += e;
    return e;
  });
  let gx = 0;
  let gy = 0;
  for (let k = 0; k < K; k++) {
    resp[k] /= z;
    if (!resp[k]) continue;
    const v = a * mix.s[k] ** 2 + 1 - a;
    const g = (ra * mix.s[k] ** 2) / v;
    gx += resp[k] * (mix.m[k][0] + g * (x[0] - ra * mix.m[k][0]));
    gy += resp[k] * (mix.m[k][1] + g * (x[1] - ra * mix.m[k][1]));
  }
  return { x0: [gx, gy], resp };
}

/** Probability the noisy point belongs to each class: the exact classifier at this noise level. */
export function classProbs(mix: Labelled, a: number, x: Pt): number[] {
  const r = guess(mix, a, x, null).resp;
  const p = mix.classNames.map(() => 0);
  r.forEach((v, k) => (p[mix.cls[k]] += v));
  return p;
}

/**
 * Classifier-free guidance on the best guess. Because ε̂ = (x − √ᾱ x̂₀)/√(1 − ᾱ) is a straight-line
 * function of x̂₀, mixing the noise predictions with weight w is the same as mixing the guesses:
 * guided = unconditional + w × (conditional − unconditional).
 */
export type Den = (x: Pt, a: number) => Pt;
export function guided(mix: Labelled, cls: number | null, w: number): Den {
  return (x, a) => {
    if (cls === null) return guess(mix, a, x, null).x0;
    const c = guess(mix, a, x, cls).x0;
    if (w === 1) return c;
    const u = guess(mix, a, x, null).x0;
    return [u[0] + w * (c[0] - u[0]), u[1] + w * (c[1] - u[1])];
  };
}

// ------------------------------------------------------------------ sampling

export const T = 1000;
export const LINEAR_ABAR: number[] = (() => {
  const out = [1];
  let a = 1;
  for (let t = 1; t <= T; t++) {
    a *= 1 - (0.0001 + ((0.02 - 0.0001) * (t - 1)) / (T - 1));
    out.push(a);
  }
  return out;
})();
export const abarAt = (t: number) => LINEAR_ABAR[Math.max(0, Math.min(T, Math.round(t)))];

/**
 * Stops for an n-step sampler, from pure noise: stop i of n sits at signal share 1 − (1 − i/n)².
 * They bunch up towards clean data, because the diffusion lesson found that evenly spaced stops
 * waste most of a small budget on noise levels where the shape has already gone.
 */
export function stops(n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(1 - (1 - i / n) ** 2);
  return out;
}

/** DDIM: guess, infer the noise, rebuild at the next stop. */
export function ddim(den: Den, levels: number[], start: Pt[], keep = false) {
  let xs = start.map((p) => [p[0], p[1]] as Pt);
  const path: Pt[][] = keep ? [xs] : [];
  for (let i = 0; i < levels.length; i++) {
    const a = levels[i];
    const an = i + 1 < levels.length ? levels[i + 1] : 1;
    xs = xs.map((x) => {
      const x0 = den(x, a);
      if (an >= 1) return x0;
      const r = Math.sqrt(1 - a);
      const e: Pt = [(x[0] - Math.sqrt(a) * x0[0]) / r, (x[1] - Math.sqrt(a) * x0[1]) / r];
      return [Math.sqrt(an) * x0[0] + Math.sqrt(1 - an) * e[0], Math.sqrt(an) * x0[1] + Math.sqrt(1 - an) * e[1]] as Pt;
    });
    if (keep) path.push(xs);
  }
  return { xs, path };
}

export function noiseStart(n: number, seed: number): Pt[] {
  const rng = createRng(seed);
  return Array.from({ length: n }, () => [rng.normal(), rng.normal()] as Pt);
}

// ------------------------------------------------------------------ judging

export const ON_SHAPE = 3;

/**
 * For samples requested as class `cls`: where did they land? Each sample is assigned to its nearest
 * blob if within ON_SHAPE SDs, else "off the shape".
 *  - onClass: share that landed on a blob of the requested class
 *  - wrongClass: share on a blob of another class
 *  - offShape: share on no blob at all
 *  - withinClass: share of each of the requested class's blobs among the on-class samples
 *  - unfair: total variation between withinClass and the class's true blob weights (0 = fair)
 */
export function judge(mix: Labelled, xs: Pt[], cls: number) {
  const K = mix.w.length;
  const counts = new Array(K).fill(0);
  let off = 0;
  for (const x of xs) {
    let best = Infinity;
    let bk = -1;
    for (let k = 0; k < K; k++) {
      const d = Math.hypot(x[0] - mix.m[k][0], x[1] - mix.m[k][1]) / mix.s[k];
      if (d < best) {
        best = d;
        bk = k;
      }
    }
    if (best <= ON_SHAPE) counts[bk]++;
    else off++;
  }
  const mine = mix.w.map((_, k) => k).filter((k) => mix.cls[k] === cls);
  const on = mine.reduce((s, k) => s + counts[k], 0);
  const n = xs.length;
  const tot = mine.reduce((s, k) => s + mix.w[k], 0);
  const withinClass = mine.map((k) => (on ? counts[k] / on : 0));
  const unfair = mine.reduce((s, k, i) => s + Math.abs(withinClass[i] - mix.w[k] / tot), 0) / 2;
  return { onClass: on / n, wrongClass: (n - on - off) / n, offShape: off / n, withinClass, unfair, counts };
}

// ------------------------------------------------------------------ Lab: tune the guidance

export const LAB_STEPS = [2, 3, 4, 5, 6, 8] as const;
export const LAB_WS = Array.from({ length: 17 }, (_, i) => i * 0.25);
export const LAB_N = 2000;

export type GuideBrief = { id: 'catalogue' | 'poster' | 'both'; title: string; story: string; minOn: number; minBlob: number };
export const GUIDE_BRIEFS: GuideBrief[] = [
  {
    id: 'catalogue',
    title: 'Brief 1 · The catalogue',
    story: 'A catalogue of blue things must show the whole range: at least 90% of samples on a blue blob, and every blue blob getting at least 25% of them.',
    minOn: 0.9,
    minBlob: 0.25,
  },
  {
    id: 'poster',
    title: 'Brief 2 · The poster',
    story: 'A poster needs one unmistakably blue image, fast. At least 95% of samples on a blue blob; variety doesn’t matter.',
    minOn: 0.95,
    minBlob: 0,
  },
  {
    id: 'both',
    title: 'Brief 3 · Both',
    story: 'At least 95% on a blue blob, and no blue blob below 10%.',
    minOn: 0.95,
    minBlob: 0.1,
  },
];

export function labTry(steps: number, w: number, seed = 5) {
  const j = judge(TRIBES, ddim(guided(TRIBES, 0, w), stops(steps), noiseStart(LAB_N, seed)).xs, 0);
  return { ...j, minBlob: Math.min(...j.withinClass) };
}

export const labPass = (b: GuideBrief, r: { onClass: number; minBlob: number }) => r.onClass >= b.minOn && r.minBlob >= b.minBlob;

/** Every setting tried: which pass, and the fewest steps that can. */
export function guideExhaustive(b: GuideBrief, seed = 5) {
  const rows = LAB_STEPS.flatMap((steps) => LAB_WS.map((w) => ({ steps, w, ...labTry(steps, w, seed) })));
  const pass = rows.filter((r) => labPass(b, r));
  const fewest = pass.length ? Math.min(...pass.map((r) => r.steps)) : -1;
  return { total: rows.length, passing: pass.length, fewest, atFewest: pass.filter((r) => r.steps === fewest).map((r) => r.w) };
}
