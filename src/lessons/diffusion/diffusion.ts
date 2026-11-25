// authored-by: Claude Opus 5.5
/**
 * Diffusion, exactly. The data are mixtures of small round Gaussian blobs in 2D, which has a
 * wonderful property: adding Gaussian noise to a mixture of Gaussians gives another mixture of
 * Gaussians. So at every noise level we know the exact distribution of noisy points, and the exact
 * best possible denoiser, in closed form. That lets us study sampling without any training error,
 * and then compare a learned denoiser against the perfect one fairly.
 *
 * Noise levels are written as ᾱ ("signal share", alpha-bar) between 0 and 1, variance-preserving:
 *   x_noisy = √ᾱ · x_clean + √(1 − ᾱ) · ε,   ε ~ N(0, I)
 * ᾱ = 1 is clean data, ᾱ = 0 is pure noise.
 */
import { createRng, type Rng } from '../../lib/rng';

export type Pt = [number, number];
export type Mixture = { name: string; w: number[]; m: Pt[]; s: number[] };

// ------------------------------------------------------------------ datasets

function normalise(w: number[]) {
  const t = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / t);
}

/** Eight small blobs on a ring: the classic diffusion toy. */
export const RING: Mixture = (() => {
  const m: Pt[] = [];
  for (let k = 0; k < 8; k++) m.push([1.4 * Math.cos((2 * Math.PI * k) / 8), 1.4 * Math.sin((2 * Math.PI * k) / 8)]);
  return { name: 'ring', w: normalise(m.map(() => 1)), m, s: m.map(() => 0.08) };
})();

/** A smiley: two eyes and a smile made of blobs. */
export const SMILEY: Mixture = (() => {
  const m: Pt[] = [
    [-0.7, 0.75],
    [0.7, 0.75],
  ];
  for (let k = 0; k < 7; k++) {
    const t = Math.PI * (1.15 + (0.7 * k) / 6);
    m.push([1.25 * Math.cos(t), 0.15 + 1.25 * Math.sin(t)]);
  }
  return { name: 'smiley', w: normalise(m.map(() => 1)), m, s: m.map(() => 0.08) };
})();

/** One big blob and one rare small one: does the sampler give the rare one its fair share? */
export const LOPSIDED: Mixture = {
  name: 'lopsided',
  w: [0.9, 0.1],
  m: [
    [-0.45, 0],
    [1.6, 0.4],
  ],
  s: [0.35, 0.08],
};

export function sampleData(mix: Mixture, n: number, rng: Rng): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    let u = rng.next();
    let k = 0;
    while (k < mix.w.length - 1 && u >= mix.w[k]) u -= mix.w[k++];
    out.push([mix.m[k][0] + mix.s[k] * rng.normal(), mix.m[k][1] + mix.s[k] * rng.normal()]);
  }
  return out;
}

// ------------------------------------------------------------------ the exact noisy distribution

/** Variance of component k after noising to signal share a. */
const vk = (mix: Mixture, k: number, a: number) => a * mix.s[k] * mix.s[k] + 1 - a;

/** Exact density of noisy points at signal share a. */
export function density(mix: Mixture, a: number, x: Pt): number {
  let p = 0;
  const ra = Math.sqrt(a);
  for (let k = 0; k < mix.w.length; k++) {
    const v = vk(mix, k, a);
    const dx = x[0] - ra * mix.m[k][0];
    const dy = x[1] - ra * mix.m[k][1];
    p += (mix.w[k] / (2 * Math.PI * v)) * Math.exp(-(dx * dx + dy * dy) / (2 * v));
  }
  return p;
}

/**
 * The perfect denoiser: given a noisy point at signal share a, the average of all clean points that
 * could have produced it (E[x_clean | x_noisy]), and how responsible each blob is for it.
 */
export function bestGuess(mix: Mixture, a: number, x: Pt): { x0: Pt; resp: number[] } {
  const K = mix.w.length;
  const ra = Math.sqrt(a);
  const logs = new Array(K);
  let mx = -Infinity;
  for (let k = 0; k < K; k++) {
    const v = vk(mix, k, a);
    const dx = x[0] - ra * mix.m[k][0];
    const dy = x[1] - ra * mix.m[k][1];
    logs[k] = Math.log(mix.w[k]) - Math.log(v) - (dx * dx + dy * dy) / (2 * v);
    if (logs[k] > mx) mx = logs[k];
  }
  let z = 0;
  const resp = logs.map((l) => {
    const e = Math.exp(l - mx);
    z += e;
    return e;
  });
  let gx = 0;
  let gy = 0;
  for (let k = 0; k < K; k++) {
    resp[k] /= z;
    const v = vk(mix, k, a);
    const g = (ra * mix.s[k] * mix.s[k]) / v; // how far to trust the noisy point within blob k
    gx += resp[k] * (mix.m[k][0] + g * (x[0] - ra * mix.m[k][0]));
    gy += resp[k] * (mix.m[k][1] + g * (x[1] - ra * mix.m[k][1]));
  }
  return { x0: [gx, gy], resp };
}

/** Noise implied by a clean-point guess: ε̂ = (x − √a · x̂₀) / √(1 − a). */
export const impliedNoise = (x: Pt, x0: Pt, a: number): Pt => {
  const r = Math.sqrt(1 - a);
  const ra = Math.sqrt(a);
  return [(x[0] - ra * x0[0]) / r, (x[1] - ra * x0[1]) / r];
};

/** The data's overall mean: what the perfect denoiser guesses from pure noise. */
export function dataMean(mix: Mixture): Pt {
  let x = 0;
  let y = 0;
  mix.w.forEach((w, k) => {
    x += w * mix.m[k][0];
    y += w * mix.m[k][1];
  });
  return [x, y];
}

// ------------------------------------------------------------------ schedules

export const T = 1000;

/** DDPM's original schedule: each step adds a little noise, β growing linearly from 0.0001 to 0.02. */
export const LINEAR_ABAR: number[] = (() => {
  const out = [1];
  let a = 1;
  for (let t = 1; t <= T; t++) {
    const beta = 0.0001 + ((0.02 - 0.0001) * (t - 1)) / (T - 1);
    a *= 1 - beta;
    out.push(a);
  }
  return out;
})();

/** Signal share after t of the T forward steps. */
export const abarAt = (t: number) => LINEAR_ABAR[Math.max(0, Math.min(T, Math.round(t)))];

// ------------------------------------------------------------------ samplers

export type Denoiser = (x: Pt, a: number) => Pt;
export const exactDenoiser = (mix: Mixture): Denoiser => (x, a) => bestGuess(mix, a, x).x0;

/**
 * Run the reverse process through a list of signal shares, from noisiest (first) to clean (1).
 * Each step: guess the clean point, work out the noise that guess implies, then rebuild the point
 * at the next, less noisy level. η = 0 is DDIM (deterministic: reuse the implied noise); η = 1
 * swaps part of it for fresh randomness, which is DDPM's ancestral sampler.
 */
export function sample(den: Denoiser, levels: number[], start: Pt[], eta: number, rng: Rng, keep = false) {
  let xs = start.map((p) => [p[0], p[1]] as Pt);
  const path: Pt[][] = keep ? [xs] : [];
  for (let i = 0; i < levels.length; i++) {
    const a = levels[i];
    const an = i + 1 < levels.length ? levels[i + 1] : 1;
    xs = xs.map((x) => {
      const x0 = den(x, a);
      if (an >= 1) return x0;
      const e = a >= 1 ? ([0, 0] as Pt) : impliedNoise(x, x0, a);
      // DDIM's variance for a fresh-noise share η (η = 1 matches DDPM's posterior variance)
      const s2 = eta * eta * ((1 - an) / (1 - a)) * (1 - a / an);
      const keepNoise = Math.sqrt(Math.max(1 - an - s2, 0));
      const s = Math.sqrt(s2);
      const zx = s > 0 ? rng.normal() : 0;
      const zy = s > 0 ? rng.normal() : 0;
      const ra = Math.sqrt(an);
      return [ra * x0[0] + keepNoise * e[0] + s * zx, ra * x0[1] + keepNoise * e[1] + s * zy] as Pt;
    });
    if (keep) path.push(xs);
  }
  return { xs, path };
}

/** Evenly spaced steps through DDPM's 1,000-step schedule, starting from pure noise. */
export function evenLevels(n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(i === 0 ? 0 : abarAt(T - (i * T) / n));
  return out;
}

export function noiseStart(n: number, seed: number): Pt[] {
  const rng = createRng(seed);
  return Array.from({ length: n }, () => [rng.normal(), rng.normal()] as Pt);
}

// ------------------------------------------------------------------ judging samples

/** A sample "belongs" to a blob if it is within this many of the blob's SDs of its centre. */
export const ON_SHAPE = 3;

/**
 * Probability out of place: the share of samples that would have to move to make the sample
 * histogram (which blob each sample landed on, plus an "off the shape" bin) match the true blob
 * weights. 0 = perfect. (Even a perfect sampler scores a little above 0 with finitely many samples.)
 */
export function outOfPlace(mix: Mixture, xs: Pt[]) {
  const K = mix.w.length;
  const counts = new Array(K + 1).fill(0);
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
    counts[best <= ON_SHAPE ? bk : K]++;
  }
  let tv = counts[K] / xs.length;
  for (let k = 0; k < K; k++) tv += Math.abs(counts[k] / xs.length - mix.w[k]);
  return { tv: tv / 2, offShape: counts[K] / xs.length, counts };
}

/** Total variation between the noisy distribution at share a and pure noise N(0, I), on a grid. */
export function distanceFromNoise(mix: Mixture, a: number, half = 5, n = 160) {
  const h = (2 * half) / n;
  let tv = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const x: Pt = [-half + (i + 0.5) * h, -half + (j + 0.5) * h];
      const q = Math.exp(-(x[0] * x[0] + x[1] * x[1]) / 2) / (2 * Math.PI);
      tv += Math.abs(density(mix, a, x) - q) * h * h;
    }
  return tv / 2;
}

/**
 * Total variation between the noisy distribution at share a and pure noise N(0, I), estimated as
 * E over noisy points of max(0, 1 − q/p). Uses a fixed set of samples, so it is reproducible.
 */
export function shapeLeft(mix: Mixture, a: number, n = 4000, seed = 17) {
  const rng = createRng(seed);
  const clean = sampleData(mix, n, rng);
  const ra = Math.sqrt(a);
  const rn = Math.sqrt(1 - a);
  let tv = 0;
  for (const c of clean) {
    const x: Pt = [ra * c[0] + rn * rng.normal(), ra * c[1] + rn * rng.normal()];
    const p = density(mix, a, x);
    const q = Math.exp(-(x[0] * x[0] + x[1] * x[1]) / 2) / (2 * Math.PI);
    tv += Math.max(0, 1 - q / p) / n;
  }
  return tv;
}

/** Noisy copies of a fixed set of clean points, all with their own fixed noise (for animating). */
export function fixedCloud(mix: Mixture, n: number, seed: number) {
  const rng = createRng(seed);
  const clean = sampleData(mix, n, rng);
  const eps = clean.map(() => [rng.normal(), rng.normal()] as Pt);
  return {
    clean,
    eps,
    at: (a: number) => clean.map((c, i) => [Math.sqrt(a) * c[0] + Math.sqrt(1 - a) * eps[i][0], Math.sqrt(a) * c[1] + Math.sqrt(1 - a) * eps[i][1]] as Pt),
  };
}

// ------------------------------------------------------------------ Lab: spend the step budget

/** Signal shares the lab's sampler may stop at (it always starts from pure noise, ᾱ = 0). */
export const LAB_LEVELS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99] as const;
export const LAB_SAMPLES = 2000;
export const LAB_NOISE_SEED = 5;
export const LAB_FRESH_SEED = 9;

export type LabBriefId = 'ring' | 'smiley' | 'lopsided';
export const LAB_MIX: Record<LabBriefId, Mixture> = { ring: RING, smiley: SMILEY, lopsided: LOPSIDED };

/** Run the lab's sampler: start at pure noise, stop at each chosen level (ascending), finish clean. */
export function labRun(id: LabBriefId, chosen: number[], eta: number, noiseSeed = LAB_NOISE_SEED, keep = false) {
  const mix = LAB_MIX[id];
  const levels = [0, ...[...chosen].sort((p, q) => p - q)];
  const res = sample(exactDenoiser(mix), levels, noiseStart(LAB_SAMPLES, noiseSeed), eta, createRng(LAB_FRESH_SEED + noiseSeed - LAB_NOISE_SEED), keep);
  return { ...res, judge: outOfPlace(mix, res.xs), steps: levels.length };
}

export function* subsets(n: number, start = 0, acc: number[] = []): Generator<number[]> {
  if (acc.length === n) {
    yield acc;
    return;
  }
  for (let i = start; i < LAB_LEVELS.length; i++) yield* subsets(n, i + 1, [...acc, LAB_LEVELS[i]]);
}
