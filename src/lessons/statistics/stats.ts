// authored-by: Claude Opus 5.5
// Exact statistics for the lesson: means, variances, exact distributions of sums (by convolution),
// binomial test-set noise, and the dice-workshop lab. Every quoted number is pinned in stats.test.ts.

// ------------------------------------------------------------------ basics

export const mean = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

/** Population variance: the average squared distance from the mean. */
export const variance = (xs: readonly number[]) => {
  const m = mean(xs);
  return xs.reduce((s, x) => s + (x - m) * (x - m), 0) / xs.length;
};

export const sd = (xs: readonly number[]) => Math.sqrt(variance(xs));

export function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** Mean absolute deviation from the mean (the "average distance" that variance is often confused with). */
export const meanAbsDev = (xs: readonly number[]) => {
  const m = mean(xs);
  return xs.reduce((s, x) => s + Math.abs(x - m), 0) / xs.length;
};

// ------------------------------------------------------------------ exact discrete distributions on the integers

/** P(X = min + i) = p[i]. */
export type Dist = { min: number; p: number[] };

export const point = (v: number): Dist => ({ min: v, p: [1] });
export const uniformInts = (lo: number, hi: number): Dist => ({ min: lo, p: new Array(hi - lo + 1).fill(1 / (hi - lo + 1)) });

export function convolve(a: Dist, b: Dist): Dist {
  const p = new Array<number>(a.p.length + b.p.length - 1).fill(0);
  for (let i = 0; i < a.p.length; i++) {
    if (a.p[i] === 0) continue;
    for (let j = 0; j < b.p.length; j++) p[i + j] += a.p[i] * b.p[j];
  }
  return { min: a.min + b.min, p };
}

/** Distribution of the sum of n independent copies (n ≥ 0). */
export function sumOf(d: Dist, n: number): Dist {
  let out = point(0);
  for (let k = 0; k < n; k++) out = convolve(out, d);
  return out;
}

export const shift = (d: Dist, k: number): Dist => ({ min: d.min + k, p: d.p });

export function distMean(d: Dist): number {
  return d.p.reduce((s, q, i) => s + q * (d.min + i), 0);
}
export function distVar(d: Dist): number {
  const m = distMean(d);
  return d.p.reduce((s, q, i) => s + q * (d.min + i - m) ** 2, 0);
}
export const distSd = (d: Dist) => Math.sqrt(distVar(d));

/** Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
export function normalCdf(z: number): number {
  const t = 1 / (1 + 0.3275911 * (Math.abs(z) / Math.SQRT2));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

export const normalPdf = (x: number, mu: number, sigma: number) => Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));

/**
 * How far a discrete distribution is from a bell curve with the same mean and SD: the share of probability that is
 * "in the wrong place" (total variation distance). Each integer outcome x is compared with the bell's mass on
 * [x − ½, x + ½]; bell mass that falls on no outcome at all also counts. 0 = identical, 1 = nothing in common.
 */
export function bellGap(d: Dist): number {
  const mu = distMean(d);
  const s = distSd(d);
  if (s === 0) return 1;
  const F = (x: number) => normalCdf((x - mu) / s);
  let diff = F(d.min - 0.5); // bell mass below the lowest outcome
  for (let i = 0; i < d.p.length; i++) {
    const x = d.min + i;
    diff += Math.abs(d.p[i] - (F(x + 0.5) - F(x - 0.5)));
  }
  diff += 1 - F(d.min + d.p.length - 0.5); // bell mass above the highest outcome
  return diff / 2;
}

// ------------------------------------------------------------------ the bell-curve factory

export type SourceId = 'coin' | 'die' | 'ticket';
export const SOURCES: Record<SourceId, { label: string; dist: Dist; note: string }> = {
  coin: { label: 'coin flip (0 or 1)', dist: uniformInts(0, 1), note: 'two outcomes, perfectly symmetric' },
  die: { label: 'die (1–6)', dist: uniformInts(1, 6), note: 'six equally likely outcomes: flat, not a bell' },
  ticket: {
    label: 'long shot (wins 1 one time in ten)',
    dist: { min: 0, p: [0.9, 0.1] },
    note: 'very lopsided: usually 0, occasionally 1',
  },
};

// ------------------------------------------------------------------ test-set noise (binomial)

/** Exact Binomial(n, p) as a Dist over the number of successes, computed in log space for stability. */
export function binomial(n: number, p: number): Dist {
  const logF = new Array<number>(n + 1).fill(0);
  for (let i = 2; i <= n; i++) logF[i] = logF[i - 1] + Math.log(i);
  const out = new Array<number>(n + 1);
  for (let k = 0; k <= n; k++) {
    out[k] = Math.exp(logF[n] - logF[k] - logF[n - k] + (k === 0 ? 0 : k * Math.log(p)) + (k === n ? 0 : (n - k) * Math.log(1 - p)));
  }
  return { min: 0, p: out };
}

/** Standard error of a measured accuracy: √(p(1 − p)/n). */
export const accuracySE = (p: number, n: number) => Math.sqrt((p * (1 - p)) / n);

/**
 * Two models with the SAME true accuracy p, each scored on its own test set of n examples: the exact probability that
 * one of them (a specific one) measures at least `gap` higher than the other.
 */
export function probLooksBetter(p: number, n: number, gap: number): number {
  const b = binomial(n, p);
  // P(K_A − K_B ≥ g·n): convolve A with the mirror of B
  const need = Math.ceil(gap * n - 1e-9);
  const cdfB = new Array<number>(n + 2).fill(0);
  for (let k = 0; k <= n; k++) cdfB[k + 1] = cdfB[k] + b.p[k];
  let total = 0;
  for (let ka = 0; ka <= n; ka++) {
    const maxKb = ka - need; // K_B ≤ ka − need
    if (maxKb < 0) continue;
    total += b.p[ka] * cdfB[Math.min(n, maxKb) + 1];
  }
  return total;
}

/** The lesson's opening question: two models scored on 500 test examples each. */
export const OPENING = { accA: 0.912, accB: 0.906, n: 500 } as const;

// ------------------------------------------------------------------ the lab: the dice workshop

export type DieId = 'd2' | 'd4' | 'd6' | 'd8' | 'd12' | 'd20';
export const DICE: { id: DieId; sides: number }[] = [
  { id: 'd2', sides: 2 },
  { id: 'd4', sides: 4 },
  { id: 'd6', sides: 6 },
  { id: 'd8', sides: 8 },
  { id: 'd12', sides: 12 },
  { id: 'd20', sides: 20 },
];
export const MAX_EACH = 8;

export type Recipe = { counts: Record<DieId, number>; plus: number };

export function recipeDist(r: Recipe): Dist {
  let d = point(r.plus);
  for (const die of DICE) d = convolve(d, sumOf(uniformInts(1, die.sides), r.counts[die.id] ?? 0));
  return d;
}

/** Means and variances add for independent dice, so a recipe's mean and SD need no convolution. */
export function recipeMoments(r: Recipe): { mean: number; sd: number; dice: number } {
  let m = r.plus;
  let v = 0;
  let dice = 0;
  for (const die of DICE) {
    const c = r.counts[die.id] ?? 0;
    m += (c * (die.sides + 1)) / 2;
    v += (c * (die.sides * die.sides - 1)) / 12;
    dice += c;
  }
  return { mean: m, sd: Math.sqrt(v), dice };
}

export type WorkshopBrief = {
  id: string;
  title: string;
  story: string;
  mean: number;
  sdLo: number;
  sdHi: number;
  maxGap: number;
  allowPlus: boolean;
};

export const WORKSHOP: WorkshopBrief[] = [
  {
    id: 'board-game',
    title: 'Brief 1 · A board-game roll',
    story: 'A board game needs a roll that averages exactly 10.5, with a standard deviation between 2.5 and 3.5, and clearly bell-shaped: middle values common, extremes rare. Dice only — no adding constants.',
    mean: 10.5,
    sdLo: 2.5,
    sdHi: 3.5,
    maxGap: 0.05,
    allowPlus: false,
  },
  {
    id: 'damage',
    title: 'Brief 2 · A dragon’s bite',
    story: 'A role-playing game wants a damage roll that averages exactly 30, with a standard deviation between 4 and 5, and convincingly bell-shaped. You may add a constant.',
    mean: 30,
    sdLo: 4,
    sdHi: 5,
    maxGap: 0.025,
    allowPlus: true,
  },
  {
    id: 'precise',
    title: 'Brief 3 · Steady hands',
    story: 'A crafting game wants a quality score averaging exactly 50 that rarely strays: standard deviation between 1.5 and 2, and bell-shaped. You may add a constant.',
    mean: 50,
    sdLo: 1.5,
    sdHi: 2,
    maxGap: 0.03,
    allowPlus: true,
  },
];

export type WorkshopCheck = { meanOk: boolean; sdOk: boolean; bellOk: boolean; pass: boolean; mean: number; sd: number; gap: number; dice: number };

export function checkRecipe(b: WorkshopBrief, r: Recipe): WorkshopCheck {
  const mo = recipeMoments(r);
  const plusOk = b.allowPlus || r.plus === 0;
  const meanOk = Math.abs(mo.mean - b.mean) < 1e-9 && plusOk;
  const sdOk = mo.sd >= b.sdLo - 1e-12 && mo.sd <= b.sdHi + 1e-12;
  const gap = mo.dice === 0 ? 1 : bellGap(recipeDist(r));
  const bellOk = gap <= b.maxGap;
  return { meanOk, sdOk, bellOk, pass: meanOk && sdOk && bellOk && mo.dice > 0, mean: mo.mean, sd: mo.sd, gap, dice: mo.dice };
}

/**
 * Exhaustive search: every combination of 0…MAX_EACH of each die (9⁶ = 531,441 recipes), with the constant chosen to
 * hit the mean when allowed. Moments first (cheap), exact distribution only for recipes whose mean and SD qualify.
 */
export function solveWorkshop(b: WorkshopBrief) {
  const ids = DICE.map((d) => d.id);
  const counts = {} as Record<DieId, number>;
  const passing: { r: Recipe; dice: number; gap: number }[] = [];
  let checkedDists = 0;
  const rec = (k: number) => {
    if (k === ids.length) {
      const base = recipeMoments({ counts, plus: 0 });
      if (base.dice === 0) return;
      const plus = b.mean - base.mean;
      if (!Number.isInteger(plus)) return;
      if (!b.allowPlus && plus !== 0) return;
      if (base.sd < b.sdLo - 1e-12 || base.sd > b.sdHi + 1e-12) return;
      const r = { counts: { ...counts }, plus };
      checkedDists++;
      const gap = bellGap(recipeDist(r));
      if (gap <= b.maxGap) passing.push({ r, dice: base.dice, gap });
      return;
    }
    for (let c = 0; c <= MAX_EACH; c++) {
      counts[ids[k]] = c;
      rec(k + 1);
    }
  };
  rec(0);
  const fewest = Math.min(...passing.map((p) => p.dice));
  const fewestRecipes = passing.filter((p) => p.dice === fewest).sort((a, b) => a.gap - b.gap);
  return { passing, fewest, fewestRecipes, best: fewestRecipes[0], checkedDists };
}
