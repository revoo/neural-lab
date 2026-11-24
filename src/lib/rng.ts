/**
 * Small seeded random number generator (mulberry32).
 * Every experiment that uses randomness takes a seed so results are reproducible.
 */
export type Rng = {
  /** Uniform in [0, 1). */
  next(): number;
  /** Uniform in [lo, hi). */
  uniform(lo: number, hi: number): number;
  /** Standard normal sample (Box–Muller). */
  normal(mean?: number, std?: number): number;
  /** Integer in [0, n). */
  int(n: number): number;
};

export function createRng(seed: number): Rng {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    normal: (mean = 0, std = 1) => {
      let u = 0;
      while (u === 0) u = next();
      const v = next();
      return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
    int: (n) => Math.floor(next() * n),
  };
}
