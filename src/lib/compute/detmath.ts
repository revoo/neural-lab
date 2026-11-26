// authored-by: Claude Opus 5.5
/**
 * Deterministic transcendental functions for the GPT engine's CPU tier.
 *
 * `Math.exp` and `Math.tanh` may differ in the last binary digit between Node and a browser (see
 * the gotchas in HANDOFF.md), and a long training run amplifies that into different samples. These
 * versions use only `+ − × ÷`, `Math.round` and bit-exact powers of two, which IEEE 754 defines
 * exactly, so every JavaScript engine returns the same bits. That makes a CPU-tier training run in
 * the learner's browser reproduce the run recorded in Node bit for bit.
 *
 * Accuracy: relative error below 1e-15 for `dexp` (Taylor series to degree 13 after reducing the
 * argument to |r| ≤ ln2/2), which is far below float32 resolution.
 */

const LN2_HI = 6.93147180369123816490e-1; // trailing zero bits, so k·LN2_HI is exact for |k| < 2¹¹
const LN2_LO = 1.90821492927058770002e-10;
const INV_LN2 = 1.4426950408889634;

const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer);
// little-endian on every platform JavaScript runs on in practice; checked once here
const HI = (() => {
  f64[0] = 1;
  return u32[1] === 0x3ff00000 ? 1 : 0;
})();

/** 2^k for an integer −1022 ≤ k ≤ 1023, built from its bit pattern (exact). */
function pow2(k: number): number {
  u32[HI] = (k + 1023) << 20;
  u32[1 - HI] = 0;
  return f64[0];
}

const C = (() => {
  // 1/n! for n = 0 … 13
  const c: number[] = [1];
  for (let n = 1; n <= 13; n++) c.push(c[n - 1] / n);
  return c;
})();

export function dexp(x: number): number {
  if (x !== x) return NaN;
  if (x > 709.78) return Infinity;
  if (x < -745) return 0;
  const k = Math.round(x * INV_LN2);
  const r = x - k * LN2_HI - k * LN2_LO;
  let p = C[13];
  for (let n = 12; n >= 0; n--) p = p * r + C[n];
  if (k > 1023) return p * pow2(1023) * pow2(k - 1023);
  if (k < -1022) return p * pow2(-1022) * pow2(k + 1022);
  return p * pow2(k);
}

export function dtanh(x: number): number {
  if (x > 20) return 1;
  if (x < -20) return -1;
  const e = dexp(2 * x);
  return (e - 1) / (e + 1);
}

/**
 * Approximately normal draw: the sum of 12 uniforms minus 6 (Irwin–Hall). Mean 0, variance 1,
 * bounded at ±6. Used for weight initialisation instead of Box–Muller, which needs `Math.log` and
 * `Math.cos` and so isn't bit-identical across engines.
 */
export function irwinHall(next: () => number): number {
  let s = 0;
  for (let i = 0; i < 12; i++) s += next();
  return s - 6;
}
