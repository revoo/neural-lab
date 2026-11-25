// authored-by: Neon High
/**
 * Bit-level floating-point formats (FP32, FP16, BF16, FP8 E4M3), symmetric per-tensor and
 * group-wise integer quantisation (INT8, INT4), outlier catastrophe demo, and a 4-block
 * mixed-precision quantisation lab with exhaustive search over all 625 configurations.
 */
import { createRng } from '../../lib/rng';

export type FloatSpec = {
  id: 'fp32' | 'fp16' | 'bf16' | 'fp8';
  name: string;
  bits: number;
  expBits: number;
  mantBits: number;
  bias: number;
  hasInf: boolean;
  maxFinite: number;
  minNormal: number;
  blurb: string;
};

export function decodeFloatBits(sign: number, exp: number, mant: number, expBits: number, mantBits: number, hasInf = true): number {
  const bias = (1 << (expBits - 1)) - 1;
  const maxExp = (1 << expBits) - 1;
  const s = sign ? -1 : 1;
  if (exp === maxExp && hasInf) {
    return mant === 0 ? s * Infinity : NaN;
  }
  if (exp === maxExp && !hasInf && mant === (1 << mantBits) - 1) return NaN;
  if (exp === 0) {
    return s * Math.pow(2, 1 - bias) * (mant / Math.pow(2, mantBits));
  }
  return s * Math.pow(2, exp - bias) * (1 + mant / Math.pow(2, mantBits));
}

export const FLOAT_FORMATS: FloatSpec[] = [
  {
    id: 'fp32',
    name: 'FP32',
    bits: 32,
    expBits: 8,
    mantBits: 23,
    bias: 127,
    hasInf: true,
    maxFinite: decodeFloatBits(0, 254, (1 << 23) - 1, 8, 23, true),
    minNormal: Math.pow(2, -126),
    blurb: 'Standard 32-bit float: huge range (up to 3.4 × 10³⁸) and ~7 decimal digits of precision.',
  },
  {
    id: 'fp16',
    name: 'FP16',
    bits: 16,
    expBits: 5,
    mantBits: 10,
    bias: 15,
    hasInf: true,
    maxFinite: decodeFloatBits(0, 30, 1023, 5, 10, true),
    minNormal: Math.pow(2, -14),
    blurb: 'Half precision: sharp ~3.3 digits, but only 5 exponent bits — overflows above 65,504.',
  },
  {
    id: 'bf16',
    name: 'BF16',
    bits: 16,
    expBits: 8,
    mantBits: 7,
    bias: 127,
    hasInf: true,
    maxFinite: decodeFloatBits(0, 254, 127, 8, 7, true),
    minNormal: Math.pow(2, -126),
    blurb: 'Brain Float 16: keeps FP32’s full 8-bit exponent range (up to 3.4 × 10³⁸) with 7 mantissa bits.',
  },
  {
    id: 'fp8',
    name: 'FP8 (E4M3)',
    bits: 8,
    expBits: 4,
    mantBits: 3,
    bias: 7,
    hasInf: false,
    maxFinite: decodeFloatBits(0, 15, 6, 4, 3, false),
    minNormal: Math.pow(2, -6),
    blurb: '8-bit float used on H100/Blackwell: max finite 448 (saturates instead of Inf), coarse steps.',
  },
];

export function encodeFloatBits(x: number, spec: FloatSpec): { sign: number; exp: number; mant: number; value: number } {
  if (Number.isNaN(x)) {
    return { sign: 0, exp: (1 << spec.expBits) - 1, mant: (1 << spec.mantBits) - 1, value: NaN };
  }
  const sign = x < 0 || Object.is(x, -0) ? 1 : 0;
  const ax = Math.abs(x);
  if (ax === 0) return { sign, exp: 0, mant: 0, value: sign ? -0 : 0 };

  const maxExp = (1 << spec.expBits) - 1;
  const maxFiniteExp = spec.hasInf ? maxExp - 1 : maxExp;
  const maxFiniteMant = spec.hasInf ? (1 << spec.mantBits) - 1 : (1 << spec.mantBits) - 2;

  if (ax > spec.maxFinite) {
    if (spec.hasInf) return { sign, exp: maxExp, mant: 0, value: sign ? -Infinity : Infinity };
    return { sign, exp: maxFiniteExp, mant: maxFiniteMant, value: sign ? -spec.maxFinite : spec.maxFinite };
  }

  const e = Math.floor(Math.log2(ax));
  const biased = e + spec.bias;
  if (biased <= 0) {
    const step = Math.pow(2, 1 - spec.bias - spec.mantBits);
    const m = Math.round(ax / step);
    if (m >= 1 << spec.mantBits) {
      return { sign, exp: 1, mant: 0, value: decodeFloatBits(sign, 1, 0, spec.expBits, spec.mantBits, spec.hasInf) };
    }
    return { sign, exp: 0, mant: m, value: decodeFloatBits(sign, 0, m, spec.expBits, spec.mantBits, spec.hasInf) };
  }
  const scale = Math.pow(2, e - spec.mantBits);
  let m = Math.round((ax - Math.pow(2, e)) / scale);
  let expOut = biased;
  if (m >= 1 << spec.mantBits) {
    m = 0;
    expOut++;
  }
  if (expOut > maxFiniteExp || (!spec.hasInf && expOut === maxFiniteExp && m > maxFiniteMant)) {
    if (spec.hasInf) return { sign, exp: maxExp, mant: 0, value: sign ? -Infinity : Infinity };
    return { sign, exp: maxFiniteExp, mant: maxFiniteMant, value: sign ? -spec.maxFinite : spec.maxFinite };
  }
  return { sign, exp: expOut, mant: m, value: decodeFloatBits(sign, expOut, m, spec.expBits, spec.mantBits, spec.hasInf) };
}

export function quantizeFloat(x: number, spec: FloatSpec): number {
  return encodeFloatBits(x, spec).value;
}

// ------------------------------------------------------------------ Symmetric integer quantisation & outlier demo

export type IntQuantResult = {
  reconstructed: number[];
  codes: number[];
  scales: number[];
  rmse: number;
  normalRmse: number;
  normalZeros: number;
  effBits: number;
};

/** Fixed 64-weight sample from N(0, 0.35²) used by the outlier explorer (index 0 is replaced by the outlier slider). */
export const BASE_WEIGHTS: readonly number[] = (() => {
  const rng = createRng(11);
  return Array.from({ length: 64 }, () => rng.normal(0, 0.35));
})();

export function quantizeIntSymmetric(weights: readonly number[], bits: 4 | 8, groupSize: number): IntQuantResult {
  const qMax = (1 << (bits - 1)) - 1;
  const reconstructed = new Array<number>(weights.length);
  const codes = new Array<number>(weights.length);
  const scales: number[] = [];

  for (let start = 0; start < weights.length; start += groupSize) {
    const end = Math.min(weights.length, start + groupSize);
    let maxAbs = 0;
    for (let i = start; i < end; i++) maxAbs = Math.max(maxAbs, Math.abs(weights[i]));
    const scale = maxAbs > 0 ? maxAbs / qMax : 1;
    scales.push(scale);
    for (let i = start; i < end; i++) {
      const q = Math.max(-qMax, Math.min(qMax, Math.round(weights[i] / scale)));
      codes[i] = q;
      reconstructed[i] = q * scale;
    }
  }

  let sqAll = 0;
  let sqNorm = 0;
  let normalZeros = 0;
  for (let i = 0; i < weights.length; i++) {
    const err = weights[i] - reconstructed[i];
    sqAll += err * err;
    if (i > 0) {
      sqNorm += err * err;
      if (codes[i] === 0) normalZeros++;
    }
  }
  const rmse = Math.sqrt(sqAll / weights.length);
  const normalRmse = weights.length > 1 ? Math.sqrt(sqNorm / (weights.length - 1)) : rmse;
  // Each group stores one FP16 scale factor (16 bits shared across `groupSize` weights)
  const effBits = bits + 16 / groupSize;
  return { reconstructed, codes, scales, rmse, normalRmse, normalZeros, effBits };
}

// ------------------------------------------------------------------ Mixed-precision 4-block lab (5^4 = 625 exhaustive)

export type QuantSchemeId = 'fp16' | 'int8' | 'int4-tensor' | 'int4-g16' | 'int4-g8';

export type QuantScheme = {
  id: QuantSchemeId;
  label: string;
  effBits: number;
  apply: (w: readonly number[]) => number[];
};

const FP16_SPEC = FLOAT_FORMATS[1];

export const LAB_SCHEMES: QuantScheme[] = [
  { id: 'fp16', label: 'FP16 (16 b)', effBits: 16, apply: (w) => w.map((x) => quantizeFloat(x, FP16_SPEC)) },
  { id: 'int8', label: 'INT8 (8.25 b)', effBits: 8.25, apply: (w) => quantizeIntSymmetric(w, 8, 64).reconstructed },
  { id: 'int4-tensor', label: 'INT4 tensor (4.25 b)', effBits: 4.25, apply: (w) => quantizeIntSymmetric(w, 4, 64).reconstructed },
  { id: 'int4-g16', label: 'INT4 g=16 (5.0 b)', effBits: 5.0, apply: (w) => quantizeIntSymmetric(w, 4, 16).reconstructed },
  { id: 'int4-g8', label: 'INT4 g=8 (6.0 b)', effBits: 6.0, apply: (w) => quantizeIntSymmetric(w, 4, 8).reconstructed },
];

export type WeightBlock = {
  id: string;
  name: string;
  character: string;
  weights: number[];
  maxAbs: number;
};

export const LAB_BLOCKS: WeightBlock[] = (() => {
  const rng = createRng(23);
  const make = (std: number, outliers: [number, number][]) => {
    const w = Array.from({ length: 64 }, () => rng.normal(0, std));
    for (const [idx, val] of outliers) w[idx] = val;
    return w;
  };
  const specs = [
    { id: 'qkv', name: 'Attn QKV', character: '2 sharp outliers (+7.2, −6.4)', weights: make(0.3, [[3, 7.2], [41, -6.4]]) },
    { id: 'out', name: 'Attn Output', character: '1 mild outlier (+2.8)', weights: make(0.35, [[12, 2.8]]) },
    { id: 'ffn-up', name: 'FFN Up', character: 'smooth Gaussian (no outliers)', weights: make(0.32, []) },
    { id: 'ffn-down', name: 'FFN Down', character: 'smooth Gaussian (no outliers)', weights: make(0.28, []) },
  ];
  return specs.map((s) => ({ ...s, maxAbs: Math.max(...s.weights.map(Math.abs)) }));
})();

export function blockRmse(weights: readonly number[], reconstructed: readonly number[]): number {
  let sq = 0;
  for (let i = 0; i < weights.length; i++) sq += (weights[i] - reconstructed[i]) ** 2;
  return Math.sqrt(sq / weights.length);
}

export function evaluateMix(choiceIndices: readonly [number, number, number, number]) {
  let sumBits = 0;
  let sumRmse = 0;
  const perBlock = choiceIndices.map((sIdx, bIdx) => {
    const scheme = LAB_SCHEMES[sIdx];
    const block = LAB_BLOCKS[bIdx];
    const rmse = blockRmse(block.weights, scheme.apply(block.weights));
    sumBits += scheme.effBits;
    sumRmse += rmse;
    return { blockId: block.id, schemeId: scheme.id, effBits: scheme.effBits, rmse };
  });
  return { avgBits: sumBits / 4, avgRmse: sumRmse / 4, perBlock };
}

export const LAB_BIT_BUDGET = 5.75;
export const LAB_RMSE_GOAL = 0.045;

/** Exhaustively check all 5⁴ = 625 mixed-precision assignments over the 4 blocks. */
export function solveQuantLab() {
  let total = 0;
  let passing = 0;
  let best = { choices: [0, 0, 0, 0] as [number, number, number, number], avgBits: 0, avgRmse: Infinity };

  for (let a = 0; a < LAB_SCHEMES.length; a++) {
    for (let b = 0; b < LAB_SCHEMES.length; b++) {
      for (let c = 0; c < LAB_SCHEMES.length; c++) {
        for (let d = 0; d < LAB_SCHEMES.length; d++) {
          total++;
          const choices: [number, number, number, number] = [a, b, c, d];
          const res = evaluateMix(choices);
          if (res.avgBits <= LAB_BIT_BUDGET + 1e-9) {
            if (res.avgRmse <= LAB_RMSE_GOAL) passing++;
            if (res.avgRmse < best.avgRmse) {
              best = { choices, avgBits: res.avgBits, avgRmse: res.avgRmse };
            }
          }
        }
      }
    }
  }
  return { total, passing, best };
}
