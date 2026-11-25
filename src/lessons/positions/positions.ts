/**
 * Position in transformers: why attention is order-blind, sinusoidal position vectors, rotary
 * position embedding (RoPE), and the "previous-word head" puzzle with exhaustively computed
 * solutions.
 */
import { createRng } from '../../lib/rng';

export type Vec = number[];
const dot = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0);
const matVec = (M: number[][], x: Vec) => M.map((r) => dot(r, x));
export function softmax(z: number[]) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// ---------------------------------------------------------------- order blindness

export const DIM = 6;
const r = createRng(2024);
const rand = (n: number, m: number) => Array.from({ length: n }, () => Array.from({ length: m }, () => Number(r.normal(0, 0.7).toFixed(2))));
/** Word vectors (embeddings) and attention matrices: random but fixed, as in an untrained-but-real layer. */
export const EMB: Record<string, Vec> = { dog: rand(1, DIM)[0], bites: rand(1, DIM)[0], man: rand(1, DIM)[0] };
export const WQ = rand(DIM, DIM);
export const WK = rand(DIM, DIM);
export const WV = rand(DIM, DIM);

/** One attention layer over a sentence; optional position vectors added to the embeddings first. */
export function selfAttention(words: string[], withPositions = false) {
  const X = words.map((w, i) => (withPositions ? EMB[w].map((v, j) => v + sinusoid(i, DIM)[j]) : EMB[w]));
  const Q = X.map((x) => matVec(WQ, x));
  const K = X.map((x) => matVec(WK, x));
  const V = X.map((x) => matVec(WV, x));
  return words.map((w, i) => {
    const wts = softmax(K.map((k) => dot(Q[i], k) / Math.sqrt(DIM)));
    return { word: w, weights: wts, out: V[0].map((_, j) => V.reduce((s, v, t) => s + wts[t] * v[j], 0)) };
  });
}

// ---------------------------------------------------------------- sinusoidal positions

/** The original transformer's position vector: pairs of (sin, cos) turning at geometrically spaced speeds. */
export function sinusoid(pos: number, d: number, base = 10000): Vec {
  return Array.from({ length: d }, (_, j) => {
    const i = Math.floor(j / 2);
    const angle = pos / base ** ((2 * i) / d);
    return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  });
}

// ---------------------------------------------------------------- RoPE in 2D

export const rotate = (v: Vec, angle: number): Vec => [v[0] * Math.cos(angle) - v[1] * Math.sin(angle), v[0] * Math.sin(angle) + v[1] * Math.cos(angle)];

/** Score between a query at position m and a key at position n after rotating each by θ·position. */
export const ropeScore = (q: Vec, k: Vec, m: number, n: number, theta: number) => dot(rotate(q, theta * m), rotate(k, theta * n));

// ---------------------------------------------------------------- the previous-word head

/**
 * A head that should attend to the word just before it. Contents are identical; only position can
 * decide. Each "hand" is a 2D rotary pair turning `speed` degrees per position, with the query aimed
 * one position back (the best possible aim for that goal). A word at the end of a context of
 * `length` words looks back over offsets 0 (itself) … length − 1; the score at offset o is
 * AMP · Σ_hands cos(speed · (o − 1)).
 */
export const AMP = 8;
const deg = Math.PI / 180;

export function previousWordWeights(speeds: number[], length: number) {
  const scores = Array.from({ length }, (_, o) => AMP * speeds.reduce((s, sp) => s + Math.cos(sp * deg * (o - 1)), 0));
  return softmax(scores);
}

export type Level = { id: string; title: string; length: number; hands: number; target: number; brief: string };
export const LEVELS: Level[] = [
  { id: 'short', title: 'A short sentence', length: 8, hands: 1, target: 0.6, brief: 'Eight words. With one hand, make the last word give at least 60% of its attention to the word just before it.' },
  { id: 'long', title: 'A long paragraph', length: 64, hands: 1, target: 0.6, brief: 'Now the context is 64 words long. Same goal, still one hand.' },
  { id: 'two', title: 'Two hands', length: 64, hands: 2, target: 0.6, brief: '64 words again, but now you have two hands turning at different speeds.' },
];

export const SPEEDS = Array.from({ length: 180 }, (_, i) => i + 1); // whole degrees per position

/** Every combination of whole-degree speeds (hands in non-decreasing order) that meets the goal. */
export function solveLevel(L: Level) {
  const sols: { speeds: number[]; weight: number }[] = [];
  let best = { speeds: [] as number[], weight: 0 };
  const consider = (speeds: number[]) => {
    const w = previousWordWeights(speeds, L.length)[1];
    if (w > best.weight) best = { speeds, weight: w };
    if (w >= L.target) sols.push({ speeds, weight: w });
  };
  if (L.hands === 1) for (const s of SPEEDS) consider([s]);
  else for (const a of SPEEDS) for (const b of SPEEDS) if (b >= a) consider([a, b]);
  return { solutions: sols, best, tried: L.hands === 1 ? SPEEDS.length : (SPEEDS.length * (SPEEDS.length + 1)) / 2 };
}

/** Previous-word share for every one-hand speed (index = speed − 1): the post-solve strip. */
export function speedStrip(L: Level): number[] {
  return SPEEDS.map((s) => previousWordWeights([s], L.length)[1]);
}

/** Previous-word share for every pair of speeds, share[a − 1][b − 1] (symmetric), for the two-hand map. */
export function speedPairMap(L: Level): Float64Array {
  const n = SPEEDS.length;
  const out = new Float64Array(n * n);
  for (let a = 1; a <= n; a++)
    for (let b = a; b <= n; b++) {
      const v = previousWordWeights([a, b], L.length)[1];
      out[(a - 1) * n + (b - 1)] = v;
      out[(b - 1) * n + (a - 1)] = v;
    }
  return out;
}
