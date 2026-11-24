/** Plain-array linear algebra. Deliberately tiny and loop-based so it mirrors what lessons teach. */
export type Vec = number[];
export type Mat = number[][]; // row-major: m[row][col]

export const dot = (a: Vec, b: Vec): number => {
  if (a.length !== b.length) throw new Error(`dot: length mismatch ${a.length} vs ${b.length}`);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export const add = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i]);
export const sub = (a: Vec, b: Vec): Vec => a.map((v, i) => v - b[i]);
export const scale = (a: Vec, k: number): Vec => a.map((v) => v * k);
export const norm = (a: Vec): number => Math.sqrt(dot(a, a));

export const cosineSimilarity = (a: Vec, b: Vec): number => {
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot(a, b) / d;
};

/** Angle between two vectors in radians, in [0, π]. */
export const angleBetween = (a: Vec, b: Vec): number =>
  Math.acos(Math.max(-1, Math.min(1, cosineSimilarity(a, b))));

export const shape = (m: Mat): [number, number] => [m.length, m[0]?.length ?? 0];

export const matVec = (m: Mat, v: Vec): Vec => {
  if (m[0].length !== v.length) throw new Error('matVec: shape mismatch');
  return m.map((row) => dot(row, v));
};

export const matMul = (a: Mat, b: Mat): Mat => {
  const [n, k] = shape(a);
  const [k2, p] = shape(b);
  if (k !== k2) throw new Error(`matMul: (${n}×${k}) · (${k2}×${p}) inner dimensions differ`);
  const out: Mat = [];
  for (let i = 0; i < n; i++) {
    out.push([]);
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += a[i][t] * b[t][j];
      out[i].push(s);
    }
  }
  return out;
};

export const transpose = (m: Mat): Mat => m[0].map((_, j) => m.map((row) => row[j]));

export const identity = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

export const det2 = (m: Mat): number => m[0][0] * m[1][1] - m[0][1] * m[1][0];

/** Linear interpolation between two matrices of the same shape. */
export const lerpMat = (a: Mat, b: Mat, t: number): Mat =>
  a.map((row, i) => row.map((v, j) => v + (b[i][j] - v) * t));

/**
 * Result shape of elementwise broadcasting between two shapes (NumPy rules), or an error message.
 * Shapes are aligned from the right; each pair of dims must be equal or one of them must be 1.
 */
export function broadcastShape(a: number[], b: number[]): { shape: number[] } | { error: string } {
  const n = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      return { error: `dimension ${n - 1 - i} from the right: ${da} vs ${db} (neither is 1)` };
    }
    out.unshift(Math.max(da, db));
  }
  return { shape: out };
}
