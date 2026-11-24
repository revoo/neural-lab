/**
 * A neural network written the way you'd write it in NumPy: whole batches as matrices, one line
 * per operation. Rows are examples; weights are stored (inputs × outputs), the NumPy convention
 * (the shared lib/mlp.ts stores them the other way round, outputs × inputs — same maths).
 *
 * Shape errors are raised exactly where NumPy would raise them, so the "write the backward pass"
 * lab can report them like a real interpreter.
 */
import { createRng } from '../../lib/rng';

export type Mat = number[][];

export class ShapeError extends Error {}

export const shape = (A: Mat): [number, number] => [A.length, A[0]?.length ?? 0];
export const fmtShape = (A: Mat) => `(${shape(A).join(', ')})`;

export function matmul(A: Mat, B: Mat): Mat {
  const [n, k] = shape(A);
  const [k2, m] = shape(B);
  if (k !== k2) throw new ShapeError(`matmul: shapes ${fmtShape(A)} and ${fmtShape(B)} not aligned: ${k} (dim 1) ≠ ${k2} (dim 0)`);
  return Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => {
    let s = 0;
    for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
    return s;
  }));
}

export const T = (A: Mat): Mat => A[0].map((_, j) => A.map((r) => r[j]));

/** Elementwise combination with NumPy-style broadcasting: a dimension of size 1 stretches to match. */
export function ew(A: Mat, B: Mat, f: (a: number, b: number) => number, op = 'operate'): Mat {
  const [n, m] = shape(A);
  const [p, q] = shape(B);
  const rows = n === p ? n : n === 1 ? p : p === 1 ? n : -1;
  const cols = m === q ? m : m === 1 ? q : q === 1 ? m : -1;
  if (rows < 0 || cols < 0) throw new ShapeError(`operands could not be broadcast together with shapes ${fmtShape(A)} ${fmtShape(B)} (${op})`);
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => f(A[n === 1 ? 0 : i][m === 1 ? 0 : j], B[p === 1 ? 0 : i][q === 1 ? 0 : j])),
  );
}
export const add = (A: Mat, B: Mat) => ew(A, B, (a, b) => a + b, '+');
export const sub = (A: Mat, B: Mat) => ew(A, B, (a, b) => a - b, '-');
export const mul = (A: Mat, B: Mat) => ew(A, B, (a, b) => a * b, '*');
export const map = (A: Mat, f: (v: number) => number): Mat => A.map((r) => r.map(f));
/** Sum over the rows (axis 0), keeping a (1, m) row: what `.sum(axis=0, keepdims=True)` gives. */
export const sum0 = (A: Mat): Mat => [A[0].map((_, j) => A.reduce((s, r) => s + r[j], 0))];
export const scale = (A: Mat, k: number) => map(A, (v) => v * k);

export const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

// ---------------------------------------------------------------- the network

export type Params = { W1: Mat; b1: Mat; W2: Mat; b2: Mat };

/** XOR: four examples, rows of X; labels as a (4, 1) column. */
export const X_XOR: Mat = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
export const Y_XOR: Mat = [[0], [1], [1], [0]];

export function init(hidden: number, seed: number): Params {
  const r = createRng(seed);
  const s1 = Math.sqrt(2 / (2 + hidden));
  const s2 = Math.sqrt(2 / (hidden + 1));
  return {
    W1: Array.from({ length: 2 }, () => Array.from({ length: hidden }, () => r.normal(0, s1))),
    b1: [Array.from({ length: hidden }, () => 0)],
    W2: Array.from({ length: hidden }, () => [r.normal(0, s2)]),
    b2: [[0]],
  };
}

export type Env = Params & { X: Mat; y: Mat; n: number; z1?: Mat; h?: Mat; z2?: Mat; p?: Mat; loss?: number; dz2?: Mat; dW2?: Mat; db2?: Mat; dh?: Mat; dz1?: Mat; dW1?: Mat; db1?: Mat };

export function forward(P: Params, X: Mat, y: Mat): Env {
  const z1 = add(matmul(X, P.W1), P.b1);
  const h = map(z1, Math.tanh);
  const z2 = add(matmul(h, P.W2), P.b2);
  const p = map(z2, sigmoid);
  const n = X.length;
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const q = Math.min(Math.max(p[i][0], 1e-12), 1 - 1e-12);
    loss += -(y[i][0] * Math.log(q) + (1 - y[i][0]) * Math.log(1 - q));
  }
  return { ...P, X, y, n, z1, h, z2, p, loss: loss / n };
}

export const lossOf = (P: Params, X = X_XOR, y = Y_XOR) => forward(P, X, y).loss!;

// ---------------------------------------------------------------- the backward pass, line by line

export type Slot = 'dz2' | 'dW2' | 'dh' | 'dz1' | 'dW1';
export type Option = { code: string; f: (e: Env) => Mat };

/**
 * Candidate expressions for each blank of the "write the backward pass" lab. Several are correct
 * (the same maths written differently); the rest are wrong in instructive ways: wrong shape (the
 * interpreter refuses), or right shape but wrong values (only the gradient check catches it).
 */
export const OPTIONS: Record<Slot, Option[]> = {
  dz2: [
    { code: '(p - y) / n', f: (e) => scale(sub(e.p!, e.y), 1 / e.n) },
    { code: '(p - y) * p * (1 - p) / n', f: (e) => scale(mul(mul(sub(e.p!, e.y), e.p!), map(e.p!, (v) => 1 - v)), 1 / e.n) },
    { code: '(y - p) / n', f: (e) => scale(sub(e.y, e.p!), 1 / e.n) },
    { code: '(p - y)', f: (e) => sub(e.p!, e.y) },
  ],
  dW2: [
    { code: 'h.T @ dz2', f: (e) => matmul(T(e.h!), e.dz2!) },
    { code: '(dz2.T @ h).T', f: (e) => T(matmul(T(e.dz2!), e.h!)) },
    { code: 'h @ dz2', f: (e) => matmul(e.h!, e.dz2!) },
    { code: 'dz2.T @ h', f: (e) => matmul(T(e.dz2!), e.h!) },
  ],
  dh: [
    { code: 'dz2 @ W2.T', f: (e) => matmul(e.dz2!, T(e.W2)) },
    { code: '(W2 @ dz2.T).T', f: (e) => T(matmul(e.W2, T(e.dz2!))) },
    { code: 'dz2 @ W2', f: (e) => matmul(e.dz2!, e.W2) },
    { code: 'dz2 * W2.T', f: (e) => mul(e.dz2!, T(e.W2)) },
  ],
  dz1: [
    { code: 'dh * (1 - h**2)', f: (e) => mul(e.dh!, map(e.h!, (v) => 1 - v * v)) },
    { code: 'dh * (1 - tanh(z1)**2)', f: (e) => mul(e.dh!, map(e.z1!, (v) => 1 - Math.tanh(v) ** 2)) },
    { code: 'dh * (1 - h)', f: (e) => mul(e.dh!, map(e.h!, (v) => 1 - v)) },
    { code: 'dh @ (1 - h**2).T', f: (e) => matmul(e.dh!, T(map(e.h!, (v) => 1 - v * v))) },
  ],
  dW1: [
    { code: 'X.T @ dz1', f: (e) => matmul(T(e.X), e.dz1!) },
    { code: '(dz1.T @ X).T', f: (e) => T(matmul(T(e.dz1!), e.X)) },
    { code: 'X @ dz1', f: (e) => matmul(e.X, e.dz1!) },
    { code: 'X.T @ dh', f: (e) => matmul(T(e.X), e.dh!) },
  ],
};

export const SLOTS: Slot[] = ['dz2', 'dW2', 'dh', 'dz1', 'dW1'];
/** What each blank is the gradient of. */
export const PARTNER: Record<Slot, 'z2' | 'W2' | 'h' | 'z1' | 'W1'> = { dz2: 'z2', dW2: 'W2', dh: 'h', dz1: 'z1', dW1: 'W1' };
export type Choice = Record<Slot, number>;
export const REFERENCE: Choice = { dz2: 0, dW2: 0, dh: 0, dz1: 0, dW1: 0 };

export type BackwardResult =
  | { ok: true; grads: { W1: Mat; b1: Mat; W2: Mat; b2: Mat }; env: Env }
  | { ok: false; slot: Slot; error: string; env: Env };

/** Run the backward pass using the chosen expression for each blank (db1, db2 are given). */
export function backward(P: Params, choice: Choice, X = X_XOR, y = Y_XOR): BackwardResult {
  const env = forward(P, X, y);
  for (const s of SLOTS) {
    try {
      env[s] = OPTIONS[s][choice[s]].f(env);
    } catch (e) {
      return { ok: false, slot: s, error: (e as Error).message, env };
    }
    // a gradient always has exactly the shape of the thing it is the gradient of
    const of = PARTNER[s];
    const want = fmtShape(env[of] as Mat);
    if (fmtShape(env[s]!) !== want)
      return { ok: false, slot: s, error: `${s} has shape ${fmtShape(env[s]!)}, but it is the gradient of ${of}, which has shape ${want}`, env };
    if (s === 'dz2') env.db2 = sum0(env.dz2!);
    if (s === 'dz1') env.db1 = sum0(env.dz1!);
  }
  return { ok: true, grads: { W1: env.dW1!, b1: env.db1!, W2: env.dW2!, b2: env.db2! }, env };
}

/** Central-difference numerical gradient of the loss for every parameter. */
export function numericalGrads(P: Params, X = X_XOR, y = Y_XOR, h = 1e-6) {
  const out = {} as Record<keyof Params, Mat>;
  for (const k of ['W1', 'b1', 'W2', 'b2'] as const) {
    out[k] = P[k].map((r, i) =>
      r.map((_, j) => {
        const orig = P[k][i][j];
        P[k][i][j] = orig + h;
        const up = lossOf(P, X, y);
        P[k][i][j] = orig - h;
        const dn = lossOf(P, X, y);
        P[k][i][j] = orig;
        return (up - dn) / (2 * h);
      }),
    );
  }
  return out;
}

/** Largest relative disagreement between two gradients (0 = identical). */
export function relError(a: Mat, b: Mat) {
  let m = 0;
  a.forEach((r, i) => r.forEach((v, j) => (m = Math.max(m, Math.abs(v - b[i][j]) / Math.max(1e-8, Math.abs(v) + Math.abs(b[i][j]))))));
  return m;
}

export function gradCheck(P: Params, choice: Choice) {
  const r = backward(P, choice);
  if (!r.ok) return { ok: false as const, error: r.error, slot: r.slot };
  const num = numericalGrads(P);
  const errs = { W1: relError(r.grads.W1, num.W1), b1: relError(r.grads.b1, num.b1), W2: relError(r.grads.W2, num.W2), b2: relError(r.grads.b2, num.b2) };
  return { ok: true as const, errs, pass: Object.values(errs).every((e) => e < 1e-5) };
}

/** Plain gradient descent with the chosen backward pass. Returns the loss after each step. */
export function train(choice: Choice, hidden: number, lr: number, steps: number, seed = 1) {
  const P = init(hidden, seed);
  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    const r = backward(P, choice);
    if (!r.ok) return { P, losses, error: r.error };
    losses.push(r.env.loss!);
    for (const k of ['W1', 'b1', 'W2', 'b2'] as const) P[k] = sub(P[k], scale(r.grads[k], lr));
    if (!Number.isFinite(r.env.loss!)) break;
  }
  return { P, losses, error: null as string | null };
}

export const accuracy = (P: Params) => forward(P, X_XOR, Y_XOR).p!.filter((r, i) => (r[0] > 0.5 ? 1 : 0) === Y_XOR[i][0]).length / 4;

/** Steps until the loss first drops below `target` (null if never within maxSteps). */
export function stepsToSolve(hidden: number, lr: number, target = 0.05, maxSteps = 5000, seed = 1) {
  const P = init(hidden, seed);
  for (let s = 0; s < maxSteps; s++) {
    const r = backward(P, REFERENCE);
    if (!r.ok) return null;
    if (!Number.isFinite(r.env.loss!)) return null;
    if (r.env.loss! < target) return s;
    for (const k of ['W1', 'b1', 'W2', 'b2'] as const) P[k] = sub(P[k], scale(r.grads[k], lr));
  }
  return null;
}
