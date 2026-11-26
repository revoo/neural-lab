// authored-by: Claude Opus 5.5
/**
 * V2-WP3, test / fallback tier: `Float32Array` tensors with plain-loop kernels and exact backward
 * passes. No toolchain, runs anywhere (Node, Vitest, every browser). Each backward pass is checked
 * against central finite differences in `compute.test.ts`, and `CpuMlpEngine` reproduces
 * `src/lib/mlp.ts` + `src/lib/optimizers.ts` step for step (to float32 rounding).
 */
import { paramLayout, type ActName, type MlpEngine, type MlpSpec, type TrainCall, type Weights } from './types';

export type Tensor2D = { rows: number; cols: number; data: Float32Array };

export const tensor = (rows: number, cols: number, data?: ArrayLike<number>): Tensor2D => ({
  rows,
  cols,
  data: data ? Float32Array.from(data) : new Float32Array(rows * cols),
});

/** C = A·B  (A: m×k, B: k×n). Loop order i-k-j keeps the inner loop on contiguous memory. */
export function matmul(A: Tensor2D, B: Tensor2D): Tensor2D {
  if (A.cols !== B.rows) throw new Error(`matmul ${A.rows}×${A.cols} · ${B.rows}×${B.cols}`);
  const C = tensor(A.rows, B.cols);
  const n = B.cols;
  for (let i = 0; i < A.rows; i++) {
    const ci = i * n;
    for (let k = 0; k < A.cols; k++) {
      const a = A.data[i * A.cols + k];
      if (a === 0) continue;
      const bk = k * n;
      for (let j = 0; j < n; j++) C.data[ci + j] += a * B.data[bk + j];
    }
  }
  return C;
}

/** C = A·Bᵀ  (A: m×k, B: n×k): a layer's pre-activations from inputs and W[out × in]. */
export function matmulTB(A: Tensor2D, B: Tensor2D): Tensor2D {
  if (A.cols !== B.cols) throw new Error('matmulTB shape');
  const C = tensor(A.rows, B.rows);
  const k = A.cols;
  for (let i = 0; i < A.rows; i++)
    for (let j = 0; j < B.rows; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += A.data[i * k + t] * B.data[j * k + t];
      C.data[i * B.rows + j] = s;
    }
  return C;
}

/** C = Aᵀ·B  (A: k×m, B: k×n): weight gradients δᵀ·a. */
export function matmulTA(A: Tensor2D, B: Tensor2D): Tensor2D {
  if (A.rows !== B.rows) throw new Error('matmulTA shape');
  const C = tensor(A.cols, B.cols);
  const n = B.cols;
  for (let t = 0; t < A.rows; t++)
    for (let i = 0; i < A.cols; i++) {
      const a = A.data[t * A.cols + i];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) C.data[i * n + j] += a * B.data[t * n + j];
    }
  return C;
}

export function addBias(Z: Tensor2D, b: Float32Array): Tensor2D {
  for (let i = 0; i < Z.rows; i++) for (let j = 0; j < Z.cols; j++) Z.data[i * Z.cols + j] += b[j];
  return Z;
}

const SQRT_2_PI = Math.sqrt(2 / Math.PI);
export function act(name: ActName, z: number): number {
  switch (name) {
    case 'linear':
      return z;
    case 'relu':
      return z > 0 ? z : 0;
    case 'tanh':
      return Math.tanh(z);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-z));
    case 'gelu':
      return 0.5 * z * (1 + Math.tanh(SQRT_2_PI * (z + 0.044715 * z * z * z)));
  }
}
/** d act / dz, given the pre-activation z and the activation a = act(z). */
export function actGrad(name: ActName, z: number, a: number): number {
  switch (name) {
    case 'linear':
      return 1;
    case 'relu':
      return z > 0 ? 1 : 0;
    case 'tanh':
      return 1 - a * a;
    case 'sigmoid':
      return a * (1 - a);
    case 'gelu': {
      const u = SQRT_2_PI * (z + 0.044715 * z * z * z);
      const t = Math.tanh(u);
      return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * SQRT_2_PI * (1 + 3 * 0.044715 * z * z);
    }
  }
}

export function actForward(name: ActName, Z: Tensor2D): Tensor2D {
  const A = tensor(Z.rows, Z.cols);
  for (let i = 0; i < Z.data.length; i++) A.data[i] = act(name, Z.data[i]);
  return A;
}
export function actBackward(name: ActName, Z: Tensor2D, A: Tensor2D, dA: Tensor2D): Tensor2D {
  const dZ = tensor(Z.rows, Z.cols);
  for (let i = 0; i < Z.data.length; i++) dZ.data[i] = dA.data[i] * actGrad(name, Z.data[i], A.data[i]);
  return dZ;
}

/** Row-wise softmax with the max subtracted for stability. */
export function softmaxRows(Z: Tensor2D): Tensor2D {
  const P = tensor(Z.rows, Z.cols);
  for (let i = 0; i < Z.rows; i++) {
    let mx = -Infinity;
    for (let j = 0; j < Z.cols; j++) mx = Math.max(mx, Z.data[i * Z.cols + j]);
    let s = 0;
    for (let j = 0; j < Z.cols; j++) s += P.data[i * Z.cols + j] = Math.exp(Z.data[i * Z.cols + j] - mx);
    for (let j = 0; j < Z.cols; j++) P.data[i * Z.cols + j] /= s;
  }
  return P;
}
/** Backward through softmax: dZ = P ⊙ (dP − Σⱼ dPⱼ Pⱼ), row by row. */
export function softmaxBackward(P: Tensor2D, dP: Tensor2D): Tensor2D {
  const dZ = tensor(P.rows, P.cols);
  for (let i = 0; i < P.rows; i++) {
    let dot = 0;
    for (let j = 0; j < P.cols; j++) dot += dP.data[i * P.cols + j] * P.data[i * P.cols + j];
    for (let j = 0; j < P.cols; j++) dZ.data[i * P.cols + j] = P.data[i * P.cols + j] * (dP.data[i * P.cols + j] - dot);
  }
  return dZ;
}

export type LayerNormCache = { xhat: Tensor2D; rstd: Float32Array };
/** Row-wise layer norm: y = γ ⊙ (x − mean)/√(var + ε) + β. */
export function layerNorm(X: Tensor2D, gamma: Float32Array, beta: Float32Array, eps = 1e-5): { Y: Tensor2D; cache: LayerNormCache } {
  const Y = tensor(X.rows, X.cols);
  const xhat = tensor(X.rows, X.cols);
  const rstd = new Float32Array(X.rows);
  const d = X.cols;
  for (let i = 0; i < X.rows; i++) {
    let mean = 0;
    for (let j = 0; j < d; j++) mean += X.data[i * d + j];
    mean /= d;
    let v = 0;
    for (let j = 0; j < d; j++) v += (X.data[i * d + j] - mean) ** 2;
    v /= d;
    const r = 1 / Math.sqrt(v + eps);
    rstd[i] = r;
    for (let j = 0; j < d; j++) {
      const h = (X.data[i * d + j] - mean) * r;
      xhat.data[i * d + j] = h;
      Y.data[i * d + j] = gamma[j] * h + beta[j];
    }
  }
  return { Y, cache: { xhat, rstd } };
}
export function layerNormBackward(dY: Tensor2D, gamma: Float32Array, cache: LayerNormCache) {
  const { xhat, rstd } = cache;
  const d = dY.cols;
  const dX = tensor(dY.rows, d);
  const dgamma = new Float32Array(d);
  const dbeta = new Float32Array(d);
  for (let i = 0; i < dY.rows; i++) {
    let s1 = 0;
    let s2 = 0;
    for (let j = 0; j < d; j++) {
      const g = dY.data[i * d + j];
      const h = xhat.data[i * d + j];
      dgamma[j] += g * h;
      dbeta[j] += g;
      const gh = g * gamma[j];
      s1 += gh;
      s2 += gh * h;
    }
    for (let j = 0; j < d; j++) {
      const gh = dY.data[i * d + j] * gamma[j];
      dX.data[i * d + j] = (rstd[i] / d) * (d * gh - s1 - xhat.data[i * d + j] * s2);
    }
  }
  return { dX, dgamma, dbeta };
}

// ------------------------------------------------------------------ the MLP engine

type Replica = { p: Float32Array; g: Float32Array; m: Float32Array; v: Float32Array; t: number };

/** One forward + backward pass of one replica on one batch; fills `g`, returns the mean loss. */
export function mlpGrad(spec: MlpSpec, p: Float32Array, g: Float32Array, Xb: Tensor2D, Yb: Tensor2D): number {
  const { sizes, acts } = spec;
  const L = sizes.length - 1;
  const { wOff, bOff } = paramLayout(sizes);
  const W = (l: number) => ({ rows: sizes[l + 1], cols: sizes[l], data: p.subarray(wOff[l], wOff[l] + sizes[l] * sizes[l + 1]) });
  const b = (l: number) => p.subarray(bOff[l], bOff[l] + sizes[l + 1]);
  const As: Tensor2D[] = [Xb];
  const Zs: Tensor2D[] = [];
  for (let l = 0; l < L; l++) {
    const Z = addBias(matmulTB(As[l], W(l)), b(l));
    Zs.push(Z);
    As.push(spec.loss === 'softmax-ce' && l === L - 1 ? softmaxRows(Z) : actForward(acts[l], Z));
  }
  const n = Xb.rows;
  const out = As[L];
  let loss = 0;
  // δ = dLoss/dZ for the output layer, already divided by n
  let D = tensor(n, sizes[L]);
  for (let i = 0; i < out.data.length; i++) {
    const o = out.data[i];
    const y = Yb.data[i];
    if (spec.loss === 'bce') {
      const q = Math.min(Math.max(o, 1e-12), 1 - 1e-12);
      loss += -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
      D.data[i] = (o - y) / n;
    } else if (spec.loss === 'softmax-ce') {
      if (y > 0) loss += -y * Math.log(Math.max(o, 1e-30));
      D.data[i] = (o - y) / n;
    } else {
      const d = o - y;
      loss += 0.5 * d * d;
      D.data[i] = (d * actGrad(acts[L - 1], Zs[L - 1].data[i], o)) / n;
    }
  }
  const l2 = spec.l2 ?? 0;
  for (let l = L - 1; l >= 0; l--) {
    const gW = matmulTA(D, As[l]); // [out × in]
    const Wl = W(l);
    for (let i = 0; i < gW.data.length; i++) {
      g[wOff[l] + i] = gW.data[i] + l2 * Wl.data[i];
      if (l2) loss += 0.5 * l2 * Wl.data[i] * Wl.data[i] * n; // divided by n below
    }
    for (let j = 0; j < sizes[l + 1]; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += D.data[i * sizes[l + 1] + j];
      g[bOff[l] + j] = s;
    }
    if (l > 0) D = actBackward(acts[l - 1], Zs[l - 1], As[l], matmul(D, Wl));
  }
  return loss / n;
}

/** One optimiser update of a flat parameter vector, as `src/lib/optimizers.ts` does it. */
export function optimiserStep(spec: MlpSpec, s: Replica, lr: number) {
  const { kind } = spec.opt;
  const b1 = spec.opt.beta1 ?? 0.9;
  const b2 = spec.opt.beta2 ?? 0.999;
  const eps = spec.opt.eps ?? 1e-8;
  const wd = spec.opt.weightDecay ?? 0;
  s.t++;
  const c1 = 1 - b1 ** s.t;
  const c2 = 1 - b2 ** s.t;
  const { p, g, m, v } = s;
  for (let j = 0; j < p.length; j++) {
    const gj = g[j];
    if (kind === 'sgd') p[j] -= lr * gj;
    else if (kind === 'momentum') {
      m[j] = b1 * m[j] + gj;
      p[j] -= lr * m[j];
    } else {
      m[j] = b1 * m[j] + (1 - b1) * gj;
      v[j] = b2 * v[j] + (1 - b2) * gj * gj;
      if (kind === 'adamw' && wd) p[j] -= lr * wd * p[j];
      p[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + eps);
    }
  }
}

export class CpuMlpEngine implements MlpEngine {
  readonly kind = 'cpu' as const;
  readonly replicas: number;
  private reps: Replica[];
  private X: Float32Array = new Float32Array(0);
  private Y: Float32Array = new Float32Array(0);
  /** Rows in the dataset. */
  rows = 0;
  constructor(readonly spec: MlpSpec) {
    this.replicas = spec.replicas ?? 1;
    const P = paramLayout(spec.sizes).total;
    this.reps = Array.from({ length: this.replicas }, () => ({ p: new Float32Array(P), g: new Float32Array(P), m: new Float32Array(P), v: new Float32Array(P), t: 0 }));
  }
  setWeights(r: number, w: Weights) {
    const { wOff, bOff } = paramLayout(this.spec.sizes);
    const s = this.reps[r];
    w.W.forEach((Wl, l) => s.p.set(Wl, wOff[l]));
    w.b.forEach((bl, l) => s.p.set(bl, bOff[l]));
    s.m.fill(0);
    s.v.fill(0);
    s.t = 0;
  }
  setData(X: Float32Array, Y: Float32Array, n: number) {
    this.X = X;
    this.Y = Y;
    this.rows = n;
  }
  /** Synchronous training, used by tests and by the browser fallback. */
  trainSync(call: TrainCall): Float32Array {
    const { sizes } = this.spec;
    const din = sizes[0];
    const dout = sizes[sizes.length - 1];
    const R = this.replicas;
    const B = call.batch;
    const out = new Float32Array(call.steps * R);
    const Xb = tensor(B, din);
    const Yb = tensor(B, dout);
    for (let k = 0; k < call.steps; k++)
      for (let r = 0; r < R; r++) {
        for (let i = 0; i < B; i++) {
          const row = call.idx ? call.idx[(k * R + r) * B + i] : i;
          Xb.data.set(this.X.subarray(row * din, row * din + din), i * din);
          Yb.data.set(this.Y.subarray(row * dout, row * dout + dout), i * dout);
        }
        const s = this.reps[r];
        out[k * R + r] = mlpGrad(this.spec, s.p, s.g, Xb, Yb);
        optimiserStep(this.spec, s, call.lr[k * R + r]);
      }
    return out;
  }
  train(call: TrainCall) {
    return Promise.resolve(this.trainSync(call));
  }
  predictSync(X: Float32Array, n: number): Float32Array {
    const { sizes, acts } = this.spec;
    const L = sizes.length - 1;
    const dout = sizes[L];
    const { wOff, bOff } = paramLayout(sizes);
    const res = new Float32Array(this.replicas * n * dout);
    this.reps.forEach((s, r) => {
      let A = tensor(n, sizes[0], X.subarray(0, n * sizes[0]));
      for (let l = 0; l < L; l++) {
        const Wl = { rows: sizes[l + 1], cols: sizes[l], data: s.p.subarray(wOff[l], wOff[l] + sizes[l] * sizes[l + 1]) };
        const Z = addBias(matmulTB(A, Wl), s.p.subarray(bOff[l], bOff[l] + sizes[l + 1]));
        A = this.spec.loss === 'softmax-ce' && l === L - 1 ? softmaxRows(Z) : actForward(acts[l], Z);
      }
      res.set(A.data, r * n * dout);
    });
    return res;
  }
  predict(X: Float32Array, n: number) {
    return Promise.resolve(this.predictSync(X, n));
  }
  readWeightsSync(): Weights[] {
    const { sizes } = this.spec;
    const { wOff, bOff } = paramLayout(sizes);
    return this.reps.map((s) => ({
      W: sizes.slice(1).map((o, l) => s.p.slice(wOff[l], wOff[l] + sizes[l] * o)),
      b: sizes.slice(1).map((o, l) => s.p.slice(bOff[l], bOff[l] + o)),
    }));
  }
  readWeights() {
    return Promise.resolve(this.readWeightsSync());
  }
  destroy() {}
}
