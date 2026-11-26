// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import {
  actBackward,
  actForward,
  addBias,
  CpuMlpEngine,
  layerNorm,
  layerNormBackward,
  matmul,
  matmulTA,
  matmulTB,
  mlpGrad,
  softmaxBackward,
  softmaxRows,
  tensor,
  type Tensor2D,
} from './cpuBackend';
import { paramLayout, type ActName, type MlpSpec, type Weights } from './types';
import { MLP } from '../mlp';
import { createOptimizer } from '../optimizers';
import { createRng } from '../rng';

const rnd = (rows: number, cols: number, seed: number, scale = 1) => {
  const r = createRng(seed);
  return tensor(rows, cols, Array.from({ length: rows * cols }, () => r.normal(0, scale)));
};
/** Σ G ⊙ f(…): a scalar whose gradient w.r.t. f's output is G. */
const dotT = (A: Tensor2D, G: Tensor2D) => A.data.reduce((s, a, i) => s + a * G.data[i], 0);

/** Central finite differences of a scalar function of x (in float64 via a copy). */
function fd(x: Float32Array, f: () => number, h = 1e-3): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    x[i] = v + h;
    const a = f();
    x[i] = v - h;
    const b = f();
    x[i] = v;
    out[i] = (a - b) / (2 * h);
  }
  return out;
}
const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i])));
  expect(worst).toBeLessThan(tol);
};

describe('V2-WP3 CPU Float32 kernels', () => {
  it('the three matmul layouts agree with each other and with a plain triple loop', () => {
    const A = rnd(5, 4, 1);
    const B = rnd(4, 3, 2);
    const C = matmul(A, B);
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += A.data[i * 4 + k] * B.data[k * 3 + j];
        expect(C.data[i * 3 + j]).toBeCloseTo(s, 5);
      }
    const Bt = tensor(3, 4);
    for (let k = 0; k < 4; k++) for (let j = 0; j < 3; j++) Bt.data[j * 4 + k] = B.data[k * 3 + j];
    close(matmulTB(A, Bt).data, C.data, 1e-6);
    const At = tensor(4, 5);
    for (let i = 0; i < 5; i++) for (let k = 0; k < 4; k++) At.data[k * 5 + i] = A.data[i * 4 + k];
    close(matmulTA(At, B).data, C.data, 1e-6);
  });

  it('matmul gradients: dA = G·Bᵀ and dB = Aᵀ·G match finite differences', () => {
    const A = rnd(3, 4, 3);
    const B = rnd(4, 2, 4);
    const G = rnd(3, 2, 5);
    const f = () => dotT(matmul(A, B), G);
    close(matmulTB(G, B).data, fd(A.data, f), 1e-3);
    close(matmulTA(A, G).data, fd(B.data, f), 1e-3);
  });

  it.each(['linear', 'relu', 'tanh', 'sigmoid', 'gelu'] as ActName[])('%s backward matches finite differences', (name) => {
    const Z = rnd(4, 5, 6);
    for (let i = 0; i < Z.data.length; i++) if (Math.abs(Z.data[i]) < 0.01) Z.data[i] = 0.3; // keep relu away from its kink
    const G = rnd(4, 5, 7);
    const f = () => dotT(actForward(name, Z), G);
    const A = actForward(name, Z);
    close(actBackward(name, Z, A, G).data, fd(Z.data, f), 2e-3);
  });

  it('softmax backward matches finite differences, and rows sum to 1', () => {
    const Z = rnd(3, 6, 8, 2);
    const P = softmaxRows(Z);
    for (let i = 0; i < 3; i++) expect(P.data.slice(i * 6, i * 6 + 6).reduce((s, x) => s + x, 0)).toBeCloseTo(1, 6);
    const G = rnd(3, 6, 9);
    close(softmaxBackward(P, G).data, fd(Z.data, () => dotT(softmaxRows(Z), G)), 2e-3);
  });

  it('layer norm backward matches finite differences for x, γ and β', () => {
    const X = rnd(3, 5, 10, 2);
    const gamma = Float32Array.from([1.2, 0.7, -0.4, 1, 0.3]);
    const beta = Float32Array.from([0.1, -0.2, 0, 0.5, 0.2]);
    const G = rnd(3, 5, 11);
    const f = () => dotT(layerNorm(X, gamma, beta).Y, G);
    const { cache } = layerNorm(X, gamma, beta);
    const { dX, dgamma, dbeta } = layerNormBackward(G, gamma, cache);
    close(dX.data, fd(X.data, f), 3e-3);
    close(dgamma, fd(gamma, f), 3e-3);
    close(dbeta, fd(beta, f), 3e-3);
  });

  it('addBias broadcasts along rows', () => {
    const Z = addBias(tensor(2, 3), Float32Array.from([1, 2, 3]));
    expect([...Z.data]).toEqual([1, 2, 3, 1, 2, 3]);
  });

  const specs: [string, MlpSpec, (r: ReturnType<typeof createRng>) => number[]][] = [
    ['bce + tanh + L2', { sizes: [3, 5, 4, 1], acts: ['tanh', 'tanh', 'sigmoid'], loss: 'bce', opt: { kind: 'adam' }, l2: 0.01 }, (r) => [r.next() < 0.5 ? 0 : 1]],
    ['softmax-ce + relu', { sizes: [3, 6, 4], acts: ['relu', 'linear'], loss: 'softmax-ce', opt: { kind: 'adam' } }, (r) => { const y = [0, 0, 0, 0]; y[r.int(4)] = 1; return y; }],
    ['mse + gelu', { sizes: [3, 5, 2], acts: ['gelu', 'linear'], loss: 'mse', opt: { kind: 'sgd' } }, (r) => [r.normal(), r.normal()]],
  ];
  it.each(specs)('the whole-network gradient (%s) matches finite differences', (_, spec, label) => {
    const r = createRng(12);
    const P = paramLayout(spec.sizes).total;
    const p = Float32Array.from({ length: P }, () => r.normal(0, 0.6));
    const g = new Float32Array(P);
    const X = rnd(7, spec.sizes[0], 13);
    const Y = tensor(7, spec.sizes.at(-1)!, Array.from({ length: 7 }, () => label(r)).flat());
    mlpGrad(spec, p, g, X, Y);
    close(g, fd(p, () => mlpGrad(spec, p, new Float32Array(P), X, Y)), 3e-3);
  });

  it('CpuMlpEngine reproduces src/lib/mlp.ts + optimizers.ts for 60 Adam steps (with L2)', () => {
    const sizes = [2, 6, 5, 1];
    const net = new MLP(sizes, ['tanh', 'tanh', 'sigmoid'], createRng(3), 'xavier');
    const w: Weights = { W: net.W.map((L) => Float32Array.from(L.flat())), b: net.b.map((b) => Float32Array.from(b)) };
    const eng = new CpuMlpEngine({ sizes, acts: ['tanh', 'tanh', 'sigmoid'], loss: 'bce', opt: { kind: 'adam' }, l2: 0.003 });
    eng.setWeights(0, w);
    const r = createRng(4);
    const n = 40;
    const X = Array.from({ length: n }, () => [r.uniform(-1.5, 1.5), r.uniform(-1.5, 1.5)]);
    const Y = X.map(([a, b]) => [a * b > 0 ? 1 : 0]);
    eng.setData(Float32Array.from(X.flat()), Float32Array.from(Y.flat()), n);
    const opt = createOptimizer({ name: 'adam', lr: 0.05, beta1: 0.9 });
    const B = 8;
    const steps = 60;
    const idx = new Int32Array(steps * B);
    const br = createRng(5);
    for (let i = 0; i < idx.length; i++) idx[i] = br.int(n);
    const losses = eng.trainSync({ steps, batch: B, idx, lr: new Float32Array(steps).fill(0.05) });
    for (let k = 0; k < steps; k++) {
      const rows = [...idx.slice(k * B, k * B + B)];
      const l = net.backprop(rows.map((i) => X[i]), rows.map((i) => Y[i]), 'bce', 0.003);
      opt.step(net.params(), net.grads());
      expect(losses[k]).toBeCloseTo(l, 4);
    }
    const got = eng.readWeightsSync()[0];
    close(got.W[0], net.W[0].flat(), 1e-4);
    close(got.b[2], net.b[2], 1e-4);
    const pred = eng.predictSync(Float32Array.from([0.3, -0.7]), 1);
    expect(pred[0]).toBeCloseTo(net.predict([0.3, -0.7])[0], 4);
  });

  it('replicas train independently: different data orders give different weights, same order the same', () => {
    const spec: MlpSpec = { sizes: [2, 4, 1], acts: ['tanh', 'sigmoid'], loss: 'bce', opt: { kind: 'momentum' }, replicas: 3 };
    const eng = new CpuMlpEngine(spec);
    const net = new MLP(spec.sizes, ['tanh', 'sigmoid'], createRng(9), 'xavier');
    const w: Weights = { W: net.W.map((L) => Float32Array.from(L.flat())), b: net.b.map((b) => Float32Array.from(b)) };
    for (let r = 0; r < 3; r++) eng.setWeights(r, w);
    eng.setData(Float32Array.from([0, 0, 1, 1, 1, 0, 0, 1]), Float32Array.from([0, 0, 1, 1]), 4);
    // replica 0 and 1 see the same rows, replica 2 a different order
    const idx = Int32Array.from([0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 0, 1]);
    eng.trainSync({ steps: 2, batch: 2, idx, lr: new Float32Array(6).fill(0.1) });
    const [a, b, c] = eng.readWeightsSync();
    expect([...a.W[0]]).toEqual([...b.W[0]]);
    expect([...a.W[0]]).not.toEqual([...c.W[0]]);
  });
});
