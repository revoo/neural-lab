import { describe, expect, it } from 'vitest';
import { createRng } from './rng';
import { dot, matMul, matVec, transpose, det2, broadcastShape, cosineSimilarity, angleBetween } from './linalg';
import { derivative, partials } from './calculus';
import { softmax, entropy, crossEntropy, logSumExp, surprise, klDivergence } from './prob';
import { ACTIVATIONS, erf } from './activations';
import { Value, topoSort } from './autodiff';
import { MLP } from './mlp';
import { createOptimizer, type OptimizerName } from './optimizers';

describe('rng', () => {
  it('is reproducible for a seed', () => {
    const a = createRng(42);
    const b = createRng(42);
    for (let i = 0; i < 10; i++) expect(a.next()).toBe(b.next());
  });
  it('normal samples have roughly the right moments', () => {
    const r = createRng(1);
    const xs = Array.from({ length: 20000 }, () => r.normal(2, 3));
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
    expect(mean).toBeCloseTo(2, 1);
    expect(Math.sqrt(v)).toBeCloseTo(3, 1);
  });
});

describe('linalg', () => {
  it('dot product', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(() => dot([1], [1, 2])).toThrow();
  });
  it('matMul matches hand calculation and associativity', () => {
    const A = [[1, 2, 3], [4, 5, 6]];
    const B = [[7, 8], [9, 10], [11, 12]];
    expect(matMul(A, B)).toEqual([[58, 64], [139, 154]]);
    const C = [[1, -1], [2, 0.5]];
    expect(matMul(matMul(A, B), C)).toEqual(matMul(A, matMul(B, C)));
    expect(() => matMul(A, A)).toThrow(/inner dimensions/);
  });
  it('matVec and transpose', () => {
    expect(matVec([[2, 0], [0, 3]], [1, 1])).toEqual([2, 3]);
    expect(transpose([[1, 2, 3]])).toEqual([[1], [2], [3]]);
    expect(det2([[2, 1], [1, 2]])).toBe(3);
  });
  it('cosine similarity and angle', () => {
    expect(cosineSimilarity([1, 0], [0, 5])).toBeCloseTo(0);
    expect(cosineSimilarity([1, 1], [2, 2])).toBeCloseTo(1);
    expect(angleBetween([1, 0], [-1, 0])).toBeCloseTo(Math.PI);
  });
  it('broadcasting follows numpy rules', () => {
    expect(broadcastShape([32, 10, 64], [64])).toEqual({ shape: [32, 10, 64] });
    expect(broadcastShape([5, 1], [1, 7])).toEqual({ shape: [5, 7] });
    expect('error' in broadcastShape([3, 4], [3])).toBe(true);
  });
});

describe('calculus', () => {
  it('numerical derivatives', () => {
    expect(derivative((x) => x ** 3, 2)).toBeCloseTo(12, 5);
    expect(partials(([x, y]) => x * x * y, [3, 2])).toEqual([expect.closeTo(12, 4), expect.closeTo(9, 4)]);
  });
});

describe('prob', () => {
  it('softmax sums to one, is shift-invariant and stable', () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b)).toBeCloseTo(1);
    softmax([1001, 1002, 1003]).forEach((v, i) => expect(v).toBeCloseTo(p[i]));
    expect(softmax([1, 2, 3], 1e-9)[2]).toBeCloseTo(1);
    softmax([1, 2, 3], 1e6).forEach((v) => expect(v).toBeCloseTo(1 / 3));
  });
  it('entropy, cross-entropy, logsumexp, surprise', () => {
    expect(entropy([0.5, 0.5])).toBeCloseTo(1);
    expect(entropy([1, 0])).toBeCloseTo(0);
    expect(entropy([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2);
    expect(crossEntropy([0, 1, 0], [0.2, 0.5, 0.3])).toBeCloseTo(-Math.log(0.5));
    expect(logSumExp([1000, 1000])).toBeCloseTo(1000 + Math.log(2));
    expect(surprise(0.125)).toBeCloseTo(3);
  });
});

describe('activations', () => {
  it('analytic derivatives match numerical ones', () => {
    for (const act of Object.values(ACTIVATIONS)) {
      if (act.name === 'step') continue;
      for (const x of [-2.3, -0.7, 0.4, 1.9]) {
        expect(act.df(x), `${act.name} at ${x}`).toBeCloseTo(derivative(act.f, x), 4);
      }
    }
  });
  it('erf approximation', () => {
    expect(erf(0)).toBeCloseTo(0, 6);
    expect(erf(1)).toBeCloseTo(0.8427007929, 6);
    expect(erf(-2)).toBeCloseTo(-0.995322265, 6);
  });
});

describe('autodiff', () => {
  it('computes gradients of a neuron', () => {
    const x1 = new Value(2, 'x1');
    const w1 = new Value(-3, 'w1');
    const x2 = new Value(0, 'x2');
    const w2 = new Value(1, 'w2');
    const b = new Value(6.8813735870195432, 'b');
    const o = x1.mul(w1).add(x2.mul(w2)).add(b).tanh();
    o.backward();
    expect(o.data).toBeCloseTo(0.7071, 4);
    expect(x1.grad).toBeCloseTo(-1.5, 4);
    expect(w1.grad).toBeCloseTo(1, 4);
    expect(x2.grad).toBeCloseTo(0.5, 4);
    expect(w2.grad).toBeCloseTo(0, 4);
  });
  it('accumulates gradients when a value is reused', () => {
    const a = new Value(3);
    const y = a.mul(a).add(a); // a² + a → dy/da = 2a + 1 = 7
    y.backward();
    expect(a.grad).toBe(7);
  });
  it('matches numerical gradients on a composite expression', () => {
    const f = (x: number, y: number) => Math.log(1 + Math.exp(x * y)) + Math.tanh(x) * y ** 2 - 1 / (1 + Math.exp(-y));
    const x = new Value(0.7);
    const y = new Value(-1.3);
    const out = x.mul(y).exp().add(1).log().add(x.tanh().mul(y.pow(2))).sub(y.sigmoid());
    out.backward();
    expect(out.data).toBeCloseTo(f(0.7, -1.3), 10);
    expect(x.grad).toBeCloseTo(derivative((t) => f(t, -1.3), 0.7), 5);
    expect(y.grad).toBeCloseTo(derivative((t) => f(0.7, t), -1.3), 5);
  });
  it('topological order puts inputs first', () => {
    const a = new Value(1);
    const b = a.mul(2);
    const c = b.add(a);
    const order = topoSort(c);
    expect(order.indexOf(a)).toBeLessThan(order.indexOf(b));
    expect(order[order.length - 1]).toBe(c);
  });
});

describe('mlp', () => {
  const gradCheck = (loss: 'mse' | 'bce', acts: ('tanh' | 'relu' | 'sigmoid' | 'gelu' | 'linear')[], l2 = 0) => {
    const rng = createRng(7);
    const net = new MLP([2, 4, 3, 1], acts, rng);
    const X = [[0.3, -0.8], [-1.2, 0.5], [0.9, 0.1]];
    const Y = [[1], [0], [1]];
    net.backprop(X, Y, loss, l2);
    const h = 1e-6;
    const lossAt = () => {
      let r = 0;
      for (const L of net.W) for (const row of L) for (const w of row) r += 0.5 * l2 * w * w;
      return net.loss(X, Y, loss) + r;
    };
    for (let l = 0; l < net.W.length; l++) {
      const j = 0;
      const i = net.W[l][0].length - 1;
      const orig = net.W[l][j][i];
      net.W[l][j][i] = orig + h;
      const up = lossAt();
      net.W[l][j][i] = orig - h;
      const dn = lossAt();
      net.W[l][j][i] = orig;
      expect(net.gW[l][j][i]).toBeCloseTo((up - dn) / (2 * h), 5);
      const ob = net.b[l][j];
      net.b[l][j] = ob + h;
      const ub = lossAt();
      net.b[l][j] = ob - h;
      const db = lossAt();
      net.b[l][j] = ob;
      expect(net.gb[l][j]).toBeCloseTo((ub - db) / (2 * h), 5);
    }
  };
  it('backprop matches numerical gradients (bce + sigmoid)', () => gradCheck('bce', ['tanh', 'gelu', 'sigmoid']));
  it('backprop matches numerical gradients (mse + l2)', () => gradCheck('mse', ['tanh', 'tanh', 'linear'], 0.01));
  it('backprop with dropout matches numerical gradients for fixed masks', () => {
    const net = new MLP([2, 5, 4, 1], ['tanh', 'relu', 'linear'], createRng(11));
    net.setDropout(0.4, createRng(5));
    const X = [[0.3, -0.8], [-1.2, 0.5], [0.9, 0.1], [0.2, 0.7]];
    const Y = [[1], [-0.5], [0.3], [0]];
    const masks = X.map(() => net.sampleMasks());
    // the masks really drop something and scale the rest by 1/(1 − p)
    const flat = masks.flat(2);
    expect(flat.some((v) => v === 0)).toBe(true);
    expect(flat.filter((v) => v !== 0).every((v) => Math.abs(v - 1 / 0.6) < 1e-12)).toBe(true);
    const l2 = 0.01;
    net.backprop(X, Y, 'mse', l2, masks);
    const lossAt = () => {
      let r = 0;
      for (const L of net.W) for (const row of L) for (const w of row) r += 0.5 * l2 * w * w;
      return net.loss(X, Y, 'mse', masks) + r;
    };
    const h = 1e-6;
    let checked = 0;
    for (let l = 0; l < net.W.length; l++)
      for (let j = 0; j < net.W[l].length; j++)
        for (let i = 0; i < net.W[l][j].length; i++) {
          const orig = net.W[l][j][i];
          net.W[l][j][i] = orig + h;
          const up = lossAt();
          net.W[l][j][i] = orig - h;
          const dn = lossAt();
          net.W[l][j][i] = orig;
          expect(net.gW[l][j][i]).toBeCloseTo((up - dn) / (2 * h), 5);
          checked++;
        }
    for (let l = 0; l < net.b.length; l++)
      for (let j = 0; j < net.b[l].length; j++) {
        const ob = net.b[l][j];
        net.b[l][j] = ob + h;
        const ub = lossAt();
        net.b[l][j] = ob - h;
        const db = lossAt();
        net.b[l][j] = ob;
        expect(net.gb[l][j]).toBeCloseTo((ub - db) / (2 * h), 5);
      }
    expect(checked).toBe(2 * 5 + 5 * 4 + 4);
  });
  it('dropout with p = 0 changes nothing, and keeps each activation’s average', () => {
    const X = [[0.3, -0.8], [-1.2, 0.5]];
    const Y = [[1], [0]];
    const a = new MLP([2, 6, 1], ['tanh', 'sigmoid'], createRng(2));
    const b = new MLP([2, 6, 1], ['tanh', 'sigmoid'], createRng(2));
    b.setDropout(0, createRng(9));
    expect(b.backprop(X, Y, 'bce')).toBe(a.backprop(X, Y, 'bce'));
    expect(b.gW).toEqual(a.gW);
    // inverted dropout: averaged over many masks, the hidden layer's output equals the unmasked one
    const net = new MLP([2, 6, 1], ['tanh', 'sigmoid'], createRng(2));
    net.setDropout(0.5, createRng(3));
    const clean = net.forward([0.4, -0.2]).a[1];
    const avg = clean.map(() => 0);
    const N = 20000;
    for (let k = 0; k < N; k++) {
      const h = net.forward([0.4, -0.2], net.sampleMasks()).a[1];
      h.forEach((v, j) => (avg[j] += v / N));
    }
    avg.forEach((v, j) => expect(v).toBeCloseTo(clean[j], 1));
    // predict never drops
    expect(net.predict([0.4, -0.2])).toEqual(net.predict([0.4, -0.2]));
  });
  it('learns XOR with a hidden layer', () => {
    const rng = createRng(3);
    const net = new MLP([2, 4, 1], ['tanh', 'sigmoid'], rng);
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const Y = [[0], [1], [1], [0]];
    const opt = createOptimizer({ name: 'adam', lr: 0.05 });
    for (let i = 0; i < 1500; i++) {
      net.backprop(X, Y, 'bce');
      opt.step(net.params(), net.grads());
    }
    X.forEach((x, k) => expect(Math.round(net.predict(x)[0])).toBe(Y[k][0]));
  });
  it('cannot learn XOR with only linear hidden layers', () => {
    const rng = createRng(3);
    const net = new MLP([2, 8, 8, 1], ['linear', 'linear', 'sigmoid'], rng);
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const Y = [[0], [1], [1], [0]];
    const opt = createOptimizer({ name: 'adam', lr: 0.05 });
    for (let i = 0; i < 1500; i++) {
      net.backprop(X, Y, 'bce');
      opt.step(net.params(), net.grads());
    }
    const correct = X.filter((x, k) => Math.round(net.predict(x)[0]) === Y[k][0]).length;
    expect(correct).toBeLessThanOrEqual(3);
  });
});

describe('optimizers', () => {
  const bowl = (p: number[]) => [2 * p[0], 20 * p[1]]; // gradient of x² + 10y²
  for (const name of ['sgd', 'momentum', 'rmsprop', 'adam'] as OptimizerName[]) {
    it(`${name} converges on a bowl`, () => {
      const lr = name === 'momentum' ? 0.01 : name === 'sgd' ? 0.04 : 0.05;
      const opt = createOptimizer({ name, lr });
      const p = [3, 2];
      for (let i = 0; i < 2000; i++) opt.step([p], [bowl(p)]);
      expect(Math.hypot(p[0], p[1])).toBeLessThan(0.05);
    });
  }
  it('plain gradient descent diverges when the step is too large', () => {
    const opt = createOptimizer({ name: 'sgd', lr: 1.1 });
    const p = [1];
    for (let i = 0; i < 20; i++) opt.step([p], [[2 * p[0]]]);
    expect(Math.abs(p[0])).toBeGreaterThan(10);
  });
});

describe('KL divergence', () => {
  it('is zero for identical distributions', () => {
    expect(klDivergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])).toBe(0);
  });
  it('is asymmetric: 0.5310 one way, 0.7370 the other (p = 90/10, q = 50/50)', () => {
    expect(klDivergence([0.9, 0.1], [0.5, 0.5])).toBeCloseTo(0.531, 4);
    expect(klDivergence([0.5, 0.5], [0.9, 0.1])).toBeCloseTo(0.737, 4);
    expect(klDivergence([0.9, 0.1], [0.5, 0.5]).toFixed(4)).toBe('0.5310');
    expect(klDivergence([0.5, 0.5], [0.9, 0.1]).toFixed(4)).toBe('0.7370');
  });
  it('equals cross-entropy minus entropy (random seeded distributions)', () => {
    const r = createRng(2026);
    for (let t = 0; t < 50; t++) {
      const raw = (n: number) => Array.from({ length: n }, () => r.next() + 0.01);
      const norm = (v: number[]) => v.map((x) => x / v.reduce((a, b) => a + b, 0));
      const p = norm(raw(5));
      const q = norm(raw(5));
      expect(klDivergence(p, q)).toBeCloseTo(crossEntropy(p, q, 2) - entropy(p, 2), 12);
      expect(klDivergence(p, q)).toBeGreaterThanOrEqual(0);
    }
  });
  it('is infinite when q rules out something p allows, and skips p = 0 terms', () => {
    expect(klDivergence([0.5, 0.5], [1, 0])).toBe(Infinity);
    expect(klDivergence([1, 0], [0.5, 0.5])).toBeCloseTo(1, 12);
  });
  it('[1/2, 1/4, 1/8, 1/8] against uniform is exactly 0.25 bits', () => {
    expect(klDivergence([0.5, 0.25, 0.125, 0.125], [0.25, 0.25, 0.25, 0.25])).toBeCloseTo(0.25, 14);
  });
});
