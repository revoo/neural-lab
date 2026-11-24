/**
 * One-input ReLU networks and an exact count of their straight-line pieces.
 *
 * A ReLU network with one input computes a piecewise-linear function. Within a region where no
 * neuron changes sign, every pre-activation is an affine function of x, so we can walk from region
 * to region, jumping to the next x where some neuron's tally crosses zero. No sampling involved.
 */
import { createRng } from '../../lib/rng';

/** W[l][j][i]: weight from unit i of layer l to unit j of layer l + 1. Hidden layers use ReLU; the output is linear. */
export type Net1D = { W: number[][][]; b: number[][] };

export const relu = (v: number) => (v > 0 ? v : 0);

export function evaluate(net: Net1D, x: number): number {
  let a = [x];
  const last = net.W.length - 1;
  net.W.forEach((L, l) => {
    a = L.map((row, j) => {
      const z = row.reduce((s, w, i) => s + w * a[i], net.b[l][j]);
      return l === last ? z : relu(z);
    });
  });
  return a[0];
}

/** Values after each hidden layer and the output, for plotting intermediate stages. */
export function evaluateLayers(net: Net1D, x: number): number[][] {
  const out: number[][] = [];
  let a = [x];
  const last = net.W.length - 1;
  net.W.forEach((L, l) => {
    a = L.map((row, j) => {
      const z = row.reduce((s, w, i) => s + w * a[i], net.b[l][j]);
      return l === last ? z : relu(z);
    });
    out.push(a);
  });
  return out;
}

export const paramCount = (net: Net1D) => net.W.reduce((s, L, l) => s + L.length * L[0].length + net.b[l].length, 0);
export const neuronCount = (net: Net1D) => net.W.slice(0, -1).reduce((s, L) => s + L.length, 0);

/**
 * Exact linear pieces of the network's output on [lo, hi]. Returns the breakpoints (x where the
 * slope actually changes) and the number of pieces = breakpoints + 1.
 */
export function linearPieces(net: Net1D, lo: number, hi: number, maxRegions = 1 << 16): { breaks: number[]; pieces: number } {
  const last = net.W.length - 1;
  const eps = 1e-10 * Math.max(1, hi - lo);
  let x = lo;
  const regions: { start: number; slope: number; icpt: number }[] = [];
  while (x < hi && regions.length < maxRegions) {
    // Activation pattern just to the right of x; affine (α, β) of every unit within that region.
    const probe = Math.min(x + eps, (x + hi) / 2);
    let A = [1]; // slopes of current layer's outputs w.r.t. x
    let B = [0]; // intercepts
    let next = hi;
    net.W.forEach((L, l) => {
      const nA: number[] = [];
      const nB: number[] = [];
      L.forEach((row, j) => {
        let al = 0;
        let be = net.b[l][j];
        row.forEach((w, i) => {
          al += w * A[i];
          be += w * B[i];
        });
        if (l === last) {
          nA.push(al);
          nB.push(be);
          return;
        }
        if (Math.abs(al) > 1e-14) {
          const root = -be / al;
          if (root > x + eps && root < next) next = root;
        }
        const on = al * probe + be > 0;
        nA.push(on ? al : 0);
        nB.push(on ? be : 0);
      });
      A = nA;
      B = nB;
    });
    regions.push({ start: x, slope: A[0], icpt: B[0] });
    x = next;
  }
  // Merge neighbouring regions whose line is the same (a neuron switched but nothing downstream noticed).
  const breaks: number[] = [];
  for (let k = 1; k < regions.length; k++) {
    const p = regions[k - 1];
    const q = regions[k];
    const scale = Math.max(1, Math.abs(p.slope), Math.abs(q.slope));
    if (Math.abs(p.slope - q.slope) > 1e-7 * scale) breaks.push(q.start);
  }
  return { breaks, pieces: breaks.length + 1 };
}

/**
 * The "folding" network: L hidden layers of 2 ReLU neurons, each computing the tent map
 * t(u) = 2·relu(u) − 4·relu(u − ½), which folds [0, 1] onto itself. Composing L folds gives a
 * sawtooth with 2^L straight pieces on [0, 1]. The combination 2h₁ − 4h₂ is absorbed into the
 * next layer's weights, so each extra fold costs 2×2 weights + 2 biases.
 */
export function foldNet(L: number): Net1D {
  const W: number[][][] = [[[1], [1]]];
  const b: number[][] = [[0, -0.5]];
  for (let l = 1; l < L; l++) {
    W.push([
      [2, -4],
      [2, -4],
    ]);
    b.push([0, -0.5]);
  }
  W.push([[2, -4]]);
  b.push([0]);
  return { W, b };
}

/**
 * The same sawtooth built from ONE wide layer: a neuron for the initial slope plus one neuron per
 * kink (2^L − 1 of them), each switching the slope between +2^L and −2^L.
 */
export function wideSawtooth(L: number): Net1D {
  const n = 2 ** L;
  const s = 2 ** L;
  const hidden: number[][] = [[1]];
  const hb: number[] = [0];
  const out: number[] = [s];
  for (let k = 1; k < n; k++) {
    hidden.push([1]);
    hb.push(-k / n);
    out.push(k % 2 ? -2 * s : 2 * s);
  }
  return { W: [hidden, [out]], b: [hb, [0]] };
}

/** One wide layer of n neurons with kinks spread evenly over [0, 1] and alternating slopes (a generic n-kink function). */
export function wideNet(n: number, height = 1): Net1D {
  // Zig-zag through n + 1 pieces of equal width between 0 and `height`.
  const pieces = n + 1;
  const s = height * pieces;
  const hidden: number[][] = [[1]];
  const hb: number[] = [0];
  const out: number[] = [s];
  for (let k = 1; hidden.length < n; k++) {
    hidden.push([1]);
    hb.push(-k / pieces);
    out.push(k % 2 ? -2 * s : 2 * s);
  }
  return { W: [hidden, [out]], b: [hb, [0]] };
}

/** Add Gaussian noise of size σ to every weight and bias (seeded), to test how fragile a construction is. */
export function perturb(net: Net1D, sigma: number, seed: number): Net1D {
  const r = createRng(seed);
  return {
    W: net.W.map((L) => L.map((row) => row.map((w) => w + r.normal(0, sigma)))),
    b: net.b.map((row) => row.map((v) => v + r.normal(0, sigma))),
  };
}

/** A randomly initialised (He) ReLU network with the given hidden widths, for "what do typical networks do?" comparisons. */
export function randomNet(hidden: number[], seed: number): Net1D {
  const r = createRng(seed);
  const sizes = [1, ...hidden, 1];
  const W: number[][][] = [];
  const b: number[][] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const std = Math.sqrt(2 / sizes[l]);
    W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => r.normal(0, std))));
    b.push(Array.from({ length: sizes[l + 1] }, () => r.normal(0, 0.5)));
  }
  return { W, b };
}

/** Parameters of a plain fully connected network with the given layer sizes (inputs first). */
export const mlpParams = (sizes: number[]) => sizes.slice(1).reduce((s, n, l) => s + sizes[l] * n + n, 0);
export const mlpMultiplyAdds = (sizes: number[]) => sizes.slice(1).reduce((s, n, l) => s + sizes[l] * n, 0);
