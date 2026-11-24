/**
 * Logic for the network playground: small 2D datasets, a forward pass that can switch individual
 * neurons off ("ablation"), per-neuron importance, parameter counting, and an exhaustive search for
 * the smallest set of hidden neurons that keeps accuracy above a threshold (the pruning challenge).
 */
import type { MLP } from '../../lib/mlp';
import { createRng } from '../../lib/rng';

export type Dataset = { X: number[][]; y: number[] };
export type DatasetName = 'circles' | 'xor' | 'moons' | 'clusters';

export const DATASET_LABELS: Record<DatasetName, string> = {
  clusters: 'Two clusters',
  circles: 'Circle',
  xor: 'XOR',
  moons: 'Moons',
};

/** Deterministic toy datasets on roughly [−1.5, 1.5]². Label 1 = orange, 0 = blue. */
export function makeDataset(name: DatasetName, n = 120, seed = 3): Dataset {
  const r = createRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let k = 0; k < n; k++) {
    const label = k % 2;
    let p: number[];
    if (name === 'circles') {
      const rad = label ? r.uniform(0.85, 1.35) : r.uniform(0, 0.55);
      const t = r.uniform(0, 2 * Math.PI);
      p = [rad * Math.cos(t), rad * Math.sin(t)];
    } else if (name === 'xor') {
      const sx = r.next() < 0.5 ? -1 : 1;
      const sy = label ? -sx : sx;
      p = [sx * r.uniform(0.15, 1.3), sy * r.uniform(0.15, 1.3)];
    } else if (name === 'moons') {
      const t = r.uniform(0, Math.PI);
      p = label ? [1 - Math.cos(t) - 0.5 + r.normal(0, 0.1), -Math.sin(t) + 0.3 + r.normal(0, 0.1)] : [Math.cos(t) - 0.5 + r.normal(0, 0.1), Math.sin(t) - 0.3 + r.normal(0, 0.1)];
    } else {
      p = label ? [r.normal(0.7, 0.3), r.normal(-0.6, 0.3)] : [r.normal(-0.7, 0.3), r.normal(0.6, 0.3)];
    }
    X.push(p);
    y.push(label);
  }
  return { X, y };
}

/** mask[col][j] === false switches neuron j of node column col off (its output is forced to 0). Column 0 is the input. */
export type Mask = boolean[][];

export const fullMask = (sizes: number[]): Mask => sizes.map((n) => Array(n).fill(true));

export type Trace = { z: number[][]; a: number[][] };

/** Forward pass honouring the mask. a[0] is the input; z[l], a[l + 1] belong to weight layer l. */
export function forwardMasked(net: MLP, x: number[], mask?: Mask): Trace {
  const a: number[][] = [x];
  const z: number[][] = [];
  for (let l = 0; l < net.W.length; l++) {
    const W = net.W[l];
    const prev = a[l];
    const f = net.acts[l].f;
    const zl = new Array(W.length);
    const al = new Array(W.length);
    for (let j = 0; j < W.length; j++) {
      let s = net.b[l][j];
      for (let i = 0; i < prev.length; i++) s += W[j][i] * prev[i];
      zl[j] = s;
      al[j] = mask && mask[l + 1] && mask[l + 1][j] === false ? 0 : f(s);
    }
    z.push(zl);
    a.push(al);
  }
  return { z, a };
}

export const predict = (net: MLP, x: number[], mask?: Mask) => {
  const t = forwardMasked(net, x, mask);
  return t.a[t.a.length - 1][0];
};

export function accuracy(net: MLP, d: Dataset, mask?: Mask): number {
  let c = 0;
  for (let k = 0; k < d.X.length; k++) if ((predict(net, d.X[k], mask) > 0.5 ? 1 : 0) === d.y[k]) c++;
  return c / d.X.length;
}

/** Accuracy lost when each hidden neuron is switched off on its own (on top of the current mask). */
export function importance(net: MLP, d: Dataset, mask: Mask): number[][] {
  const base = accuracy(net, d, mask);
  return mask.map((col, c) =>
    col.map((on, j) => {
      if (c === 0 || c === mask.length - 1 || !on) return 0;
      const m = mask.map((r) => r.slice());
      m[c][j] = false;
      return base - accuracy(net, d, m);
    }),
  );
}

/** Weights + biases per layer, e.g. [{ nin: 2, nout: 3, weights: 6, biases: 3 }, …]. */
export function paramBreakdown(sizes: number[]) {
  return sizes.slice(1).map((nout, l) => ({ nin: sizes[l], nout, weights: sizes[l] * nout, biases: nout }));
}
export const paramTotal = (sizes: number[]) => paramBreakdown(sizes).reduce((s, p) => s + p.weights + p.biases, 0);

/**
 * Exhaustive pruning search: the fewest hidden neurons that can stay switched on (all others off)
 * while accuracy stays ≥ threshold. Tries every subset, smallest first, so the answer is exact.
 * Returns null if there are too many hidden neurons to search (limit) or no subset reaches the threshold.
 */
export function minimalNeurons(net: MLP, d: Dataset, threshold: number, limit = 14): { count: number; mask: Mask; evaluated: number } | null {
  const sizes = net.sizes;
  const hidden: [number, number][] = [];
  for (let c = 1; c < sizes.length - 1; c++) for (let j = 0; j < sizes[c]; j++) hidden.push([c, j]);
  const n = hidden.length;
  if (n > limit) return null;
  let evaluated = 0;
  // Enumerate subsets by size (number of neurons kept on).
  for (let k = 0; k <= n; k++) {
    let found: Mask | null = null;
    const pick = (start: number, chosen: number[]) => {
      if (found) return;
      if (chosen.length === k) {
        const m = fullMask(sizes);
        for (const [c, j] of hidden) m[c][j] = false;
        for (const idx of chosen) m[hidden[idx][0]][hidden[idx][1]] = true;
        evaluated++;
        if (accuracy(net, d, m) >= threshold) found = m;
        return;
      }
      for (let i = start; i <= n - (k - chosen.length); i++) pick(i + 1, [...chosen, i]);
    };
    pick(0, []);
    if (found) return { count: k, mask: found, evaluated };
  }
  return null;
}
