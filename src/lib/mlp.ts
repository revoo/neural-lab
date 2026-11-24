/**
 * A small fully connected network with hand-written backpropagation.
 * Plain arrays, no tricks: the backprop lesson walks through this exact algorithm.
 */
import { ACTIVATIONS, type Activation, type ActivationName } from './activations';
import type { Rng } from './rng';

export type LossName = 'mse' | 'bce';
export type InitName = 'xavier' | 'he' | 'small' | 'zeros';

export type Forward = {
  /** a[0] is the input; a[l] the output of layer l (after any dropout mask). */
  a: number[][];
  /** z[l-1] is the pre-activation of layer l. */
  z: number[][];
};

/**
 * Dropout masks for one example: masks[l][j] multiplies the output of neuron j in hidden layer l+1
 * (layer index as in W). Each entry is 0 (dropped) or 1/(1 − p) (kept and scaled up, so the expected
 * value of every activation is unchanged — "inverted dropout"). The output layer is never dropped.
 */
export type Masks = number[][];

export class MLP {
  readonly sizes: number[];
  readonly acts: Activation[];
  /** W[l][j][i]: weight from neuron i of layer l to neuron j of layer l+1 */
  W: number[][][];
  b: number[][];
  gW: number[][][];
  gb: number[][];
  /**
   * Dropout probability for hidden neurons during training (0 = off). Only `backprop` applies it;
   * `forward`/`predict` without explicit masks always use the full network, as at test time.
   */
  dropout = 0;
  /** The dropout masks used for each example in the most recent `backprop` call (empty when dropout is off). */
  lastMasks: Masks[] = [];
  /** Random source for dropout masks; set with `setDropout`. */
  private dropRng: Rng | null = null;

  constructor(sizes: number[], activations: ActivationName[], rng: Rng, init: InitName = 'xavier') {
    if (activations.length !== sizes.length - 1) throw new Error('need one activation per layer');
    this.sizes = sizes;
    this.acts = activations.map((a) => ACTIVATIONS[a]);
    this.W = [];
    this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l];
      const nout = sizes[l + 1];
      const std =
        init === 'he' ? Math.sqrt(2 / nin) : init === 'xavier' ? Math.sqrt(2 / (nin + nout)) : init === 'small' ? 0.01 : 0;
      this.W.push(Array.from({ length: nout }, () => Array.from({ length: nin }, () => (std ? rng.normal(0, std) : 0))));
      this.b.push(Array.from({ length: nout }, () => 0));
    }
    this.gW = this.W.map((L) => L.map((r) => r.map(() => 0)));
    this.gb = this.b.map((r) => r.map(() => 0));
  }

  /** Switch dropout on (p > 0) or off (p = 0) for subsequent `backprop` calls. */
  setDropout(p: number, rng: Rng) {
    if (p < 0 || p >= 1) throw new Error('dropout probability must be in [0, 1)');
    this.dropout = p;
    this.dropRng = rng;
  }

  /** Sample one example's dropout masks for the current dropout probability. */
  sampleMasks(rng: Rng = this.dropRng!, p = this.dropout): Masks {
    const keep = 1 / (1 - p);
    return this.W.slice(0, -1).map((L) => L.map(() => (rng.next() < p ? 0 : keep)));
  }

  get paramCount(): number {
    return this.W.reduce((s, L, l) => s + L.length * L[0].length + this.b[l].length, 0);
  }

  forward(x: number[], masks?: Masks): Forward {
    const a: number[][] = [x];
    const z: number[][] = [];
    for (let l = 0; l < this.W.length; l++) {
      const W = this.W[l];
      const prev = a[l];
      const zl = new Array(W.length);
      const al = new Array(W.length);
      const f = this.acts[l].f;
      for (let j = 0; j < W.length; j++) {
        let s = this.b[l][j];
        const row = W[j];
        for (let i = 0; i < row.length; i++) s += row[i] * prev[i];
        zl[j] = s;
        al[j] = masks && l < masks.length ? f(s) * masks[l][j] : f(s);
      }
      z.push(zl);
      a.push(al);
    }
    return { a, z };
  }

  predict(x: number[]): number[] {
    return this.forward(x).a[this.W.length];
  }

  /**
   * Accumulate gradients of the mean loss over a batch into gW/gb. Returns mean loss.
   * l2 adds 0.5·λ·Σw² to the loss (weights only).
   * With dropout on, each example gets freshly sampled masks — unless `masks` (one per example) is
   * given, which makes the computation deterministic (used by the gradient check).
   */
  backprop(X: number[][], Y: number[][], loss: LossName, l2 = 0, masks?: Masks[]): number {
    for (const L of this.gW) for (const r of L) r.fill(0);
    for (const r of this.gb) r.fill(0);
    const n = X.length;
    const last = this.W.length - 1;
    let total = 0;
    this.lastMasks = [];
    for (let s = 0; s < n; s++) {
      const m = masks ? masks[s] : this.dropout > 0 && this.dropRng ? this.sampleMasks() : undefined;
      if (m) this.lastMasks.push(m);
      const { a, z } = this.forward(X[s], m);
      const out = a[last + 1];
      const y = Y[s];
      // delta = dLoss/dz for the output layer
      let delta: number[] = new Array(out.length);
      for (let j = 0; j < out.length; j++) {
        if (loss === 'mse') {
          const d = out[j] - y[j];
          total += 0.5 * d * d;
          delta[j] = d * this.acts[last].df(z[last][j]);
        } else {
          const p = Math.min(Math.max(out[j], 1e-12), 1 - 1e-12);
          total += -(y[j] * Math.log(p) + (1 - y[j]) * Math.log(1 - p));
          // sigmoid + BCE simplifies to (p - y); otherwise use the general chain rule
          delta[j] =
            this.acts[last].name === 'sigmoid'
              ? out[j] - y[j]
              : ((p - y[j]) / (p * (1 - p))) * this.acts[last].df(z[last][j]);
        }
      }
      for (let l = last; l >= 0; l--) {
        const W = this.W[l];
        const prev = a[l];
        const gW = this.gW[l];
        const gb = this.gb[l];
        for (let j = 0; j < W.length; j++) {
          gb[j] += delta[j];
          const gr = gW[j];
          for (let i = 0; i < prev.length; i++) gr[i] += delta[j] * prev[i];
        }
        if (l > 0) {
          const nd = new Array(prev.length).fill(0);
          for (let j = 0; j < W.length; j++) for (let i = 0; i < prev.length; i++) nd[i] += W[j][i] * delta[j];
          const df = this.acts[l - 1].df;
          for (let i = 0; i < nd.length; i++) nd[i] *= df(z[l - 1][i]) * (m ? m[l - 1][i] : 1);
          delta = nd;
        }
      }
    }
    let reg = 0;
    for (let l = 0; l < this.W.length; l++)
      for (let j = 0; j < this.W[l].length; j++)
        for (let i = 0; i < this.W[l][j].length; i++) {
          this.gW[l][j][i] = this.gW[l][j][i] / n + l2 * this.W[l][j][i];
          reg += 0.5 * l2 * this.W[l][j][i] ** 2;
        }
    for (const r of this.gb) for (let j = 0; j < r.length; j++) r[j] /= n;
    return total / n + reg;
  }

  /** Parameter arrays (references) in a fixed order, paired with gradient arrays in the same order. */
  params(): number[][] {
    return [...this.W.flat(), ...this.b];
  }
  grads(): number[][] {
    return [...this.gW.flat(), ...this.gb];
  }

  loss(X: number[][], Y: number[][], loss: LossName, masks?: Masks[]): number {
    let t = 0;
    for (let s = 0; s < X.length; s++) {
      const out = this.forward(X[s], masks?.[s]).a[this.W.length];
      for (let j = 0; j < out.length; j++) {
        if (loss === 'mse') t += 0.5 * (out[j] - Y[s][j]) ** 2;
        else {
          const p = Math.min(Math.max(out[j], 1e-12), 1 - 1e-12);
          t += -(Y[s][j] * Math.log(p) + (1 - Y[s][j]) * Math.log(1 - p));
        }
      }
    }
    return t / X.length;
  }
}
