// authored-by: Claude Opus 5.5
/**
 * V2-WP3: the shared interface of the two compute tiers.
 *
 * - `cpu`: pure TypeScript `Float32Array` kernels (`cpuBackend.ts`). Runs in Node/Vitest and in any
 *   browser. It is the reference: tests pin numbers against it and finite-difference-check it.
 * - `webgpu`: WGSL compute shaders (`webgpuBackend.ts`) with weights, gradients and optimiser state
 *   kept in GPU buffers. K training steps for R replicas are recorded into one command encoder and
 *   only the small loss summary is read back.
 *
 * Both implement `MlpEngine`, so a widget doesn't care which one it got.
 */

/** `square` (z²) is the activation of the minimal model that groks modular addition (Gromov 2023). */
export type ActName = 'linear' | 'relu' | 'tanh' | 'sigmoid' | 'gelu' | 'square';
export type LossKind = 'bce' | 'softmax-ce' | 'mse';
export type OptKind = 'sgd' | 'momentum' | 'adam' | 'adamw';

export type MlpSpec = {
  /** Layer widths, input first: [in, h1, …, out]. */
  sizes: number[];
  /** One activation per layer after the input (the last is the output activation). */
  acts: ActName[];
  /**
   * bce: sigmoid output and binary cross-entropy (one output); softmax-ce: softmax over the
   * outputs (output act must be 'linear') with one-hot targets; mse: ½(out − y)² summed over outputs.
   */
  loss: LossKind;
  opt: { kind: OptKind; beta1?: number; beta2?: number; eps?: number; weightDecay?: number };
  /** L2 penalty ½·λ·Σw² on weights (not biases), added to the loss as in `src/lib/mlp.ts`. */
  l2?: number;
  /** Independent copies trained side by side (different seeds or settings). Default 1. */
  replicas?: number;
};

/** Weights of one replica: W[l] is row-major [out × in] (same as `MLP.W[l][j][i]`), b[l] is [out]. */
export type Weights = { W: Float32Array[]; b: Float32Array[] };

export type TrainCall = {
  /** Number of optimiser steps to run in this call. */
  steps: number;
  /** Rows per step (mini-batch size). */
  batch: number;
  /**
   * Row indices into the dataset, laid out [step][replica][row] (length steps × R × batch).
   * Omit for full-batch training on rows 0 … batch−1 every step.
   */
  idx?: Int32Array;
  /** Learning rate per step and replica, laid out [step][replica] (length steps × R). */
  lr: Float32Array;
};

export interface MlpEngine {
  readonly kind: 'cpu' | 'webgpu';
  readonly spec: MlpSpec;
  readonly replicas: number;
  /** Upload one replica's weights (and reset its optimiser state). */
  setWeights(replica: number, w: Weights): void;
  /** Upload the dataset shared by all replicas: X is [n × in], Y is [n × out] (row-major). */
  setData(X: Float32Array, Y: Float32Array, n: number): void;
  /** Run optimiser steps; resolves to the training loss of each step, laid out [step][replica]. */
  train(call: TrainCall): Promise<Float32Array>;
  /** Forward pass for every replica: resolves to outputs laid out [replica][row][out]. */
  predict(X: Float32Array, n: number): Promise<Float32Array>;
  /** Read back every replica's weights. */
  readWeights(): Promise<Weights[]>;
  destroy(): void;
}

export const paramCount = (sizes: number[]) => sizes.slice(1).reduce((t, n, l) => t + sizes[l] * n + n, 0);

/**
 * Flat layout of one replica's parameters, shared by both tiers: all W (layer by layer, row-major)
 * then all b. Offsets are in floats.
 */
export function paramLayout(sizes: number[]) {
  const wOff: number[] = [];
  const bOff: number[] = [];
  let o = 0;
  for (let l = 0; l < sizes.length - 1; l++) {
    wOff.push(o);
    o += sizes[l] * sizes[l + 1];
  }
  for (let l = 0; l < sizes.length - 1; l++) {
    bOff.push(o);
    o += sizes[l + 1];
  }
  return { wOff, bOff, total: o };
}
