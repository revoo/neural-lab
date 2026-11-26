// authored-by: Claude Opus 5.5
/**
 * The tiny GPT's trainer interface and the inline CPU trainer, with no GPU imports, so Node
 * scripts and tests can use them (`gptTrainer.ts` adds the worker and WebGPU trainers).
 */
import { Gpt, type AdamWOpts, type GptConfig } from './gpt';

export interface GptTrainer {
  readonly kind: 'cpu' | 'webgpu';
  /** Where the work runs, for the status chip. */
  readonly where: 'gpu' | 'worker' | 'main thread';
  /** Upload parameters (flat `gptLayout` order) and reset the optimiser. */
  setParams(p: Float32Array): Promise<void>;
  /** K AdamW steps on K batches laid out [K × B × (T+1)]; resolves to the K training losses. */
  train(batches: Int32Array, K: number, B: number, T: number, lr: Float32Array, o?: AdamWOpts): Promise<Float32Array>;
  evalLoss(batch: Int32Array, B: number, T: number): Promise<number>;
  readParams(): Promise<Float32Array>;
  destroy(): void;
}

export class InlineCpuTrainer implements GptTrainer {
  readonly kind = 'cpu' as const;
  readonly where = 'main thread' as const;
  readonly model: Gpt;
  constructor(cfg: GptConfig) {
    this.model = new Gpt(cfg);
  }
  async setParams(p: Float32Array) {
    this.model.load(p);
    this.model.resetOptimiser();
  }
  async train(batches: Int32Array, K: number, B: number, T: number, lr: Float32Array, o?: AdamWOpts) {
    const row = B * (T + 1);
    const out = new Float32Array(K);
    for (let k = 0; k < K; k++) out[k] = this.model.trainStep(batches.subarray(k * row, (k + 1) * row), B, T, lr[k], o);
    return out;
  }
  async evalLoss(batch: Int32Array, B: number, T: number) {
    return this.model.evalLoss(batch, B, T);
  }
  async readParams() {
    return this.model.snapshot();
  }
  destroy() {}
}

