// authored-by: Claude Opus 5.5
/**
 * One interface over the tiny GPT's three ways to train (V1 capstone, A2):
 *  - `webgpu`: `GptGpu`, everything resident on the GPU;
 *  - `cpu` in a Web Worker (the browser default without WebGPU), so the page stays responsive;
 *  - `cpu` inline (Node, tests, or a browser without workers).
 * Widgets make batches on the main thread from a seeded RNG, so every tier sees the same data.
 * The learner's tier preference is shared with the MLP widgets (`nl-compute-v1`).
 */
import type { AdamWOpts, GptConfig } from './gpt';
import { InlineCpuTrainer, type GptTrainer } from './gptTrainerCore';
export { InlineCpuTrainer, type GptTrainer } from './gptTrainerCore';
import { computePref, type ComputePref } from './selectBackend';
import { getGpu } from './webgpuBackend';
import type { GptGpu } from './gptGpu';

class WorkerCpuTrainer implements GptTrainer {
  readonly kind = 'cpu' as const;
  readonly where = 'worker' as const;
  private w: Worker;
  private next = 1;
  private pending = new Map<number, { res: (v: unknown) => void; rej: (e: Error) => void }>();
  constructor(cfg: GptConfig) {
    this.w = new Worker(new URL('./gptWorker.ts', import.meta.url), { type: 'module' });
    this.w.onmessage = (e: MessageEvent<{ id: number; result?: unknown; error?: string }>) => {
      const p = this.pending.get(e.data.id);
      if (!p) return;
      this.pending.delete(e.data.id);
      if (e.data.error) p.rej(new Error(e.data.error));
      else p.res(e.data.result);
    };
    void this.call({ op: 'init', cfg });
  }
  private call<T>(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const id = this.next++;
    return new Promise<T>((res, rej) => {
      this.pending.set(id, { res: res as (v: unknown) => void, rej });
      this.w.postMessage({ id, ...msg }, transfer);
    });
  }
  setParams(p: Float32Array) {
    return this.call<void>({ op: 'setParams', p: Float32Array.from(p) });
  }
  train(batches: Int32Array, K: number, B: number, T: number, lr: Float32Array, o: AdamWOpts = {}) {
    return this.call<Float32Array>({ op: 'train', batches, K, B, T, lr, o });
  }
  evalLoss(batch: Int32Array, B: number, T: number) {
    return this.call<number>({ op: 'evalLoss', batch, B, T });
  }
  readParams() {
    return this.call<Float32Array>({ op: 'readParams' });
  }
  destroy() {
    this.w.terminate();
    for (const p of this.pending.values()) p.rej(new Error('trainer destroyed'));
    this.pending.clear();
  }
}

class GpuTrainer implements GptTrainer {
  readonly kind = 'webgpu' as const;
  readonly where = 'gpu' as const;
  private g: GptGpu;
  constructor(g: GptGpu) {
    this.g = g;
  }
  async setParams(p: Float32Array) {
    this.g.setParams(p);
  }
  train(batches: Int32Array, K: number, B: number, T: number, lr: Float32Array, o?: AdamWOpts) {
    return this.g.train(batches, K, B, T, lr, o);
  }
  evalLoss(batch: Int32Array, B: number, T: number) {
    return this.g.evalLoss(batch, B, T);
  }
  readParams() {
    return this.g.readParams();
  }
  destroy() {
    this.g.destroy();
  }
}

/** The trainer a widget should use: WebGPU when available and not overridden, else a CPU worker. */
export async function createGptTrainer(cfg: GptConfig, pref: ComputePref = computePref.get()): Promise<GptTrainer> {
  if (pref === 'auto') {
    const gpu = await getGpu();
    if (gpu) {
      const { GptGpu } = await import('./gptGpu');
      return new GpuTrainer(new GptGpu(cfg, gpu.device));
    }
  }
  if (typeof Worker !== 'undefined') {
    try {
      return new WorkerCpuTrainer(cfg);
    } catch {
      /* fall through */
    }
  }
  return new InlineCpuTrainer(cfg);
}
