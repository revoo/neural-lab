// authored-by: Claude Opus 5.5
/**
 * Parity checks for the tiny GPT (V1 capstone, A2), run in a real Chromium by
 * `scripts/check-webgpu.cjs`:
 *  1. the WebGPU tier against the CPU tier (same start, same batches): losses, eval loss and
 *     parameters after two training calls, within float32 tolerances;
 *  2. the CPU tier in the browser against Node: a reference run must give bit-identical parameters
 *     (`cpuDeterminismHash`, pinned in `gpt.test.ts`), which is what lets the lesson promise that
 *     the learner's CPU run is the recorded run.
 */
import { Gpt, type GptConfig } from './gpt';
export { cpuDeterminismHash } from './gptPinned';
import { GptGpu } from './gptGpu';
import { createRng } from '../rng';

type Check = { name: string; worst: number; tol: number; ok: boolean };

const worstRel = (a: ArrayLike<number>, b: ArrayLike<number>, floor = 1e-3) => {
  let w = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]) / Math.max(floor, Math.abs(b[i]));
    w = Math.max(w, Number.isFinite(d) ? d : Infinity);
  }
  return w;
};
const worstAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let w = 0;
  for (let i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i] - b[i]));
  return w;
};

const CASES: { name: string; cfg: GptConfig; B: number; T: number }[] = [
  { name: 'V 11 · d 8 · 2 heads · 2 layers · T 6', cfg: { vocab: 11, ctx: 6, d: 8, heads: 2, layers: 2 }, B: 2, T: 6 },
  { name: 'V 65 · d 32 · 4 heads · 2 layers · T 16 of 24', cfg: { vocab: 65, ctx: 24, d: 32, heads: 4, layers: 2 }, B: 4, T: 16 },
  { name: 'V 30 · d 24 · 1 head · 1 layer · ff 2 · T 20', cfg: { vocab: 30, ctx: 20, d: 24, heads: 1, layers: 1, ff: 2 }, B: 3, T: 20 },
];

export async function gptParity(device: GPUDevice): Promise<Check[]> {
  const checks: Check[] = [];
  const push = (name: string, worst: number, tol: number) => checks.push({ name, worst, tol, ok: worst <= tol });
  for (const c of CASES) {
    const cpu = new Gpt(c.cfg);
    cpu.init(3);
    const gpu = new GptGpu(c.cfg, device);
    gpu.setParams(cpu.snapshot());
    const r = createRng(11);
    const opts = { weightDecay: 0.1, clip: 1 };
    const row = c.T + 1;
    for (let call = 0; call < 2; call++) {
      const K = 6;
      const batches = Int32Array.from({ length: K * c.B * row }, () => r.int(c.cfg.vocab));
      const lr = Float32Array.from({ length: K }, (_, k) => 2e-3 * (1 + k / K));
      const cpuLoss = Array.from({ length: K }, (_, k) => cpu.trainStep(batches.subarray(k * c.B * row, (k + 1) * c.B * row), c.B, c.T, lr[k], opts));
      const gpuLoss = await gpu.train(batches, K, c.B, c.T, lr, opts);
      push(`GPT ${c.name}: losses, call ${call + 1}`, worstRel(gpuLoss, cpuLoss), 1e-3);
    }
    const evalBatch = Int32Array.from({ length: c.B * row }, () => r.int(c.cfg.vocab));
    push(`GPT ${c.name}: eval loss`, worstRel([await gpu.evalLoss(evalBatch, c.B, c.T)], [cpu.evalLoss(evalBatch, c.B, c.T)]), 1e-3);
    // after 12 Adam steps of size ≈ lr, parameters agree to a small fraction of one step
    push(`GPT ${c.name}: parameters after 12 steps (max abs)`, worstAbs(await gpu.readParams(), cpu.snapshot()), 2e-4);
    gpu.destroy();
  }
  return checks;
}
