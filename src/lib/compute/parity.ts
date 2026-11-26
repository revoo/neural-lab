// authored-by: Claude Opus 5.5
/**
 * V2-WP3: WebGPU ↔ CPU parity checks. Vitest can't run WGSL, so `scripts/check-webgpu.cjs` loads
 * this module into a real Chromium (with WebGPU enabled; headless it is the SwiftShader adapter)
 * and calls `runParity()`. Every GPU kernel is compared with the CPU reference tier.
 */
import { CpuMlpEngine, matmul, tensor } from './cpuBackend';
import { getGpu, gpuGemm, GpuMlpEngine } from './webgpuBackend';
import type { MlpSpec, Weights } from './types';
import { createRng } from '../rng';

type Check = { name: string; worst: number; tol: number; ok: boolean };

function worstRel(a: ArrayLike<number>, b: ArrayLike<number>) {
  let w = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]) / Math.max(1e-3, Math.abs(b[i]));
    w = Math.max(w, Number.isFinite(d) ? d : Infinity);
  }
  return w;
}

function randomWeights(sizes: number[], seed: number): Weights {
  const r = createRng(seed);
  return {
    W: sizes.slice(1).map((o, l) => Float32Array.from({ length: o * sizes[l] }, () => r.normal(0, Math.sqrt(2 / (o + sizes[l]))))),
    b: sizes.slice(1).map((o) => Float32Array.from({ length: o }, () => r.normal(0, 0.1))),
  };
}

const CASES: { name: string; spec: MlpSpec; label: (r: ReturnType<typeof createRng>, x: number[]) => number[] }[] = [
  { name: 'bce · tanh · adam · L2', spec: { sizes: [2, 8, 8, 1], acts: ['tanh', 'tanh', 'sigmoid'], loss: 'bce', opt: { kind: 'adam' }, l2: 0.002 }, label: (_, x) => [x[0] * x[1] > 0 ? 1 : 0] },
  { name: 'softmax-ce · relu · adamw · 3 replicas', spec: { sizes: [6, 20, 5], acts: ['relu', 'linear'], loss: 'softmax-ce', opt: { kind: 'adamw', weightDecay: 0.1 }, replicas: 3 }, label: (r) => { const y = [0, 0, 0, 0, 0]; y[r.int(5)] = 1; return y; } },
  { name: 'mse · gelu · momentum', spec: { sizes: [3, 17, 2], acts: ['gelu', 'linear'], loss: 'mse', opt: { kind: 'momentum' } }, label: (_, x) => [Math.sin(x[0]), x[1] * x[2]] },
  { name: 'mse · square · adamw (the grokking model)', spec: { sizes: [10, 24, 5], acts: ['square', 'linear'], loss: 'mse', opt: { kind: 'adamw', weightDecay: 0.5 } }, label: (r) => { const y = [0, 0, 0, 0, 0]; y[r.int(5)] = 1; return y; } },
  { name: 'bce · relu · sgd · wide (40×33)', spec: { sizes: [5, 40, 33, 1], acts: ['relu', 'relu', 'sigmoid'], loss: 'bce', opt: { kind: 'sgd' } }, label: (_, x) => [x[0] + x[1] > 0 ? 1 : 0] },
];

export async function runParity(): Promise<{ adapter: string; checks: Check[] }> {
  const gpu = await getGpu();
  if (!gpu) return { adapter: 'none', checks: [{ name: 'WebGPU adapter', worst: Infinity, tol: 0, ok: false }] };
  const checks: Check[] = [];
  const push = (name: string, worst: number, tol: number) => checks.push({ name, worst, tol, ok: worst <= tol });

  for (const c of CASES) {
    const R = c.spec.replicas ?? 1;
    const cpu = new CpuMlpEngine(c.spec);
    const gpuE = new GpuMlpEngine(c.spec, gpu.device);
    for (let r = 0; r < R; r++) {
      const w = randomWeights(c.spec.sizes, 100 + r);
      cpu.setWeights(r, w);
      gpuE.setWeights(r, w);
    }
    const rng = createRng(7);
    const n = 50;
    const din = c.spec.sizes[0];
    const xs = Array.from({ length: n }, () => Array.from({ length: din }, () => rng.uniform(-1.5, 1.5)));
    const X = Float32Array.from(xs.flat());
    const Y = Float32Array.from(xs.flatMap((x) => c.label(rng, x)));
    cpu.setData(X, Y, n);
    gpuE.setData(X, Y, n);
    const B = 12;
    // two calls of 15 steps: state (weights, moments, Adam's step count) must persist on the GPU between calls
    for (let call = 0; call < 2; call++) {
      const steps = 15;
      const idx = Int32Array.from({ length: steps * R * B }, () => rng.int(n));
      const lr = Float32Array.from({ length: steps * R }, (_, i) => 0.02 * (1 + (i % R)));
      const a = cpu.trainSync({ steps, batch: B, idx, lr });
      const g = await gpuE.train({ steps, batch: B, idx, lr });
      push(`${c.name}: losses, call ${call + 1}`, worstRel(g, a), 2e-3);
    }
    // full-batch path (no idx)
    const fl = Float32Array.from({ length: 5 * R }, () => 0.01);
    push(`${c.name}: full-batch losses`, worstRel(await gpuE.train({ steps: 5, batch: n, lr: fl }), cpu.trainSync({ steps: 5, batch: n, lr: fl })), 2e-3);
    const wc = cpu.readWeightsSync();
    const wg = await gpuE.readWeights();
    const flat = (ws: Float32Array[]) => ws.flatMap((x) => Array.from(x));
    push(`${c.name}: weights after 35 steps`, Math.max(...wc.map((w, r) => worstRel(flat(wg[r].W), flat(w.W)))), 5e-3);
    const grid = Float32Array.from({ length: 4100 * din }, () => rng.uniform(-2, 2)); // crosses the 4096-row chunk
    push(`${c.name}: predictions on 4,100 rows`, worstRel(await gpuE.predict(grid, 4100), cpu.predictSync(grid, 4100)), 2e-3);
    gpuE.destroy();
  }

  for (const n of [37, 128]) {
    const r = createRng(n);
    const A = Float32Array.from({ length: n * n }, () => r.normal());
    const B = Float32Array.from({ length: n * n }, () => r.normal());
    const { out } = await gpuGemm(gpu.device, A, B, { M: n, N: n, K: n }, 3);
    const naive = await gpuGemm(gpu.device, A, B, { M: n, N: n, K: n }, 2, false);
    const ref = matmul(tensor(n, n, A), tensor(n, n, B)).data;
    let worst = 0;
    for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(out[i] - ref[i]), Math.abs(naive.out[i] - ref[i]));
    push(`GEMM ${n}×${n}, batched and one-per-submit (max abs error)`, worst, 1e-3);
  }
  {
    const n = 300;
    const r = createRng(5);
    const A = Float32Array.from({ length: n * n }, () => r.normal());
    const v = Float32Array.from({ length: n }, () => r.normal());
    const { out } = await gpuGemm(gpu.device, A, v, { M: n, N: 1, K: n }, 2);
    const ref = matmul(tensor(n, n, A), tensor(n, 1, v)).data;
    let worst = 0;
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(out[i] - ref[i]));
    push('matrix × vector 300 (max abs error)', worst, 1e-3);
  }
  return { adapter: gpu.name, checks };
}
