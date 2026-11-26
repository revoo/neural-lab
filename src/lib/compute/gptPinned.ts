// authored-by: Claude Opus 5.5
/**
 * The CPU tier's determinism reference (V1 capstone, A2). `cpuDeterminismHash()` trains a small
 * model for 30 steps and hashes its parameters. Node's value is pinned below and in `gpt.test.ts`;
 * `scripts/check-webgpu.cjs` runs the same function in Chromium and requires the same hash.
 */
import { Gpt, type GptConfig } from './gpt';
import { createRng } from '../rng';

export const CPU_DETERMINISM_HASH = '463ddb7d';

/** FNV-1a over the parameter bytes, as a hex string. */
export function paramHash(p: Float32Array): string {
  const bytes = new Uint8Array(p.buffer, p.byteOffset, p.byteLength);
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193) >>> 0;
  return h.toString(16).padStart(8, '0');
}

/** The determinism reference: 30 AdamW steps of a small model on seeded random batches. */
export function determinismSetup() {
  const cfg: GptConfig = { vocab: 23, ctx: 12, d: 16, heads: 2, layers: 2 };
  const m = new Gpt(cfg);
  m.init(42);
  const r = createRng(7);
  const B = 3;
  const T = 12;
  const K = 30;
  const batches = Int32Array.from({ length: K * B * (T + 1) }, () => r.int(cfg.vocab));
  return { cfg, init: m.snapshot(), batches, K, B, T, lr: new Float32Array(K).fill(3e-3), opts: { weightDecay: 0.1, clip: 1 } };
}
export function cpuDeterminismHash(): string {
  const s = determinismSetup();
  const m = new Gpt(s.cfg);
  m.load(s.init);
  const row = s.B * (s.T + 1);
  for (let k = 0; k < s.K; k++) m.trainStep(s.batches.subarray(k * row, (k + 1) * row), s.B, s.T, s.lr[k], s.opts);
  return paramHash(m.snapshot());
}

