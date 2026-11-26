// authored-by: Claude Opus 5.5
/**
 * The tiny GPT's reference runs (V1 capstone, A3). Settled by measurement (sizing runs recorded in
 * docs/ROADMAP.md): the CPU run is what a learner without WebGPU trains live in a Web Worker, and is
 * bit-identical to the run recorded by `gptReference.run.ts` into `gptReferenceResults.ts`.
 * Seeds were fixed before any reference run was looked at.
 */
import { Gpt } from './gpt';
import type { RunSpec } from './gptRun';
import { decode, encode, VOCAB } from '../corpora/grimm';

export const CPU_RUN: RunSpec = {
  cfg: { vocab: VOCAB, ctx: 64, d: 32, heads: 4, layers: 2 },
  T: 64,
  batch: 16,
  steps: 2000,
  lr: 3e-3,
  warmup: 50,
  minLrFrac: 0.1,
  opt: { beta1: 0.9, beta2: 0.99, weightDecay: 0.1, clip: 1 },
  initSeed: 1,
  dataSeed: 1001,
  evalRows: 32,
};

/**
 * The larger model for learners with WebGPU (about 0.35M parameters). Its reference was recorded on
 * the CPU tier in Node (same maths); no GPU timing has been measured for it (headless Chromium only
 * offers SwiftShader, a CPU emulation).
 */
export const GPU_RUN: RunSpec = {
  ...CPU_RUN,
  cfg: { vocab: VOCAB, ctx: 64, d: 96, heads: 4, layers: 3 },
  steps: 3000,
  lr: 2e-3,
};

/** The fixed prompt and sampling seeds used for every checkpoint's samples. */
export const PROMPT = 'There was once';
export const SAMPLE_SEEDS = [1, 2, 3, 4];
export const SAMPLE_LEN = 200;
export const SAMPLE_TEMPERATURE = 1;

/** The four fixed-seed samples of a checkpoint, exactly as the reference run records them. */
export function samplesAt(spec: RunSpec, params: Float32Array, prompt = PROMPT): string[] {
  const m = new Gpt(spec.cfg);
  m.load(params);
  return SAMPLE_SEEDS.map((seed) => decode(m.generate(encode(prompt), SAMPLE_LEN, { seed, temperature: SAMPLE_TEMPERATURE })));
}
