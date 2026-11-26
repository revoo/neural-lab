// authored-by: Claude Opus 5.5
/**
 * One training run of the tiny GPT, shared by the Node reference script and the browser widgets
 * (V1 capstone, A3). Same seeds, same batches, same learning rates and the same float32 maths, so
 * on the CPU tier the browser run *is* the recorded run, bit for bit.
 *
 * The run is a sequence of chunks of K steps; after each chunk that ends on a checkpoint step it
 * reads the parameters back and reports them. Checkpoints are spaced roughly logarithmically, so
 * the fast early changes are not missed.
 */
import { Gpt, type AdamWOpts, type GptConfig } from './gpt';
import type { GptTrainer } from './gptTrainerCore';
import { createRng } from '../rng';
import { evalWindows, sampleBatches, type Corpus } from '../corpora/grimm';

export type RunSpec = {
  cfg: GptConfig;
  /** Sequence length used in training (≤ cfg.ctx). */
  T: number;
  batch: number;
  steps: number;
  lr: number;
  warmup: number;
  /** Learning rate at the end of the cosine, as a fraction of `lr`. */
  minLrFrac: number;
  opt: AdamWOpts;
  /** Seeds: weights, and the batch sampler. Fixed before any run was looked at. */
  initSeed: number;
  dataSeed: number;
  /** Validation windows (evenly spaced over the held-out tales). */
  evalRows: number;
};

/** Warmup from lr/warmup up to lr, then a cosine down to minLrFrac·lr at the last step. Rounded to float32. */
export function lrAt(s: RunSpec, step: number): number {
  const { lr, warmup, steps, minLrFrac } = s;
  if (step < warmup) return Math.fround((lr * (step + 1)) / warmup);
  const p = (step - warmup) / Math.max(1, steps - warmup);
  return Math.fround(lr * (minLrFrac + (1 - minLrFrac) * 0.5 * (1 + Math.cos(Math.PI * p))));
}

/** Roughly log-spaced checkpoint steps from 0 to `steps` inclusive (about 9 per decade). */
export function checkpointSteps(steps: number): number[] {
  const out = new Set<number>([0]);
  for (let k = 0; ; k++) {
    const v = Math.round(10 ** (k / 9));
    if (v >= steps) break;
    out.add(v);
  }
  out.add(steps);
  return [...out].sort((a, b) => a - b);
}

export type Checkpoint = { step: number; params: Float32Array; trainLoss: number; valLoss: number; lr: number; ms: number };

/** The initial parameters of a run (computed on the CPU, so identical for every tier). */
export function initialParams(s: RunSpec): Float32Array {
  const m = new Gpt(s.cfg);
  m.init(s.initSeed);
  return m.snapshot();
}

/**
 * Train `spec` on `trainer`, calling `onCheckpoint` at every checkpoint step (step 0 included).
 * Return false from the callback to stop early. `chunk` caps how many steps go into one call.
 */
export async function runTraining(
  spec: RunSpec,
  corpus: Corpus,
  trainer: GptTrainer,
  onCheckpoint: (c: Checkpoint) => boolean | void | Promise<boolean | void>,
  chunk = 25,
): Promise<void> {
  const { T, batch: B, steps } = spec;
  await trainer.setParams(initialParams(spec));
  const rng = createRng(spec.dataSeed);
  const ev = evalWindows(corpus.val, spec.evalRows, T);
  const marks = checkpointSteps(steps);
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
  let lastTrain = NaN;
  const report = async (step: number) => {
    const valLoss = await trainer.evalLoss(ev, spec.evalRows, T);
    return onCheckpoint({ step, params: await trainer.readParams(), trainLoss: lastTrain, valLoss, lr: step < steps ? lrAt(spec, step) : lrAt(spec, steps - 1), ms: now() });
  };
  if ((await report(0)) === false) return;
  let step = 0;
  let mi = 1;
  while (step < steps) {
    const K = Math.min(chunk, marks[mi] - step);
    const batches = sampleBatches(corpus.train, K, B, T, rng);
    const lr = Float32Array.from({ length: K }, (_, k) => lrAt(spec, step + k));
    const losses = await trainer.train(batches, K, B, T, lr, spec.opt);
    // mean training loss of this chunk
    lastTrain = losses.reduce((a, b) => a + b, 0) / K;
    step += K;
    if (step === marks[mi]) {
      mi++;
      if ((await report(step)) === false) return;
    }
  }
}
