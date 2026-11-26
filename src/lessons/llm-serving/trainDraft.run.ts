// authored-by: Gemini Antigravity
/**
 * Trains the small character-level draft GPT (5,408 parameters: d = 16, 2 heads, 1 layer, ctx = 64)
 * on the exact same Grimm's Fairy Tales training split as `tinyGpt.cpu.bin` (29,696 parameters),
 * writing `draftGpt.bin` (~21 KiB). Seeds were fixed before training.
 *
 * Run with:
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs "$(pwd)/src/lessons/llm-serving/trainDraft.run.ts"
 */
import { gptParamCount, type GptConfig } from '../../lib/compute/gpt';
import { runTraining, type Checkpoint, type RunSpec } from '../../lib/compute/gptRun';
import { InlineCpuTrainer } from '../../lib/compute/gptTrainerCore';
import { prepareCorpus, VOCAB } from '../../lib/corpora/grimm';
import { nodeFs } from '../../lib/nodeIo';
import { paramHash } from '../../lib/compute/gptPinned';

export const DRAFT_CFG: GptConfig = {
  vocab: VOCAB,
  ctx: 64,
  d: 16,
  heads: 2,
  layers: 1,
};

export const DRAFT_RUN: RunSpec = {
  cfg: DRAFT_CFG,
  T: 64,
  batch: 16,
  steps: 1500,
  lr: 5e-3,
  warmup: 50,
  minLrFrac: 0.1,
  opt: { beta1: 0.9, beta2: 0.99, weightDecay: 0.1, clip: 1 },
  initSeed: 7,
  dataSeed: 1007,
  evalRows: 32,
};

const procArgv = (globalThis as unknown as { process?: { argv?: string[] } }).process?.argv;
const isMain = Boolean(procArgv?.[1] && import.meta.url === `file://${procArgv[1]}`);
if (isMain) {
  const fs = await nodeFs();
  const corpus = prepareCorpus(fs.readFileSync(new URL('../../lib/corpora/grimm.txt', import.meta.url), 'utf8'));
  let finalParams: Float32Array<ArrayBufferLike> = new Float32Array(0);
  let lastVal = NaN;
  const t0 = Date.now();
  await runTraining(DRAFT_RUN, corpus, new InlineCpuTrainer(DRAFT_CFG), (c: Checkpoint) => {
    finalParams = c.params;
    lastVal = c.valLoss;
    if (c.step === 0 || c.step === DRAFT_RUN.steps || c.step % 500 === 0) {
      console.log(`step ${c.step}: train ${c.trainLoss.toFixed(4)} val ${c.valLoss.toFixed(4)} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    }
  });
  fs.writeFileSync(
    new URL('./draftGpt.bin', import.meta.url),
    new Uint8Array(finalParams.buffer, finalParams.byteOffset, finalParams.byteLength),
  );
  console.log(`Saved draftGpt.bin: ${gptParamCount(DRAFT_CFG)} params, val ${lastVal.toFixed(4)}, hash ${paramHash(finalParams)}`);
}
