// authored-by: Gemini Antigravity
/**
 * Measures speculative decoding across the full 48-cell grid:
 *   drafter ∈ ['bigram', 'small-gpt'] × temperature ∈ [0, 0.7, 1.0] × k ∈ [1..8]
 * on 12 fixed held-out prompts from Grimm's validation tales (36 new characters per prompt,
 * total 60 ≤ 64 ctx). Fixed seeds [11, 22, 33] for stochastic temperatures (1,296 chars/cell).
 * Writes `specResults.ts`.
 *
 * Run with:
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs "$(pwd)/src/lessons/llm-serving/spec.run.ts"
 */
import { Gpt } from '../../lib/compute/gpt';
import { prepareCorpus } from '../../lib/corpora/grimm';
import { nodeFs } from '../../lib/nodeIo';
import {
  BigramDrafter,
  DRAFT_CFG,
  GptDrafter,
  TARGET_CFG,
  buildBigramLogProbs,
  expectedTokensFormula,
  heldOutPrompts,
  modelledSpeedup,
  productionCostRatio,
  runPlainTarget,
  runSpeculative,
  type DrafterId,
} from './spec';

export const SPEC_DRAFTERS: readonly DrafterId[] = ['bigram', 'small-gpt'];
export const SPEC_TEMPS = [0, 0.7, 1.0] as const;
export const SPEC_KS = [1, 2, 3, 4, 5, 6, 7, 8] as const;
export const SPEC_MAX_NEW = 36;
export const SPEC_STOCHASTIC_SEEDS = [11, 22, 33] as const;

export interface SpecGridCell {
  drafter: DrafterId;
  temperature: number;
  k: number;
  alpha: number;
  alphaWordStart: number;
  alphaMidWord: number;
  tokensPerPass: number;
  formulaTokensPerPass: number;
  modelledSpeedup: number;
  formulaSpeedup: number;
  costRatio: number;
  totalEmitted: number;
  totalTargetPasses: number;
  /** Measured Node CPU wall-clock ratio (plain / speculative; < 1 means speculation was SLOWER). */
  cpuWallSpeedup: number;
}

export function measureCell(params: {
  targetModel: Gpt;
  draftModel: Gpt;
  bigramLogProbs: Float64Array;
  prompts: { title: string; prompt: string }[];
  drafter: DrafterId;
  temperature: number;
  k: number;
}): SpecGridCell {
  const { targetModel, draftModel, bigramLogProbs, prompts, drafter, temperature, k } = params;
  const seeds = temperature === 0 ? [11] : [...SPEC_STOCHASTIC_SEEDS];

  let totalEmitted = 0;
  let totalTargetPasses = 0;
  let totalDrafted = 0;
  let totalAccepted = 0;
  let wsDrafted = 0;
  let wsAccepted = 0;
  let mwDrafted = 0;
  let mwAccepted = 0;
  let specWallMs = 0;
  let plainWallMs = 0;

  for (const p of prompts) {
    for (const seed of seeds) {
      const dr = drafter === 'bigram' ? new BigramDrafter(bigramLogProbs) : new GptDrafter(draftModel);
      const r = runSpeculative({
        targetModel,
        drafter: dr,
        prompt: p.prompt,
        maxNewTokens: SPEC_MAX_NEW,
        k,
        temperature,
        seed,
      });
      totalEmitted += r.totalEmitted;
      totalTargetPasses += r.targetPasses;
      totalDrafted += r.totalDrafted;
      totalAccepted += r.totalAccepted;
      wsDrafted += r.wordStartDrafted;
      wsAccepted += r.wordStartAccepted;
      mwDrafted += r.midWordDrafted;
      mwAccepted += r.midWordAccepted;
      specWallMs += r.wallMs;

      const plain = runPlainTarget({
        targetModel,
        prompt: p.prompt,
        maxNewTokens: SPEC_MAX_NEW,
        temperature,
        seed,
      });
      plainWallMs += plain.wallMs;
    }
  }

  const alpha = totalDrafted > 0 ? totalAccepted / totalDrafted : 0;
  const alphaWordStart = wsDrafted > 0 ? wsAccepted / wsDrafted : 0;
  const alphaMidWord = mwDrafted > 0 ? mwAccepted / mwDrafted : 0;
  const tokensPerPass = totalTargetPasses > 0 ? totalEmitted / totalTargetPasses : 0;
  const formulaTokensPerPass = expectedTokensFormula(alpha, k);
  const spd = modelledSpeedup(tokensPerPass, drafter, k);
  const fSpd = modelledSpeedup(formulaTokensPerPass, drafter, k);

  return {
    drafter,
    temperature,
    k,
    alpha: +alpha.toFixed(4),
    alphaWordStart: +alphaWordStart.toFixed(4),
    alphaMidWord: +alphaMidWord.toFixed(4),
    tokensPerPass: +tokensPerPass.toFixed(4),
    formulaTokensPerPass: +formulaTokensPerPass.toFixed(4),
    modelledSpeedup: +spd.speedup.toFixed(4),
    formulaSpeedup: +fSpd.formulaSpeedup.toFixed(4),
    costRatio: +spd.c.toFixed(4),
    totalEmitted,
    totalTargetPasses,
    cpuWallSpeedup: +(plainWallMs / Math.max(1e-6, specWallMs)).toFixed(3),
  };
}

const procArgv = (globalThis as unknown as { process?: { argv?: string[] } }).process?.argv;
const isMain = Boolean(procArgv?.[1] && import.meta.url === `file://${procArgv[1]}`);
if (isMain) {
  const fs = await nodeFs();
  const corpus = prepareCorpus(fs.readFileSync(new URL('../../lib/corpora/grimm.txt', import.meta.url), 'utf8'));
  const prompts = heldOutPrompts(corpus.text, corpus.split);
  const bigramLogProbs = buildBigramLogProbs(corpus.train);

  const loadBin = (url: URL) => {
    const b = fs.readFileSync(url);
    return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  };

  const targetModel = new Gpt(TARGET_CFG);
  targetModel.load(loadBin(new URL('../../lib/compute/tinyGpt.cpu.bin', import.meta.url)));

  const draftModel = new Gpt(DRAFT_CFG);
  draftModel.load(loadBin(new URL('./draftGpt.bin', import.meta.url)));

  const cells: SpecGridCell[] = [];
  for (const drafter of SPEC_DRAFTERS) {
    for (const temperature of SPEC_TEMPS) {
      for (const k of SPEC_KS) {
        const cell = measureCell({ targetModel, draftModel, bigramLogProbs, prompts, drafter, temperature, k });
        cells.push(cell);
        console.log(
          `${drafter.padEnd(9)} T=${temperature.toFixed(1)} k=${k} | α=${(100 * cell.alpha).toFixed(1).padStart(4)}% (start ${(100 * cell.alphaWordStart).toFixed(1).padStart(4)}% / mid ${(100 * cell.alphaMidWord).toFixed(1).padStart(4)}%) | E=${cell.tokensPerPass.toFixed(2)} (iid ${cell.formulaTokensPerPass.toFixed(2)}) | prod=${cell.modelledSpeedup.toFixed(2)}× | cpu=${cell.cpuWallSpeedup.toFixed(2)}×`,
        );
      }
    }
  }

  const out = {
    prompts,
    costRatios: {
      bigram: +productionCostRatio('bigram').c.toFixed(4),
      'small-gpt': +productionCostRatio('small-gpt').c.toFixed(4),
    },
    cells,
  };

  const src = `// authored-by: Gemini Antigravity
// Generated by spec.run.ts: exhaustive 48-cell speculative decoding grid over 12 held-out Grimm prompts.
import type { SpecGridCell } from './spec.run';

export const SPEC_PROMPTS = ${JSON.stringify(out.prompts, null, 2)} as const;
export const SPEC_COST_RATIOS = ${JSON.stringify(out.costRatios, null, 2)} as const;
export const SPEC_GRID: readonly SpecGridCell[] = ${JSON.stringify(out.cells, null, 2)};
`;
  fs.writeFileSync(new URL('./specResults.ts', import.meta.url), src);
  console.log('\nWrote specResults.ts with', cells.length, 'cells.');
}
