// authored-by: Claude Opus 5.5
/**
 * Records the grokking demo's reference runs (CPU Float32 tier, 4 seeds, with and without weight decay).
 * Run: node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/open-questions/grok.run.ts
 * and paste the printed object into grokResults.ts.
 */
import { CpuMlpEngine } from '../../lib/compute/cpuBackend';
import { firstReach, GROK, grokSpec, runGrokking, type GrokCheckpoint } from './grokking';

const out: Record<string, unknown> = {};
for (const wd of [GROK.weightDecay, 0]) {
  const hist: GrokCheckpoint[] = [];
  await runGrokking(new CpuMlpEngine(grokSpec(wd, GROK.seeds.length)), GROK.seeds, (c) => (hist.push(c), true));
  out[`wd${wd}`] = {
    memorised: firstReach(hist, 'trainAcc', 1, GROK.seeds.length),
    grokked: firstReach(hist, 'valAcc', 0.99, GROK.seeds.length),
    finalVal: hist.at(-1)!.valAcc.map((v) => +v.toFixed(3)),
    bestVal: GROK.seeds.map((_, r) => +Math.max(...hist.map((h) => h.valAcc[r])).toFixed(3)),
  };
}
console.log(JSON.stringify(out));
