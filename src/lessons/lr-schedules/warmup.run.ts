// authored-by: Claude Opus 5.5
/**
 * Records the warmup experiment over 20 seeds at each peak step size (CPU Float32 tier).
 * Run one peak per process, in parallel:
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/lr-schedules/warmup.run.ts 0.02
 * and paste each printed line into `WARM_RESULTS` in warmupResults.ts.
 */
import { CpuMlpEngine } from '../../lib/compute/cpuBackend';
import { runWarmup, WARM_SEEDS_RECORDED, warmSpec } from './warmup';

declare const process: { argv: string[] };
const peak = Number(process.argv[2]);
// optional: first seed and count, to re-check a finding on fresh seeds
const seeds = process.argv[3] ? Array.from({ length: Number(process.argv[4] ?? 20) }, (_, i) => Number(process.argv[3]) + i) : WARM_SEEDS_RECORDED;
const run = await runWarmup(new CpuMlpEngine(warmSpec(2 * seeds.length)), seeds, peak);
const last = run.history.at(-1)!;
const r3 = (v: number) => +v.toFixed(3);
console.log(
  JSON.stringify({
    peak,
    earlyMax: [0, 1].map((a) => run.earlyMax.map((e) => r3(e[a]))),
    finalAcc: [0, 1].map((a) => last.acc.map((e) => r3(e[a]))),
  }),
);
