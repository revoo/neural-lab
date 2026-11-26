// authored-by: Claude Opus 5.5
/**
 * Exhaustive search of the lab's 120 settings × 5 seeds. Run in shards, in parallel:
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/moe/search.run.ts <shard> <shards>
 * then join the printed rows (in shard order) into LAB_RESULTS in results.ts.
 */
import { allSettings, scoreSetting } from './lab';

declare const process: { argv: string[] };
const shard = Number(process.argv[2] ?? 0);
const shards = Number(process.argv[3] ?? 1);
const r4 = (v: number) => +v.toFixed(4);
allSettings().forEach((s, i) => {
  if (i % shards !== shard) return;
  const r = scoreSetting(s);
  console.log(JSON.stringify({ i, k: r.k, alpha: r.alpha, capacity: r.capacity, noise: r.noise, acc: r4(r.acc), maxLoad: r4(r.maxLoad), minLoad: r4(r.minLoad), drop: r4(r.drop) }) + ',');
});
