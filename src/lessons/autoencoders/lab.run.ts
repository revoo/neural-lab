// authored-by: Claude Opus 5.5
/**
 * Exhaustive search for Lab: The Face Printer. Trains every design (3 code sizes × 3 widths ×
 * 6 regularisers) with each of the lab's three seeds. Both briefs judge the same runs.
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/autoencoders/lab.run.ts [shard] [shards]
 * Prints JSON lines: save shard i as scratch/lab-<i>.jsonl, then run `node scripts/ae-lab-results.mjs`
 * to regenerate labResults.ts. Run 4–8 shards in parallel; the whole search takes about 8 minutes.
 */
import { LAB_SEEDS, allDesigns, designKey, labTrainer, printAndJudge } from './lab';

declare const process: { argv: string[] };

const shard = Number(process.argv[2] ?? 0);
const shards = Number(process.argv[3] ?? 1);
const designs = allDesigns();
let k = 0;
for (const d of designs) {
  if (k++ % shards !== shard) continue;
  const seeds: { recon: number; realShare: number; cells: number }[] = [];
  for (const seed of LAB_SEEDS) {
    const t = labTrainer(d, seed);
    t.advance(Infinity);
    const r = printAndJudge(t.net);
    seeds.push({ recon: +r.recon.toFixed(5), realShare: r.realShare, cells: r.cells });
  }
  console.log(JSON.stringify({ key: designKey(d), seeds }));
}
