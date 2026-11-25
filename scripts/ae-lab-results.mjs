// authored-by: Claude Opus 5.5
// Collects the JSON lines printed by src/lessons/autoencoders/lab.run.ts (saved as scratch/lab-<shard>.jsonl)
// into src/lessons/autoencoders/labResults.ts. Usage: node scripts/ae-lab-results.mjs
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
const rows = readdirSync('scratch').filter((f) => /^lab-\d+\.jsonl$/.test(f)).flatMap((f) => readFileSync('scratch/' + f, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)));
const out = {};
for (const r of rows) if (!r.brief || r.brief === 'printer') out[r.key] = { seeds: r.seeds };
const sortKeys = (o) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b, 'en', { numeric: true })));
const body = `// authored-by: Claude Opus 5.5
/**
 * Recorded by \`lab.run.ts\` (every design × seeds 1, 2, 3, in Node). Do not edit by hand: re-run
 * the script and regenerate. \`lab.test.ts\` re-trains a sample of designs and checks them.
 */
import { BRIEFS, LAB_SEEDS, allDesigns, designKey, paramCount, verdict, verdictAll, type BriefId, type Design, type Measured } from './lab';

export const LAB_RESULTS: Record<string, { seeds: Measured[] }> = ${JSON.stringify(sortKeys(out), null, 1)};

const brief = (id: BriefId) => BRIEFS.find((b) => b.id === id)!;

/** Designs that meet the brief's spec with every one of the lab's seeds. */
export function passingDesigns(id: BriefId): Design[] {
  return allDesigns().filter((d) => verdictAll(brief(id), LAB_RESULTS[designKey(d)].seeds));
}

/** Designs that would have passed if only seed 1 had been tried (lucky ones included). */
export function passingSeed1(id: BriefId): Design[] {
  return allDesigns().filter((d) => verdict(brief(id), LAB_RESULTS[designKey(d)].seeds[0]).pass);
}

/** The passing design with the fewest parameters. */
export function labOptimum(id: BriefId) {
  const pass = passingDesigns(id);
  const min = Math.min(...pass.map(paramCount));
  const best = pass.filter((d) => paramCount(d) === min);
  const lucky = passingSeed1(id);
  const luckyMin = Math.min(...lucky.map(paramCount));
  return { design: best[0], ties: best.length, params: min, passing: pass.length, seeds: LAB_SEEDS.length, seed1Passing: lucky.length, seed1Smallest: lucky.find((d) => paramCount(d) === luckyMin)! };
}
`;
writeFileSync('src/lessons/autoencoders/labResults.ts', body);
console.log('rows', Object.keys(out).length);
