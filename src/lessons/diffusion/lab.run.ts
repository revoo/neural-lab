// authored-by: Claude Opus 5.5
/**
 * Exhaustive search for Lab: Spend the Step Budget, for the lab's noise seed (5) and two others
 * (6, 7) to check that the fewest-steps answer doesn't depend on the particular noise drawn.
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/diffusion/lab.run.ts > scratch/diffusion-lab.json
 * Then paste the JSON into labResults.ts (lab.test.ts recomputes seed 5 and checks it).
 */
import { BRIEFS, exhaustive } from './lab';

const out: Record<string, unknown> = {};
for (const b of BRIEFS) {
  const main = exhaustive(b, 5);
  const others = [6, 7].map((s) => exhaustive(b, s).fewest);
  out[b.id] = {
    fewest: main.fewest,
    perSteps: main.perSteps.map((p) => ({ ...p, best: Number(p.best.toFixed(4)) })),
    fewestWithOtherSeeds: others,
  };
}
console.log(JSON.stringify(out, null, 1));
