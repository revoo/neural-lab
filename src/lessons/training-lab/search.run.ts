// Runs the challenge search and prints results to paste into BEST_FOUND (challenges.ts).
// node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/training-lab/search.run.ts
import { searchAll } from './search';

const t0 = Date.now();
const results = searchAll((s) => console.log(s));
console.log(JSON.stringify(results, null, 2));
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
