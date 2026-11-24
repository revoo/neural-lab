/**
 * Search for the fewest-parameter configuration that meets each challenge's target.
 * Run it with:  node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/training-lab/search.run.ts
 * and paste the printed results into BEST_FOUND in challenges.ts. challenges.test.ts re-verifies them.
 *
 * Every candidate is trained exactly as the lab trains (Adam, step size 0.03, batches of 10, no L2),
 * checking test accuracy every 100 steps for up to 4,000 steps, with random starts 1–3.
 * Candidates are tried in order of parameter count, so the first success is the smallest in the space.
 */
import { CHALLENGES, challengeData, type Challenge } from './challenges';
import { DEFAULT_CONFIG, paramCount, trainUntil, type Act, type Config, type FeatureName } from './lab';

const ALL: FeatureName[] = ['x', 'y', 'x2', 'y2', 'xy', 'sinx', 'siny'];

function subsets(items: FeatureName[]): FeatureName[][] {
  const out: FeatureName[][] = [];
  for (let m = 1; m < 1 << items.length; m++) out.push(items.filter((_, i) => m & (1 << i)));
  return out;
}

function architectures(maxWidth: number, maxDepth: number): number[][] {
  const out: number[][] = [[]];
  const rec = (prefix: number[]) => {
    if (prefix.length >= maxDepth) return;
    for (let w = 1; w <= maxWidth; w++) {
      const a = [...prefix, w];
      out.push(a);
      rec(a);
    }
  };
  rec([]);
  return out;
}

export function searchSpace(c: Challenge): Config[] {
  const base = { ...DEFAULT_CONFIG };
  const cands: Config[] = [];
  const acts: Act[] = ['tanh', 'relu'];
  if (c.id === 'circle-flat') {
    for (const features of subsets(ALL)) cands.push({ ...base, features, hidden: [], activation: 'tanh' });
  } else if (c.id === 'xor-raw') {
    for (const hidden of architectures(6, 2)) for (const activation of acts) cands.push({ ...base, features: ['x', 'y'], hidden, activation });
  } else {
    // Spiral: every feature subset with no hidden layer or one hidden layer of up to 6 neurons.
    for (const features of subsets(ALL)) for (const hidden of architectures(6, 1)) for (const activation of acts) cands.push({ ...base, features, hidden, activation });
  }
  return cands;
}

export function searchChallenge(c: Challenge, seeds = [1, 2, 3], log?: (s: string) => void) {
  const { train, test } = challengeData(c);
  const cands = searchSpace(c)
    .filter((cfg) => c.rules(cfg) === null)
    .sort((a, b) => paramCount(a) - paramCount(b));
  let tried = 0;
  for (const cfg of cands) {
    for (const seed of seeds) {
      tried++;
      const r = trainUntil({ ...cfg, seed }, train, test, c.target);
      if (r.reachedAt !== null) {
        log?.(`${c.id}: ${paramCount(cfg)} params ${JSON.stringify({ ...cfg, seed })} at step ${r.reachedAt} after ${tried} runs`);
        return { params: paramCount(cfg), cfg: { ...cfg, seed }, reachedAt: r.reachedAt, searched: tried, spaceSize: cands.length * seeds.length };
      }
    }
  }
  return null;
}

export function searchAll(log: (s: string) => void) {
  return CHALLENGES.map((c) => searchChallenge(c, [1, 2, 3], log));
}
