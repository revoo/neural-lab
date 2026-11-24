/**
 * The fair comparison of cures. Each cure's strength is chosen from a grid by its mean VALIDATION
 * error over datasets 1..DATASETS (the test set plays no part in the choice); the test error of
 * that choice is then reported. Results are recorded in results.ts and spot-checked by
 * reg.test.ts.
 *
 * Run:  node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/regularization/search.run.ts
 * (about three minutes), then paste the printed object into results.ts.
 */
import { NO_CURE, trainAndScore, type Cure } from './reg';

export const DATASETS = 10;

export type Setting = { method: 'none' | 'early' | 'l2' | 'dropout'; strength: number | null; label: string };

export const GRID: Setting[] = [
  { method: 'none', strength: null, label: 'no cure' },
  ...[50, 100, 200, 400, 800, 4000].map((p) => ({ method: 'early' as const, strength: p, label: `patience ${p}` })),
  ...[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1].map((l) => ({ method: 'l2' as const, strength: l, label: `λ = ${l}` })),
  ...[0.05, 0.1, 0.2, 0.3, 0.4, 0.5].map((p) => ({ method: 'dropout' as const, strength: p, label: `p = ${p}` })),
];

export function runSetting(s: Setting, k: number) {
  const cure: Cure = s.method === 'l2' ? { l2: s.strength!, dropout: 0 } : s.method === 'dropout' ? { l2: 0, dropout: s.strength! } : NO_CURE;
  return trainAndScore(k, cure, s.method === 'early' ? s.strength : null);
}

export function searchAll(log?: (s: string) => void) {
  return GRID.map((s) => {
    const per = Array.from({ length: DATASETS }, (_, i) => runSetting(s, i + 1));
    const mean = (f: (o: (typeof per)[number]) => number) => per.reduce((t, o) => t + f(o), 0) / per.length;
    const row = {
      ...s,
      val: mean((o) => o.val),
      test: mean((o) => o.test),
      stoppedAt: mean((o) => o.stoppedAt),
      perDataset: per.map((o) => ({ val: o.val, test: o.test, stoppedAt: o.stoppedAt })),
    };
    log?.(`${s.method} ${s.label}: val ${row.val.toFixed(4)} test ${row.test.toFixed(4)}`);
    return row;
  });
}
