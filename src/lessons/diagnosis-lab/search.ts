/**
 * Exhaustive search over single changes: for every case, train every configuration that differs
 * from the broken one in exactly one knob (cases.ts → singleChanges) and record its accuracy on
 * fresh data. The lab uses this to say how many one-change fixes exist for each case.
 *
 * Run:  node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/diagnosis-lab/search.run.ts
 * (a few minutes) and paste the output into fixes.ts. clinic.test.ts re-verifies a sample.
 */
import { Clinic } from './clinic';
import { CASES, singleChanges } from './cases';

export function searchCase(id: string) {
  const c = CASES.find((k) => k.id === id)!;
  return singleChanges(c.cfg).map((ch) => {
    const run = new Clinic(ch.apply(c.cfg)).run(100);
    return { label: ch.label, fresh: Number(run.last.freshAcc.toFixed(4)) };
  });
}
