import { CASES } from './cases';
import { searchCase } from './search';

const out: Record<string, { label: string; fresh: number }[]> = {};
for (const c of CASES) {
  const t0 = Date.now();
  out[c.id] = searchCase(c.id);
  console.error(c.id, `${Date.now() - t0} ms`, out[c.id].filter((r) => r.fresh >= c.target).map((r) => `${r.label} (${r.fresh})`).join('; '));
}
console.log(JSON.stringify(out));
