import { searchAll } from './search';

const rows = searchAll((s) => console.error(s));
const round = (v: number) => Number(v.toPrecision(6));
const out = rows.map((r) => ({
  method: r.method,
  strength: r.strength,
  label: r.label,
  val: round(r.val),
  test: round(r.test),
  stoppedAt: round(r.stoppedAt),
  perDataset: r.perDataset.map((o) => ({ val: round(o.val), test: round(o.test), stoppedAt: o.stoppedAt })),
}));
console.log(JSON.stringify(out));
