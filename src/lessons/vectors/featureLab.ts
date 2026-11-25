// authored-by: Claude Opus 5.5
/**
 * Lab: The Feature Workbench. Twelve emails, six measurements each. The learner
 * picks two measurements as axes and a unit for each; the score is how many
 * emails have a nearest neighbour (Euclidean distance) with the same label.
 * The solver scores every configuration (15 pairs × 5 × 5 units = 375).
 */

export const FEATURES = [
  { id: 'links', label: 'links' },
  { id: 'excl', label: 'exclamation marks' },
  { id: 'caps', label: 'ALL-CAPS words' },
  { id: 'name', label: 'mentions of your name' },
  { id: 'length', label: 'length (characters)' },
  { id: 'hour', label: 'hour sent' },
] as const;
export type FeatureId = (typeof FEATURES)[number]['id'];
export const FEATURE_IDS = FEATURES.map((f) => f.id) as FeatureId[];

export const UNITS = ['0.01', '0.1', '1', '10', 'std'] as const;
export type Unit = (typeof UNITS)[number];
export const UNIT_LABEL: Record<Unit, string> = { '0.01': '× 0.01', '0.1': '× 0.1', '1': '× 1', '10': '× 10', std: 'standardise' };

export type Email = { subject: string; spam: boolean; f: Record<FeatureId, number> };

const E = (subject: string, spam: boolean, links: number, excl: number, caps: number, name: number, length: number, hour: number): Email => ({
  subject,
  spam,
  f: { links, excl, caps, name, length, hour },
});

export const EMAILS: Email[] = [
  E('Sam, your parcel is waiting', true, 7, 4, 4, 2, 835, 11),
  E('Cheap meds, no prescription!!', true, 5, 8, 5, 1, 1795, 3),
  E('URGENT, Sam: verify your account', true, 7, 3, 5, 2, 1046, 6),
  E('CLAIM YOUR PRIZE now!!!', true, 6, 7, 5, 0, 393, 2),
  E('Last chance: 90% OFF!!!', true, 3, 7, 4, 0, 1768, 18),
  E('Sam, a business proposal', true, 4, 1, 3, 2, 1856, 6),
  E('Sam, LUNCH moved to Thursday', false, 2, 1, 3, 3, 632, 17),
  E('Can you review my PR, Sam?', false, 1, 1, 0, 3, 737, 19),
  E('This week’s newsletter', false, 4, 2, 2, 0, 645, 16),
  E('Party this Saturday!!!', false, 2, 5, 2, 0, 1916, 7),
  E('Happy birthday!!!!', false, 2, 4, 2, 0, 1523, 13),
  E('Notes from today’s meeting, Sam', false, 2, 1, 0, 3, 1872, 9),
];

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = (xs: number[]) => {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
};

/** One axis: a measurement in a unit, applied to every email. */
export function axisValues(feature: FeatureId, unit: Unit, emails = EMAILS): number[] {
  const raw = emails.map((e) => e.f[feature]);
  if (unit === 'std') {
    const m = mean(raw);
    const s = sd(raw) || 1;
    return raw.map((v) => (v - m) / s);
  }
  return raw.map((v) => v * Number(unit));
}

export type Config = { x: FeatureId; y: FeatureId; ux: Unit; uy: Unit };

/** Points in the plane for a configuration. */
export function points(c: Config, emails = EMAILS): [number, number][] {
  const xs = axisValues(c.x, c.ux, emails);
  const ys = axisValues(c.y, c.uy, emails);
  return xs.map((x, i) => [x, ys[i]]);
}

/** Each email's nearest other email (ties go to the earlier email in the list) and the distance to it. */
export function nearest(pts: [number, number][]): { j: number; d: number }[] {
  return pts.map((p, i) => {
    let best = { j: -1, d: Infinity };
    pts.forEach((q, j) => {
      if (j === i) return;
      const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d < best.d - 1e-12) best = { j, d };
    });
    return best;
  });
}

/** Neighbour agreement: how many emails have a nearest neighbour with the same label. */
export function score(c: Config, emails = EMAILS): number {
  if (c.x === c.y) return 0;
  const nn = nearest(points(c, emails));
  return nn.filter((n, i) => emails[n.j].spam === emails[i].spam).length;
}

export type Brief = {
  id: string;
  tab: string;
  title: string;
  text: string;
  /** Fixed axes (only the units may change). */
  locked?: { x: FeatureId; y: FeatureId };
  pass: (c: Config, s: number) => boolean;
};

export const BRIEFS: Brief[] = [
  { id: 'ten', tab: 'Brief 1', title: 'Brief 1 · Most neighbours agree', text: 'Choose two measurements and their units so that at least 10 of the 12 emails have a nearest neighbour with the same label.', pass: (c, s) => c.x !== c.y && s >= 10 },
  { id: 'all', tab: 'Brief 2', title: 'Brief 2 · Every neighbour agrees', text: 'Now all 12: every email’s nearest neighbour should share its label.', pass: (c, s) => c.x !== c.y && s === 12 },
  {
    id: 'units',
    tab: 'Brief 3',
    title: 'Brief 3 · The units trap',
    text: 'The axes are locked to links and length. In raw units the score is poor. Reach 12 / 12 by changing only the units.',
    locked: { x: 'links', y: 'length' },
    pass: (c, s) => c.x === 'links' && c.y === 'length' && s === 12,
  },
  {
    id: 'nolinks',
    tab: 'Brief 4 (optional)',
    title: 'Brief 4 · Optional: without links',
    text: 'Find a second route: reach 12 / 12 with a pair of measurements that doesn’t use links at all.',
    pass: (c, s) => c.x !== c.y && c.x !== 'links' && c.y !== 'links' && s === 12,
  },
];

/** Every unordered pair of different measurements, with every unit for each: 15 × 25 = 375 configurations. */
export function allConfigs(): Config[] {
  const out: Config[] = [];
  for (let a = 0; a < FEATURE_IDS.length; a++)
    for (let b = a + 1; b < FEATURE_IDS.length; b++)
      for (const ux of UNITS) for (const uy of UNITS) out.push({ x: FEATURE_IDS[a], y: FEATURE_IDS[b], ux, uy });
  return out;
}

/** Exhaustive search: every configuration with its score. */
export function solve() {
  const all = allConfigs().map((c) => ({ c, s: score(c) }));
  return {
    all,
    perfect: all.filter((r) => r.s === 12),
    atLeast10: all.filter((r) => r.s >= 10),
    byBrief: BRIEFS.map((b) => all.filter((r) => b.pass(r.c, r.s) && (!b.locked || (r.c.x === b.locked.x && r.c.y === b.locked.y)))),
  };
}
