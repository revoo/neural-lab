// authored-by: Claude Opus 5.5
/**
 * Lab: Tune the Spam Scorer. Emails are count vectors (links, exclamation
 * marks, mentions of your name); a weight vector w scores each one by w · x
 * (or by cosine similarity in brief 4). A ranking is perfect when every spam
 * email scores above every real one. In 2D a dot-product ranking depends only
 * on the direction of w, so the solver sweeps the angle of w over 0–360° in
 * 0.01° steps (exhaustive at that resolution).
 */

export type Mail = { subject: string; spam: boolean; x: number[] };

const M = (subject: string, spam: boolean, ...x: number[]): Mail => ({ subject, spam, x });

/** Briefs 1, 2 and 4 use the first two counts (links, exclamation marks). */
export const MAILS: Mail[] = [
  M('WIN a free cruise!!!', true, 4, 6, 0),
  M('Your parcel is held, click here', true, 7, 2, 0),
  M('Cheap meds!!!', true, 3, 7, 0),
  M('URGENT: account suspended', true, 6, 4, 0),
  M('Investment opportunity!', true, 5, 5, 0),
  M('Lunch on Thursday?', false, 0, 1, 2),
  M('Minutes from the meeting', false, 3, 0, 1),
  M('Happy birthday!!', false, 0, 3, 3),
  M('Weekly newsletter', false, 4, 1, 0),
  M('Can you review this?', false, 1, 0, 2),
];

/** Brief 3 adds a real email that looks spammy on links and exclamation marks but mentions you by name. */
export const BANK: Mail = M('Sam, your statement is ready!!!', false, 5, 5, 3);

/** Brief 4 adds a long real email: a copy of one real email with every count doubled. */
export const LONG_COPY_OF = 8; // 'Weekly newsletter'
export const LONG: Mail = { subject: 'Weekly newsletter (long edition)', spam: false, x: MAILS[LONG_COPY_OF].x.map((v) => 2 * v) };

export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0);
export const norm = (a: number[]) => Math.sqrt(dot(a, a));
export const cosine = (w: number[], x: number[]) => {
  const n = norm(w) * norm(x);
  return n === 0 ? 0 : dot(w, x) / n;
};

export type Scorer = 'dot' | 'cosine';
export const scoreOf = (w: number[], m: Mail, d: number, scorer: Scorer) => {
  const x = m.x.slice(0, d);
  return scorer === 'dot' ? dot(w, x) : cosine(w, x);
};

/** Lowest spam score minus highest real-mail score: positive means a perfect ranking. */
export function gap(w: number[], mails: Mail[], d = 2, scorer: Scorer = 'dot'): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (const m of mails) {
    const s = scoreOf(w, m, d, scorer);
    if (m.spam) lo = Math.min(lo, s);
    else hi = Math.max(hi, s);
  }
  return lo - hi;
}

export const unit = (deg: number) => [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)];

/** Sweep the angle of a unit w in 0.01° steps; return the angles that rank perfectly and the widest gap. */
export function sweep(mails: Mail[], scorer: Scorer = 'dot') {
  const pass: number[] = [];
  let best = { deg: 0, gap: -Infinity };
  for (let k = 0; k < 36000; k++) {
    const deg = k / 100;
    const g = gap(unit(deg), mails, 2, scorer);
    if (g > 0) pass.push(deg);
    if (g > best.gap) best = { deg, gap: g };
  }
  return { pass, best };
}

/** Contiguous runs of passing angles, as [from, to] in degrees. */
export function bands(pass: number[]): [number, number][] {
  const out: [number, number][] = [];
  for (const d of pass) {
    const last = out[out.length - 1];
    if (last && Math.abs(d - last[1] - 0.01) < 1e-6) last[1] = d;
    else out.push([d, d]);
  }
  return out;
}

/** Brief 3: a grid over 3D directions (azimuth and elevation in 1° steps). */
export function sweep3(mails: Mail[]) {
  let posOrZero = 0;
  let negative = 0;
  let example: number[] | null = null;
  for (let el = -89; el <= 89; el++) {
    for (let az = 0; az < 360; az++) {
      const c = Math.cos((el * Math.PI) / 180);
      const w = [c * Math.cos((az * Math.PI) / 180), c * Math.sin((az * Math.PI) / 180), Math.sin((el * Math.PI) / 180)];
      if (gap(w, mails, 3) > 0) {
        if (w[2] >= 0) posOrZero++;
        else {
          negative++;
          example ??= w;
        }
      }
    }
  }
  return { posOrZero, negative, example };
}

export type Brief = { id: string; tab: string; title: string; text: string };
export const BRIEFS: Brief[] = [
  { id: 'rank', tab: 'Brief 1', title: 'Brief 1 · A perfect ranking', text: 'Drag the weight arrow w until every spam email scores above every real one.' },
  { id: 'gap', tab: 'Brief 2', title: 'Brief 2 · The widest gap', text: 'w is now locked to length 1, so only its direction matters. Make the gap between the lowest spam score and the highest real score as wide as you can: at least 95% of the widest possible.' },
  { id: 'negative', tab: 'Brief 3', title: 'Brief 3 · Negative evidence', text: 'A real email from your bank has as many links and exclamation marks as spam. A third measurement, mentions of your name, is now on the scorer. Rank every spam above every real email.' },
  { id: 'cosine', tab: 'Brief 4', title: 'Brief 4 · The length cheat', text: 'A long edition of the newsletter has every count doubled, and its score doubles with it. Switch the scorer to cosine similarity and rank perfectly again.' },
];
