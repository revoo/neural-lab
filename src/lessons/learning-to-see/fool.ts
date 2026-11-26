// authored-by: Claude Opus 5.5
/**
 * Lab: Fool the classifier. A change sets one pixel to blank (0) or to full ink (1). The attack succeeds
 * when the trained network's top class is no longer the photo's true class.
 * Reference: `fool.run.ts` checks every single change, then (if none works) every pair of changes,
 * then extends greedily; results in `foolResults.ts`, re-verified by `fool.test.ts`.
 */
import { Cnn } from './cnn';
import type { ImageSet } from './fashion';

export type Change = { px: number; to: 0 | 1 };
export const FOOL_CLASSES = [1, 7, 6] as const; // Trouser, Sneaker, Shirt: fixed before any search

export function topClass(net: Cnn, img: ArrayLike<number>, tr = net.newTrace()): number {
  const p = net.forward(img, tr);
  let best = 0;
  for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
  return best;
}

/** The true class's lead: its logit minus the best other logit (negative = fooled). */
export function lead(net: Cnn, img: ArrayLike<number>, y: number, tr = net.newTrace()): number {
  net.forward(img, tr);
  let other = -Infinity;
  for (let k = 0; k < tr.logits.length; k++) if (k !== y) other = Math.max(other, tr.logits[k]);
  return tr.logits[y] - other;
}

/** The brief photos: the first correctly classified test photo of each FOOL_CLASSES class. */
export function briefImages(net: Cnn, d: ImageSet): number[] {
  return FOOL_CLASSES.map((k) => {
    for (let i = 0; i < d.test.length; i++) if (d.testY[i] === k && topClass(net, d.test[i]) === k) return i;
    throw new Error(`no correctly classified ${k}`);
  });
}

export function apply(img: ArrayLike<number>, changes: readonly Change[]): Float32Array {
  const out = Float32Array.from(img);
  for (const c of changes) out[c.px] = c.to;
  return out;
}

export type FoolSearch = { singles: number; pairsChecked: number; best: Change[]; method: 'single' | 'pair' | 'greedy' };

/** The search the lab's optimum comes from (see the file comment). */
export function searchFool(net: Cnn, img: Float32Array, y: number, greedyMax = 60): FoolSearch {
  const tr = net.newTrace();
  const cands: { c: Change; lead: number }[] = [];
  for (let px = 0; px < img.length; px++)
    for (const to of [0, 1] as const) {
      if (img[px] === to) continue;
      const c = { px, to };
      const l = lead(net, apply(img, [c]), y, tr);
      if (l < 0) return { singles: 1, pairsChecked: 0, best: [c], method: 'single' };
      cands.push({ c, lead: l });
    }
  let pairs = 0;
  for (let a = 0; a < cands.length; a++) {
    const base = apply(img, [cands[a].c]);
    for (let b = a + 1; b < cands.length; b++) {
      if (cands[b].c.px === cands[a].c.px) continue;
      pairs++;
      const v = base[cands[b].c.px];
      base[cands[b].c.px] = cands[b].c.to;
      const l = lead(net, base, y, tr);
      base[cands[b].c.px] = v;
      if (l < 0) return { singles: cands.length, pairsChecked: pairs, best: [cands[a].c, cands[b].c], method: 'pair' };
    }
  }
  // greedy: repeatedly add the single change that lowers the lead most
  const chosen: Change[] = [];
  let cur: Float32Array = Float32Array.from(img);
  for (let k = 0; k < greedyMax; k++) {
    let best: { c: Change; l: number } | null = null;
    for (let px = 0; px < cur.length; px++) {
      if (chosen.some((c) => c.px === px)) continue;
      for (const to of [0, 1] as const) {
        if (cur[px] === to) continue;
        const v = cur[px];
        cur[px] = to;
        const l = lead(net, cur, y, tr);
        cur[px] = v;
        if (!best || l < best.l) best = { c: { px, to }, l };
      }
    }
    chosen.push(best!.c);
    cur = apply(cur, [best!.c]);
    if (best!.l < 0) return { singles: cands.length, pairsChecked: pairs, best: chosen, method: 'greedy' };
  }
  return { singles: cands.length, pairsChecked: pairs, best: chosen, method: 'greedy' };
}
