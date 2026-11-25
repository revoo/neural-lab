// authored-by: Claude Opus 5.5
import { klDivergence } from '../../lib/prob';

export const KL_NAMES = ['sun', 'cloud', 'rain'];
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

/** The single biggest term p_i log2(p_i / q_i) of D(p‖q), and which outcome it belongs to. */
export function worstTerm(p: number[], q: number[]) {
  let best = { i: 0, v: -Infinity };
  p.forEach((pi, i) => {
    if (pi <= 0) return;
    const v = q[i] <= 0 ? Infinity : pi * Math.log2(pi / q[i]);
    if (v > best.v) best = { i, v };
  });
  return best;
}

const fmtBits = (x: number) => (Number.isFinite(x) ? x.toFixed(3) : '∞');

/** Live note for the KL explorer: which direction is bigger, and the term that makes it so. */
export function klNote(p: number[], q: number[], names = KL_NAMES): string {
  const a = klDivergence(p, q);
  const b = klDivergence(q, p);
  if (a < 5e-4 && b < 5e-4) return 'q matches p, so there is no extra surprise in either direction: the cross-entropy is just the entropy of the truth, the part no model can avoid.';
  if (!Number.isFinite(a)) {
    const w = worstTerm(p, q);
    return `The belief q gives ${names[w.i]} 0%, but it really happens ${pct(p[w.i])} of the time. When it does, the model is infinitely surprised, so D(p‖q) is infinite. Never rule out something that can happen.`;
  }
  if (a >= b) {
    const w = worstTerm(p, q);
    return `D(p‖q) is the bigger one (${fmtBits(a)} vs ${fmtBits(b)} bits). Its costliest term: the belief q gives ${names[w.i]} ${pct(q[w.i])} when it really happens ${pct(p[w.i])} of the time, which alone costs ${w.v.toFixed(3)} bits. Being confidently wrong costs more than being vaguely wrong.`;
  }
  const w = worstTerm(q, p);
  if (!Number.isFinite(b)) return `D(p‖q) is ${fmtBits(a)} bits, but D(q‖p) is infinite: read that way round, p is the belief, and it gives ${names[w.i]} 0% although q expects it ${pct(q[w.i])} of the time.`;
  return `D(q‖p) is the bigger one (${fmtBits(b)} vs ${fmtBits(a)} bits). Read that way round, p is the belief and q the truth: p gives ${names[w.i]} only ${pct(p[w.i])} when q says it happens ${pct(q[w.i])} of the time, which alone costs ${w.v.toFixed(3)} bits. Being confidently wrong costs more than being vaguely wrong.`;
}
