// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for cross-entropy: three ways to grade
 * the classifier's guess on a photo that really is a cat, as a function of the
 * probability p it gave to "cat". Slopes are taken with respect to cat's score
 * (logit) behind a softmax, where ∂p/∂z = p(1 − p).
 */
import type { Rival } from '../../components/RivalTable';

export const LOSSES = {
  /** Right or wrong: 0 if cat got the most probability. Here "most" means more than half, the two-way case. */
  accuracy: (p: number) => (p > 0.5 ? 0 : 1),
  squared: (p: number) => (1 - p) ** 2,
  crossEntropy: (p: number) => -Math.log(p),
};
export type LossId = keyof typeof LOSSES;

/** Slope of the loss with respect to cat's score: dLoss/dp × p(1 − p). */
export const SCORE_SLOPE: Record<LossId, (p: number) => number> = {
  accuracy: () => 0,
  squared: (p) => -2 * (1 - p) * p * (1 - p),
  crossEntropy: (p) => -(1 - p),
};

/** Measure the slope with respect to the score by nudging the score (two classes, p = sigmoid(z)). */
export function measuredSlope(id: LossId, p: number, h = 1e-6): number {
  const z = Math.log(p / (1 - p));
  const sig = (v: number) => 1 / (1 + Math.exp(-v));
  return (LOSSES[id](sig(z + h)) - LOSSES[id](sig(z - h))) / (2 * h);
}

const n2 = (v: number) => v.toFixed(2);
const s4 = (v: number) => (v < 0 ? '−' : '') + Math.abs(v).toPrecision(1);

export function rivals(): Rival[] {
  const L = LOSSES;
  const S = SCORE_SLOPE;
  return [
    {
      name: 'Accuracy: right or wrong',
      result: `Saying 60% cat and saying 95% cat both score ${L.accuracy(0.6)}; saying 40% scores ${L.accuracy(0.4)}. It is what we report at the end, but it is flat almost everywhere: the slope with respect to cat’s score is ${S.accuracy(0.6)}, so training gets no hint which way to move.`,
      verdicts: ['pass', 'fail', 'pass'],
    },
    {
      name: 'Squared error on the probability',
      result: `(1 − p)² gives ${n2(L.squared(0.01))} for 1% on cat and ${n2(L.squared(0.0001))} for 0.01%: a hundred times more wrong, the same grade. Worse, the confidently wrong model gets almost no push: at p = 0.0001 the slope on cat’s score is only ${s4(S.squared(0.0001))}.`,
      verdicts: ['partly', 'fail', 'pass'],
    },
    {
      name: 'Cross-entropy',
      result: `−ln p gives ${n2(L.crossEntropy(0.01))} for 1% and ${n2(L.crossEntropy(0.0001))} for 0.01%, and keeps climbing towards infinity. Behind a softmax its slope is just p − y: ${n2(S.crossEntropy(0.0001)).replace('-', '−')} for that confidently wrong model, the biggest push exactly where it is needed.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
