// authored-by: Gemini Antigravity
/**
 * Multi-class softmax + cross-entropy backprop engine for Module 5
 * (`network-from-scratch`). Computes exact probabilities, cross-entropy loss,
 * analytic `p - y` gradients, and live central finite-difference verification.
 */

export interface SoftmaxPreset {
  id: 'predict-case' | 'confident-wrong' | 'confident-right' | 'flat-tie' | 'python-example';
  label: string;
  logits: [number, number, number];
  targetClass: 0 | 1 | 2;
}

export const SOFTMAX_PRESETS: readonly SoftmaxPreset[] = [
  {
    id: 'predict-case',
    label: 'Prediction case',
    // Gives p = [0.200, 0.700, 0.100] to 3 decimal places when target = 0
    logits: [0.0, 1.253, -0.693],
    targetClass: 0,
  },
  {
    id: 'confident-wrong',
    label: 'Confident & wrong',
    logits: [-0.8, 2.5, 0.2],
    targetClass: 0,
  },
  {
    id: 'confident-right',
    label: 'Confident & right',
    logits: [2.8, -0.4, -0.6],
    targetClass: 0,
  },
  {
    id: 'flat-tie',
    label: 'Flat tie',
    logits: [0.0, 0.0, 0.0],
    targetClass: 0,
  },
  {
    id: 'python-example',
    label: 'Code example',
    logits: [2.0, 1.0, -0.5],
    targetClass: 1,
  },
];

export const CLASS_NAMES = ['Class A', 'Class B', 'Class C'] as const;

export function softmax(logits: readonly number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function crossEntropyLoss(logits: readonly number[], targetClass: number): number {
  const m = Math.max(...logits);
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) sumExp += Math.exp(logits[i] - m);
  // -log(exp(z_c - m) / sumExp) = log(sumExp) - (z_c - m)
  return Math.log(sumExp) - (logits[targetClass] - m);
}

export interface SoftmaxBackpropResult {
  logits: number[];
  targetClass: number;
  probs: number[];
  oneHot: number[];
  loss: number;
  analyticGrad: number[];
  numericGrad: number[];
  maxDiff: number;
  gradSum: number;
}

export function softmaxBackprop(
  logits: readonly number[],
  targetClass: number,
  eps = 1e-6,
): SoftmaxBackpropResult {
  const probs = softmax(logits);
  const oneHot = logits.map((_, i) => (i === targetClass ? 1 : 0));
  const loss = crossEntropyLoss(logits, targetClass);
  const analyticGrad = probs.map((p, i) => p - oneHot[i]);

  const numericGrad = logits.map((_, i) => {
    const plus = [...logits];
    const minus = [...logits];
    plus[i] += eps;
    minus[i] -= eps;
    return (crossEntropyLoss(plus, targetClass) - crossEntropyLoss(minus, targetClass)) / (2 * eps);
  });

  let maxDiff = 0;
  for (let i = 0; i < logits.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(analyticGrad[i] - numericGrad[i]));
  }
  const gradSum = analyticGrad.reduce((a, b) => a + b, 0);

  return {
    logits: [...logits],
    targetClass,
    probs,
    oneHot,
    loss,
    analyticGrad,
    numericGrad,
    maxDiff,
    gradSum,
  };
}

export function stepSoftmaxLogits(
  logits: readonly number[],
  targetClass: number,
  lr: number,
): number[] {
  const res = softmaxBackprop(logits, targetClass);
  return logits.map((z, i) => z - lr * res.analyticGrad[i]);
}
