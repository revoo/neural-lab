// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for matrix multiplication: three ways
 * to push a batch of 32 images (784 pixels each) through a two-layer network,
 * 784 → 128 → 10.
 */
import type { Rival } from '../../components/RivalTable';
import { createRng } from '../../lib/rng';

export const BATCH = 32;
export const SIZES = [784, 128, 10];

type M = number[][];
export const matmul = (A: M, B: M): M => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
const relu = (A: M): M => A.map((r) => r.map((v) => Math.max(0, v)));
const random = (rows: number, cols: number, seed: number): M => {
  const rng = createRng(seed);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => rng.normal(0, 1)));
};

/** Multiply-adds for one example through every layer. */
export const perExample = () => SIZES.slice(1).reduce((s, out, i) => s + SIZES[i] * out, 0);

/** Rival (a): one example at a time. */
export function oneAtATime() {
  const layers = SIZES.length - 1;
  return { operations: BATCH * layers, multiplyAdds: BATCH * perExample(), weightReads: BATCH * perExample() };
}

/** Rival (c): one batched multiply per layer. */
export function batched() {
  const layers = SIZES.length - 1;
  return { operations: layers, multiplyAdds: BATCH * perExample(), weightReads: perExample() };
}

/** Rival (b): collapse the layers into one matrix ahead of time. Numbers stored, before and after. */
export function collapsedSize() {
  return { stack: perExample(), collapsed: SIZES[0] * SIZES[SIZES.length - 1] };
}

/**
 * Check rival (b) with real numbers on small seeded matrices (a batch of 4, 6 → 5 → 3):
 * without an activation, the collapsed matrix gives the same outputs; with a ReLU between
 * the layers, it doesn't.
 */
export function collapseCheck(seed = 1) {
  const X = random(4, 6, seed);
  const W1 = random(6, 5, seed + 1);
  const W2 = random(5, 3, seed + 2);
  const maxDiff = (A: M, B: M) => Math.max(...A.flatMap((r, i) => r.map((v, j) => Math.abs(v - B[i][j]))));
  const collapsed = matmul(X, matmul(W1, W2));
  return {
    linear: maxDiff(matmul(matmul(X, W1), W2), collapsed),
    withRelu: maxDiff(matmul(relu(matmul(X, W1)), W2), collapsed),
  };
}

const fmt = (n: number) => n.toLocaleString('en-GB');

export function rivals(): Rival[] {
  const a = oneAtATime();
  const c = batched();
  const s = collapsedSize();
  const chk = collapseCheck();
  return [
    {
      name: 'One example at a time',
      result: `Push each of the ${BATCH} images through on its own: ${a.operations} separate matrix–vector jobs. The answers and the ${fmt(a.multiplyAdds)} multiply-adds are exactly the same, but the ${fmt(perExample())} weights are read ${BATCH} times over instead of once, and the hardware never gets one big job to spread out.`,
      verdicts: ['pass', 'pass', 'fail'],
    },
    {
      name: 'Collapse the layers first',
      result: `Multiply the two weight matrices together once, into a single 784 × 10 matrix: ${fmt(s.collapsed)} numbers instead of ${fmt(s.stack)}, with outputs identical to the stack (on small random matrices, every output agrees to ${chk.linear < 1e-12 ? 'within 10⁻¹², just rounding' : 'only ' + chk.linear.toExponential(1)}). So a stack of plain matrices is secretly one layer. Put a ReLU between those small layers and the collapsed matrix is off by as much as ${chk.withRelu.toFixed(1)}. That is why activation functions exist.`,
      verdicts: ['fail', 'pass', 'pass'],
    },
    {
      name: 'Batched matrix multiplication',
      result: `Stack the ${BATCH} images as the rows of one (32 × 784) matrix and do ${c.operations} multiplies, one per layer. Same ${fmt(c.multiplyAdds)} multiply-adds, each weight read once, and each layer is one big grid of independent dot products that a GPU spreads across thousands of cores.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
