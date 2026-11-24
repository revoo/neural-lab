/**
 * Shape rules for NumPy/PyTorch-style matmul and element-wise broadcasting,
 * returning either the result shape or a human-readable explanation of the failure.
 */
import { broadcastShape } from '../../lib/linalg';

export type ShapeResult = { shape: number[]; note?: string } | { error: string };

export function matmulShape(a: number[], b: number[]): ShapeResult {
  if (a.length === 0 || b.length === 0) return { error: 'Scalars can’t be matrix-multiplied; use * instead.' };
  // 1-D operands are treated as a row (left) or column (right) vector, then that dim is dropped.
  const A = a.length === 1 ? [1, a[0]] : a;
  const B = b.length === 1 ? [b[0], 1] : b;
  const n = A[A.length - 2];
  const k1 = A[A.length - 1];
  const k2 = B[B.length - 2];
  const m = B[B.length - 1];
  if (k1 !== k2)
    return {
      error: `Inner dimensions don’t match: the left operand’s last dimension is ${k1}, but the right operand’s second-to-last is ${k2}. Every row of the left side must have as many entries as every column of the right side, or the dot products are undefined.`,
    };
  const batch = broadcastShape(A.slice(0, -2), B.slice(0, -2));
  if ('error' in batch) return { error: `The leading “batch” dimensions can’t be broadcast together — ${batch.error}.` };
  const out = [...batch.shape, n, m];
  if (a.length === 1) out.splice(out.length - 2, 1);
  if (b.length === 1) out.splice(out.length - 1, 1);
  const batchCount = batch.shape.reduce((s, d) => s * d, 1);
  return {
    shape: out,
    note: `${batchCount > 1 ? `${batchCount.toLocaleString()} separate ` : ''}(${n}×${k1}) · (${k2}×${m}) product${batchCount > 1 ? 's' : ''}, each needing ${(n * m * k1).toLocaleString()} multiply-adds — ${(batchCount * n * m * k1).toLocaleString()} in total.`,
  };
}

export function elementwiseShape(a: number[], b: number[]): ShapeResult {
  const r = broadcastShape(a, b);
  if ('error' in r) return { error: `Shapes can’t be broadcast: at ${r.error}. Dimensions are compared from the right; each pair must be equal, or one of them must be 1.` };
  const stretched: string[] = [];
  const n = r.shape.length;
  for (let i = 0; i < n; i++) {
    const da = a[a.length - n + i];
    const db = b[b.length - n + i];
    if (da === undefined || (da === 1 && r.shape[i] !== 1)) stretched.push(`left operand repeated along dim ${i}`);
    if (db === undefined || (db === 1 && r.shape[i] !== 1)) stretched.push(`right operand repeated along dim ${i}`);
  }
  return { shape: r.shape, note: stretched.length ? `Broadcasting: ${stretched.join('; ')}.` : 'Shapes match exactly; no broadcasting needed.' };
}

export function parseShape(s: string): number[] | null {
  const t = s.replace(/[()[\]\s]/g, '');
  if (t === '') return [];
  const parts = t.split(/[,×x]/).filter(Boolean).map(Number);
  return parts.every((p) => Number.isInteger(p) && p > 0) ? parts : null;
}
