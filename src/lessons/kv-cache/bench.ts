// authored-by: Claude Opus 5.5
/**
 * V2-WP3 (required by the owner): the CPU vs WebGPU throughput benchmark. The timings come from the
 * learner's own machine, so nothing here is pinned except the arithmetic that explains them.
 */
export type Shape = 'mm' | 'mv';

export const SHAPE_LABEL: Record<Shape, string> = {
  mm: 'matrix × matrix (like prefill)',
  mv: 'matrix × vector (like decoding one token)',
};

/** Sizes n tried for each shape: n×n times n×n, or n×n times a length-n vector. */
export const SIZES: Record<Shape, number[]> = { mm: [32, 64, 128, 256, 512, 1024], mv: [256, 512, 1024, 2048, 4096] };

/** The CPU tier skips matrix products bigger than this: 2·1024³ FLOPs would freeze the page for seconds. */
export const CPU_MAX_MM = 512;

/** Multiply-adds count as 2 FLOPs. */
export const flops = (shape: Shape, n: number) => (shape === 'mm' ? 2 * n ** 3 : 2 * n ** 2);

/** The least memory traffic in bytes (32-bit floats): read both inputs once, write the output once. */
export const bytes = (shape: Shape, n: number) => 4 * (shape === 'mm' ? 3 * n * n : n * n + 2 * n);

/** Arithmetic intensity: FLOPs per byte moved. */
export const intensity = (shape: Shape, n: number) => flops(shape, n) / bytes(shape, n);

export const gflops = (shape: Shape, n: number, reps: number, ms: number) => (flops(shape, n) * reps) / (ms * 1e6);

export type BenchRow = { n: number; cpu?: number; gpu?: number; naive?: number; err?: number };

/** The smallest size at which the batched GPU beat the CPU, or null. */
export function crossover(rows: BenchRow[]): number | null {
  for (const r of rows) if (r.cpu !== undefined && r.gpu !== undefined && r.gpu > r.cpu) return r.n;
  return null;
}

/** Choose a repetition count that should take about `targetMs`, given one timed trial. */
export const repsFor = (trialMs: number, trialReps: number, targetMs: number, maxReps: number) =>
  Math.max(1, Math.min(maxReps, Math.round((trialReps * targetMs) / Math.max(trialMs, 0.05))));
