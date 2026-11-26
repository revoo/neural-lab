// authored-by: Gemini Antigravity
/**
 * Pure helpers for the Training Lab's Custom (paint) brush mode.
 * Enforces domain clamping, drag-spacing deduplication, a 160-point cap,
 * a two-per-class minimum before training, starter cloud seeding, and a
 * deterministic stratified train/test split.
 */
import { DOMAIN, type Pt } from './lab';

export const DEFAULT_MIN_SPACING = 0.06;
export const DEFAULT_MAX_POINTS = 160;

/** Clamp a 2D coordinate inside [-bound, bound]^2. */
export function clampPoint(x: number, y: number, bound = DOMAIN): { x: number; y: number } {
  return {
    x: Math.max(-bound, Math.min(bound, x)),
    y: Math.max(-bound, Math.min(bound, y)),
  };
}

/**
 * Append a painted point if the canvas is under `maxPoints` and no existing point
 * sits strictly closer than `minSpacing` (preventing dense pileups while dragging).
 */
export function addPaintedPoint(
  pts: readonly Pt[],
  x: number,
  y: number,
  label: 0 | 1,
  minSpacing = DEFAULT_MIN_SPACING,
  maxPoints = DEFAULT_MAX_POINTS,
  bound = DOMAIN,
): Pt[] {
  if (pts.length >= maxPoints) return pts.slice();
  const c = clampPoint(x, y, bound);
  const minSq = minSpacing * minSpacing;
  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i].x - c.x;
    const dy = pts[i].y - c.y;
    if (dx * dx + dy * dy < minSq) return pts.slice();
  }
  return [...pts, { x: c.x, y: c.y, label }];
}

/** Requires at least 2 points of Class A (0) and 2 points of Class B (1) before training. */
export function hasBothClasses(pts: readonly Pt[]): boolean {
  let c0 = 0;
  let c1 = 0;
  for (let i = 0; i < pts.length; i++) {
    if (pts[i].label === 0) c0++;
    else c1++;
    if (c0 >= 2 && c1 >= 2) return true;
  }
  return false;
}

/** Drop a single stubborn outlier onto an existing painted dataset (respecting maxPoints). */
export function seedWithOutlier(
  basePoints: readonly Pt[],
  x: number,
  y: number,
  label: 0 | 1,
  maxPoints = DEFAULT_MAX_POINTS,
  bound = DOMAIN,
): Pt[] {
  if (basePoints.length >= maxPoints) return basePoints.slice();
  const c = clampPoint(x, y, bound);
  return [...basePoints, { x: c.x, y: c.y, label }];
}

/** Deterministic 12-point two-cluster starter cloud (6 Class A, 6 Class B). */
export function makeStarterCloud(): Pt[] {
  return [
    // Class A (blue, label 0): lower-left cluster
    { x: -0.95, y: -0.55, label: 0 },
    { x: -0.70, y: -0.20, label: 0 },
    { x: -0.55, y: -0.65, label: 0 },
    { x: -0.85, y: -0.85, label: 0 },
    { x: -0.40, y: -0.35, label: 0 },
    { x: -1.05, y: -0.25, label: 0 },
    // Class B (orange, label 1): upper-right cluster
    { x: 0.95, y: 0.55, label: 1 },
    { x: 0.70, y: 0.20, label: 1 },
    { x: 0.55, y: 0.65, label: 1 },
    { x: 0.85, y: 0.85, label: 1 },
    { x: 0.40, y: 0.35, label: 1 },
    { x: 1.05, y: 0.25, label: 1 },
  ];
}

/**
 * Deterministic stratified train/validation split (default 80/20) so both train and
 * test curves stay balanced even on small custom painted datasets.
 */
export function splitCustomTrainVal(pts: readonly Pt[], trainFrac = 0.8): { train: Pt[]; test: Pt[] } {
  const byClass: [Pt[], Pt[]] = [[], []];
  for (const p of pts) byClass[p.label].push(p);

  const train: Pt[] = [];
  const test: Pt[] = [];
  const f = Math.max(0.1, Math.min(0.95, trainFrac));

  for (const group of byClass) {
    if (group.length === 0) continue;
    if (group.length === 1) {
      train.push(group[0]);
      continue;
    }
    const nTrain = Math.max(1, Math.min(group.length - 1, Math.round(group.length * f)));
    // Interleave deterministically by picking evenly spaced indices for test
    const nTest = group.length - nTrain;
    const testIdx = new Set<number>();
    for (let k = 0; k < nTest; k++) {
      testIdx.add(Math.floor(((k + 0.5) * group.length) / nTest));
    }
    group.forEach((pt, idx) => {
      if (testIdx.has(idx)) test.push(pt);
      else train.push(pt);
    });
  }
  return { train, test };
}
