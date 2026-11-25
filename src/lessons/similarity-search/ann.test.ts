// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  annLabExhaustive,
  benchmarkIvf16D,
  benchmarkLsh16D,
  compareMetrics,
  cosine,
  evalAnnBrief1,
  evalAnnBrief2,
  evalAnnBrief3,
  exactTopK,
  getIvf2DScene,
  lshCollisionProb,
  normalize,
  queryIvf,
  sqDist,
  vectorDbGiB,
} from './ann';

describe('similarity-search: metrics, unit-norm equivalence, and 2D IVF boundary', () => {
  it('proves ||a - b||^2 === 2(1 - cos) on unit vectors and shows unnormalised Dot disagrees with Cosine', () => {
    const a = normalize([3, -4, 1.5, 2]);
    const b = normalize([-1, 2, 5, -0.5]);
    expect(sqDist(a, b)).toBeCloseTo(2 * (1 - cosine(a, b)), 10);

    const cands = [
      { id: 'A', label: 'Close angle, short', vec: [0.95, 0.25] as [number, number] },
      { id: 'B', label: 'Wider angle, very long', vec: [2.4, 1.5] as [number, number] },
      { id: 'C', label: 'Same length, medium angle', vec: [0.85, 0.55] as [number, number] },
    ];
    const raw = compareMetrics([1, 0], cands, false);
    expect(raw.bestCos).toBe('A');
    expect(raw.bestDot).toBe('B');
    expect(raw.allAgree).toBe(false);

    const normed = compareMetrics([1, 0], cands, true);
    expect(normed.bestCos).toBe('A');
    expect(normed.bestDot).toBe('A');
    expect(normed.bestL2).toBe('A');
    expect(normed.allAgree).toBe(true);
  });

  it('pins 2D Voronoi boundary miss at nprobe=1 (1/5) rescued to 5/5 at nprobe=2', () => {
    const { data, ivf, boundaryQuery } = getIvf2DScene();
    const truth = new Set(exactTopK(data, boundaryQuery, 5));
    const p1 = queryIvf(data, ivf, boundaryQuery, 1, 5);
    expect(p1.topK.filter((i) => truth.has(i)).length).toBe(1);
    const p2 = queryIvf(data, ivf, boundaryQuery, 2, 5);
    expect(p2.topK.filter((i) => truth.has(i)).length).toBe(5);
    expect(p2.distEvals).toBe(52);
  });

  it('pins exact SimHash LSH collision formula and 10M x 1536D RAM sizes', () => {
    expect(lshCollisionProb(20, 8, 1)).toBeCloseTo((1 - 20 / 180) ** 8, 6); // 0.390
    expect(lshCollisionProb(20, 8, 8)).toBeCloseTo(0.981, 3);
    expect(vectorDbGiB(10_000_000, 1536 * 4)).toBeCloseTo(57.22, 2);
    expect(vectorDbGiB(10_000_000, 1536 * 1)).toBeCloseTo(14.31, 2);
    expect(vectorDbGiB(10_000_000, 64)).toBeCloseTo(0.60, 2);
  });
});

describe('similarity-search: Lab (Tune the Vector Index) exhaustive checks', () => {
  it('Brief 1 (IVF 16D): unique passing config nlist=32, nprobe=6 (91.3% Recall@5, 24.5% scan)', () => {
    const r = evalAnnBrief1(32, 6);
    expect(r.pass).toBe(true);
    expect(r.recall5).toBeCloseTo(0.9133, 3);
    expect(r.scanFrac).toBeCloseTo(0.245, 2);
    expect(benchmarkIvf16D(16, 1).recall5).toBeCloseTo(0.583, 2);

    const ex = annLabExhaustive('ivf');
    expect(ex.total).toBe(15);
    expect(ex.passing).toBe(1);
  });

  it('Brief 2 (LSH 16D): unique passing config b=6, L=16 (90.7% Recall@5, 29.3% scan)', () => {
    const r = evalAnnBrief2(6, 16);
    expect(r.pass).toBe(true);
    expect(r.recall5).toBeCloseTo(0.9067, 3);
    expect(r.scanFrac).toBeCloseTo(0.293, 2);
    expect(benchmarkLsh16D(6, 1).recall5).toBeCloseTo(0.123, 2);

    const ex = annLabExhaustive('lsh');
    expect(ex.total).toBe(12);
    expect(ex.passing).toBe(1);
  });

  it('Brief 3 (10M x 1536D <= 16 GiB, >= 95% recall): unique Scalar INT8 (14.31 GiB)', () => {
    expect(evalAnnBrief3('fp32').pass).toBe(false);
    expect(evalAnnBrief3('pq64').pass).toBe(false);
    expect(evalAnnBrief3('int8').pass).toBe(true);
    expect(annLabExhaustive('ram')).toMatchObject({ total: 5, passing: 1 });
  });
});
