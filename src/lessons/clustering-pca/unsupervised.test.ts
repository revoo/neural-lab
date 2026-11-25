// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  BAD_INIT_BLOBS4,
  center,
  elbowSweep,
  evalBrief1,
  evalBrief2,
  evalBrief3,
  fitPCA,
  getBlobs4,
  getInitRace,
  getPca2DData,
  getRingsData,
  jacobiEigen,
  labExhaustive,
  pca2DAtAngle,
  runKMeans,
  runKMeansFrom,
} from './unsupervised';

describe('clustering-pca: Jacobi eigensolver and PCA identities', () => {
  it('diagonalises a symmetric covariance matrix accurately (C v = λ v and V^T V = I)', () => {
    const pts = getPca2DData();
    const { data } = center(pts);
    const n = data.length;
    const cov = [
      [0, 0],
      [0, 0],
    ];
    for (const [x, y] of data) {
      cov[0][0] += x * x;
      cov[0][1] += x * y;
      cov[1][0] += y * x;
      cov[1][1] += y * y;
    }
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) cov[i][j] /= n;

    const { values, vectors } = jacobiEigen(cov);
    expect(values[0]).toBeGreaterThan(values[1]);
    // Orthonormality V^T V = I
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const dot = vectors[i][0] * vectors[j][0] + vectors[i][1] * vectors[j][1];
        expect(dot).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
    // Eigen-equation C v = λ v
    for (let i = 0; i < 2; i++) {
      const v = vectors[i];
      const cv0 = cov[0][0] * v[0] + cov[0][1] * v[1];
      const cv1 = cov[1][0] * v[0] + cov[1][1] * v[1];
      expect(cv0).toBeCloseTo(values[i] * v[0], 10);
      expect(cv1).toBeCloseTo(values[i] * v[1], 10);
    }
  });

  it('satisfies Pythagoras (projected variance + reconstruction MSE = total variance) at every angle', () => {
    const pca = fitPCA(getPca2DData());
    const refTotal = pca.totalVar;
    for (const deg of [0, 30.8, 45, 90, 120.8, 165]) {
      const r = pca2DAtAngle(deg);
      expect(r.projVar + r.reconMse).toBeCloseTo(refTotal, 10);
    }
    const atOpt = pca2DAtAngle(pca2DAtAngle(0).optDeg);
    expect(atOpt.optDeg).toBeCloseTo(30.8, 1);
    expect(atOpt.projVar).toBeCloseTo(3.532, 3);
    expect(atOpt.reconMse).toBeCloseTo(0.271, 3);
    expect(atOpt.totalVar).toBeCloseTo(3.803, 3);
    expect(atOpt.share).toBeCloseTo(0.9288, 3);

    const perp = pca2DAtAngle(atOpt.optDeg + 90);
    expect(perp.projVar).toBeCloseTo(0.271, 3);
    expect(perp.reconMse).toBeCloseTo(3.532, 3);
    expect(perp.share).toBeCloseTo(0.0712, 3);
  });
});

describe('clustering-pca: k-Means init trap, elbow and concentric rings', () => {
  it('pins the bad-init trap and 40-seed Random vs k-Means++ race on 4 blobs', () => {
    const { pts } = getBlobs4();
    const badTrace = runKMeansFrom(pts, BAD_INIT_BLOBS4);
    expect(badTrace[badTrace.length - 1].inertia).toBeCloseTo(415.6, 1);
    expect(badTrace.length - 1).toBe(3);

    const ppTrace = runKMeans(pts, 4, 'kmeans++', 17).trace;
    expect(ppTrace[ppTrace.length - 1].inertia).toBeCloseTo(73.2, 1);

    const race = getInitRace();
    expect(race.seeds).toBe(40);
    expect(race.globalInertia).toBeCloseTo(73.2, 1);
    expect(race.randomOptimal).toBe(29);
    expect(race.ppOptimal).toBe(38);
    expect(race.randomMeanInertia).toBeCloseTo(158.0, 1);
    expect(race.ppMeanInertia).toBeCloseTo(87.9, 1);
  });

  it('pins the elbow and silhouette curves for 4 blobs vs concentric rings', () => {
    const blobs = elbowSweep(getBlobs4().pts);
    expect(blobs[0].k).toBe(2);
    expect(blobs[0].inertia).toBeCloseTo(636.1, 0);
    expect(blobs[1].inertia).toBeCloseTo(352.3, 1);
    expect(blobs[2].k).toBe(4);
    expect(blobs[2].inertia).toBeCloseTo(73.2, 1);
    expect(blobs[2].silhouette).toBeCloseTo(0.763, 3);
    // Silhouette peaks sharply at k = 4
    const bestBlob = blobs.reduce((a, b) => (b.silhouette > a.silhouette ? b : a));
    expect(bestBlob.k).toBe(4);

    const rings = elbowSweep(getRingsData().pts);
    expect(rings[0].k).toBe(2);
    expect(rings[0].silhouette).toBeCloseTo(0.326, 3);
    // k-Means cannot separate concentric rings: Silhouette stays in 0.31-0.41 across k = 2..7
    for (const row of rings) {
      expect(row.silhouette).toBeLessThan(0.42);
      expect(row.silhouette).toBeGreaterThan(0.30);
    }
  });
});

describe('clustering-pca: Lab (Compress & Segment) exhaustive checks', () => {
  it('Brief 1 (PCA scaling trap): raw m=1 fools apparent variance (99.97%) while losing true variance (17.3%)', () => {
    const raw1 = evalBrief1(false, 1);
    expect(raw1.apparentVar).toBeGreaterThan(0.999);
    expect(raw1.trueVar).toBeCloseTo(0.173, 3);
    expect(raw1.pass).toBe(false);

    const raw2 = evalBrief1(false, 2);
    expect(raw2.trueVar).toBeCloseTo(0.664, 3);
    expect(raw2.pass).toBe(false);

    const std2 = evalBrief1(true, 2);
    expect(std2.trueVar).toBeCloseTo(0.983, 3);
    expect(std2.pass).toBe(true);

    const ex = labExhaustive('pca-scale');
    expect(ex.total).toBe(12);
    expect(ex.passing).toBe(9);
    expect(ex.summary).toContain('m = 2');
  });

  it('Brief 2 (unscaled customers): ONLY standardised k=4 k-Means++ passes both Silhouette and 90% stability', () => {
    const rawPP = evalBrief2(false, 4, 'kmeans++');
    expect(rawPP.meanSil).toBeCloseTo(0.123, 3);
    expect(rawPP.pass).toBe(false);

    const stdRand = evalBrief2(true, 4, 'random');
    expect(stdRand.meanSil).toBeCloseTo(0.721, 3);
    expect(stdRand.stableSeeds).toBe(29);
    expect(stdRand.pass).toBe(false);

    const stdPP = evalBrief2(true, 4, 'kmeans++');
    expect(stdPP.meanSil).toBeCloseTo(0.777, 3);
    expect(stdPP.stableSeeds).toBe(36);
    expect(stdPP.pass).toBe(true);

    const ex = labExhaustive('segment');
    expect(ex.total).toBe(20);
    expect(ex.passing).toBe(1);
  });

  it('Brief 3 (PCA -> k-Means denoising): m=2, k=4 beats raw 8D cleanly', () => {
    const noPca = evalBrief3(8, 4);
    expect(noPca.pass).toBe(false);
    expect(noPca.sil).toBeLessThan(0.5);

    const opt = evalBrief3(2, 4);
    expect(opt.pass).toBe(true);
    expect(opt.sil).toBeCloseTo(0.805, 3);
    expect(opt.purity).toBeCloseTo(1.0, 3);

    const ex = labExhaustive('denoise');
    expect(ex.total).toBe(24);
    expect(ex.passing).toBe(3);
    expect(ex.summary).toContain('m = 2');
  });
});
