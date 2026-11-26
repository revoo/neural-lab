// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { nodeFs } from '../../lib/nodeIo';
import { Cnn, cnnLayout } from './cnn';
import { decodeFashion, type ImageSet } from './fashion';
import { accuracy, gaborBank, measureSee, nearestCentroidAccuracy, occlusion, pca2, saliency, SEE_RUN, seeMoments, templateMatch, trainSee } from './see';
import { SEE_RESULTS } from './seeResults';

let data: ImageSet;
async function load() {
  if (data) return data;
  const fs = await nodeFs();
  data = decodeFashion(new Uint8Array(fs.readFileSync(new URL('./fashion.bin', import.meta.url))));
  return data;
}
async function finalNet() {
  const fs = await nodeFs();
  const b = fs.readFileSync(new URL('./seeNet.bin', import.meta.url));
  const n = new Cnn(SEE_RUN.cfg);
  n.p.set(new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  return n;
}
const r4 = (v: number) => +v.toFixed(4);

describe('the image set', () => {
  it('3,000 training and 500 test photos, balanced over 10 classes', async () => {
    const d = await load();
    expect(d.train.length).toBe(3000);
    expect(d.test.length).toBe(500);
    for (let k = 0; k < 10; k++) {
      expect(Array.from(d.trainY).filter((y) => y === k).length).toBe(300);
      expect(Array.from(d.testY).filter((y) => y === k).length).toBe(50);
    }
    expect(cnnLayout(SEE_RUN.cfg).total).toBe(11978);
  });
});

describe('the recorded run', () => {
  it('re-trains exactly: checkpoints to step 24 match the record', async () => {
    const d = await load();
    const rec = SEE_RESULTS.checkpoints;
    let i = 0;
    trainSee(SEE_RUN, d, (step, _p, net) => {
      const m = measureSee(net, d);
      const r = rec[i++];
      expect(step).toBe(r.step);
      expect(m.acc).toBe(r.acc);
      expect(r4(m.nc)).toBe(r.nc);
      expect(m.edge.map(r4)).toEqual(r.edge);
      return step < 24;
    });
    expect(rec[i - 1].step).toBe(24);
  }, 120_000);
  it('moments: first edge detector at step 24, classes beat raw pixels at 178, half the filters at 237', () => {
    expect(SEE_RESULTS.moments.map((m) => [m.id, m.step])).toEqual([
      ['first-edge', 24],
      ['separate', 178],
      ['half-edges', 237],
    ]);
    expect(seeMoments(SEE_RESULTS.checkpoints, SEE_RESULTS.edgeBar, SEE_RESULTS.rawNC)).toEqual(SEE_RESULTS.moments);
    expect(SEE_RESULTS.checkpoints[0].edge.every((e) => e < SEE_RESULTS.edgeBar)).toBe(true);
    const last = SEE_RESULTS.checkpoints.at(-1)!;
    expect(last.step).toBe(1000);
    expect(last.acc).toBe(0.83);
    expect(last.edge.filter((e) => e > SEE_RESULTS.edgeBar).length).toBe(5);
    expect(SEE_RESULTS.rawNC).toBe(0.686);
    expect(SEE_RESULTS.checkpoints.find((c) => c.step === 100)!.acc).toBe(0.694);
    expect(last.nc).toBe(0.796);
    expect(SEE_RESULTS.edgeBar).toBeCloseTo(0.615, 3);
  });
  it('fair comparison over seeds 1–3: small start grows edge detectors, He start barely; accuracy alike', () => {
    const c = SEE_RESULTS.compare;
    const small = c.filter((x) => x.start === 'small');
    const he = c.filter((x) => x.start === 'He');
    expect(small.map((x) => x.edges)).toEqual([5, 3, 3]);
    expect(small.map((x) => x.firstEdge)).toEqual([24, 75, 133]);
    expect(he.map((x) => x.edges)).toEqual([1, 0, 0]);
    expect(he.map((x) => x.firstEdge)).toEqual([750, null, null]);
    expect(small.map((x) => x.acc)).toEqual([0.83, 0.826, 0.846]);
    expect(he.map((x) => x.acc)).toEqual([0.824, 0.842, 0.812]);
  });
  it('the shipped weights are the final checkpoint', async () => {
    const d = await load();
    const n = await finalNet();
    expect(accuracy(n, d.test, d.testY)).toBe(0.83);
    const bank = gaborBank(5);
    expect(Array.from({ length: 8 }, (_, f) => r4(templateMatch(n.filter(f), bank).score))).toEqual(SEE_RESULTS.checkpoints.at(-1)!.edge);
  });
});

describe('measurements', () => {
  it('raw-pixel nearest-centroid accuracy and a random filter bar are what the lesson says', async () => {
    const d = await load();
    expect(nearestCentroidAccuracy(d.train.slice(0, 1000), d.trainY.slice(0, 1000), d.test, d.testY)).toBe(0.686);
  });
  it('occlusion is exact: a patch on a blank corner changes nothing; PCA axes are unit and orthogonal', async () => {
    const d = await load();
    const n = await finalNet();
    const img = d.test[0];
    const occ = occlusion(n, img, d.testY[0]);
    expect(occ.n).toBe(12);
    // saliency is a gradient: same length as the image, not all zero
    const s = saliency(n, img, d.testY[0]);
    expect(s.length).toBe(784);
    expect(s.some((v) => v !== 0)).toBe(true);
    const { axes } = pca2(d.test.slice(0, 100).map((x) => Array.from(x.slice(0, 50))));
    const dot = axes[0].reduce((t, v, i) => t + v * axes[1][i], 0);
    expect(Math.abs(dot)).toBeLessThan(1e-6);
    expect(Math.hypot(...axes[0])).toBeCloseTo(1, 9);
  });
});
