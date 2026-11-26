// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { Cnn, makeSet, patchify, shift, Vit, type Model } from './vision';
import { clipLoss, initClip, makePairs } from './clip';
import { attnCost, ATTN_CAP, CLIP_BAR, clipMean, PATCH_BAR, RESCUE_BAR, runOne, type Arch } from './visionLab';
import { CLIP_RESULTS, VISION_RESULTS } from './visionResults';

const fdCheck = (m: Model, x: Float64Array, y: number) => {
  const g = new Float64Array(m.params.length);
  m.lossGrad(x, y, g);
  let worst = 0;
  const h = 1e-6;
  for (let j = 0; j < m.params.length; j++) {
    const o = m.params[j];
    m.params[j] = o + h;
    const a = m.lossGrad(x, y);
    m.params[j] = o - h;
    const b = m.lossGrad(x, y);
    m.params[j] = o;
    worst = Math.max(worst, Math.abs((a - b) / (2 * h) - g[j]));
  }
  return worst;
};
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

describe('vision models: gradients', () => {
  it('CNN and ViT backprop match central finite differences', () => {
    const x = makeSet(2, 5)[1].x;
    const cnn = new Cnn(3, 1);
    // move the filter biases off zero so no pre-activation sits exactly on the ReLU kink
    for (let k = 0; k < 3; k++) cnn.params[k * 10 + 9] = 0.013 * (k + 1);
    expect(fdCheck(cnn, x, 1)).toBeLessThan(1e-7);
    for (const v of [new Vit(2, 4, true, 1), new Vit(4, 3, false, 2)]) expect(fdCheck(v, x, 0)).toBeLessThan(1e-7);
  });
  it('mini-CLIP’s contrastive loss gradient matches finite differences', () => {
    const m = initClip(4, 1);
    const b = makePairs(6, 3);
    const gi = new Float64Array(m.Wi.length);
    const ge = new Float64Array(m.We.length);
    clipLoss(m, b, 0.3, gi, ge);
    let worst = 0;
    const h = 1e-6;
    for (const [P, G] of [
      [m.Wi, gi],
      [m.We, ge],
    ] as const)
      for (let j = 0; j < P.length; j++) {
        const o = P[j];
        P[j] = o + h;
        const a = clipLoss(m, b, 0.3).loss;
        P[j] = o - h;
        const c = clipLoss(m, b, 0.3).loss;
        P[j] = o;
        worst = Math.max(worst, Math.abs((a - c) / (2 * h) - G[j]));
      }
    expect(worst).toBeLessThan(1e-7);
  });
});

describe('patches and positions', () => {
  it('patchify, shuffling and the position-blindness of a ViT without position embeddings', () => {
    const x = makeSet(1, 7)[0].x;
    expect(patchify(x, 2)).toHaveLength(16);
    expect(patchify(x, 4)[0]).toHaveLength(16);
    const order = [3, 0, 15, 7, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14];
    const blind = new Vit(2, 6, false, 3);
    const a = blind.pooledFrom(x);
    const b = blind.pooledFrom(x, order);
    a.forEach((v, i) => expect(b[i]).toBeCloseTo(v, 12));
    const sighted = new Vit(2, 6, true, 3);
    const c = sighted.pooledFrom(x);
    const d = sighted.pooledFrom(x, order);
    expect(Math.max(...c.map((v, i) => Math.abs(v - d[i])))).toBeGreaterThan(0.01);
    expect(shift(x, 8, 0).every((v) => v === 0)).toBe(true);
    expect([attnCost(1, 4), attnCost(2, 8), attnCost(4, 16)]).toEqual([16384, 2048, 256]);
  });
});

describe('recorded vision results', () => {
  const row = (g: string, f: (a: Arch, n: number) => boolean) => VISION_RESULTS.filter((r) => r.group === g && f(r.arch, r.n));
  it('the race: CNN ahead at every size, gap shrinking from 25.5 to 4.4 points', () => {
    const race = [16, 64, 256, 1024].map((n) => [mean(row('race', (a, m) => a.kind === 'cnn' && m === n)[0].accs), mean(row('race', (a, m) => a.kind === 'vit' && m === n)[0].accs)]);
    expect(race.map(([c, v]) => [+(100 * c).toFixed(1), +(100 * v).toFixed(1)])).toEqual([
      [81.2, 55.7],
      [91.3, 69.2],
      [91.5, 81.8],
      [92.7, 88.3],
    ]);
  });
  it('rescue brief: four solutions, all with augmentation and 2-pixel patches; positions hurt without augmentation', () => {
    const r = row('rescue', () => true);
    const pass = r.filter((x) => mean(x.accs) >= RESCUE_BAR);
    expect(pass).toHaveLength(4);
    expect(pass.every((x) => x.arch.kind === 'vit' && x.arch.augment && x.arch.P === 2)).toBe(true);
    const get = (pos: boolean, aug: boolean, d: number) => +(100 * mean(r.find((x) => x.arch.kind === 'vit' && x.arch.P === 2 && x.arch.d === d && x.arch.pos === pos && x.arch.augment === aug)!.accs)).toFixed(1);
    expect([get(false, false, 8), get(true, false, 8), get(true, true, 8), get(false, true, 8)]).toEqual([76.4, 67.3, 84.9, 86.8]);
  });
  it('patch brief: big patches lose the detail; the cheapest passing layout is 2-pixel patches, width 8', () => {
    const r = row('patch', () => true).filter((x) => x.arch.kind === 'vit' && attnCost(x.arch.P, x.arch.d) <= ATTN_CAP);
    const pass = r.filter((x) => mean(x.accs) >= PATCH_BAR);
    expect(pass.map((x) => (x.arch.kind === 'vit' ? [x.arch.P, x.arch.d] : []))).toEqual([
      [2, 8],
      [2, 16],
    ]);
    expect(Math.max(...r.filter((x) => x.arch.kind === 'vit' && x.arch.P === 4).map((x) => mean(x.accs)))).toBeLessThan(0.71);
  });
  it('CLIP brief: six settings reach 90%, the smallest passing batch is 8', () => {
    const pass = CLIP_RESULTS.filter((r) => r.acc >= CLIP_BAR);
    expect(pass).toHaveLength(6);
    expect(Math.min(...pass.map((r) => r.batch))).toBe(8);
    expect(Math.max(...CLIP_RESULTS.map((r) => r.acc))).toBe(0.9358);
    expect(CLIP_RESULTS.filter((r) => r.k === 2).every((r) => r.acc < 0.8)).toBe(true);
  });
  it('re-training a sample reproduces the record', () => {
    const r = VISION_RESULTS[13];
    expect(runOne(r.arch, r.n, 1)).toBeCloseTo(r.accs[0], 4);
    const c = CLIP_RESULTS.find((x) => x.tau === 0.3 && x.k === 4 && x.batch === 8)!;
    expect(clipMean(c)).toBeCloseTo(c.acc, 4);
  }, 60_000);
});

describe('widget readouts', () => {
  it('model sizes and the workbench’s CLIP setting', async () => {
    const { trainClip, zeroShot } = await import('./clip');
    expect([new Cnn(4, 1).params.length, new Vit(2, 8, true, 1).params.length]).toEqual([45, 369]);
    expect([1, 2, 3].map((s) => zeroShot(trainClip({ tau: 0.3, k: 8, batch: 16 }, s)))).toEqual([0.93, 0.94, 0.9375]);
  }, 30_000);
});

describe('race gap', () => {
  it('25.5 points at 16 images, 4.4 at 1,024', () => {
    const m = (kind: string, n: number) => mean(VISION_RESULTS.find((r) => r.group === 'race' && r.arch.kind === kind && r.n === n)!.accs);
    expect(+(100 * (m('cnn', 16) - m('vit', 16))).toFixed(1)).toBe(25.5);
    expect(+(100 * (m('cnn', 1024) - m('vit', 1024))).toFixed(1)).toBe(4.4);
  });
});
