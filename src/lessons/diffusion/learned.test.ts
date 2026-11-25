// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import { RING, evenLevels, exactDenoiser, noiseStart, sample, sampleData, type Pt } from './diffusion';
import { LEARN, NoiseNet, lossFloor, noiseTrainer } from './learned';

const median = (xs: Pt[]) => {
  const d = xs.map((p) => Math.min(...RING.m.map((m) => Math.hypot(p[0] - m[0], p[1] - m[1])))).sort((a, b) => a - b);
  return d[Math.floor(d.length / 2)];
};

describe('the learned denoiser', () => {
  it('backprop matches central finite differences', () => {
    const net = new NoiseNet(7, 3);
    const x: Pt = [0.4, -1.1];
    const eps: Pt = [0.3, 0.9];
    net.g.fill(0);
    net.lossOne(x, 420, eps, true, 1);
    const an = Float64Array.from(net.g);
    const h = 1e-6;
    let worst = 0;
    for (let i = 0; i < net.p.length; i++) {
      const o = net.p[i];
      net.p[i] = o + h;
      const up = net.lossOne(x, 420, eps);
      net.p[i] = o - h;
      const dn = net.lossOne(x, 420, eps);
      net.p[i] = o;
      worst = Math.max(worst, Math.abs((up - dn) / (2 * h) - an[i]));
    }
    expect(worst).toBeLessThan(1e-7);
  });

  it('a 4,738-parameter network gets the ring but not the blobs', () => {
    expect(new NoiseNet(LEARN.H, 1).size).toBe(4738);
    const tr = noiseTrainer(RING, LEARN);
    tr.advance(Infinity);
    const lf = lossFloor(RING, tr.net);
    expect(lf.exact).toBeCloseTo(0.399, 2);
    expect(lf.learned).toBeCloseTo(0.476, 2);
    const levels = evenLevels(100);
    const learned = sample(tr.net.denoiser(), levels, noiseStart(2000, 5), 0, createRng(9)).xs;
    const exact = sample(exactDenoiser(RING), levels, noiseStart(2000, 5), 0, createRng(9)).xs;
    const data = sampleData(RING, 2000, createRng(3));
    expect(median(data)).toBeCloseTo(0.097, 2);
    expect(median(exact)).toBeCloseTo(0.078, 2);
    expect(median(learned)).toBeCloseTo(0.246, 2);
  }, 60_000);
});
