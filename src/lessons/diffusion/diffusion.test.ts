// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import {
  LOPSIDED,
  RING,
  SMILEY,
  abarAt,
  bestGuess,
  dataMean,
  density,
  evenLevels,
  exactDenoiser,
  impliedNoise,
  noiseStart,
  outOfPlace,
  sample,
  sampleData,
  shapeLeft,
  type Mixture,
  type Pt,
} from './diffusion';

const run = (mix: Mixture, steps: number, eta: number) => outOfPlace(mix, sample(exactDenoiser(mix), evenLevels(steps), noiseStart(2000, 5), eta, createRng(9)).xs).tv;

describe('the exact noisy distribution and the perfect denoiser', () => {
  it('the noisy density integrates to 1', () => {
    for (const a of [0.2, 0.7]) {
      let s = 0;
      const h = 0.04;
      for (let x = -6; x < 6; x += h) for (let y = -6; y < 6; y += h) s += density(SMILEY, a, [x + h / 2, y + h / 2]) * h * h;
      expect(s).toBeCloseTo(1, 3);
    }
  });

  it('the best guess matches a brute-force weighted average over clean points', () => {
    const rng = createRng(2);
    const clean = sampleData(RING, 200000, rng);
    for (const [a, x] of [
      [0.4, [0.8, 0.3]],
      [0.9, [1.2, -0.2]],
    ] as [number, Pt][]) {
      let wx = 0;
      let wy = 0;
      let w = 0;
      for (const c of clean) {
        const d2 = (x[0] - Math.sqrt(a) * c[0]) ** 2 + (x[1] - Math.sqrt(a) * c[1]) ** 2;
        const k = Math.exp(-d2 / (2 * (1 - a)));
        wx += k * c[0];
        wy += k * c[1];
        w += k;
      }
      const g = bestGuess(RING, a, x).x0;
      expect(Math.abs(g[0] - wx / w)).toBeLessThan(0.02);
      expect(Math.abs(g[1] - wy / w)).toBeLessThan(0.02);
    }
  });

  it('Tweedie: the implied noise equals −√(1 − ᾱ) times the gradient of log density (central differences)', () => {
    const h = 1e-5;
    for (const mix of [RING, SMILEY, LOPSIDED])
      for (const a of [0.05, 0.5, 0.95])
        for (const x of [
          [0.3, -0.7],
          [1.1, 0.4],
        ] as Pt[]) {
          const gx = (Math.log(density(mix, a, [x[0] + h, x[1]])) - Math.log(density(mix, a, [x[0] - h, x[1]]))) / (2 * h);
          const gy = (Math.log(density(mix, a, [x[0], x[1] + h])) - Math.log(density(mix, a, [x[0], x[1] - h]))) / (2 * h);
          const e = impliedNoise(x, bestGuess(mix, a, x).x0, a);
          expect(e[0]).toBeCloseTo(-Math.sqrt(1 - a) * gx, 5);
          expect(e[1]).toBeCloseTo(-Math.sqrt(1 - a) * gy, 5);
        }
  });

  it('from pure noise the best guess is the data mean', () => {
    const g = bestGuess(RING, 0, [1.7, -0.4]).x0;
    expect(Math.hypot(g[0] - dataMean(RING)[0], g[1] - dataMean(RING)[1])).toBeLessThan(1e-12);
  });
});

describe('numbers quoted in the lesson', () => {
  it('forward: 3.3% of the ring left at step 300, under 0.1% at step 500', () => {
    expect(shapeLeft(RING, abarAt(300))).toBeCloseTo(0.0334, 3);
    expect(shapeLeft(RING, abarAt(500))).toBeLessThan(0.001);
    expect(abarAt(1000)).toBeLessThan(5e-5);
  });

  it('one step puts every sample on the mean', () => {
    const xs = sample(exactDenoiser(RING), evenLevels(1), noiseStart(2000, 5), 0, createRng(9)).xs;
    expect(Math.max(...xs.map((p) => Math.hypot(p[0], p[1])))).toBeLessThan(1e-12);
  });

  it('ring with evenly spaced DDIM steps: 10 → 31.6%, 20 → 7.0%, 100 → 3.8%; perfect sampler 2.7%', () => {
    const p1 = (v: number) => (100 * v).toFixed(1);
    expect(p1(run(RING, 10, 0))).toBe('31.6');
    expect(p1(run(RING, 20, 0))).toBe('7.0');
    expect(p1(run(RING, 100, 0))).toBe('3.8');
    expect(p1(outOfPlace(RING, sampleData(RING, 2000, createRng(3))).tv)).toBe('2.7');
  });

  it('DDIM beats DDPM at 10 and 20 steps on every shape; at 50, DDPM wins on ring and smiley but not the rare blob', () => {
    for (const mix of [RING, SMILEY, LOPSIDED]) for (const s of [10, 20]) expect(run(mix, s, 0)).toBeLessThan(run(mix, s, 1));
    expect((100 * run(RING, 50, 1)).toFixed(1)).toBe('2.1');
    expect((100 * run(RING, 50, 0)).toFixed(1)).toBe('3.9');
    expect(run(SMILEY, 50, 1)).toBeLessThan(run(SMILEY, 50, 0));
    expect(run(LOPSIDED, 50, 0)).toBeLessThan(run(LOPSIDED, 50, 1));
  });

  it('four evenly spaced steps stop at 0, 0.003, 0.08, 0.52 and leave 95% of the ring out of place', () => {
    const lv = evenLevels(4);
    expect(lv[0]).toBe(0);
    expect(lv[1]).toBeCloseTo(0.003, 3);
    expect(lv[2]).toBeCloseTo(0.08, 2);
    expect(lv[3]).toBeCloseTo(0.52, 2);
    expect(Math.round(100 * run(RING, 4, 0))).toBe(95);
  });
});
