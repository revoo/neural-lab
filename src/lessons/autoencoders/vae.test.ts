// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { PIX, REAL_TOL, VAE, face, fitBell, makeDataset, mix, nearestFace, rmse } from './vae';
import { createRng } from '../../lib/rng';

describe('the hand-written backward pass', () => {
  for (const mode of ['ae', 'vae'] as const)
    for (const latent of [1, 2, 3])
      it(`matches central finite differences (${mode}, ${latent}-number code)`, () => {
        const net = new VAE(5, mode, 0.7, 3, latent);
        // nudge the log-variance biases off their starting value so every path is exercised
        const x = face(0.3, 0.6);
        const eps = Array.from({ length: latent }, (_, k) => 0.8 - 0.9 * k);
        net.g.fill(0);
        net.lossOne(x, eps, true, 1);
        const analytic = Float64Array.from(net.g);
        const h = 1e-6;
        let worst = 0;
        const rng = createRng(11);
        // check 400 random parameters plus every parameter of the small layers
        const idx = new Set<number>();
        for (let i = 0; i < 400; i++) idx.add(rng.int(net.p.length));
        for (let i = 5 * PIX; i < net.p.length - 5 * PIX - PIX; i++) idx.add(i);
        for (const i of idx) {
          const old = net.p[i];
          net.p[i] = old + h;
          const up = net.lossOne(x, eps).loss;
          net.p[i] = old - h;
          const down = net.lossOne(x, eps).loss;
          net.p[i] = old;
          const numeric = (up - down) / (2 * h);
          worst = Math.max(worst, Math.abs(numeric - analytic[i]) / Math.max(1, Math.abs(numeric)));
        }
        expect(worst).toBeLessThan(1e-6);
      });
});

describe('faces and the space they live in', () => {
  it('a pixel blend of two faces is not a face, a knob blend is', () => {
    const a = face(1, 1);
    const b = face(-1, 0);
    expect(nearestFace(mix(a, b, 0.5)).dist).toBeGreaterThan(REAL_TOL);
    expect(nearestFace(face(0, 0.5)).dist).toBe(0);
  });

  it('a fitted bell reproduces the mean and covariance of its data', () => {
    const rng = createRng(4);
    const pts = Array.from({ length: 4000 }, () => {
      const a = rng.normal();
      return [3 + 2 * a, -1 + a + 0.5 * rng.normal(), rng.normal()];
    });
    const bell = fitBell(pts);
    const s = Array.from({ length: 20000 }, () => bell.sample(rng));
    const again = fitBell(s);
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(again.mean[i] - bell.mean[i])).toBeLessThan(0.06);
      for (let j = 0; j < 3; j++) expect(Math.abs(again.cov[i][j] - bell.cov[i][j])).toBeLessThan(0.08);
    }
  });

  it('datasets are reproducible', () => {
    const a = makeDataset(5, 7);
    const b = makeDataset(5, 7);
    expect(rmse(a.imgs[3], b.imgs[3])).toBe(0);
  });
});
