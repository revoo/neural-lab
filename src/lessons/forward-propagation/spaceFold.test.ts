// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  deformGrid,
  forwardStages,
  interpolateBounds,
  interpolateStage,
  lossAndGrad,
  makeDataset,
  outputBoundaryLine,
  trainSpaceFold,
  type FoldNet,
} from './spaceFold';

describe('spaceFold engine', () => {
  it('analytic gradients match central finite differences to < 1e-8 on every parameter', () => {
    const net: FoldNet = {
      W1: [[0.7, -0.4], [0.3, 0.9]],
      b1: [0.15, -0.25],
      W2: [[-0.6, 0.8], [0.5, -0.35]],
      b2: [-0.1, 0.2],
      W3: [0.9, -0.75],
      b3: 0.12,
    };
    const pts = makeDataset('rings', 8, 1);
    const wd = 0.0005;
    const { grad } = lossAndGrad(net, pts, wd);
    const eps = 1e-6;

    const check = (get: () => number, set: (v: number) => void, analytic: number) => {
      const orig = get();
      set(orig + eps);
      const lPlus = lossAndGrad(net, pts, wd).loss;
      set(orig - eps);
      const lMinus = lossAndGrad(net, pts, wd).loss;
      set(orig);
      const fd = (lPlus - lMinus) / (2 * eps);
      expect(Math.abs(analytic - fd)).toBeLessThan(1e-8);
    };

    check(() => net.W1[0][0], (v) => { net.W1[0][0] = v; }, grad.W1[0][0]);
    check(() => net.W1[0][1], (v) => { net.W1[0][1] = v; }, grad.W1[0][1]);
    check(() => net.W1[1][0], (v) => { net.W1[1][0] = v; }, grad.W1[1][0]);
    check(() => net.W1[1][1], (v) => { net.W1[1][1] = v; }, grad.W1[1][1]);
    check(() => net.b1[0], (v) => { net.b1[0] = v; }, grad.b1[0]);
    check(() => net.b1[1], (v) => { net.b1[1] = v; }, grad.b1[1]);
    check(() => net.W2[0][0], (v) => { net.W2[0][0] = v; }, grad.W2[0][0]);
    check(() => net.W2[0][1], (v) => { net.W2[0][1] = v; }, grad.W2[0][1]);
    check(() => net.W2[1][0], (v) => { net.W2[1][0] = v; }, grad.W2[1][0]);
    check(() => net.W2[1][1], (v) => { net.W2[1][1] = v; }, grad.W2[1][1]);
    check(() => net.b2[0], (v) => { net.b2[0] = v; }, grad.b2[0]);
    check(() => net.b2[1], (v) => { net.b2[1] = v; }, grad.b2[1]);
    check(() => net.W3[0], (v) => { net.W3[0] = v; }, grad.W3[0]);
    check(() => net.W3[1], (v) => { net.W3[1] = v; }, grad.W3[1]);
    check(() => net.b3, (v) => { net.b3 = v; }, grad.b3);
  });

  it('pins progressive linear separability on Concentric rings (70.8% -> 89.6% -> 100%)', () => {
    const r = trainSpaceFold('rings');
    expect(r.paramCount).toBe(15);
    expect(r.points.length).toBe(48);
    expect(r.acc0).toBeCloseTo(34 / 48, 4); // 70.8%
    expect(r.acc1).toBeCloseTo(43 / 48, 4); // 89.6%
    expect(r.acc2).toBe(1);                 // 100.0%
    expect(r.denseAcc).toBe(1);             // 100.0% on 400-point continuous ring
  });

  it('pins progressive linear separability on Two moons (87.5% -> 91.7% -> 100%)', () => {
    const m = trainSpaceFold('moons');
    expect(m.paramCount).toBe(15);
    expect(m.points.length).toBe(48);
    expect(m.acc0).toBeCloseTo(42 / 48, 4); // 87.5%
    expect(m.acc1).toBeCloseTo(44 / 48, 4); // 91.7%
    expect(m.acc2).toBe(1);                 // 100.0%
    expect(m.denseAcc).toBe(1);             // 100.0% on 400-point continuous moons
  });

  it('deformGrid is continuous across t = 0, 0.5, 1, 1.5, 2 and matches forwardStages', () => {
    const r = trainSpaceFold('rings');
    const g0 = deformGrid(r.net, 0, 13);
    const g05 = deformGrid(r.net, 0.5, 13);
    const g1 = deformGrid(r.net, 1, 13);
    const g15 = deformGrid(r.net, 1.5, 13);
    const g2 = deformGrid(r.net, 2, 13);

    expect(g0.length).toBe(13);
    expect(g0[0].length).toBe(13);

    // Midpoint vertex (row 6, col 6) starts at (0, 0)
    expect(g0[6][6][0]).toBeCloseTo(0, 12);
    expect(g0[6][6][1]).toBeCloseTo(0, 12);

    const origin = forwardStages(r.net, 0, 0);
    expect(g1[6][6][0]).toBeCloseTo(origin.h1[0], 12);
    expect(g1[6][6][1]).toBeCloseTo(origin.h1[1], 12);
    expect(g2[6][6][0]).toBeCloseTo(origin.h2[0], 12);
    expect(g2[6][6][1]).toBeCloseTo(origin.h2[1], 12);

    // Exact linear midpoint checks at t = 0.5 and t = 1.5
    expect(g05[6][6][0]).toBeCloseTo(0.5 * (g0[6][6][0] + g1[6][6][0]), 12);
    expect(g05[6][6][1]).toBeCloseTo(0.5 * (g0[6][6][1] + g1[6][6][1]), 12);
    expect(g15[6][6][0]).toBeCloseTo(0.5 * (g1[6][6][0] + g2[6][6][0]), 12);
    expect(g15[6][6][1]).toBeCloseTo(0.5 * (g1[6][6][1] + g2[6][6][1]), 12);

    // Boundary segments unbend onto the straight output line W3.h2 + b3 = 0
    expect(r.boundarySegments.length).toBeGreaterThan(20);
    for (const seg of r.boundarySegments) {
      const zA = r.net.W3[0] * seg.a[2][0] + r.net.W3[1] * seg.a[2][1] + r.net.b3;
      const zB = r.net.W3[0] * seg.b[2][0] + r.net.W3[1] * seg.b[2][1] + r.net.b3;
      expect(Math.abs(zA)).toBeLessThan(0.15);
      expect(Math.abs(zB)).toBeLessThan(0.15);
    }

    // Straight knife cut clips cleanly inside the final stage box
    const line = outputBoundaryLine(r.net, interpolateBounds(r.stageBounds, 2));
    expect(line).not.toBeNull();
    const mid = interpolateStage(origin, 1.5);
    expect(mid[0]).toBeCloseTo(g15[6][6][0], 12);
  });
});
