// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  MICROSCOPE_INIT,
  MICROSCOPE_KEYS,
  MICROSCOPE_XOR,
  TINY222_PRESETS,
  applyMicroscopeStep,
  cloneTiny222Params,
  compareCoordinatedVsSingle,
  hiddenLineSegment,
  microscopeStepInfo,
  runMicroscopeSteps,
  stepTiny222,
  tiny222ViaAutodiff,
  traceTiny222,
  twinGap,
  type MicroscopeKey,
  type Tiny222Params,
  type TinyAct,
} from './tinyMlp';

function fdTiny222(params: Tiny222Params, x: [number, number], t: [number, number], act: TinyAct, eps = 1e-6) {
  const lossAt = (p: Tiny222Params) => traceTiny222(p, x, t, act).loss;
  const dW1: [[number, number], [number, number]] = [
    [0, 0],
    [0, 0],
  ];
  const db1: [number, number] = [0, 0];
  const dW2: [[number, number], [number, number]] = [
    [0, 0],
    [0, 0],
  ];
  const db2: [number, number] = [0, 0];

  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const pp = cloneTiny222Params(params);
      const pm = cloneTiny222Params(params);
      pp.W1[i][j] += eps;
      pm.W1[i][j] -= eps;
      dW1[i][j] = (lossAt(pp) - lossAt(pm)) / (2 * eps);

      const qp = cloneTiny222Params(params);
      const qm = cloneTiny222Params(params);
      qp.W2[i][j] += eps;
      qm.W2[i][j] -= eps;
      dW2[i][j] = (lossAt(qp) - lossAt(qm)) / (2 * eps);
    }
    const bp = cloneTiny222Params(params);
    const bm = cloneTiny222Params(params);
    bp.b1[i] += eps;
    bm.b1[i] -= eps;
    db1[i] = (lossAt(bp) - lossAt(bm)) / (2 * eps);

    const cp = cloneTiny222Params(params);
    const cm = cloneTiny222Params(params);
    cp.b2[i] += eps;
    cm.b2[i] -= eps;
    db2[i] = (lossAt(cp) - lossAt(cm)) / (2 * eps);
  }
  return { dW1, db1, dW2, db2 };
}

describe('2 → 2 → 2 scalar credit-assignment engine', () => {
  it('matches Value.backward() and central finite differences (< 1e-8) across all presets and activations', () => {
    for (const preset of Object.values(TINY222_PRESETS)) {
      for (const act of ['tanh', 'relu'] as const) {
        const tr = traceTiny222(preset.params, preset.x, preset.t, act);
        const ad = tiny222ViaAutodiff(preset.params, preset.x, preset.t, act);
        const fd = fdTiny222(preset.params, preset.x, preset.t, act);

        expect(Math.abs(tr.loss - ad.loss)).toBeLessThan(1e-12);
        for (let i = 0; i < 2; i++) {
          expect(Math.abs(tr.dh[i] - ad.dh[i])).toBeLessThan(1e-12);
          expect(Math.abs(tr.dz1[i] - ad.dz1[i])).toBeLessThan(1e-12);
          expect(Math.abs(tr.db1[i] - ad.db1[i])).toBeLessThan(1e-12);
          expect(Math.abs(tr.db2[i] - ad.db2[i])).toBeLessThan(1e-12);
          expect(Math.abs(tr.db1[i] - fd.db1[i])).toBeLessThan(1e-8);
          expect(Math.abs(tr.db2[i] - fd.db2[i])).toBeLessThan(1e-8);
          for (let j = 0; j < 2; j++) {
            expect(Math.abs(tr.dW1[i][j] - ad.dW1[i][j])).toBeLessThan(1e-12);
            expect(Math.abs(tr.dW2[i][j] - ad.dW2[i][j])).toBeLessThan(1e-12);
            expect(Math.abs(tr.dW1[i][j] - fd.dW1[i][j])).toBeLessThan(1e-8);
            expect(Math.abs(tr.dW2[i][j] - fd.dW2[i][j])).toBeLessThan(1e-8);
          }
        }
      }
    }
  });

  it('pins the Balanced preset fan-in sums (explicit dy @ W2.T)', () => {
    const p = TINY222_PRESETS.balanced;
    const tr = traceTiny222(p.params, p.x, p.t, 'tanh');
    expect(tr.dhPaths[0][0]).toBeCloseTo(0.2374, 3);
    expect(tr.dhPaths[0][1]).toBeCloseTo(0.2376, 3);
    expect(tr.dh[0]).toBeCloseTo(0.475, 3);
    expect(tr.dh[1]).toBeCloseTo(-0.4091, 3);
  });

  it('pins the Tug-of-war preset: opposite paths into h1 cancel by >97% while reinforcing in h2', () => {
    const p = TINY222_PRESETS.cancelling;
    const tr = traceTiny222(p.params, p.x, p.t, 'tanh');
    expect(tr.dhPaths[0][0]).toBeCloseTo(-0.8052, 3);
    expect(tr.dhPaths[0][1]).toBeCloseTo(0.7884, 3);
    expect(tr.dh[0]).toBeCloseTo(-0.0168, 3);
    expect(Math.abs(tr.dh[0])).toBeLessThan(0.03 * Math.abs(tr.dhPaths[0][0]));
    // Meanwhile h2's two paths reinforce
    expect(tr.dhPaths[1][0]).toBeGreaterThan(0.3);
    expect(tr.dhPaths[1][1]).toBeGreaterThan(0.7);
    expect(tr.dh[1]).toBeCloseTo(1.0317, 3);
  });

  it('pins the Identical Twins preset: 10 steps of SGD leave |w_h1 - w_h2| === 0 bit-for-bit', () => {
    const p = TINY222_PRESETS.twins;
    let cur = cloneTiny222Params(p.params);
    expect(twinGap(cur)).toBe(0);
    for (let step = 0; step < 10; step++) {
      cur = stepTiny222(cur, p.x, p.t, 'tanh', 0.25);
      expect(twinGap(cur)).toBe(0);
    }
    // Weights actually moved, just identically
    expect(cur.W1[0][0]).not.toBe(p.params.W1[0][0]);
  });

  it('pins the Saturated tanh preset: equal upstream dh, >600x starvation on h1 incoming weights', () => {
    const p = TINY222_PRESETS.saturated;
    const tr = traceTiny222(p.params, p.x, p.t, 'tanh');
    expect(tr.z1[0]).toBeCloseTo(4.0, 12);
    expect(tr.z1[1]).toBeCloseTo(0.3, 12);
    expect(tr.dh[0]).toBeCloseTo(tr.dh[1], 12);
    expect(tr.actDeriv[0]).toBeCloseTo(0.001341, 5);
    expect(tr.actDeriv[1]).toBeCloseTo(0.91514, 4);
    const ratio = Math.abs(tr.dW1[0][1]) / Math.abs(tr.dW1[0][0]);
    expect(ratio).toBeGreaterThan(680);
    expect(ratio).toBeCloseTo(682.45, 1);
  });
});

describe('2 → 2 → 1 geometry microscope engine', () => {
  it('matches central finite differences (< 1e-8) in both full-batch and every single-point mode', () => {
    const eps = 1e-6;
    for (const sel of [null, 0, 1, 2, 3] as const) {
      const info = microscopeStepInfo(MICROSCOPE_INIT, sel);
      for (const k of MICROSCOPE_KEYS) {
        const plus = { ...MICROSCOPE_INIT, [k]: MICROSCOPE_INIT[k] + eps };
        const minus = { ...MICROSCOPE_INIT, [k]: MICROSCOPE_INIT[k] - eps };
        const num = (microscopeStepInfo(plus, sel).selectedLoss - microscopeStepInfo(minus, sel).selectedLoss) / (2 * eps);
        expect(Math.abs(info.grads[k] - num)).toBeLessThan(1e-8);
      }
    }
  });

  it('computes exact zero-contour line segments satisfying w1*x + w2*y + b = 0', () => {
    const l1 = hiddenLineSegment(MICROSCOPE_INIT.w11, MICROSCOPE_INIT.w12, MICROSCOPE_INIT.b1);
    const l2 = hiddenLineSegment(MICROSCOPE_INIT.w21, MICROSCOPE_INIT.w22, MICROSCOPE_INIT.b2);
    expect(l1).not.toBeNull();
    expect(l2).not.toBeNull();
    if (l1 && l2) {
      expect(MICROSCOPE_INIT.w11 * l1.x1 + MICROSCOPE_INIT.w12 * l1.y1 + MICROSCOPE_INIT.b1).toBeCloseTo(0, 10);
      expect(MICROSCOPE_INIT.w11 * l1.x2 + MICROSCOPE_INIT.w12 * l1.y2 + MICROSCOPE_INIT.b1).toBeCloseTo(0, 10);
      expect(Math.hypot(l1.nx, l1.ny)).toBeCloseTo(1, 10);
      expect(MICROSCOPE_INIT.w21 * l2.x1 + MICROSCOPE_INIT.w22 * l2.y1 + MICROSCOPE_INIT.b2).toBeCloseTo(0, 10);
      expect(MICROSCOPE_INIT.w21 * l2.x2 + MICROSCOPE_INIT.w22 * l2.y2 + MICROSCOPE_INIT.b2).toBeCloseTo(0, 10);
    }
    expect(hiddenLineSegment(0, 0, 1)).toBeNull();
  });

  it('pins the single-point tug-of-war on P1 and full-batch convergence on XOR', () => {
    const init = microscopeStepInfo(MICROSCOPE_INIT, null);
    expect(init.meanLoss).toBeCloseTo(0.6674, 4);
    expect(init.accuracy).toBe(0.25);

    // Single-point step on P1 (-1, +1) -> 1 drops P1's loss sharply while tugging P0 and P3 worse
    const p1Before = microscopeStepInfo(MICROSCOPE_INIT, 1);
    const afterP1 = microscopeStepInfo(applyMicroscopeStep(MICROSCOPE_INIT, p1Before.grads, 0.8), 1);
    expect(afterP1.points[1].loss - p1Before.points[1].loss).toBeCloseTo(-0.834, 3);
    expect(afterP1.points[0].loss - p1Before.points[0].loss).toBeCloseTo(0.542, 3);
    expect(afterP1.points[3].loss - p1Before.points[3].loss).toBeCloseTo(0.500, 3);

    // Coordinated 9-weight step drops batch loss nearly 3x more than moving only the steepest weight
    const cmp = compareCoordinatedVsSingle(MICROSCOPE_INIT, null, 0.9);
    expect(cmp.steepestKey).toBe<MicroscopeKey>('w12');
    expect(cmp.fullDrop).toBeCloseTo(0.0823, 4);
    expect(cmp.singleDrop).toBeCloseTo(0.0284, 4);
    expect(cmp.fullDrop / cmp.singleDrop).toBeGreaterThan(2.8);

    // Full-batch trajectory reaches 100% accuracy by step 5 and loss ~0.102 by step 40
    const hist = runMicroscopeSteps(MICROSCOPE_INIT, 40, 0.9, null);
    expect(microscopeStepInfo(hist[5], null).accuracy).toBe(1);
    expect(microscopeStepInfo(hist[25], null).meanLoss).toBeCloseTo(0.1644, 4);
    expect(microscopeStepInfo(hist[40], null).meanLoss).toBeCloseTo(0.1019, 4);
    expect(MICROSCOPE_XOR).toHaveLength(4);
  });
});
