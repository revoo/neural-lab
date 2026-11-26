// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  BRIEF1_NET,
  BRIEF1_PROBES,
  BRIEF2_ACTS,
  BRIEF2_INIT_MODES,
  BRIEF2_LRS,
  BRIEF3_INIT,
  BRIEF3_LRS,
  checkBrief1,
  cloneLabMlp,
  evalAndGradMlp,
  evaluateBrief1,
  evaluateBrief2,
  evaluateBrief3,
  isXorHiddenSeparable,
  makeBrief2Init,
  type Brief1WeightKey,
  type CreditAct,
  type LabMlp,
} from './creditLab';
import { BRIEF2_GRID, BRIEF2_SUMMARY, BRIEF3_GRID, BRIEF3_SUMMARY } from './results';

function fdLabMlp(m: LabMlp, act: CreditAct, eps = 1e-6) {
  const lossAt = (net: LabMlp) => evalAndGradMlp(net, act).loss;
  const dW1 = Array.from({ length: m.H }, () => [0, 0]);
  const db1 = Array(m.H).fill(0);
  const dW2 = Array(m.H).fill(0);

  for (let j = 0; j < m.H; j++) {
    for (let i = 0; i < 2; i++) {
      const p = cloneLabMlp(m);
      const q = cloneLabMlp(m);
      p.W1[j][i] += eps;
      q.W1[j][i] -= eps;
      dW1[j][i] = (lossAt(p) - lossAt(q)) / (2 * eps);
    }
    const bp = cloneLabMlp(m);
    const bm = cloneLabMlp(m);
    bp.b1[j] += eps;
    bm.b1[j] -= eps;
    db1[j] = (lossAt(bp) - lossAt(bm)) / (2 * eps);

    const wp = cloneLabMlp(m);
    const wm = cloneLabMlp(m);
    wp.W2[j] += eps;
    wm.W2[j] -= eps;
    dW2[j] = (lossAt(wp) - lossAt(wm)) / (2 * eps);
  }
  const cp = cloneLabMlp(m);
  const cm = cloneLabMlp(m);
  cp.b2 += eps;
  cm.b2 -= eps;
  const db2 = (lossAt(cp) - lossAt(cm)) / (2 * eps);
  return { dW1, db1, dW2, db2 };
}

describe('Credit-Assignment Workbench: gradients and Brief 1', () => {
  it('Brief 1 analytic gradients match central finite differences (< 1e-8) and pin exact values', () => {
    const tr = evaluateBrief1(BRIEF1_NET);
    expect(tr.z1).toBeCloseTo(1.5, 12);
    expect(tr.z2).toBeCloseTo(-1.8, 12);
    expect(tr.h1).toBeCloseTo(1.5, 12);
    expect(tr.h2).toBe(0);
    expect(tr.yHat).toBeCloseTo(-1.0, 12);
    expect(tr.error).toBeCloseTo(-2.0, 12);
    expect(tr.dh1).toBeCloseTo(1.6, 12);
    expect(tr.dh2).toBeCloseTo(-1.4, 12);
    expect(tr.dz1).toBeCloseTo(1.6, 12);
    expect(tr.dz2).toBe(0);

    expect(tr.grads.v1).toBeCloseTo(-3.0, 12);
    expect(tr.grads.w11).toBeCloseTo(3.2, 12);
    expect(tr.grads.w21).toBeCloseTo(-1.6, 12);
    expect(tr.grads.w12).toBe(0);

    expect(tr.updateSigns).toEqual({ v1: '+', w11: '-', w21: '+', w12: '0' });
    expect(tr.largestKey).toBe<Brief1WeightKey>('w11');

    const eps = 1e-6;
    for (const k of BRIEF1_PROBES) {
      const plus = { ...BRIEF1_NET, [k]: BRIEF1_NET[k] + eps };
      const minus = { ...BRIEF1_NET, [k]: BRIEF1_NET[k] - eps };
      const num = (evaluateBrief1(plus).loss - evaluateBrief1(minus).loss) / (2 * eps);
      expect(Math.abs(tr.grads[k] - num)).toBeLessThan(1e-8);
    }

    const ok = checkBrief1({ v1: '+', w11: '-', w21: '+', w12: '0' }, 'w11');
    expect(ok.allSignsCorrect).toBe(true);
    expect(ok.largestCorrect).toBe(true);
  });

  it('shared MLP backward pass matches central finite differences (< 1e-8) for tanh, sigmoid, and relu', () => {
    const net = makeBrief2Init('asym-small', 1);
    for (const act of BRIEF2_ACTS) {
      const g = evalAndGradMlp(net, act);
      const fd = fdLabMlp(net, act);
      expect(Math.abs(g.db2 - fd.db2)).toBeLessThan(1e-8);
      for (let j = 0; j < net.H; j++) {
        expect(Math.abs(g.dW2[j] - fd.dW2[j])).toBeLessThan(1e-8);
        expect(Math.abs(g.db1[j] - fd.db1[j])).toBeLessThan(1e-8);
        expect(Math.abs(g.dW1[j][0] - fd.dW1[j][0])).toBeLessThan(1e-8);
        expect(Math.abs(g.dW1[j][1] - fd.dW1[j][1])).toBeLessThan(1e-8);
      }
    }
  });
});

describe('Credit-Assignment Workbench: exhaustive verification of Brief 2 and Brief 3 results.ts', () => {
  it('re-verifies all 48 Brief 2 settings (x 5 seeds) against recorded results.ts', () => {
    expect(BRIEF2_GRID).toHaveLength(48);
    let idx = 0;
    let passing = 0;
    for (const initMode of BRIEF2_INIT_MODES) {
      for (const act of BRIEF2_ACTS) {
        for (const lr of BRIEF2_LRS) {
          const live = evaluateBrief2({ initMode, act, lr });
          const rec = BRIEF2_GRID[idx++];
          expect(rec.initMode).toBe(initMode);
          expect(rec.act).toBe(act);
          expect(rec.lr).toBe(lr);
          expect(live.pass).toBe(rec.pass);
          expect(live.solvedSeeds).toBe(rec.solvedSeeds);
          expect(live.meanLoss).toBeCloseTo(rec.meanLoss, 5);
          expect(live.meanAcc).toBeCloseTo(rec.meanAcc, 4);
          if (live.pass) {
            passing++;
            expect(live.meanSolveStep).toBeCloseTo(rec.meanSolveStep!, 2);
          } else {
            expect(rec.meanSolveStep).toBeNull();
          }
        }
      }
    }
    expect(passing).toBe(BRIEF2_SUMMARY.passing);
    expect(BRIEF2_SUMMARY.passing).toBe(11);
    expect(BRIEF2_SUMMARY.brokenPassing).toBe(0);
    expect(BRIEF2_SUMMARY.zerosPassing).toBe(0);
    expect(BRIEF2_SUMMARY.asymLargePassing).toBe(3);
    expect(BRIEF2_SUMMARY.asymSmallPassing).toBe(8);
    // Honest finding: saturated large-scale fails 0/8 on tanh & sigmoid, yet passes on ReLU (fastest 11.6 steps)
    expect(BRIEF2_GRID.filter((r) => r.initMode === 'asym-large' && r.act !== 'relu' && r.pass)).toHaveLength(0);
    expect(BRIEF2_SUMMARY.fastest.initMode).toBe('asym-large');
    expect(BRIEF2_SUMMARY.fastest.act).toBe('relu');
    expect(BRIEF2_SUMMARY.fastest.lr).toBe(1);
    expect(BRIEF2_SUMMARY.fastest.meanSolveStep).toBeCloseTo(11.6, 1);
  });

  it('re-verifies all 32 Brief 3 settings, initial non-separability, and hidden-space untangling', () => {
    const initEval = evalAndGradMlp(BRIEF3_INIT, 'tanh');
    expect(isXorHiddenSeparable(initEval.hiddenCoords)).toBe(false);

    // Output-only cannot solve XOR even after 2,000 steps
    const longFrozen = evaluateBrief3({ trainH1: false, trainH2: false, trainOut: true, lr: 2.5 }, 2000);
    expect(longFrozen.pass).toBe(false);
    expect(longFrozen.finalSeparable).toBe(false);
    expect(longFrozen.finalAcc).toBeLessThanOrEqual(0.75);

    expect(BRIEF3_GRID).toHaveLength(32);
    let idx = 0;
    let passing = 0;
    for (const trainH1 of [false, true]) {
      for (const trainH2 of [false, true]) {
        for (const trainOut of [false, true]) {
          for (const lr of BRIEF3_LRS) {
            const live = evaluateBrief3({ trainH1, trainH2, trainOut, lr });
            const rec = BRIEF3_GRID[idx++];
            expect(live.pass).toBe(rec.pass);
            expect(live.solvedStep).toBe(rec.solvedStep);
            expect(live.finalSeparable).toBe(rec.finalSeparable);
            expect(live.finalLoss).toBeCloseTo(rec.finalLoss, 5);
            if (live.pass) {
              passing++;
              // Every passing route must have untangled the hidden space!
              expect(live.finalSeparable).toBe(true);
            }
          }
        }
      }
    }
    expect(passing).toBe(BRIEF3_SUMMARY.passing);
    expect(BRIEF3_SUMMARY.passing).toBe(7);
    expect(BRIEF3_SUMMARY.outputOnlyPassing).toBe(0);
    expect(BRIEF3_SUMMARY.outputFrozenPassing).toBe(0);
    expect(BRIEF3_SUMMARY.minUnfrozenHidden).toBe(1);
    expect(BRIEF3_SUMMARY.minUnfrozenPassing).toBe(3);
    expect(BRIEF3_SUMMARY.fastestMinUnfrozen.solvedStep).toBe(68);
    expect(BRIEF3_SUMMARY.fastestOverall.solvedStep).toBe(19);
  });
});
