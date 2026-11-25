// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  TWO_TREASURE_CLIFF,
  POLICY_LAB_LEVELS,
  cellIndex,
  evaluateLabLevel,
  evaluatePolicyExact,
  findCriticalGamma,
  inspectBellmanBackup,
  solveValueIteration,
} from './mdp';

describe('MDP & Bellman engine (mdps)', () => {
  it('Value Iteration converges to the exact Bellman optimality fixed point and agrees with linear policy evaluation', () => {
    const sol = solveValueIteration(TWO_TREASURE_CLIFF, 0.9, 0.2);
    const ev = evaluatePolicyExact(TWO_TREASURE_CLIFF, sol.policy, 0.9, 0.2);

    for (let i = 0; i < sol.V.length; i++) {
      expect(sol.V[i]).toBeCloseTo(ev.V[i], 8);
    }

    // Verify Bellman optimality residual < 1e-9 at every non-terminal cell
    for (let r = 0; r < TWO_TREASURE_CLIFF.rows; r++) {
      for (let c = 0; c < TWO_TREASURE_CLIFF.cols; c++) {
        if (TWO_TREASURE_CLIFF.cells[r][c].kind !== 'empty') continue;
        const idx = cellIndex(TWO_TREASURE_CLIFF, r, c);
        const insp = inspectBellmanBackup(TWO_TREASURE_CLIFF, sol.V, idx, 0.9, 0.2);
        expect(Math.abs(insp.bestQ - sol.V[idx])).toBeLessThan(1e-9);
      }
    }
  });

  it('pins the Two-Treasure Cliff numbers quoted in prose (critical gamma, myopia, and slip detour)', () => {
    // Exact deterministic critical discount factor
    const gStar = findCriticalGamma(TWO_TREASURE_CLIFF, 0);
    expect(gStar).toBeCloseTo(0.6034, 3);

    // Myopic regime: gamma = 0.50, slip = 0% -> takes Near Gem (+2) in 2 steps: -0.1 + 0.5*2 = 0.90
    const myopic = solveValueIteration(TWO_TREASURE_CLIFF, 0.5, 0);
    const evMyopic = evaluatePolicyExact(TWO_TREASURE_CLIFF, myopic.policy, 0.5, 0);
    expect(evMyopic.startValue).toBeCloseTo(0.9, 4);
    expect(evMyopic.probNear).toBeCloseTo(1.0, 4);
    expect(evMyopic.probGoal).toBeCloseTo(0.0, 4);

    // Patient deterministic regime: gamma = 0.90, slip = 0% -> marches straight to +10 Vault: 6.2171
    const patient = solveValueIteration(TWO_TREASURE_CLIFF, 0.9, 0);
    const evPatient = evaluatePolicyExact(TWO_TREASURE_CLIFF, patient.policy, 0.9, 0);
    expect(evPatient.startValue).toBeCloseTo(6.2171, 3);
    expect(evPatient.probGoal).toBeCloseTo(1.0, 4);

    // Deterministic cliff-edge policy evaluated under 20% slip crashes to 2.3790 (19.0% pit!)
    const evCliffOnSlip = evaluatePolicyExact(TWO_TREASURE_CLIFF, patient.policy, 0.9, 0.2);
    expect(evCliffOnSlip.startValue).toBeCloseTo(2.379, 3);
    expect(evCliffOnSlip.probHazard).toBeCloseTo(0.19, 2);

    // Optimal stochastic policy at gamma = 0.90, slip = 20% detours inland: 3.5146 (97.6% Vault, 1.0% pit)
    const stoch = solveValueIteration(TWO_TREASURE_CLIFF, 0.9, 0.2);
    const evStoch = evaluatePolicyExact(TWO_TREASURE_CLIFF, stoch.policy, 0.9, 0.2);
    expect(evStoch.startValue).toBeCloseTo(3.5146, 3);
    expect(evStoch.probGoal).toBeCloseTo(0.976, 2);
    expect(evStoch.probHazard).toBeCloseTo(0.01, 2);
  });

  it('pins all 3 Policy Architect lab levels, including the Level 3 wall-bounce guardrail trick', () => {
    // Level 1: Fork
    const l1 = evaluateLabLevel(POLICY_LAB_LEVELS[0], POLICY_LAB_LEVELS[0].initialPolicy);
    expect(l1.passed).toBe(false);
    expect(l1.evalResult.startValue).toBeCloseTo(1.7, 3);
    expect(l1.optStartValue).toBeCloseTo(5.4954, 3);

    // Level 2: Slippery Bridge
    const l2 = evaluateLabLevel(POLICY_LAB_LEVELS[1], POLICY_LAB_LEVELS[1].initialPolicy);
    expect(l2.passed).toBe(false);
    expect(l2.evalResult.startValue).toBeCloseTo(2.22, 2);
    expect(l2.optStartValue).toBeCloseTo(3.1036, 3);

    // Level 3: Ice Corridor — wall-bounce at (0,2) beats moving right across the pit
    const lvl3 = POLICY_LAB_LEVELS[2];
    const l3 = evaluateLabLevel(lvl3, lvl3.initialPolicy);
    expect(l3.passed).toBe(false);
    expect(l3.optStartValue).toBeCloseTo(2.3522, 3);
    expect(l3.optimal.policy[cellIndex(lvl3.grid, 0, 2)]).toBe(0); // Up into the north wall!

    const noBounce = l3.optimal.policy.slice();
    noBounce[cellIndex(lvl3.grid, 0, 2)] = 1; // Right instead of Up
    const evNoBounce = evaluatePolicyExact(lvl3.grid, noBounce, lvl3.gamma, lvl3.slip);
    expect(evNoBounce.startValue).toBeCloseTo(2.2785, 3);
    expect(evNoBounce.probHazard).toBeCloseTo(0.185, 2);
    expect(evNoBounce.startValue).toBeGreaterThan(lvl3.targetReturn); // still passes primary goal
    expect(evNoBounce.startValue).toBeLessThan(l3.optStartValue - 0.05); // misses exact Bellman optimum
  });

  it('handles policy cycles cleanly with exact finite discounted return', () => {
    // Point (2,0) -> Right and (2,1) -> Left: deterministic infinite loop at -0.1 per step
    const loopPolicy = new Array<0 | 1 | 2 | 3 | null>(TWO_TREASURE_CLIFF.rows * TWO_TREASURE_CLIFF.cols).fill(0);
    loopPolicy[cellIndex(TWO_TREASURE_CLIFF, 2, 0)] = 1;
    loopPolicy[cellIndex(TWO_TREASURE_CLIFF, 2, 1)] = 3;
    const ev = evaluatePolicyExact(TWO_TREASURE_CLIFF, loopPolicy, 0.9, 0);
    // -0.1 / (1 - 0.9) = -1.0 exactly
    expect(ev.startValue).toBeCloseTo(-1.0, 6);
    expect(ev.probLoop).toBeCloseTo(1.0, 6);
  });
});
