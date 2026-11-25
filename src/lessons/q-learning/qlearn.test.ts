// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  Q_MAZE_GRID,
  QLAB_BRIEFS,
  createQSession,
  evaluateGreedyMazeReturn,
  evaluateQLabConfig,
  runCliffWalkingRace,
  runFairBanditRace,
  runQEpisodes,
  solveQLabBriefExhaustive,
  stepQSession,
} from './qlearn';

describe('Q-learning, Bandits & SARSA engine (q-learning)', () => {
  it('pins the fair 40-seed bandit race results quoted in prose', () => {
    const res = runFairBanditRace(200, 40);
    const byId = Object.fromEntries(res.map((r) => [r.id, r]));

    // Pure greedy gets trapped on suboptimal arms
    expect(byId.greedy.finalRegret).toBeCloseTo(94.9, 0);
    expect(byId.greedy.optimalPullPct).toBeCloseTo(29.8, 0);

    // Every exploration strategy dramatically beats pure greedy
    expect(byId['eps-greedy'].finalRegret).toBeCloseTo(47.0, 0);
    expect(byId['eps-decay'].finalRegret).toBeCloseTo(32.0, 0);
    expect(byId.optimistic.finalRegret).toBeCloseTo(31.3, 0);

    // UCB1 wins the race with lowest regret and highest optimal-arm rate
    expect(byId.ucb1.bestParam).toBe(0.8);
    expect(byId.ucb1.finalRegret).toBeCloseTo(20.2, 0);
    expect(byId.ucb1.optimalPullPct).toBeCloseTo(82.5, 0);
    expect(byId.ucb1.finalRegret).toBeLessThan(byId.optimistic.finalRegret);
  });

  it('verifies exact single-step TD error update and maze convergence', () => {
    let sess = createQSession(Q_MAZE_GRID, 0, 42);
    sess = stepQSession(sess, 0.3, 0.15);
    const u = sess.lastUpdate!;
    expect(u.target).toBeCloseTo(u.r + Q_MAZE_GRID.gamma * u.maxNextQ, 10);
    expect(u.tdError).toBeCloseTo(u.target - u.oldQ, 10);
    expect(u.newQ).toBeCloseTo(u.oldQ + 0.3 * u.tdError, 10);

    // After 50 episodes Q-learning solves the 4x5 maze to optimal 7-step return +4.846
    sess = runQEpisodes(createQSession(Q_MAZE_GRID, 0, 42), 50, 0.3, 0.15);
    const ev = evaluateGreedyMazeReturn(Q_MAZE_GRID, sess.Q);
    expect(ev.reachedGoal).toBe(true);
    expect(ev.steps).toBe(7);
    expect(ev.discountedReturn).toBeCloseTo(4.846, 2);
  });

  it('pins the fair 20-seed Cliff Walking race: Q-learning wins greedy eval, SARSA wins online safety', () => {
    const cliff = runCliffWalkingRace(0.1, 400, 20);
    expect(cliff.qLearning.bestAlpha).toBe(0.2);
    expect(cliff.sarsa.bestAlpha).toBe(0.2);

    // Q-learning learns the exact 9-step cliff edge (greedy eval -8.00) but suffers online (-19.86)
    expect(cliff.qLearning.greedyEvalMean).toBeCloseTo(-8.0, 1);
    expect(cliff.qLearning.lateOnlineMean).toBeCloseTo(-19.86, 1);

    // SARSA takes the safer inland detour: worse greedy eval (-10.40) but +5.36 better online (-14.50)
    expect(cliff.sarsa.lateOnlineMean).toBeCloseTo(-14.5, 1);
    expect(cliff.sarsa.greedyEvalMean).toBeCloseTo(-10.4, 1);
    expect(cliff.sarsa.lateOnlineMean).toBeGreaterThan(cliff.qLearning.lateOnlineMean + 4.0);
    expect(cliff.qLearning.greedyEvalMean).toBeGreaterThan(cliff.sarsa.greedyEvalMean + 1.5);
  });

  it('verifies exhaustive solution counts across all 3 Q-Learner lab briefs (36 configs each)', () => {
    const fast = solveQLabBriefExhaustive('fast-maze');
    expect(fast.totalConfigs).toBe(36);
    expect(fast.passingCount).toBe(14);
    expect(fast.bestScore).toBeCloseTo(4.846, 2);

    const safe = solveQLabBriefExhaustive('safe-cliff');
    expect(safe.totalConfigs).toBe(36);
    expect(safe.passingCount).toBe(4);
    expect(safe.bestConfig.algo).toBe('sarsa');
    expect(safe.bestScore).toBeCloseTo(-15.237, 2);

    // Verify 0 of 18 Q-learning configs pass safe-cliff
    let qPassCliff = 0;
    for (const alpha of [0.1, 0.3, 0.6]) {
      for (const epsSchedule of ['const-0.02', 'const-0.15', 'decay'] as const) {
        for (const qInit of [0, 5]) {
          if (evaluateQLabConfig('safe-cliff', { algo: 'qlearning', alpha, epsSchedule, qInit }).passed) {
            qPassCliff++;
          }
        }
      }
    }
    expect(qPassCliff).toBe(0);

    const shift = solveQLabBriefExhaustive('shifting-goal');
    expect(shift.totalConfigs).toBe(36);
    expect(shift.passingCount).toBe(20);
    expect(shift.bestScore).toBeCloseTo(9.657, 2);

    // Verify Q0=0 contrast on shifting-goal: const-0.15+alpha=0.6 passes, decay fails
    const q0Const = evaluateQLabConfig('shifting-goal', { algo: 'qlearning', alpha: 0.6, epsSchedule: 'const-0.15', qInit: 0 });
    const q0Decay = evaluateQLabConfig('shifting-goal', { algo: 'qlearning', alpha: 0.6, epsSchedule: 'decay', qInit: 0 });
    expect(q0Const.passed).toBe(true);
    expect(q0Const.score).toBeCloseTo(7.02, 1);
    expect(q0Decay.passed).toBe(false);
    expect(q0Decay.score).toBeCloseTo(3.82, 1);

    expect(QLAB_BRIEFS.length).toBe(3);
  });
});

describe('Q-learner lab machinery', () => {
  it('the learning curves are the same runs the score averages', async () => {
    const { evaluateQLabConfig: ev, qlabLayout, qlabGreedyPath } = await import('./qlearn');
    const cfg = { algo: 'sarsa' as const, alpha: 0.3, epsSchedule: 'const-0.15' as const, qInit: 0 };
    for (const id of ['safe-cliff', 'shifting-goal'] as const) {
      const e = ev(id, cfg);
      const L = qlabLayout(id);
      expect(e.curves.length).toBe(10);
      expect(e.curves[0].length).toBe(L.episodes);
      let sum = 0;
      for (const c of e.curves) sum += c.slice(L.window[0], L.window[1]).reduce((a, b) => a + b, 0) / (L.window[1] - L.window[0]);
      expect(sum / 10).toBeCloseTo(e.score, 9);
    }
    const fast = ev('fast-maze', { algo: 'qlearning', alpha: 0.6, epsSchedule: 'const-0.02', qInit: 0 });
    expect(fast.curves.length).toBe(10);
    expect(fast.curves[0].length).toBe(25);
    const p = qlabGreedyPath(qlabLayout('fast-maze'), fast.seedQ);
    expect(p.outcome).toBe('goal');
    expect(p.path[p.path.length - 1]).toEqual([0, 4]);
  });
});
