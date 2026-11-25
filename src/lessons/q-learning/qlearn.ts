// authored-by: Gemini Antigravity
import { createRng, type Rng } from '../../lib/rng';

export type Action = 0 | 1 | 2 | 3; // 0: Up, 1: Right, 2: Down, 3: Left
export const ACTIONS: readonly Action[] = [0, 1, 2, 3];
export const ACTION_NAMES = ['Up', 'Right', 'Down', 'Left'] as const;
export const ACTION_ARROWS = ['↑', '→', '↓', '←'] as const;

const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];

// ------------------------------------------------------------------ 1. Multi-Armed Bandit

export const BANDIT_MEANS = [0.4, 0.9, 1.5, 0.7, 1.1] as const;
export const BANDIT_OPTIMAL_ARM = 2; // mean = 1.5
export const BANDIT_OPTIMAL_MEAN = 1.5;

export type BanditAlgoId = 'greedy' | 'eps-greedy' | 'eps-decay' | 'optimistic' | 'ucb1';

export type BanditAlgoSummary = {
  id: BanditAlgoId;
  label: string;
  paramLabel: string;
  bestParam: number;
  meanRewardCurve: number[]; // length = steps
  cumRegretCurve: number[]; // length = steps
  avgReward: number;
  finalRegret: number;
  optimalPullPct: number;
};

function runBanditEpisode(
  algo: BanditAlgoId,
  param: number,
  steps: number,
  rng: Rng,
): { rewards: number[]; optimalFlags: number[] } {
  const k = BANDIT_MEANS.length;
  const qInit = algo === 'optimistic' ? param : 0;
  const Q = new Array<number>(k).fill(qInit);
  const N = new Array<number>(k).fill(0);
  const rewards = new Array<number>(steps);
  const optimalFlags = new Array<number>(steps);

  for (let t = 0; t < steps; t++) {
    let arm = 0;
    if (algo === 'ucb1') {
      const unpulled = N.findIndex((c) => c === 0);
      if (unpulled !== -1) {
        arm = unpulled;
      } else {
        let bestScore = -Infinity;
        for (let a = 0; a < k; a++) {
          const score = Q[a] + param * Math.sqrt(Math.log(t + 1) / N[a]);
          if (score > bestScore) {
            bestScore = score;
            arm = a;
          }
        }
      }
    } else {
      const eps =
        algo === 'greedy' || algo === 'optimistic'
          ? 0
          : algo === 'eps-greedy'
            ? param
            : 1 / (1 + t / param);
      if (eps > 0 && rng.next() < eps) {
        arm = rng.int(k);
      } else {
        // Tie-break randomly among equal max Q values
        let bestVal = -Infinity;
        let bestCount = 0;
        for (let a = 0; a < k; a++) {
          if (Q[a] > bestVal + 1e-12) {
            bestVal = Q[a];
            arm = a;
            bestCount = 1;
          } else if (Math.abs(Q[a] - bestVal) <= 1e-12) {
            bestCount++;
            if (rng.next() < 1 / bestCount) arm = a;
          }
        }
      }
    }

    const r = rng.normal(BANDIT_MEANS[arm], 1.0);
    N[arm]++;
    Q[arm] += (r - Q[arm]) / N[arm];
    rewards[t] = r;
    optimalFlags[t] = arm === BANDIT_OPTIMAL_ARM ? 1 : 0;
  }

  return { rewards, optimalFlags };
}

function evalBanditParam(
  algo: BanditAlgoId,
  param: number,
  steps: number,
  numSeeds: number,
): Omit<BanditAlgoSummary, 'id' | 'label' | 'paramLabel' | 'bestParam'> {
  const meanRewardCurve = new Array<number>(steps).fill(0);
  const cumRegretCurve = new Array<number>(steps).fill(0);
  let totalReward = 0;
  let totalOpt = 0;

  for (let seed = 1; seed <= numSeeds; seed++) {
    const rng = createRng(seed * 7919 + 17);
    const { rewards, optimalFlags } = runBanditEpisode(algo, param, steps, rng);
    let cumRegret = 0;
    for (let t = 0; t < steps; t++) {
      meanRewardCurve[t] += rewards[t] / numSeeds;
      cumRegret += BANDIT_OPTIMAL_MEAN - rewards[t];
      cumRegretCurve[t] += cumRegret / numSeeds;
      totalReward += rewards[t];
      totalOpt += optimalFlags[t];
    }
  }

  return {
    meanRewardCurve,
    cumRegretCurve,
    avgReward: totalReward / (numSeeds * steps),
    finalRegret: cumRegretCurve[steps - 1],
    optimalPullPct: (100 * totalOpt) / (numSeeds * steps),
  };
}

/**
 * Fair multi-seed bandit race (default: 40 fixed seeds × 200 pulls).
 * Each algorithm gets its own grid-searched best hyperparameter on cumulative regret.
 */
export function runFairBanditRace(steps = 200, numSeeds = 40): BanditAlgoSummary[] {
  const specs: { id: BanditAlgoId; label: string; grid: number[]; fmt: (p: number) => string }[] = [
    { id: 'greedy', label: 'Pure greedy (ε=0)', grid: [0], fmt: () => 'ε = 0' },
    { id: 'eps-greedy', label: 'ε-greedy', grid: [0.02, 0.05, 0.1, 0.15, 0.2, 0.3], fmt: (p) => `ε = ${p}` },
    { id: 'eps-decay', label: 'Decaying ε', grid: [5, 10, 20, 40, 80], fmt: (p) => `τ = ${p}` },
    { id: 'optimistic', label: 'Optimistic init', grid: [1.5, 2.0, 3.0, 4.0, 5.0], fmt: (p) => `Q₀ = ${p}` },
    { id: 'ucb1', label: 'UCB1', grid: [0.3, 0.5, 0.8, 1.0, 1.4, 2.0], fmt: (p) => `c = ${p}` },
  ];

  return specs.map((sp) => {
    let bestParam = sp.grid[0];
    let bestRes = evalBanditParam(sp.id, bestParam, steps, numSeeds);
    for (let i = 1; i < sp.grid.length; i++) {
      const cand = evalBanditParam(sp.id, sp.grid[i], steps, numSeeds);
      if (cand.finalRegret < bestRes.finalRegret) {
        bestParam = sp.grid[i];
        bestRes = cand;
      }
    }
    return {
      id: sp.id,
      label: sp.label,
      paramLabel: sp.fmt(bestParam),
      bestParam,
      ...bestRes,
    };
  });
}

// ------------------------------------------------------------------ 2. Step-by-Step Q-Learning Gridworld

export type QCellKind = 'empty' | 'wall' | 'goal' | 'hazard';

export type QGridSpec = {
  rows: number;
  cols: number;
  start: [number, number];
  stepCost: number;
  gamma: number;
  cells: { kind: QCellKind; reward: number; label?: string }[][];
};

function buildQGrid(
  rows: number,
  cols: number,
  start: [number, number],
  stepCost: number,
  gamma: number,
  specials: { r: number; c: number; kind: QCellKind; reward?: number; label?: string }[],
): QGridSpec {
  const cells: { kind: QCellKind; reward: number; label?: string }[][] = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({ kind: 'empty' as QCellKind, reward: stepCost })),
  );
  for (const sp of specials) {
    cells[sp.r][sp.c] = {
      kind: sp.kind,
      reward: sp.reward ?? (sp.kind === 'wall' ? 0 : stepCost),
      label: sp.label,
    };
  }
  return { rows, cols, start, stepCost, gamma, cells };
}

/** 4x5 Maze for interactive step-by-step Q-table exploration. */
export const Q_MAZE_GRID: QGridSpec = buildQGrid(4, 5, [3, 0], -0.1, 0.9, [
  { r: 1, c: 1, kind: 'wall' },
  { r: 2, c: 1, kind: 'wall' },
  { r: 1, c: 3, kind: 'hazard', reward: -10, label: 'Pit' },
  { r: 0, c: 4, kind: 'goal', reward: 10, label: 'Goal' },
]);

export type LastTdUpdate = {
  s: [number, number];
  a: Action;
  r: number;
  sNext: [number, number];
  terminal: boolean;
  wasExplore: boolean;
  oldQ: number;
  maxNextQ: number;
  target: number;
  tdError: number;
  newQ: number;
};

export type QSession = {
  grid: QGridSpec;
  Q: number[][]; // [rows*cols][4]
  pos: [number, number];
  episodes: number;
  totalSteps: number;
  episodeReturn: number;
  recentReturns: number[];
  lastUpdate: LastTdUpdate | null;
  rngState: number;
};

function qIdx(grid: QGridSpec, r: number, c: number): number {
  return r * grid.cols + c;
}

function chooseEpsGreedy(Qrow: number[], eps: number, rng: Rng): { action: Action; wasExplore: boolean } {
  if (rng.next() < eps) {
    return { action: rng.int(4) as Action, wasExplore: true };
  }
  let best = -Infinity;
  let bestA: Action = 0;
  let ties = 0;
  for (let a = 0; a < 4; a++) {
    if (Qrow[a] > best + 1e-12) {
      best = Qrow[a];
      bestA = a as Action;
      ties = 1;
    } else if (Math.abs(Qrow[a] - best) <= 1e-12) {
      ties++;
      if (rng.next() < 1 / ties) bestA = a as Action;
    }
  }
  return { action: bestA, wasExplore: false };
}

export function createQSession(grid: QGridSpec = Q_MAZE_GRID, qInit = 0, seed = 42): QSession {
  const n = grid.rows * grid.cols;
  return {
    grid,
    Q: Array.from({ length: n }, () => [qInit, qInit, qInit, qInit]),
    pos: [grid.start[0], grid.start[1]],
    episodes: 0,
    totalSteps: 0,
    episodeReturn: 0,
    recentReturns: [],
    lastUpdate: null,
    rngState: seed,
  };
}

export function stepQSession(session: QSession, alpha: number, eps: number): QSession {
  const { grid } = session;
  const rng = createRng(session.rngState + session.totalSteps * 31 + 7);
  const Q = session.Q.map((row) => row.slice());
  const [r, c] = session.pos;
  const sIdx = qIdx(grid, r, c);

  const { action, wasExplore } = chooseEpsGreedy(Q[sIdx], eps, rng);
  let nr = r + DR[action];
  let nc = c + DC[action];
  if (nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols || grid.cells[nr][nc].kind === 'wall') {
    nr = r;
    nc = c;
  }
  const nCell = grid.cells[nr][nc];
  const terminal = nCell.kind === 'goal' || nCell.kind === 'hazard';
  const reward = nCell.reward;
  const nIdx = qIdx(grid, nr, nc);

  const oldQ = Q[sIdx][action];
  const maxNextQ = terminal ? 0 : Math.max(Q[nIdx][0], Q[nIdx][1], Q[nIdx][2], Q[nIdx][3]);
  const target = reward + grid.gamma * maxNextQ;
  const tdError = target - oldQ;
  const newQ = oldQ + alpha * tdError;
  Q[sIdx][action] = newQ;

  const epRet = session.episodeReturn + reward;
  const nextEpisodes = terminal ? session.episodes + 1 : session.episodes;
  const nextPos: [number, number] = terminal ? [grid.start[0], grid.start[1]] : [nr, nc];
  const recentReturns = terminal
    ? [...session.recentReturns.slice(-39), epRet]
    : session.recentReturns;

  return {
    grid,
    Q,
    pos: nextPos,
    episodes: nextEpisodes,
    totalSteps: session.totalSteps + 1,
    episodeReturn: terminal ? 0 : epRet,
    recentReturns,
    lastUpdate: {
      s: [r, c],
      a: action,
      r: reward,
      sNext: [nr, nc],
      terminal,
      wasExplore,
      oldQ,
      maxNextQ,
      target,
      tdError,
      newQ,
    },
    rngState: session.rngState + 101,
  };
}

export function runQEpisodes(session: QSession, count: number, alpha: number, eps: number, maxStepsPerEp = 80): QSession {
  let cur = session;
  const targetEp = cur.episodes + count;
  let guard = 0;
  while (cur.episodes < targetEp && guard < count * maxStepsPerEp) {
    cur = stepQSession(cur, alpha, eps);
    guard++;
  }
  return cur;
}

export function evaluateGreedyMazeReturn(grid: QGridSpec, Q: number[][], maxSteps = 40): { reachedGoal: boolean; discountedReturn: number; steps: number } {
  let r = grid.start[0];
  let c = grid.start[1];
  let g = 0;
  let disc = 1;
  for (let step = 1; step <= maxSteps; step++) {
    const idx = qIdx(grid, r, c);
    let bestA: Action = 0;
    let bestQ = -Infinity;
    for (let a = 0; a < 4; a++) {
      if (Q[idx][a] > bestQ) {
        bestQ = Q[idx][a];
        bestA = a as Action;
      }
    }
    let nr = r + DR[bestA];
    let nc = c + DC[bestA];
    if (nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols || grid.cells[nr][nc].kind === 'wall') {
      nr = r;
      nc = c;
    }
    const cell = grid.cells[nr][nc];
    g += disc * cell.reward;
    disc *= grid.gamma;
    r = nr;
    c = nc;
    if (cell.kind === 'goal') return { reachedGoal: true, discountedReturn: g, steps: step };
    if (cell.kind === 'hazard') return { reachedGoal: false, discountedReturn: g, steps: step };
  }
  return { reachedGoal: false, discountedReturn: g, steps: maxSteps };
}

// ------------------------------------------------------------------ 3. Cliff Walking: Q-Learning vs SARSA

export const CLIFF_ROWS = 4;
export const CLIFF_COLS = 8;
export const CLIFF_START: [number, number] = [3, 0];
export const CLIFF_GOAL: [number, number] = [3, 7];

export function isCliffCell(r: number, c: number): boolean {
  return r === 3 && c >= 1 && c <= 6;
}

export type TdAlgo = 'qlearning' | 'sarsa';

function runCliffSeed(
  algo: TdAlgo,
  alpha: number,
  eps: number,
  episodes: number,
  rng: Rng,
  gamma = 0.99,
  qInit = 0,
): { epReturns: number[]; Q: number[][]; greedyReturn: number; greedyPath: [number, number][] } {
  const n = CLIFF_ROWS * CLIFF_COLS;
  const Q = Array.from({ length: n }, () => [qInit, qInit, qInit, qInit]);
  const epReturns = new Array<number>(episodes);

  const stepCliff = (r: number, c: number, a: Action): { nr: number; nc: number; reward: number; done: boolean } => {
    let nr = r + DR[a];
    let nc = c + DC[a];
    if (nr < 0 || nr >= CLIFF_ROWS || nc < 0 || nc >= CLIFF_COLS) {
      nr = r;
      nc = c;
    }
    if (isCliffCell(nr, nc)) {
      return { nr: CLIFF_START[0], nc: CLIFF_START[1], reward: -50, done: false };
    }
    if (nr === CLIFF_GOAL[0] && nc === CLIFF_GOAL[1]) {
      return { nr, nc, reward: 0, done: true };
    }
    return { nr, nc, reward: -1, done: false };
  };

  for (let ep = 0; ep < episodes; ep++) {
    let r = CLIFF_START[0];
    let c = CLIFF_START[1];
    let sIdx = r * CLIFF_COLS + c;
    let a = chooseEpsGreedy(Q[sIdx], eps, rng).action;
    let totalR = 0;

    for (let step = 0; step < 120; step++) {
      const { nr, nc, reward, done } = stepCliff(r, c, a);
      totalR += reward;
      const nIdx = nr * CLIFF_COLS + nc;
      const nextAction = chooseEpsGreedy(Q[nIdx], eps, rng).action;

      const bootstrap = done
        ? 0
        : algo === 'qlearning'
          ? Math.max(Q[nIdx][0], Q[nIdx][1], Q[nIdx][2], Q[nIdx][3])
          : Q[nIdx][nextAction];

      Q[sIdx][a] += alpha * (reward + gamma * bootstrap - Q[sIdx][a]);

      r = nr;
      c = nc;
      sIdx = nIdx;
      a = nextAction;
      if (done) break;
    }
    epReturns[ep] = totalR;
  }

  // Greedy evaluation rollout
  let gr = CLIFF_START[0];
  let gc = CLIFF_START[1];
  let greedyReturn = 0;
  const greedyPath: [number, number][] = [[gr, gc]];
  const visited = new Set<number>([gr * CLIFF_COLS + gc]);
  for (let s = 0; s < 40; s++) {
    const idx = gr * CLIFF_COLS + gc;
    let bestA: Action = 0;
    let bestQ = -Infinity;
    for (let a = 0; a < 4; a++) {
      if (Q[idx][a] > bestQ) {
        bestQ = Q[idx][a];
        bestA = a as Action;
      }
    }
    const { nr, nc, reward, done } = stepCliff(gr, gc, bestA);
    greedyReturn += reward;
    gr = nr;
    gc = nc;
    greedyPath.push([gr, gc]);
    if (done) break;
    const nIdx = gr * CLIFF_COLS + gc;
    if (visited.has(nIdx)) {
      greedyReturn = -40;
      break;
    }
    visited.add(nIdx);
  }

  return { epReturns, Q, greedyReturn, greedyPath };
}

export type CliffRaceResult = {
  qLearning: {
    bestAlpha: number;
    onlineCurve: number[]; // smoothed or per-episode average over seeds
    lateOnlineMean: number; // episodes 150..399
    greedyEvalMean: number;
    samplePolicy: Action[];
    samplePath: [number, number][];
  };
  sarsa: {
    bestAlpha: number;
    onlineCurve: number[];
    lateOnlineMean: number;
    greedyEvalMean: number;
    samplePolicy: Action[];
    samplePath: [number, number][];
  };
};

/**
 * Fair Q-learning vs SARSA race on 4x8 Cliff Walking (ε = 0.10, 400 episodes, 20 fixed seeds).
 * Each algorithm uses its own grid-searched best α over [0.1, 0.2, 0.3, 0.5, 0.7] on late online return.
 */
export function runCliffWalkingRace(eps = 0.1, episodes = 400, numSeeds = 20): CliffRaceResult {
  const alphaGrid = [0.1, 0.2, 0.3, 0.5, 0.7];

  const evalAlgo = (algo: TdAlgo) => {
    let bestAlpha = alphaGrid[0];
    let bestScore = -Infinity;
    let bestCurve: number[] = [];
    let bestGreedy = -Infinity;
    let samplePolicy: Action[] = [];
    let samplePath: [number, number][] = [];

    for (const alpha of alphaGrid) {
      const curve = new Array<number>(episodes).fill(0);
      let greedySum = 0;
      let sPol: Action[] = [];
      let sPath: [number, number][] = [];

      for (let seed = 1; seed <= numSeeds; seed++) {
        const res = runCliffSeed(algo, alpha, eps, episodes, createRng(seed * 3571 + 13));
        for (let e = 0; e < episodes; e++) curve[e] += res.epReturns[e] / numSeeds;
        greedySum += res.greedyReturn;
        if (seed === 1) {
          sPol = res.Q.map((row) => {
            let b = 0;
            for (let a = 1; a < 4; a++) if (row[a] > row[b]) b = a;
            return b as Action;
          });
          sPath = res.greedyPath;
        }
      }

      const startLate = Math.floor(episodes * 0.4);
      let lateSum = 0;
      for (let e = startLate; e < episodes; e++) lateSum += curve[e];
      const lateMean = lateSum / (episodes - startLate);

      if (lateMean > bestScore) {
        bestScore = lateMean;
        bestAlpha = alpha;
        bestCurve = curve;
        bestGreedy = greedySum / numSeeds;
        samplePolicy = sPol;
        samplePath = sPath;
      }
    }

    return {
      bestAlpha,
      onlineCurve: bestCurve,
      lateOnlineMean: bestScore,
      greedyEvalMean: bestGreedy,
      samplePolicy,
      samplePath,
    };
  };

  return {
    qLearning: evalAlgo('qlearning'),
    sarsa: evalAlgo('sarsa'),
  };
}

// ------------------------------------------------------------------ 4. Lab: Tune the Q-Learner

export type LabEpsSchedule = 'const-0.02' | 'const-0.15' | 'decay';

export type QLabConfig = {
  algo: TdAlgo;
  alpha: number;
  epsSchedule: LabEpsSchedule;
  qInit: number;
};

export const QLAB_ALGOS: readonly TdAlgo[] = ['qlearning', 'sarsa'];
export const QLAB_ALPHAS: readonly number[] = [0.1, 0.3, 0.6];
export const QLAB_EPS: readonly LabEpsSchedule[] = ['const-0.02', 'const-0.15', 'decay'];
export const QLAB_QINITS: readonly number[] = [0, 5];

export type QLabBriefId = 'fast-maze' | 'safe-cliff' | 'shifting-goal';

export type QLabBrief = {
  id: QLabBriefId;
  title: string;
  description: string;
  metricLabel: string;
  target: number;
  hint: string;
};

export const QLAB_BRIEFS: QLabBrief[] = [
  {
    id: 'fast-maze',
    title: 'Brief 1 · Fast Maze Convergence',
    description:
      '4×5 maze with a wall and −10 pit, tight budget of only 25 training episodes (averaged over 10 seeds). Tune the agent to reach greedy discounted evaluation return ≥ +4.30.',
    metricLabel: 'Greedy eval V(S)',
    target: 4.3,
    hint: 'With only 25 episodes, a tiny learning rate α = 0.1 propagates rewards too slowly from the Goal back to Start. Pair a responsive α with optimistic init (Q₀ = +5) or active exploration.',
  },
  {
    id: 'safe-cliff',
    title: 'Brief 2 · Safe Online Cliff Learning',
    description:
      '4×8 Cliff Walking with a mandatory 10% safety floor on exploration (ε ≥ 0.10, 250 episodes, 10 seeds). Beat average ONLINE return ≥ −17.0 during episodes 101–250.',
    metricLabel: 'Online return (ep 101–250)',
    target: -17.0,
    hint: 'Because exploration stays at least 10%, off-policy Q-learning keeps hugging the cliff and falling in (−50!). Switch to on-policy SARSA so the agent prices in its own exploration risk.',
  },
  {
    id: 'shifting-goal',
    title: 'Brief 3 · Shifting Goal (Non-Stationary)',
    description:
      'At episode 100, the old Goal at (0,4) shrinks from +10 down to +2 while a new +10 Vault opens at (3,4)! Achieve average online return ≥ +5.50 across episodes 141–220 (10 seeds).',
    metricLabel: 'Post-shift return (ep 141–220)',
    target: 5.5,
    hint: 'With Q₀ = 0, decaying ε or tiny ε = 0.02 gets trapped on the downgraded +2 corner — you need constant ε = 0.15 AND a responsive α = 0.6 to rediscover the new +10 Vault. (Or try optimistic init Q₀ = +5 to keep unvisited paths enticing!)',
  },
];

function epsForSchedule(sched: LabEpsSchedule, ep: number, minEps = 0): number {
  let e: number;
  if (sched === 'const-0.02') e = 0.02;
  else if (sched === 'const-0.15') e = 0.15;
  else e = Math.max(0.005, 0.6 * Math.exp(-ep / 18));
  return Math.max(minEps, e);
}

export type QLabEval = {
  score: number;
  secondaryLabel: string;
  secondaryValue: string;
  passed: boolean;
};

export function evaluateQLabConfig(briefId: QLabBriefId, cfg: QLabConfig): QLabEval {
  const numSeeds = 10;

  if (briefId === 'fast-maze') {
    let evalSum = 0;
    let solvedCount = 0;
    const grid = Q_MAZE_GRID;
    const n = grid.rows * grid.cols;

    for (let seed = 1; seed <= numSeeds; seed++) {
      const rng = createRng(seed * 4127 + 9);
      const Q = Array.from({ length: n }, () => [cfg.qInit, cfg.qInit, cfg.qInit, cfg.qInit]);
      for (let ep = 0; ep < 25; ep++) {
        const eps = epsForSchedule(cfg.epsSchedule, ep);
        let r = grid.start[0];
        let c = grid.start[1];
        let sIdx = qIdx(grid, r, c);
        let a = chooseEpsGreedy(Q[sIdx], eps, rng).action;

        for (let step = 0; step < 60; step++) {
          let nr = r + DR[a];
          let nc = c + DC[a];
          if (nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols || grid.cells[nr][nc].kind === 'wall') {
            nr = r;
            nc = c;
          }
          const cell = grid.cells[nr][nc];
          const done = cell.kind === 'goal' || cell.kind === 'hazard';
          const nIdx = qIdx(grid, nr, nc);
          const nextA = chooseEpsGreedy(Q[nIdx], eps, rng).action;
          const boot = done
            ? 0
            : cfg.algo === 'qlearning'
              ? Math.max(Q[nIdx][0], Q[nIdx][1], Q[nIdx][2], Q[nIdx][3])
              : Q[nIdx][nextA];
          Q[sIdx][a] += cfg.alpha * (cell.reward + grid.gamma * boot - Q[sIdx][a]);
          r = nr;
          c = nc;
          sIdx = nIdx;
          a = nextA;
          if (done) break;
        }
      }
      const ev = evaluateGreedyMazeReturn(grid, Q);
      evalSum += ev.discountedReturn;
      if (ev.reachedGoal) solvedCount++;
    }
    const score = evalSum / numSeeds;
    return {
      score,
      secondaryLabel: 'Seeds reaching Goal',
      secondaryValue: `${solvedCount}/${numSeeds}`,
      passed: score >= 4.3,
    };
  }

  if (briefId === 'safe-cliff') {
    let lateSum = 0;
    let greedySum = 0;
    for (let seed = 1; seed <= numSeeds; seed++) {
      const rng = createRng(seed * 2897 + 5);
      // Enforced minimum exploration 0.10 on this safety brief
      const eps = cfg.epsSchedule === 'const-0.15' ? 0.15 : 0.1;
      const res = runCliffSeed(cfg.algo, cfg.alpha, eps, 250, rng, 0.99, cfg.qInit);
      let s = 0;
      for (let ep = 100; ep < 250; ep++) s += res.epReturns[ep];
      lateSum += s / 150;
      greedySum += res.greedyReturn;
    }
    const score = lateSum / numSeeds;
    return {
      score,
      secondaryLabel: 'Greedy path return',
      secondaryValue: (greedySum / numSeeds).toFixed(1),
      passed: score >= -17.0,
    };
  }

  // Brief 3: Shifting Goal (at ep 100, old Goal (0,4) shrinks +10 -> +2 and new +10 opens at (3,4))
  let postSum = 0;
  let preSum = 0;
  const rows = 4;
  const cols = 5;
  const n = rows * cols;
  for (let seed = 1; seed <= numSeeds; seed++) {
    const rng = createRng(seed * 5381 + 21);
    const Q = Array.from({ length: n }, () => [cfg.qInit, cfg.qInit, cfg.qInit, cfg.qInit]);
    for (let ep = 0; ep < 220; ep++) {
      const shifted = ep >= 100;
      const eps = epsForSchedule(cfg.epsSchedule, ep);
      let r = 3;
      let c = 0;
      let sIdx = r * cols + c;
      let a = chooseEpsGreedy(Q[sIdx], eps, rng).action;
      let epRet = 0;

      for (let step = 0; step < 50; step++) {
        let nr = r + DR[a];
        let nc = c + DC[a];
        const isWall = (nr === 1 && nc === 2) || (nr === 2 && nc === 2);
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols || isWall) {
          nr = r;
          nc = c;
        }
        const hitOldGoal = nr === 0 && nc === 4;
        const hitNewGoal = shifted && nr === 3 && nc === 4;
        const done = hitOldGoal || hitNewGoal;
        const reward = hitNewGoal ? 10 : hitOldGoal ? (shifted ? 2 : 10) : -0.1;
        epRet += reward;
        const nIdx = nr * cols + nc;
        const nextA = chooseEpsGreedy(Q[nIdx], eps, rng).action;
        const boot = done
          ? 0
          : cfg.algo === 'qlearning'
            ? Math.max(Q[nIdx][0], Q[nIdx][1], Q[nIdx][2], Q[nIdx][3])
            : Q[nIdx][nextA];
        Q[sIdx][a] += cfg.alpha * (reward + 0.9 * boot - Q[sIdx][a]);
        r = nr;
        c = nc;
        sIdx = nIdx;
        a = nextA;
        if (done) break;
      }
      if (ep >= 60 && ep < 100) preSum += epRet / 40;
      if (ep >= 140 && ep < 220) postSum += epRet / 80;
    }
  }
  const score = postSum / numSeeds;
  return {
    score,
    secondaryLabel: 'Pre-shift return (ep 61–100)',
    secondaryValue: (preSum / numSeeds).toFixed(2),
    passed: score >= 5.5,
  };
}

export type QLabGridSummary = {
  totalConfigs: number;
  passingCount: number;
  bestConfig: QLabConfig;
  bestScore: number;
};

/** Exhaustive evaluation over all 2 × 3 × 3 × 2 = 36 grid configurations for a brief. */
export function solveQLabBriefExhaustive(briefId: QLabBriefId): QLabGridSummary {
  let totalConfigs = 0;
  let passingCount = 0;
  let bestConfig: QLabConfig = { algo: 'qlearning', alpha: 0.3, epsSchedule: 'const-0.15', qInit: 0 };
  let bestScore = -Infinity;

  for (const algo of QLAB_ALGOS) {
    for (const alpha of QLAB_ALPHAS) {
      for (const epsSchedule of QLAB_EPS) {
        for (const qInit of QLAB_QINITS) {
          const cfg: QLabConfig = { algo, alpha, epsSchedule, qInit };
          const ev = evaluateQLabConfig(briefId, cfg);
          totalConfigs++;
          if (ev.passed) passingCount++;
          if (ev.score > bestScore) {
            bestScore = ev.score;
            bestConfig = cfg;
          }
        }
      }
    }
  }

  return { totalConfigs, passingCount, bestConfig, bestScore };
}
