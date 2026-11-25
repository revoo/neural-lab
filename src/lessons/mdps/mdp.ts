// authored-by: Gemini Antigravity

export type Action = 0 | 1 | 2 | 3; // 0: Up, 1: Right, 2: Down, 3: Left

export const ACTIONS: readonly Action[] = [0, 1, 2, 3];
export const ACTION_NAMES = ['Up', 'Right', 'Down', 'Left'] as const;
export const ACTION_ARROWS = ['↑', '→', '↓', '←'] as const;

const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];

export type CellKind = 'empty' | 'wall' | 'goal' | 'near' | 'hazard';

export type CellSpec = {
  kind: CellKind;
  /** Reward granted upon transitioning into this cell. Ignored for walls. */
  reward: number;
  label?: string;
};

export type GridMdp = {
  id: string;
  name: string;
  rows: number;
  cols: number;
  start: [number, number];
  stepCost: number;
  cells: CellSpec[][];
};

export function isTerminal(cell: CellSpec): boolean {
  return cell.kind === 'goal' || cell.kind === 'near' || cell.kind === 'hazard';
}

export function cellIndex(grid: GridMdp, r: number, c: number): number {
  return r * grid.cols + c;
}

export function indexToRC(grid: GridMdp, idx: number): [number, number] {
  return [Math.floor(idx / grid.cols), idx % grid.cols];
}

export type TransitionOutcome = {
  nextIdx: number;
  r: number;
  c: number;
  prob: number;
  reward: number;
  terminal: boolean;
  label: string;
};

/**
 * Exact transition distribution P(s' | s, a) for a non-terminal, non-wall cell.
 * Intended direction has probability (1 - slip); 90° left and 90° right each have slip / 2.
 * Bumping a wall or boundary keeps the agent in state s (still paying stepCost).
 * Duplicate landing states are merged so outcomes sum to 1 with unique nextIdx.
 */
export function getTransitions(grid: GridMdp, idx: number, action: Action, slip: number): TransitionOutcome[] {
  const [r, c] = indexToRC(grid, idx);
  const cell = grid.cells[r][c];
  if (cell.kind === 'wall' || isTerminal(cell)) return [];

  const leftA = ((action + 3) % 4) as Action;
  const rightA = ((action + 1) % 4) as Action;
  const raw: { a: Action; prob: number; dirLabel: string }[] =
    slip <= 0
      ? [{ a: action, prob: 1, dirLabel: ACTION_NAMES[action] }]
      : [
          { a: action, prob: 1 - slip, dirLabel: `${ACTION_NAMES[action]} (intended)` },
          { a: leftA, prob: slip / 2, dirLabel: `${ACTION_NAMES[leftA]} (slip)` },
          { a: rightA, prob: slip / 2, dirLabel: `${ACTION_NAMES[rightA]} (slip)` },
        ];

  const merged = new Map<number, TransitionOutcome>();
  for (const item of raw) {
    if (item.prob <= 0) continue;
    let nr = r + DR[item.a];
    let nc = c + DC[item.a];
    let bounced = false;
    if (nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols || grid.cells[nr][nc].kind === 'wall') {
      nr = r;
      nc = c;
      bounced = true;
    }
    const nIdx = cellIndex(grid, nr, nc);
    const nCell = grid.cells[nr][nc];
    const existing = merged.get(nIdx);
    const lbl = bounced ? `stay (${r},${c})` : `(${nr},${nc})`;
    if (existing) {
      existing.prob += item.prob;
    } else {
      merged.set(nIdx, {
        nextIdx: nIdx,
        r: nr,
        c: nc,
        prob: item.prob,
        reward: nCell.reward,
        terminal: isTerminal(nCell),
        label: lbl,
      });
    }
  }
  return Array.from(merged.values());
}

export type ActionBackup = {
  action: Action;
  q: number;
  outcomes: (TransitionOutcome & { futureV: number; contribution: number })[];
};

export type BellmanInspection = {
  idx: number;
  r: number;
  c: number;
  actions: ActionBackup[];
  bestAction: Action;
  bestQ: number;
};

/** Exact Bellman backup breakdown for one cell given current value table V. */
export function inspectBellmanBackup(
  grid: GridMdp,
  V: number[],
  idx: number,
  gamma: number,
  slip: number,
): BellmanInspection {
  const [r, c] = indexToRC(grid, idx);
  const actions: ActionBackup[] = ACTIONS.map((action) => {
    const trans = getTransitions(grid, idx, action, slip);
    let q = 0;
    const outcomes = trans.map((t) => {
      const futureV = t.terminal ? 0 : V[t.nextIdx];
      const contribution = t.prob * (t.reward + gamma * futureV);
      q += contribution;
      return { ...t, futureV, contribution };
    });
    return { action, q, outcomes };
  });

  let bestAction: Action = 0;
  let bestQ = -Infinity;
  for (const a of actions) {
    if (a.q > bestQ + 1e-12) {
      bestQ = a.q;
      bestAction = a.action;
    }
  }
  return { idx, r, c, actions, bestAction, bestQ };
}

export type ValueIterationSnapshot = {
  sweep: number;
  V: number[];
  policy: (Action | null)[];
  maxDelta: number;
};

export function valueIterationStep(
  grid: GridMdp,
  V: number[],
  gamma: number,
  slip: number,
): { nextV: number[]; policy: (Action | null)[]; Q: number[][]; maxDelta: number } {
  const n = grid.rows * grid.cols;
  const nextV = new Array<number>(n).fill(0);
  const policy = new Array<Action | null>(n).fill(null);
  const Q: number[][] = Array.from({ length: n }, () => [0, 0, 0, 0]);
  let maxDelta = 0;

  for (let idx = 0; idx < n; idx++) {
    const [r, c] = indexToRC(grid, idx);
    const cell = grid.cells[r][c];
    if (cell.kind === 'wall' || isTerminal(cell)) {
      nextV[idx] = 0;
      policy[idx] = null;
      continue;
    }
    const insp = inspectBellmanBackup(grid, V, idx, gamma, slip);
    for (const ab of insp.actions) Q[idx][ab.action] = ab.q;
    nextV[idx] = insp.bestQ;
    policy[idx] = insp.bestAction;
    const delta = Math.abs(nextV[idx] - V[idx]);
    if (delta > maxDelta) maxDelta = delta;
  }

  return { nextV, policy, Q, maxDelta };
}

export type ValueIterationResult = {
  V: number[];
  Q: number[][];
  policy: (Action | null)[];
  iterations: number;
  snapshots: ValueIterationSnapshot[];
};

/** Exact synchronous Value Iteration to tolerance 1e-10. */
export function solveValueIteration(
  grid: GridMdp,
  gamma: number,
  slip: number,
  tol = 1e-10,
  maxIter = 500,
  keepSnapshots = false,
): ValueIterationResult {
  const n = grid.rows * grid.cols;
  let V = new Array<number>(n).fill(0);
  let policy: (Action | null)[] = new Array<Action | null>(n).fill(null);
  let Q: number[][] = Array.from({ length: n }, () => [0, 0, 0, 0]);
  const snapshots: ValueIterationSnapshot[] = keepSnapshots
    ? [{ sweep: 0, V: V.slice(), policy: policy.slice(), maxDelta: 0 }]
    : [];

  let iter = 0;
  for (let k = 1; k <= maxIter; k++) {
    const step = valueIterationStep(grid, V, gamma, slip);
    V = step.nextV;
    policy = step.policy;
    Q = step.Q;
    iter = k;
    if (keepSnapshots && (k <= 30 || k % 5 === 0 || step.maxDelta < tol)) {
      snapshots.push({ sweep: k, V: V.slice(), policy: policy.slice(), maxDelta: step.maxDelta });
    }
    if (step.maxDelta < tol) break;
  }

  return { V, Q, policy, iterations: iter, snapshots };
}

/**
 * Gaussian elimination with partial pivoting to solve A x = b.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(M[row][col]);
      if (val > best) {
        best = val;
        pivot = row;
      }
    }
    if (best < 1e-14) continue;
    if (pivot !== col) {
      const tmp = M[col];
      M[col] = M[pivot];
      M[pivot] = tmp;
    }
    const diag = M[col][col];
    for (let j = col; j <= n; j++) M[col][j] /= diag;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = M[row][col];
      if (f === 0) continue;
      for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row[n]);
}

export type PolicyEvaluation = {
  V: number[];
  startValue: number;
  /** Exact absorption probabilities from start state (computed via Markov absorption / power iteration). */
  probGoal: number;
  probNear: number;
  probHazard: number;
  probLoop: number;
  /** Greedy trajectory (following intended direction, up to 40 steps) for visual highlighting. */
  greedyPath: number[];
};

/**
 * Exact policy evaluation solving (I - gamma * P^pi) V^pi = R^pi,
 * plus exact terminal outcome probabilities from grid.start.
 */
export function evaluatePolicyExact(
  grid: GridMdp,
  policy: (Action | null)[],
  gamma: number,
  slip: number,
): PolicyEvaluation {
  const n = grid.rows * grid.cols;
  const A: number[][] = Array.from({ length: n }, (_, i) => {
    const row = new Array<number>(n).fill(0);
    row[i] = 1;
    return row;
  });
  const b = new Array<number>(n).fill(0);

  for (let idx = 0; idx < n; idx++) {
    const [r, c] = indexToRC(grid, idx);
    const cell = grid.cells[r][c];
    if (cell.kind === 'wall' || isTerminal(cell)) continue;
    const act = policy[idx];
    if (act === null || act === undefined) continue;
    const trans = getTransitions(grid, idx, act, slip);
    for (const t of trans) {
      b[idx] += t.prob * t.reward;
      if (!t.terminal) {
        A[idx][t.nextIdx] -= gamma * t.prob;
      }
    }
  }

  const V = solveLinearSystem(A, b);
  const startIdx = cellIndex(grid, grid.start[0], grid.start[1]);

  // Exact absorption probabilities via 400 steps of Markov distribution propagation
  let dist = new Array<number>(n).fill(0);
  dist[startIdx] = 1;
  let probGoal = 0;
  let probNear = 0;
  let probHazard = 0;
  for (let step = 0; step < 400; step++) {
    const nextDist = new Array<number>(n).fill(0);
    let active = 0;
    for (let idx = 0; idx < n; idx++) {
      const p = dist[idx];
      if (p < 1e-14) continue;
      const act = policy[idx];
      if (act === null || act === undefined) continue;
      const trans = getTransitions(grid, idx, act, slip);
      for (const t of trans) {
        const mass = p * t.prob;
        if (t.terminal) {
          const k = grid.cells[t.r][t.c].kind;
          if (k === 'goal') probGoal += mass;
          else if (k === 'near') probNear += mass;
          else if (k === 'hazard') probHazard += mass;
        } else {
          nextDist[t.nextIdx] += mass;
          active += mass;
        }
      }
    }
    dist = nextDist;
    if (active < 1e-12) break;
  }
  const probLoop = Math.max(0, 1 - probGoal - probNear - probHazard);

  // Intended-direction trace from start (for visual path highlight)
  const greedyPath: number[] = [startIdx];
  const seen = new Set<number>([startIdx]);
  let cur = startIdx;
  for (let s = 0; s < 40; s++) {
    const act = policy[cur];
    if (act === null || act === undefined) break;
    const [r, c] = indexToRC(grid, cur);
    const nr = r + DR[act];
    const nc = c + DC[act];
    if (nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols || grid.cells[nr][nc].kind === 'wall') break;
    const nextIdx = cellIndex(grid, nr, nc);
    greedyPath.push(nextIdx);
    if (isTerminal(grid.cells[nr][nc]) || seen.has(nextIdx)) break;
    seen.add(nextIdx);
    cur = nextIdx;
  }

  return {
    V,
    startValue: V[startIdx],
    probGoal,
    probNear,
    probHazard,
    probLoop,
    greedyPath,
  };
}

// ------------------------------------------------------------------ Grid builders

function makeGrid(
  id: string,
  name: string,
  rows: number,
  cols: number,
  start: [number, number],
  stepCost: number,
  specials: { r: number; c: number; kind: CellKind; reward?: number; label?: string }[],
): GridMdp {
  const cells: CellSpec[][] = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({ kind: 'empty' as CellKind, reward: stepCost })),
  );
  for (const sp of specials) {
    cells[sp.r][sp.c] = {
      kind: sp.kind,
      reward: sp.reward ?? (sp.kind === 'wall' ? 0 : stepCost),
      label: sp.label,
    };
  }
  return { id, name, rows, cols, start, stepCost, cells };
}

/**
 * Primary teaching map (6x4): "The Two-Treasure Cliff".
 * - Start at (2, 0).
 * - Near Gem (+2) at (3, 1): 2 steps from Start.
 * - Three Hazard Pits (-10) along Row 3: (3,2), (3,3), (3,4).
 * - Big Treasure (+10) at (2, 5): 5 steps straight along the cliff edge (Row 2),
 *   or 7 steps via the safe inland road (Row 1).
 */
export const TWO_TREASURE_CLIFF: GridMdp = makeGrid(
  'two-treasure-cliff',
  'The Two-Treasure Cliff',
  4,
  6,
  [2, 0],
  -0.1,
  [
    { r: 3, c: 1, kind: 'near', reward: 2, label: 'Gem' },
    { r: 3, c: 2, kind: 'hazard', reward: -10, label: 'Pit' },
    { r: 3, c: 3, kind: 'hazard', reward: -10, label: 'Pit' },
    { r: 3, c: 4, kind: 'hazard', reward: -10, label: 'Pit' },
    { r: 2, c: 5, kind: 'goal', reward: 10, label: 'Vault' },
  ],
);

/**
 * Computes the exact critical discount factor gamma* where the deterministic optimal policy
 * on TWO_TREASURE_CLIFF switches from the 2-step Near Gem (+2) to the 5-step Vault (+10).
 * At slip = 0:
 *   Near Gem return = -0.1 + 2 * gamma
 *   Vault return    = -0.1 * (1 + gamma + gamma^2 + gamma^3) + 10 * gamma^4
 */
export function findCriticalGamma(grid: GridMdp = TWO_TREASURE_CLIFF, slip = 0): number {
  let lo = 0.0;
  let hi = 0.99;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    const sol = solveValueIteration(grid, mid, slip);
    const ev = evaluatePolicyExact(grid, sol.policy, mid, slip);
    if (ev.probGoal > 0.5) hi = mid;
    else lo = mid;
  }
  return (lo + hi) / 2;
}

// ------------------------------------------------------------------ Lab: Wire the Policy

export type PolicyLabLevel = {
  id: string;
  title: string;
  brief: string;
  hint: string;
  grid: GridMdp;
  gamma: number;
  slip: number;
  /** Target expected discounted return V^pi(start) to pass the primary objective. */
  targetReturn: number;
  /** Default starting policy (a naive or broken policy the learner improves). */
  initialPolicy: (Action | null)[];
};

function policyFromArrows(grid: GridMdp, rows: string[]): (Action | null)[] {
  const map: Record<string, Action> = { U: 0, R: 1, D: 2, L: 3 };
  const out = new Array<Action | null>(grid.rows * grid.cols).fill(null);
  for (let r = 0; r < grid.rows; r++) {
    for (let c = 0; c < grid.cols; c++) {
      const ch = rows[r][c];
      if (ch in map && grid.cells[r][c].kind === 'empty') {
        out[cellIndex(grid, r, c)] = map[ch];
      }
    }
  }
  return out;
}

// Level 1: The Two-Gem Fork (deterministic, beat myopia)
const LAB_GRID_1 = makeGrid(
  'lab-fork',
  'The Two-Gem Fork',
  3,
  5,
  [1, 0],
  -0.1,
  [
    { r: 2, c: 1, kind: 'near', reward: 2, label: 'Gem' },
    { r: 1, c: 2, kind: 'wall' },
    { r: 2, c: 3, kind: 'hazard', reward: -8, label: 'Pit' },
    { r: 1, c: 4, kind: 'goal', reward: 10, label: 'Vault' },
  ],
);

// Level 2: The Slippery Bridge (stochastic, safe detour beats risky shortcut)
const LAB_GRID_2 = makeGrid(
  'lab-bridge',
  'The Slippery Bridge',
  4,
  5,
  [2, 0],
  -0.1,
  [
    { r: 1, c: 1, kind: 'wall' },
    { r: 1, c: 3, kind: 'wall' },
    { r: 3, c: 1, kind: 'hazard', reward: -10, label: 'Pit' },
    { r: 3, c: 2, kind: 'hazard', reward: -10, label: 'Pit' },
    { r: 3, c: 3, kind: 'hazard', reward: -10, label: 'Pit' },
    { r: 2, c: 4, kind: 'goal', reward: 10, label: 'Vault' },
  ],
);

// Level 3: The Guardrail Wall (high slip 30%: aiming safely vs hugging hazards)
const LAB_GRID_3 = makeGrid(
  'lab-guardrail',
  'The Ice Corridor',
  4,
  5,
  [2, 0],
  -0.08,
  [
    { r: 1, c: 2, kind: 'hazard', reward: -12, label: 'Pit' },
    { r: 3, c: 1, kind: 'hazard', reward: -12, label: 'Pit' },
    { r: 3, c: 2, kind: 'hazard', reward: -12, label: 'Pit' },
    { r: 3, c: 3, kind: 'hazard', reward: -12, label: 'Pit' },
    { r: 0, c: 4, kind: 'goal', reward: 10, label: 'Vault' },
  ],
);

export const POLICY_LAB_LEVELS: PolicyLabLevel[] = [
  {
    id: 'fork',
    title: 'Level 1 · Beat the Near-Term Trap',
    brief:
      'Discount γ = 0.90, deterministic (0% slip). The initial policy grabs the nearby +2 gem. Rewire the arrows to route around the central wall and reach the +10 Vault with expected return V(S) ≥ 5.20.',
    hint: 'Click cells along Row 0 to build a clean bridge over the wall at (1,2), then drop down into the Vault at (1,4).',
    grid: LAB_GRID_1,
    gamma: 0.9,
    slip: 0.0,
    targetReturn: 5.2,
    initialPolicy: policyFromArrows(LAB_GRID_1, [
      'RRRRD',
      'DR#RD',
      'R.R.U',
    ]),
  },
  {
    id: 'bridge',
    title: 'Level 2 · The Slippery Bridge',
    brief:
      'Discount γ = 0.90, 20% slip. The initial policy sprints straight across Row 2 along three −10 pits (V(S) = 2.22, 25.7% pit rate). Rewire the arrows to push V(S) ≥ 2.90 — and see if you can match the exact Bellman optimum!',
    hint: 'Each step across Row 2 has a 10% chance of slipping south into a −10 pit. Route north up Row 0 above the two wall pillars, and make sure every recovery cell points safely back onto the road.',
    grid: LAB_GRID_2,
    gamma: 0.9,
    slip: 0.2,
    targetReturn: 2.9,
    initialPolicy: policyFromArrows(LAB_GRID_2, [
      'RRRRD',
      'U#U#D',
      'RRRR.',
      'U...U',
    ]),
  },
  {
    id: 'guardrail',
    title: 'Level 3 · The Ice Corridor',
    brief:
      'Discount γ = 0.92, 30% slip! A central −12 pit at (1,2) guards the middle and three −12 pits line the bottom. Wire every cell to beat V(S) ≥ 2.10 — then hunt for a sneaky stochastic trick to reach the exact Bellman optimum (2.352)!',
    hint: 'Go straight north to Row 0 so the top wall acts as a guardrail. To beat 2.28 and hit the true Bellman optimum 2.352, look closely at cell (0,2) directly above the −12 pit: what happens if you deliberately aim ↑ into the north wall instead of → across the gap?',
    grid: LAB_GRID_3,
    gamma: 0.92,
    slip: 0.3,
    targetReturn: 2.1,
    initialPolicy: policyFromArrows(LAB_GRID_3, [
      'RRRR.',
      'RR.RD',
      'RRRRU',
      'U...U',
    ]),
  },
];

export type LabEvaluation = {
  evalResult: PolicyEvaluation;
  optimal: ValueIterationResult;
  optStartValue: number;
  passed: boolean;
  isOptimal: boolean;
};

export function evaluateLabLevel(level: PolicyLabLevel, policy: (Action | null)[]): LabEvaluation {
  const optimal = solveValueIteration(level.grid, level.gamma, level.slip);
  const startIdx = cellIndex(level.grid, level.grid.start[0], level.grid.start[1]);
  const optStartValue = optimal.V[startIdx];
  const evalResult = evaluatePolicyExact(level.grid, policy, level.gamma, level.slip);
  return {
    evalResult,
    optimal,
    optStartValue,
    passed: evalResult.startValue >= level.targetReturn - 1e-6,
    isOptimal: evalResult.startValue >= optStartValue - 1e-3,
  };
}

// ------------------------------------------------------------------ Sampled episodes (the lab's "Run 20 episodes")

export type SampledEpisode = {
  /** Cells visited, start first. */
  cells: number[];
  /** slipped[t] is true when step t went sideways instead of the intended way. */
  slipped: boolean[];
  outcome: 'goal' | 'near' | 'hazard' | 'timeout';
  /** Undiscounted reward collected. */
  reward: number;
};

/**
 * Play the policy `n` times from the start with real slips: each step goes the
 * intended way with probability 1 − slip, else 90° left or right (slip/2 each),
 * exactly as getTransitions describes. Seeded, so a run can be replayed.
 */
export function sampleEpisodes(grid: GridMdp, policy: (Action | null)[], slip: number, n: number, seed: number, maxSteps = 40): SampledEpisode[] {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const eps: SampledEpisode[] = [];
  for (let k = 0; k < n; k++) {
    let idx = cellIndex(grid, grid.start[0], grid.start[1]);
    const cells = [idx];
    const slipped: boolean[] = [];
    let reward = 0;
    let outcome: SampledEpisode['outcome'] = 'timeout';
    for (let t = 0; t < maxSteps; t++) {
      const act = policy[idx] ?? 0;
      const u = rand();
      const dir = (u < 1 - slip ? act : u < 1 - slip / 2 ? (act + 3) % 4 : (act + 1) % 4) as Action;
      const [r, c] = indexToRC(grid, idx);
      let nr = r + DR[dir];
      let nc = c + DC[dir];
      if (nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols || grid.cells[nr][nc].kind === 'wall') {
        nr = r;
        nc = c;
      }
      idx = cellIndex(grid, nr, nc);
      const cell = grid.cells[nr][nc];
      reward += cell.reward;
      cells.push(idx);
      slipped.push(dir !== act);
      if (isTerminal(cell)) {
        outcome = cell.kind as 'goal' | 'near' | 'hazard';
        break;
      }
    }
    eps.push({ cells, slipped, outcome, reward });
  }
  return eps;
}
