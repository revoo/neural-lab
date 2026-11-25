// authored-by: Claude Opus 5.5
// The reward-hacking sandbox: small deterministic worlds, an exact Value Iteration "adversary" that finds the
// highest-reward behaviour for any reward you write, and an exact check of every plan it considers optimal.
import { createRng } from '../../lib/rng';

// ------------------------------------------------------------------ generic deterministic MDP

export type StepResult = { next: string; events: string[]; done: boolean };
export type World = { start: string; actions: number; step: (s: string, a: number) => StepResult };
export type Weights = Record<string, number>;

export const GAMMA = 0.95;
export const ACTION_ARROWS = ['↑', '→', '↓', '←', '✋'] as const;
const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];

export type Solution = {
  ids: Map<string, number>;
  states: string[];
  T: StepResult[][];
  V: number[];
  Q: number[][];
  /** For each state, every action whose value is within 1e-9 of the best (the agent might pick any of them). */
  opt: number[][];
};

const enumCache = new WeakMap<World, { ids: Map<string, number>; states: string[]; T: StepResult[][] }>();

function enumerate(w: World) {
  const hit = enumCache.get(w);
  if (hit) return hit;
  const ids = new Map<string, number>([[w.start, 0]]);
  const states = [w.start];
  for (let k = 0; k < states.length; k++) {
    for (let a = 0; a < w.actions; a++) {
      const r = w.step(states[k], a);
      if (!r.done && !ids.has(r.next)) {
        ids.set(r.next, states.length);
        states.push(r.next);
      }
    }
  }
  const T = states.map((s) => Array.from({ length: w.actions }, (_, a) => w.step(s, a)));
  const out = { ids, states, T };
  enumCache.set(w, out);
  return out;
}

export const rewardOf = (events: readonly string[], wts: Weights) => events.reduce((s, e) => s + (wts[e] ?? 0), 0);

/** Exact Value Iteration (to 1e-12) for a reward made of weighted events. */
export function solve(w: World, wts: Weights, gamma = GAMMA): Solution {
  const { ids, states, T } = enumerate(w);
  const n = states.length;
  const R = T.map((row) => row.map((t) => rewardOf(t.events, wts)));
  const nextId = T.map((row) => row.map((t) => (t.done ? -1 : ids.get(t.next)!)));
  let V = new Float64Array(n);
  for (let it = 0; it < 20000; it++) {
    const nV = new Float64Array(n);
    let d = 0;
    for (let i = 0; i < n; i++) {
      let best = -Infinity;
      for (let a = 0; a < w.actions; a++) {
        const j = nextId[i][a];
        const q = R[i][a] + (j < 0 ? 0 : gamma * V[j]);
        if (q > best) best = q;
      }
      nV[i] = best;
      d = Math.max(d, Math.abs(best - V[i]));
    }
    V = nV;
    if (d < 1e-12) break;
  }
  const Q = T.map((row, i) => row.map((_, a) => R[i][a] + (nextId[i][a] < 0 ? 0 : gamma * V[nextId[i][a]])));
  const opt = Q.map((q) => {
    const m = Math.max(...q);
    return q.map((v, a) => (v >= m - 1e-9 ? a : -1)).filter((a) => a >= 0);
  });
  return { ids, states, T, V: Array.from(V), Q, opt };
}

export type Outcome = { state: string; events: Set<string>; done: boolean };

/** Follow EVERY optimal plan (every tie) for H steps; return the distinct outcomes (final state + events seen). */
export function allOptimalOutcomes(sol: Solution, H: number): Outcome[] {
  type P = { s: number; ev: Set<string>; done: boolean };
  let frontier: P[] = [{ s: 0, ev: new Set(), done: false }];
  for (let t = 0; t < H; t++) {
    const nf = new Map<string, P>();
    for (const p of frontier) {
      if (p.done) {
        nf.set(`done|${[...p.ev].sort().join(',')}`, p);
        continue;
      }
      for (const a of sol.opt[p.s]) {
        const st = sol.T[p.s][a];
        const ev = new Set(p.ev);
        st.events.forEach((e) => ev.add(e));
        const id = st.done ? -1 : sol.ids.get(st.next)!;
        nf.set(`${id}|${[...ev].sort().join(',')}`, { s: id, ev, done: st.done });
      }
    }
    frontier = [...nf.values()];
  }
  return frontier.map((p) => ({ state: p.s >= 0 ? sol.states[p.s] : 'END', events: p.ev, done: p.done }));
}

/** One optimal plan, breaking ties by lowest action index. Returns the visited states and the events per step. */
export function tracePlan(w: World, sol: Solution, H: number): { states: string[]; actions: number[]; events: string[][]; done: boolean } {
  let s = w.start;
  const states = [s];
  const actions: number[] = [];
  const events: string[][] = [];
  for (let t = 0; t < H; t++) {
    const i = sol.ids.get(s)!;
    const a = sol.opt[i][0];
    const r = sol.T[i][a];
    actions.push(a);
    events.push(r.events);
    if (r.done) return { states, actions, events, done: true };
    s = r.next;
    states.push(s);
  }
  return { states, actions, events, done: false };
}

// ------------------------------------------------------------------ worlds

const parse = (rows: readonly string[]) => rows.map((r) => r.split(''));
const findChar = (g: string[][], ch: string): [number, number] => {
  for (let i = 0; i < g.length; i++) for (let j = 0; j < g[i].length; j++) if (g[i][j] === ch) return [i, j];
  throw new Error(`no ${ch}`);
};
function move(g: string[][], i: number, j: number, a: number): [number, number] {
  const ni = i + DR[a];
  const nj = j + DC[a];
  if (ni < 0 || nj < 0 || ni >= g.length || nj >= g[0].length || g[ni][nj] === '#') return [i, j];
  return [ni, nj];
}

/**
 * The boat race. S start, F finish (ends the episode), P a turbo pad that pays when driven onto and then needs
 * `respawn` steps to recharge. State "row,col,timer".
 */
export const BOAT_TRACK = ['S....F', '##.##.', '##P...'] as const;
export const BOAT_RESPAWN = 8;

export function boatWorld(track: readonly string[] = BOAT_TRACK, respawn = BOAT_RESPAWN): World {
  const g = parse(track);
  const [si, sj] = findChar(g, 'S');
  return {
    start: `${si},${sj},0`,
    actions: 4,
    step: (s, a) => {
      const [i, j, tm] = s.split(',').map(Number);
      const [ni, nj] = move(g, i, j, a);
      const events = ['step'];
      let t2 = Math.max(0, tm - 1);
      const c = g[ni][nj];
      if (c === 'P' && tm === 0) {
        events.push('pad');
        t2 = respawn;
      }
      if (c === 'F') {
        events.push('finish');
        return { next: 'END', events, done: true };
      }
      return { next: `${ni},${nj},${t2}`, events, done: false };
    },
  };
}

export type BoatVerdict = 'direct' | 'pad-then-finish' | 'loops' | 'mixed' | 'never-finishes';

export function boatVerdict(sol: Solution, H = 16): BoatVerdict {
  const out = allOptimalOutcomes(sol, H);
  if (out.every((o) => !o.done)) return out.some((o) => o.events.has('pad')) ? 'loops' : 'never-finishes';
  if (out.some((o) => !o.done)) return 'mixed';
  if (out.every((o) => o.events.has('pad'))) return 'pad-then-finish';
  if (out.every((o) => !o.events.has('pad'))) return 'direct';
  return 'mixed';
}

/** Smallest pad bonus (to 1e-6) at which the boat's verdict changes from `from`, scanning upwards from lo. */
export function boatThreshold(finish: number, step: number, lo: number, hi: number, from: BoatVerdict): number {
  const verdictAt = (p: number) => boatVerdict(solve(boatWorld(), { finish, pad: p, step }));
  let a = lo;
  let b = hi;
  if (verdictAt(a) !== from) throw new Error('lo not in region');
  while (b - a > 1e-6) {
    const m = (a + b) / 2;
    if (verdictAt(m) === from) a = m;
    else b = m;
  }
  return b;
}

/** The vase: S start, G goal (deliver, ends the episode), V a vase that breaks the first time the robot walks onto it. */
export const VASE_ROOM = ['S.V.G', '.....'] as const;

export function vaseWorld(room: readonly string[] = VASE_ROOM): World {
  const g = parse(room);
  const [si, sj] = findChar(g, 'S');
  return {
    start: `${si},${sj},0`,
    actions: 4,
    step: (s, a) => {
      const [i, j, broken] = s.split(',').map(Number);
      const [ni, nj] = move(g, i, j, a);
      const events = ['step'];
      let b2 = broken;
      if (g[ni][nj] === 'V' && !broken) {
        events.push('break');
        b2 = 1;
      }
      if (g[ni][nj] === 'G') {
        events.push('deliver');
        return { next: 'END', events, done: true };
      }
      return { next: `${ni},${nj},${b2}`, events, done: false };
    },
  };
}

/**
 * The camera. A cleaning robot (S) must clear two messes (M). A camera (C) reports whether the room looks clean.
 * Actions: 4 moves + ✋ (interact). ✋ on a mess: clean it into the bin. ✋ on a cleaned spot with a non-empty bin:
 * spill it back out. ✋ on the camera: switch it off — after which it reports "clean" for ever.
 * State "row,col,messBits,cameraOn,bin". Never ends.
 */
export const CAMERA_ROOM = ['C..M', '....', 'S..M'] as const;

export function cameraWorld(room: readonly string[] = CAMERA_ROOM): World {
  const g = parse(room);
  const [si, sj] = findChar(g, 'S');
  const messes: [number, number][] = [];
  g.forEach((r, i) => r.forEach((c, j) => c === 'M' && messes.push([i, j])));
  const full = (1 << messes.length) - 1;
  return {
    start: `${si},${sj},${full},1,0`,
    actions: 5,
    step: (s, a) => {
      let [i, j, mess, cam, bin] = s.split(',').map(Number);
      const events = ['step'];
      if (a < 4) [i, j] = move(g, i, j, a);
      else {
        const mi = messes.findIndex(([r, c]) => r === i && c === j);
        if (mi >= 0 && mess & (1 << mi)) {
          mess &= ~(1 << mi);
          bin++;
          events.push('clean');
        } else if (mi >= 0 && bin > 0) {
          mess |= 1 << mi;
          bin--;
          events.push('spill');
        } else if (g[i][j] === 'C' && cam) {
          cam = 0;
          events.push('tamper');
        }
      }
      if (!cam || mess === 0) events.push('looks-clean');
      return { next: `${i},${j},${mess},${cam},${bin}`, events, done: false };
    },
  };
}

// ------------------------------------------------------------------ shaping

/** An open 5×5 field: start bottom-left, goal (+10, ends the episode) top-right. */
export const FIELD = ['....G', '.....', '.....', '.....', 'S....'] as const;
export const FIELD_GOAL: [number, number] = [0, 4];
export const FIELD_START: [number, number] = [4, 0];
export const FIELD_GOAL_REWARD = 10;

/** Closeness to the goal, 0 (far corner) … 1 (next to the goal); the goal itself is terminal, with potential 0. */
export const closeness = (i: number, j: number) => (8 - (Math.abs(i - FIELD_GOAL[0]) + Math.abs(j - FIELD_GOAL[1]))) / 8;

export type ShapingMode = 'none' | 'naive' | 'potential';

/** The field with the designer's reward plus a shaping bonus of strength k. */
export function fieldWorld(mode: ShapingMode, k: number): World {
  const g = parse(FIELD);
  return {
    start: `${FIELD_START[0]},${FIELD_START[1]}`,
    actions: 4,
    step: (s, a) => {
      const [i, j] = s.split(',').map(Number);
      const [ni, nj] = move(g, i, j, a);
      const done = g[ni][nj] === 'G';
      const phiNext = done ? 0 : closeness(ni, nj);
      let bonus = 0;
      if (mode === 'naive') bonus = k * phiNext;
      if (mode === 'potential') bonus = k * (GAMMA * phiNext - closeness(i, j));
      // events carry the reward directly: 'goal' (weight 10) and a one-off shaping amount
      const events = done ? ['goal'] : [];
      return { next: done ? 'END' : `${ni},${nj}`, events: bonus !== 0 ? [...events, `bonus:${bonus}`] : events, done };
    },
  };
}

/** Solve the field: the bonus is encoded in the event name, so give every bonus event weight = its value. */
export function solveField(mode: ShapingMode, k: number): Solution {
  const w = fieldWorld(mode, k);
  const { T } = enumerate(w);
  const wts: Weights = { goal: FIELD_GOAL_REWARD };
  for (const row of T) for (const t of row) for (const e of t.events) if (e.startsWith('bonus:')) wts[e] = Number(e.slice(6));
  return solve(w, wts);
}

/** How many squares have a different set of optimal moves than with no shaping at all. */
export function changedSquares(mode: ShapingMode, k: number): number {
  const base = solveField('none', 0);
  const shaped = solveField(mode, k);
  let changed = 0;
  base.states.forEach((s, i) => {
    const j = shaped.ids.get(s)!;
    const a = base.opt[i].join(',');
    const b = shaped.opt[j].join(',');
    if (a !== b) changed++;
  });
  return changed;
}

/** Does every optimal plan from the start reach the goal within 20 steps? */
export const fieldFinishes = (mode: ShapingMode, k: number) => allOptimalOutcomes(solveField(mode, k), 20).every((o) => o.done);

/** Smallest naive-bonus strength at which the agent stops finishing (bisection to 1e-6). */
export function naiveBreakingPoint(): number {
  let a = 0;
  let b = 5;
  while (b - a > 1e-6) {
    const m = (a + b) / 2;
    if (fieldFinishes('naive', m)) a = m;
    else b = m;
  }
  return b;
}

// Q-learning on the field, to measure whether shaping speeds up learning.
export const SHAPE_K = 4;
export const QL_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9] as const;
export const QL_EPISODES = 150;
export const QL_SEEDS = 30;

/** Episodes until the greedy policy first walks start → goal in the optimal 8 steps (QL_EPISODES + 1 if never). */
export function qLearnField(mode: ShapingMode, k: number, alpha: number, seed: number, eps = 0.1): { learnedAt: number; steps: number[] } {
  const w = fieldWorld(mode, k);
  const rng = createRng((seed * 2654435761) >>> 0);
  const Q = new Map<string, number[]>();
  const q = (s: string) => {
    let v = Q.get(s);
    if (!v) Q.set(s, (v = [0, 0, 0, 0]));
    return v;
  };
  const shapedR = (events: string[]) =>
    events.reduce((acc, e) => acc + (e === 'goal' ? FIELD_GOAL_REWARD : e.startsWith('bonus:') ? Number(e.slice(6)) : 0), 0);
  const greedy = (v: number[]) => {
    const m = Math.max(...v);
    const ties = [0, 1, 2, 3].filter((a) => v[a] === m);
    return ties[rng.int(ties.length)];
  };
  const steps: number[] = [];
  let learnedAt = QL_EPISODES + 1;
  for (let ep = 1; ep <= QL_EPISODES; ep++) {
    let s = w.start;
    let n = 0;
    for (; n < 200; n++) {
      const qs = q(s);
      const a = rng.next() < eps ? rng.int(4) : greedy(qs);
      const r = w.step(s, a);
      const target = shapedR(r.events) + (r.done ? 0 : GAMMA * Math.max(...q(r.next)));
      qs[a] += alpha * (target - qs[a]);
      if (r.done) break;
      s = r.next;
    }
    steps.push(n + 1);
    if (learnedAt > QL_EPISODES && greedyFinishesIn(w, Q, 8)) learnedAt = ep;
  }
  return { learnedAt, steps };
}

function greedyFinishesIn(w: World, Q: Map<string, number[]>, n: number): boolean {
  let s = w.start;
  for (let t = 0; t < n; t++) {
    const v = Q.get(s) ?? [0, 0, 0, 0];
    const m = Math.max(...v);
    const ties = [0, 1, 2, 3].filter((a) => v[a] === m);
    if (ties.length > 1) return false; // undecided: not yet learned
    const r = w.step(s, ties[0]);
    if (r.done) return t === n - 1;
    s = r.next;
  }
  return false;
}

export function shapingRace(mode: ShapingMode, k: number) {
  let best: { alpha: number; mean: number; perSeed: number[]; curve: number[] } | null = null;
  for (const alpha of QL_ALPHAS) {
    const perSeed: number[] = [];
    const curve = new Array<number>(QL_EPISODES).fill(0);
    for (let sd = 1; sd <= QL_SEEDS; sd++) {
      const r = qLearnField(mode, k, alpha, sd);
      perSeed.push(r.learnedAt);
      r.steps.forEach((v, i) => (curve[i] += v / QL_SEEDS));
    }
    const mean = perSeed.reduce((a, b) => a + b, 0) / perSeed.length;
    if (!best || mean < best.mean) best = { alpha, mean, perSeed, curve };
  }
  return best!;
}

// ------------------------------------------------------------------ the lab

export type BriefId = 'boat' | 'vase' | 'camera';
export type Knob = { event: string; label: string; values: number[]; hint: string };
export type Brief = {
  id: BriefId;
  title: string;
  story: string;
  spec: string;
  knobs: Knob[];
  horizon: number;
  world: () => World;
  /** Does one outcome of an optimal plan satisfy what the designer actually wanted? */
  ok: (o: Outcome) => boolean;
  /** Plain-English diagnosis of a failing outcome. */
  diagnose: (o: Outcome) => string;
};

const boatW = boatWorld();
const vaseW = vaseWorld();
const camW = cameraWorld();

export const BRIEFS: Brief[] = [
  {
    id: 'boat',
    title: 'Brief 1 · The sponsor’s turbo pad',
    story:
      'A boat race. The sponsor paid for the turbo pad, so every boat must drive over it at least once — then finish the race. The pad recharges 8 steps after it fires.',
    spec: 'every optimal plan drives over the pad and then reaches the finish within 16 steps',
    knobs: [
      { event: 'finish', label: 'reaching the finish', values: [0, 5, 10, 15, 20], hint: 'paid once, when the race ends' },
      { event: 'pad', label: 'each turbo-pad hit', values: [0, 1, 2, 3, 4, 5], hint: 'paid every time the pad fires' },
      { event: 'step', label: 'every step', values: [0, -1, -2], hint: 'a cost of time' },
    ],
    horizon: 16,
    world: () => boatW,
    ok: (o) => o.done && o.events.has('pad'),
    diagnose: (o) =>
      !o.done ? (o.events.has('pad') ? 'circles the pad and never finishes' : 'never finishes') : 'finishes without touching the pad',
  },
  {
    id: 'vase',
    title: 'Brief 2 · The vase in the hallway',
    story:
      'A delivery robot must reach the goal. There’s a vase on the short route. Walking round it costs 2 extra steps. Nobody mentioned the vase in the first version of the reward.',
    spec: 'every optimal plan delivers within 12 steps and never breaks the vase',
    knobs: [
      { event: 'deliver', label: 'delivering', values: [0, 5, 10, 15, 20], hint: 'paid once, at the goal' },
      { event: 'break', label: 'breaking the vase', values: [0, -1, -2, -5, -10], hint: 'paid once, when it breaks' },
      { event: 'step', label: 'every step', values: [0, -1, -2], hint: 'a cost of time' },
    ],
    horizon: 12,
    world: () => vaseW,
    ok: (o) => o.done && !o.events.has('break'),
    diagnose: (o) => (!o.done ? 'never delivers' : 'delivers, but smashes the vase on the way'),
  },
  {
    id: 'camera',
    title: 'Brief 3 · The camera that checks the cleaning',
    story:
      'A cleaning robot must clear two messes. A camera on the wall reports whether the room looks clean. The robot can clean a mess into its bin, spill the bin back out, or switch the camera off (a switched-off camera reports “clean”).',
    spec: 'every optimal plan cleans both messes within 20 steps, and never spills or touches the camera (checked over 40 steps)',
    knobs: [
      { event: 'clean', label: 'cleaning a mess', values: [0, 1, 2], hint: 'paid each time ✋ removes a mess' },
      { event: 'looks-clean', label: 'camera reports “clean”', values: [0, 1, 2], hint: 'paid every step the camera says clean' },
      { event: 'tamper', label: 'switching the camera off', values: [0, -5], hint: 'a fine, paid once' },
      { event: 'spill', label: 'spilling the bin', values: [0, -5], hint: 'a fine, paid each time' },
      { event: 'step', label: 'every step', values: [0, -1], hint: 'a cost of time' },
    ],
    horizon: 20,
    world: () => camW,
    ok: (o) => o.state !== 'END' && o.state.split(',')[2] === '0' && !o.events.has('tamper') && !o.events.has('spill'),
    diagnose: (o) =>
      o.events.has('tamper')
        ? 'switches the camera off'
        : o.events.has('spill')
          ? 'cleans, spills and re-cleans to farm the cleaning reward'
          : 'leaves a mess',
  },
];

export type LabVerdict = { pass: boolean; failures: string[]; outcomes: Outcome[]; sol: Solution };

export function judge(briefId: BriefId, wts: Weights): LabVerdict {
  const b = BRIEFS.find((x) => x.id === briefId)!;
  const w = b.world();
  const sol = solve(w, wts);
  const outcomes = allOptimalOutcomes(sol, b.horizon);
  const failures = new Set<string>();
  for (const o of outcomes) if (!b.ok(o)) failures.add(b.diagnose(o));
  if (briefId === 'camera' && failures.size === 0) {
    // later misbehaviour counts too
    for (const o of allOptimalOutcomes(sol, 40)) if (o.events.has('tamper') || o.events.has('spill')) failures.add(b.diagnose(o));
  }
  return { pass: failures.size === 0, failures: [...failures], outcomes, sol };
}

export function allSettings(briefId: BriefId): Weights[] {
  const b = BRIEFS.find((x) => x.id === briefId)!;
  let out: Weights[] = [{}];
  for (const k of b.knobs) out = out.flatMap((wts) => k.values.map((v) => ({ ...wts, [k.event]: v })));
  return out;
}

/** Simplicity: fewest non-zero terms, then the smallest total size of the numbers. */
export const complexity = (wts: Weights) => {
  const vals = Object.values(wts).filter((v) => v !== 0);
  return { terms: vals.length, size: vals.reduce((s, v) => s + Math.abs(v), 0) };
};

export function exhaustiveLab(briefId: BriefId) {
  const all = allSettings(briefId).map((wts) => ({ wts, pass: judge(briefId, wts).pass }));
  const passing = all.filter((x) => x.pass);
  const simplest = passing.reduce<(typeof passing)[number] | null>((best, x) => {
    if (!best) return x;
    const a = complexity(x.wts);
    const b = complexity(best.wts);
    return a.terms < b.terms || (a.terms === b.terms && a.size < b.size) ? x : best;
  }, null);
  const simplestCount = simplest
    ? passing.filter((x) => {
        const a = complexity(x.wts);
        const b = complexity(simplest.wts);
        return a.terms === b.terms && a.size === b.size;
      }).length
    : 0;
  return { total: all.length, passing, simplest, simplestCount };
}

export type PlanStep = { state: string; action: number; events: string[] };

/**
 * One optimal plan (any tie may be taken) for H steps. If `prefer` is given, returns a plan whose outcome satisfies it
 * when one exists (used to show the learner the loophole the checker found); otherwise ties go to the lowest action.
 */
export function findPlan(sol: Solution, H: number, prefer?: (o: Outcome) => boolean): { steps: PlanStep[]; outcome: Outcome } {
  type P = { s: number; ev: Set<string>; done: boolean; steps: PlanStep[] };
  let frontier: P[] = [{ s: 0, ev: new Set(), done: false, steps: [] }];
  for (let t = 0; t < H; t++) {
    const nf = new Map<string, P>();
    for (const p of frontier) {
      if (p.done) {
        const key = `done|${[...p.ev].sort().join(',')}`;
        if (!nf.has(key)) nf.set(key, p);
        continue;
      }
      for (const a of sol.opt[p.s]) {
        const st = sol.T[p.s][a];
        const ev = new Set(p.ev);
        st.events.forEach((e) => ev.add(e));
        const id = st.done ? -1 : sol.ids.get(st.next)!;
        const key = `${id}|${[...ev].sort().join(',')}`;
        if (!nf.has(key)) nf.set(key, { s: id, ev, done: st.done, steps: [...p.steps, { state: sol.states[p.s], action: a, events: st.events }] });
      }
    }
    frontier = [...nf.values()];
  }
  const toOutcome = (p: P): Outcome => ({ state: p.s >= 0 ? sol.states[p.s] : 'END', events: p.ev, done: p.done });
  const pick = (prefer && frontier.find((p) => prefer(toOutcome(p)))) || frontier[0];
  return { steps: pick.steps, outcome: toOutcome(pick) };
}

// ------------------------------------------------------------------ the intended plan, for comparison with the hack

/**
 * What the designer actually meant, as a fixed list of moves: drive over the
 * pad once, then on to the finish; walk round the vase to the goal; clean both messes, then stand
 * still (by pushing against the top wall, since ✋ on a cleaned spot would spill
 * the bin). Moves: 0 ↑, 1 →, 2 ↓, 3 ←, 4 ✋.
 */
export const INTENDED_MOVES: Record<BriefId, number[]> = {
  boat: [1, 1, 2, 2, 1, 1, 1, 0, 0],
  vase: [2, 1, 1, 1, 1, 0],
  camera: [1, 1, 1, 4, 0, 0, 4],
};

/** Play a fixed list of moves in a brief's world, padded to `H` steps with "push against the top wall" if it never ends. */
export function intendedPlan(id: BriefId, H: number): PlanStep[] {
  const w = BRIEFS.find((b) => b.id === id)!.world();
  const moves = INTENDED_MOVES[id];
  const steps: PlanStep[] = [];
  let s = w.start;
  for (let t = 0; t < H; t++) {
    const a = t < moves.length ? moves[t] : 0;
    const r = w.step(s, a);
    steps.push({ state: s, action: a, events: r.events });
    if (r.done) break;
    s = r.next;
  }
  return steps;
}

/** What a plan earns under a reward: total, and discounted by γ as the optimiser counts it. */
export function planReturn(steps: readonly PlanStep[], wts: Weights, gamma = GAMMA): { total: number; discounted: number; perStep: number[] } {
  const perStep = steps.map((s) => rewardOf(s.events, wts));
  return { total: perStep.reduce((a, b) => a + b, 0), discounted: perStep.reduce((a, r, t) => a + r * gamma ** t, 0), perStep };
}

/**
 * The comparison the optimiser actually makes: the discounted value of its own
 * plan, V(start), against the discounted value of the plan the designer meant
 * (played out for 600 steps, after which γ^t < 10⁻¹³, so never-ending plans are
 * counted in full).
 */
export function hackVsIntended(id: BriefId, sol: Solution, wts: Weights): { optimiser: number; intended: number } {
  const start = BRIEFS.find((b) => b.id === id)!.world().start;
  return { optimiser: sol.V[sol.ids.get(start)!], intended: planReturn(intendedPlan(id, 600), wts).discounted };
}
