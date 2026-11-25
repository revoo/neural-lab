/**
 * Scaled dot-product attention on small, readable vectors: the soft lookup, a hand-built sentence,
 * the √d experiment, and the attention puzzles with exhaustively computed optimums.
 */
import { createRng } from '../../lib/rng';

export type Vec = number[];
export const dot = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0);

export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** One query against a set of keys: scores q·k/√d, weights softmax(scores), output Σ wᵢ vᵢ. */
export function attend(q: Vec, keys: Vec[], values: Vec[], scale = true) {
  const d = q.length;
  const scores = keys.map((k) => dot(q, k) / (scale ? Math.sqrt(d) : 1));
  const weights = softmax(scores);
  const out = values[0].map((_, j) => values.reduce((s, v, i) => s + weights[i] * v[j], 0));
  return { scores, weights, out };
}

const matVec = (M: number[][], x: Vec) => M.map((row) => dot(row, x));

// ---------------------------------------------------------------- a hand-built sentence

/**
 * Features of each word, by hand: [noun, animate, pronoun, verb or adjective, function word].
 * A trained model learns its own (hundreds of) features; these five are chosen so you can read them.
 */
export const FEATURES = ['noun', 'animate', 'pronoun', 'verb/adj', 'function word'];
export const WORDS: { w: string; x: Vec }[] = [
  { w: 'the', x: [0, 0, 0, 0, 1] },
  { w: 'animal', x: [1, 1, 0, 0, 0] },
  { w: "didn't", x: [0, 0, 0, 1, 1] },
  { w: 'cross', x: [0, 0, 0, 1, 0] },
  { w: 'the', x: [0, 0, 0, 0, 1] },
  { w: 'street', x: [1, 0, 0, 0, 0] },
  { w: 'because', x: [0, 0, 0, 0, 1] },
  { w: 'it', x: [0, 0, 1, 0, 0] },
  { w: 'was', x: [0, 0, 0, 1, 1] },
  { w: 'tired', x: [0, 0, 0, 1, 0] },
];

/**
 * Hand-set projections into a 2-dimensional query/key space. Key dimension 0 = "I am a noun",
 * key dimension 1 = "I am animate". A pronoun's query asks for an animate noun; a verb or
 * adjective's query asks for a noun of any kind; function words ask for nothing in particular.
 */
export const W_Q = [
  // query dim 0 ("want a noun") and dim 1 ("want something animate"), from the five features
  [0, 0, 2, 2, 0],
  [0, 0, 3, 0, 0],
];
export const W_K = [
  [2, 0, 0, 0, 0],
  [0, 2, 0, 0, 0],
];
/** Values: what each word passes on when attended to — here simply its own features. */
export const W_V = [
  [1, 0, 0, 0, 0],
  [0, 1, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 1],
];

export function sentenceAttention(words = WORDS) {
  const Q = words.map((t) => matVec(W_Q, t.x));
  const K = words.map((t) => matVec(W_K, t.x));
  const V = words.map((t) => matVec(W_V, t.x));
  return words.map((_, i) => ({ q: Q[i], ...attend(Q[i], K, V) }));
}

// ---------------------------------------------------------------- why divide by √d

/**
 * For random query and key vectors with independent standard-normal entries, the dot product has
 * spread √d. Returns, averaged over trials, the largest softmax weight among n keys — with and
 * without dividing by √d. Without scaling, softmax saturates (one weight near 1) as d grows.
 */
export function maxWeightStudy(d: number, n = 10, trials = 400, seed = 1) {
  const r = createRng(seed * 1009 + d);
  let raw = 0;
  let scaled = 0;
  let spread = 0;
  for (let t = 0; t < trials; t++) {
    const q = Array.from({ length: d }, () => r.normal());
    const keys = Array.from({ length: n }, () => Array.from({ length: d }, () => r.normal()));
    const s = keys.map((k) => dot(q, k));
    spread += Math.sqrt(s.reduce((a, v) => a + v * v, 0) / n);
    raw += Math.max(...softmax(s));
    scaled += Math.max(...softmax(s.map((v) => v / Math.sqrt(d))));
  }
  return { d, raw: raw / trials, scaled: scaled / trials, spread: spread / trials };
}

// ---------------------------------------------------------------- the puzzles

export type Puzzle = {
  id: string;
  title: string;
  brief: string;
  keys: { name: string; k: Vec }[];
  /** succeed when this returns true for the attention weights */
  goal: (w: number[]) => boolean;
  goalText: string;
};

export const GRID = 5; // queries have integer components in [−GRID, GRID]

export const PUZZLES: Puzzle[] = [
  {
    id: 'spotlight',
    title: 'Spotlight',
    brief: 'Make the query attend to "cat": at least 80% of the attention.',
    keys: [
      { name: 'cat', k: [-1, 3] },
      { name: 'sat', k: [3, 1] },
      { name: 'mat', k: [-2, -2] },
    ],
    goal: (w) => w[0] >= 0.8,
    goalText: 'cat ≥ 80%',
  },
  {
    id: 'neighbours',
    title: 'Close neighbours',
    brief: '"dog" and "wolf" have similar keys. Give "dog" at least 80% of the attention anyway.',
    keys: [
      { name: 'dog', k: [3, 1] },
      { name: 'wolf', k: [3, 2] },
      { name: 'cup', k: [-2, 1] },
      { name: 'run', k: [0, -3] },
    ],
    goal: (w) => w[0] >= 0.8,
    goalText: 'dog ≥ 80%',
  },
  {
    id: 'split',
    title: 'Fair split',
    brief: 'Share the attention between "salt" and "pepper": each between 40% and 60%, and everything else under 10% in total.',
    keys: [
      { name: 'salt', k: [2, 2] },
      { name: 'pepper', k: [2, -2] },
      { name: 'sugar', k: [-2, 1] },
      { name: 'flour', k: [-3, 0] },
    ],
    goal: (w) => w[0] >= 0.4 && w[0] <= 0.6 && w[1] >= 0.4 && w[1] <= 0.6 && w[2] + w[3] < 0.1,
    goalText: 'salt and pepper each 40–60%, the rest < 10%',
  },
  {
    id: 'hidden',
    title: 'The hidden key',
    brief: 'Give "middle" at least 80% of the attention.',
    keys: [
      { name: 'middle', k: [1, 1] },
      { name: 'left', k: [-1, 3] },
      { name: 'right', k: [3, -1] },
      { name: 'low', k: [-2, -2] },
    ],
    goal: (w) => w[0] >= 0.8,
    goalText: 'middle ≥ 80%',
  },
];

export const weightsFor = (p: Puzzle, q: Vec) => attend(q, p.keys.map((k) => k.k), p.keys.map((k) => k.k)).weights;

/** All integer queries on the grid, smallest first (by |q₁| + |q₂|). */
export function gridQueries(): Vec[] {
  const out: Vec[] = [];
  for (let a = -GRID; a <= GRID; a++) for (let b = -GRID; b <= GRID; b++) out.push([a, b]);
  return out.sort((u, v) => Math.abs(u[0]) + Math.abs(u[1]) - (Math.abs(v[0]) + Math.abs(v[1])));
}

/**
 * Exhaustive search over the grid: the smallest query (fewest units, |q₁| + |q₂|) that solves the
 * puzzle, how many grid queries solve it, and the best weight the first key can get at all.
 */
export function solve(p: Puzzle) {
  const qs = gridQueries();
  const solutions = qs.filter((q) => p.goal(weightsFor(p, q)));
  const smallest = solutions.length ? Math.abs(solutions[0][0]) + Math.abs(solutions[0][1]) : null;
  const bestFirst = Math.max(...qs.map((q) => weightsFor(p, q)[0]));
  return { solutions, smallest, bestFirst, total: qs.length };
}

/** `tie`: the top share is held by two or more keys (to 1e-9). */
export type MapCell = { q: Vec; pass: boolean; winner: number; tie: boolean; weights: number[] };

/**
 * The solution map: every query on the 11×11 grid, whether it passes, and
 * which key wins (gets the most attention). Shown after a puzzle is solved.
 */
export function queryMap(p: Puzzle): MapCell[] {
  const cells: MapCell[] = [];
  for (let b = GRID; b >= -GRID; b--)
    for (let a = -GRID; a <= GRID; a++) {
      const q = [a, b];
      const w = weightsFor(p, q);
      let winner = 0;
      for (let i = 1; i < w.length; i++) if (w[i] > w[winner] + 1e-12) winner = i;
      const tie = w.some((v, i) => i !== winner && Math.abs(v - w[winner]) < 1e-9);
      cells.push({ q, pass: p.goal(w), winner, tie, weights: w });
    }
  return cells;
}
