/**
 * Training-lab challenges. Each fixes a dataset and some rules, sets an accuracy target on the TEST
 * set, and (optionally) asks for the fewest parameters. The "best found" values come from the
 * exhaustive-within-a-space search in search.ts, recorded in BEST_FOUND below and re-verified by
 * challenges.test.ts. They are the best found in that search space — not proven optima.
 */
import { generate, split, type Config, type DatasetName, type FeatureName, type Pt } from './lab';

export type Challenge = {
  id: string;
  title: string;
  brief: string;
  dataset: Exclude<DatasetName, 'draw'>;
  noise: number;
  target: number;
  /** Rule check; returns a reason string if the configuration breaks the rules. */
  rules: (cfg: Config) => string | null;
  rulesText: string;
  hint: string;
};

export const CHALLENGE_DATA = { n: 200, trainFrac: 0.5, dataSeed: 1 };

export const CHALLENGES: Challenge[] = [
  {
    id: 'circle-flat',
    title: 'A circle with no hidden layer',
    brief: 'Reach 95% test accuracy on the circle using a network with no hidden layers at all — a single neuron.',
    dataset: 'circle',
    noise: 0.1,
    target: 0.95,
    rules: (c) => (c.hidden.length ? 'Remove every hidden layer.' : null),
    rulesText: 'No hidden layers. Any input features.',
    hint: 'One neuron can only draw a straight line — in the space of its inputs. Which inputs would turn “inside the circle” into something a straight line can separate?',
  },
  {
    id: 'xor-raw',
    title: 'XOR from raw coordinates',
    brief: 'Reach 95% test accuracy on XOR using only x and y as inputs.',
    dataset: 'xor',
    noise: 0.1,
    target: 0.95,
    rules: (c) => (c.features.some((f) => f !== 'x' && f !== 'y') ? 'Use only the x and y inputs.' : null),
    rulesText: 'Inputs x and y only. Any architecture.',
    hint: 'You built XOR by hand in the logic lab with a hidden layer. How many hidden neurons did it take there?',
  },
  {
    id: 'spiral',
    title: 'The spiral',
    brief: 'Reach 90% test accuracy on the two interlocking spirals.',
    dataset: 'spiral',
    noise: 0.05,
    target: 0.9,
    rules: () => null,
    rulesText: 'Anything goes: any features, any architecture, any settings.',
    hint: 'Raw x and y need a fairly deep or wide network here. Some input features make the spiral much easier — think about what changes as you go round it.',
  },
];

const cache = new Map<string, { train: Pt[]; test: Pt[] }>();
export function challengeData(c: Challenge) {
  const key = c.id;
  if (!cache.has(key)) cache.set(key, split(generate(c.dataset, CHALLENGE_DATA.n, c.noise, CHALLENGE_DATA.dataSeed), CHALLENGE_DATA.trainFrac, CHALLENGE_DATA.dataSeed));
  return cache.get(key)!;
}

/** Results of running search.ts (see that file for the exact search spaces). */
export const BEST_FOUND: Record<string, { params: number; cfg: Config; reachedAt: number; searched: number; spaceSize: number; space: string }> = {
  'circle-flat': {
    params: 3,
    cfg: { features: ['x2', 'y2'], hidden: [], activation: 'tanh', optimizer: 'adam', lr: 0.03, batch: 10, l2: 0, seed: 1 },
    reachedAt: 100,
    searched: 37,
    spaceSize: 381,
    space: 'every one of the 127 combinations of input features, no hidden layer, random starts 1–3',
  },
  'xor-raw': {
    params: 13,
    cfg: { features: ['x', 'y'], hidden: [3], activation: 'tanh', optimizer: 'adam', lr: 0.03, batch: 10, l2: 0, seed: 3 },
    reachedAt: 300,
    searched: 45,
    spaceSize: 258,
    space: 'one or two hidden layers of 1–6 neurons, tanh or ReLU, random starts 1–3',
  },
  spiral: {
    params: 13,
    cfg: { features: ['x', 'x2', 'y2', 'xy'], hidden: [2], activation: 'tanh', optimizer: 'adam', lr: 0.03, batch: 10, l2: 0, seed: 1 },
    reachedAt: 1900,
    searched: 2053,
    spaceSize: 5334,
    space: 'every combination of input features with no hidden layer or one hidden layer of 1–6 neurons, tanh or ReLU, random starts 1–3',
  },
};

export type FeatureSet = FeatureName[];
