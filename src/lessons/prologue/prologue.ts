// authored-by: Claude Opus 5.5
/**
 * The Prologue's demo: a small network learning the two-circles dataset.
 * Kept independent of other lessons (the dataset is duplicated on purpose).
 */
import { MLP } from '../../lib/mlp';
import { createOptimizer, type Optimizer } from '../../lib/optimizers';
import { createRng } from '../../lib/rng';
import { ERAS } from '../../history';

export type Dataset = { X: number[][]; y: number[] };

/** Inner disc (label 0) inside a ring (label 1), on roughly [−1.4, 1.4]². Fixed: only the network's seed changes. */
export function makeCircles(n = 120, seed = 7): Dataset {
  const r = createRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let k = 0; k < n; k++) {
    const label = k % 2;
    const rad = label ? r.uniform(0.85, 1.35) : r.uniform(0, 0.6);
    const t = r.uniform(0, 2 * Math.PI);
    X.push([rad * Math.cos(t), rad * Math.sin(t)]);
    y.push(label);
  }
  return { X, y };
}

/** Hidden layer size and learning rate of the demo network. */
export const HIDDEN = 6;
export const LEARNING_RATE = 0.01;
/** Training steps per animation frame in the widget, and the cap on a run. */
export const STEPS_PER_FRAME = 2;
export const MAX_STEPS = 2000;

export type Demo = {
  net: MLP;
  opt: Optimizer;
  data: Dataset;
  Y: number[][];
  steps: number;
};

export function createDemo(seed: number, data: Dataset = makeCircles()): Demo {
  const net = new MLP([2, HIDDEN, 1], ['tanh', 'sigmoid'], createRng(seed), 'xavier');
  return { net, opt: createOptimizer({ name: 'adam', lr: LEARNING_RATE }), data, Y: data.y.map((v) => [v]), steps: 0 };
}

/** Run k full-batch training steps; returns the loss measured before the last step. */
export function train(d: Demo, k: number): number {
  let loss = 0;
  for (let i = 0; i < k; i++) {
    loss = d.net.backprop(d.data.X, d.Y, 'bce');
    d.opt.step(d.net.params(), d.net.grads());
    d.steps++;
  }
  return loss;
}

export const lossOf = (d: Demo) => d.net.loss(d.data.X, d.Y, 'bce');

export function accuracy(d: Demo): number {
  let ok = 0;
  d.data.X.forEach((p, i) => {
    if ((d.net.predict(p)[0] > 0.5 ? 1 : 0) === d.data.y[i]) ok++;
  });
  return ok / d.data.X.length;
}

/** Steps until every training point is on the right side (checked every frame's worth of steps), or null within the cap. */
export function stepsToPerfect(seed: number): number | null {
  const d = createDemo(seed);
  while (d.steps < MAX_STEPS) {
    train(d, STEPS_PER_FRAME);
    if (accuracy(d) === 1) return d.steps;
  }
  return null;
}

/** "1974–1980": an era's span from the timeline's own data, so the prose and the timeline agree. */
export function eraSpan(label: string): string {
  const e = ERAS.find((x) => x.label === label);
  if (!e) throw new Error(`no era called ${label}`);
  return `${e.from}–${e.to}`;
}
export const FIRST_WINTER = eraSpan('First AI winter');
export const SECOND_WINTER = eraSpan('Second AI winter');
