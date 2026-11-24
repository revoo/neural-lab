/**
 * Pure logic for the hand-wired neuron workbench: a tiny network of step neurons
 * (fires 1 when its weighted sum is strictly positive) with an optional hidden layer.
 */
export type Neuron = { w: number[]; b: number };
export type Circuit = { hidden: Neuron[]; out: Neuron };

export const fires = (n: Neuron, inputs: number[]) => (n.w.reduce((s, w, i) => s + w * inputs[i], n.b) > 0 ? 1 : 0);
export const tally = (n: Neuron, inputs: number[]) => n.w.reduce((s, w, i) => s + w * inputs[i], n.b);

export function run(c: Circuit, a: number, b: number) {
  const inputs = [a, b];
  const h = c.hidden.map((n) => fires(n, inputs));
  const outIn = c.hidden.length ? h : inputs;
  return { hidden: h, out: fires(c.out, outIn), outTally: tally(c.out, outIn) };
}

export const ROWS: [number, number][] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

export function score(c: Circuit, target: (a: number, b: number) => number) {
  const results = ROWS.map(([a, b]) => run(c, a, b).out === target(a, b));
  return { results, solved: results.every(Boolean) };
}

/** "Material cost": total size of all weights and biases, like the cost of parts in a machine. */
export function cost(c: Circuit) {
  const nc = (n: Neuron) => n.w.reduce((s, w) => s + Math.abs(w), Math.abs(n.b));
  return c.hidden.reduce((s, n) => s + nc(n), nc(c.out));
}
