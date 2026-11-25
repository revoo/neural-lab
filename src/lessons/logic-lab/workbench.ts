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

// ---------------------------------------------------------------- decision lines

/** The values the − / + buttons can reach: −2 … +2 in steps of 0.5. */
export const STEPS = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];

/** Signed distance of point p from the neuron's line w·p + b = 0 (positive on the firing side). */
export const lineDistance = (n: Neuron, p: [number, number]) => (n.w[0] * p[0] + n.w[1] * p[1] + n.b) / Math.hypot(n.w[0], n.w[1]);

/**
 * Snap a dragged line (through p1 and p2) to the nearest neuron the buttons
 * can build. Keeps the firing side pointing the same way as `prev` (normals
 * agree), minimises the two points' distances from the line, and breaks ties
 * by the cheaper parts (smaller |w₁| + |w₂| + |b|).
 */
export function snapLine(p1: [number, number], p2: [number, number], prev: Neuron): Neuron {
  let best: Neuron = prev;
  let bestErr = Infinity;
  let bestCost = Infinity;
  const pw = prev.w[0] === 0 && prev.w[1] === 0 ? [1, 1] : prev.w;
  for (const w1 of STEPS)
    for (const w2 of STEPS) {
      if (w1 === 0 && w2 === 0) continue;
      if (w1 * pw[0] + w2 * pw[1] <= 0) continue;
      for (const b of STEPS) {
        const n = { w: [w1, w2], b };
        const err = Math.abs(lineDistance(n, p1)) + Math.abs(lineDistance(n, p2));
        const c = Math.abs(w1) + Math.abs(w2) + Math.abs(b);
        if (err < bestErr - 1e-9 || (Math.abs(err - bestErr) <= 1e-9 && c < bestCost)) {
          best = n;
          bestErr = err;
          bestCost = c;
        }
      }
    }
  return best;
}

/** One machine-log line for a truth-table row, e.g. "A=1, B=1: 1×1 + 1×1 − 1.5 = 0.5 → fires". */
export function tallyLine(name: string, n: Neuron, inputs: number[], labels: string[]): string {
  const terms = n.w.map((w, i) => `${w}×${inputs[i]}`).join(' + ');
  const t = tally(n, inputs);
  const bias = n.b < 0 ? ` − ${-n.b}` : ` + ${n.b}`;
  const lhs = labels.map((l, i) => `${l}=${inputs[i]}`).join(', ');
  return `${name} · ${lhs}: ${terms}${bias} = ${t} → ${t > 0 ? 'fires' : 'silent'}`;
}
