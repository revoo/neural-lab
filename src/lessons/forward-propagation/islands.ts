/**
 * A hand-built three-layer network that answers "is this point on land?" for a map with two
 * triangular islands. Layer 1 detects coastlines (one straight stretch each), layer 2 detects
 * islands (all three of an island's coastlines agree), layer 3 says "land" if either island does.
 * Every weight was chosen by hand so each neuron has a clear job.
 */
export type Layer = { W: number[][]; b: number[]; names: string[] };

const sig = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
export const sigmoid = sig;

/** Steepness of the coastline neurons. Larger = sharper coastlines. */
const K = 4;

export const LAYERS: Layer[] = [
  {
    // Rows are neurons; columns are the inputs (east, north).
    // West island: y > −0.8, 2x − y + 3 > 0, −2x − y − 0.6 > 0  (vertices (−1.9,−0.8), (0.1,−0.8), (−0.9,1.2))
    // East island: 1 − y > 0, −2x + y + 2.6 > 0, 2x + y − 1.4 > 0 (vertices (0.2,1), (1.8,1), (1,−0.6))
    W: [
      [0, K],
      [2 * K, -K],
      [-2 * K, -K],
      [0, -K],
      [-2 * K, K],
      [2 * K, K],
    ],
    b: [0.8 * K, 3 * K, -0.6 * K, 1 * K, 2.6 * K, -1.4 * K],
    names: ['west island, south coast', 'west island, west coast', 'west island, east coast', 'east island, north coast', 'east island, east coast', 'east island, west coast'],
  },
  {
    W: [
      [5, 5, 5, 0, 0, 0],
      [0, 0, 0, 5, 5, 5],
    ],
    b: [-12.5, -12.5],
    names: ['west island', 'east island'],
  },
  {
    W: [[6, 6]],
    b: [-3],
    names: ['land'],
  },
];

export const INPUT_NAMES = ['east (x)', 'north (y)'];

export type Trace = { z: number[][]; a: number[][] };

/** Forward pass. a[0] is the input; z[l] and a[l+1] belong to layer l. */
export function forward(x: number[], layers: Layer[] = LAYERS): Trace {
  const a: number[][] = [x];
  const z: number[][] = [];
  for (const L of layers) {
    const prev = a[a.length - 1];
    const zl = L.W.map((row, j) => row.reduce((s, w, i) => s + w * prev[i], L.b[j]));
    z.push(zl);
    a.push(zl.map(sig));
  }
  return { z, a };
}

/** Is (x, y) inside the triangle (exact geometry, for testing the network against). */
export function onLand(x: number, y: number): boolean {
  const west = y > -0.8 && 2 * x - y + 3 > 0 && -2 * x - y - 0.6 > 0;
  const east = 1 - y > 0 && -2 * x + y + 2.6 > 0 && 2 * x + y - 1.4 > 0;
  return west || east;
}

export const paramCount = (layers: Layer[] = LAYERS) => layers.reduce((s, L) => s + L.W.length * L.W[0].length + L.b.length, 0);
export const multiplyAdds = (layers: Layer[] = LAYERS) => layers.reduce((s, L) => s + L.W.length * L.W[0].length, 0);
