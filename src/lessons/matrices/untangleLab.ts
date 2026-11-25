// authored-by: Codex Meridian 6
export type Pair = [number, number];
export type Matrix = [Pair, Pair];
export type Cloud = { positive: Pair[]; negative: Pair[] };
export const tilted: Cloud = {
  positive: [
    [-1, 2],
    [2, -1],
    [1, 1],
  ],
  negative: [
    [1, -2],
    [-2, 1],
    [-1, -1],
  ],
};
export const offset: Cloud = {
  positive: [
    [2, 1],
    [1, 2],
    [2, 2],
  ],
  negative: [
    [1, 0],
    [0, 1],
    [0.5, 0.5],
  ],
};
export const corners: Cloud = {
  positive: [
    [1, 1],
    [-1, -1],
  ],
  negative: [
    [1, -1],
    [-1, 1],
  ],
};
export const identity: Matrix = [
  [1, 0],
  [0, 1],
];
export const dot = ([a, b]: Pair, [x, y]: Pair) => a * x + b * y;
export const determinant = ([a, b]: Matrix) => a[0] * b[1] - a[1] * b[0];
export function transform(p: Pair, m: Matrix, shift: Pair, kink: boolean): Pair {
  const x = dot(m[0], p) + shift[0];
  const y = dot(m[1], p) + shift[1];
  return kink ? [Math.max(0, x), Math.max(0, y)] : [x, y];
}

export function margin(cloud: Cloud, score: (p: Pair) => number) {
  return Math.min(...cloud.positive.map(score), ...cloud.negative.map((p) => -score(p)));
}

// Signed distance to the line through the handles. Either class can be on either side.
export function lineMargin(cloud: Cloud, move: (p: Pair) => Pair, [a, b]: [Pair, Pair]) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const length = Math.hypot(dx, dy);
  if (length < 1e-9) return 0;
  const side = (p: Pair) => {
    const [x, y] = move(p);
    return (dx * (y - a[1]) - dy * (x - a[0])) / length;
  };
  return Math.max(
    margin(cloud, side),
    margin(cloud, (p) => -side(p)),
  );
}

// Enumerate integer ticks to include strict boundaries exactly. Only the first
// row affects the output x. Count the free second rows by multiplicity.
export function search() {
  const ticks = Array.from({ length: 81 }, (_, i) => i - 40);
  let split = 0,
    area = 0,
    shifted = 0;
  for (const a of ticks)
    for (const b of ticks) {
      const row: Pair = [a, b];
      if (margin(tilted, (p) => dot(row, p)) > 0) {
        split += ticks.length ** 2;
        for (const c of ticks)
          for (const d of ticks) {
            const det = Math.abs(a * d - b * c);
            if (det >= 360 && det <= 440) area++;
          }
      }
      for (const bias of ticks) {
        if (margin(offset, (p) => dot(row, p) + bias) > 0) shifted++;
      }
    }
  return { total: ticks.length ** 4, split, area, shifted, shiftTrials: ticks.length ** 3 };
}
