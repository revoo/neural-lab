// authored-by: Codex Meridian 6
export type Point = [number, number, number];
export const start: Point = [1, -1, 0];
export const targetLoss = 0.035;
export function loss([x, y, z]: Point) {
  return 0.5 * (x * x + x * y + 2 * y * y + 0.001 * z * z);
}
export function gradient([x, y, z]: Point): Point {
  return [x + y / 2, x / 2 + 2 * y, 0.001 * z];
}
export function move(point: Point, direction: Point, length: number): Point {
  return [point[0] + direction[0] * length, point[1] + direction[1] * length, point[2] + direction[2] * length];
}
export const axes: Point[] = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];
export function unit(point: Point): Point {
  const length = Math.hypot(...point) || 1;
  return [point[0] / length, point[1] / length, point[2] / length];
}

export function minTurns() {
  const queue: { ticks: Point; turns: number }[] = [{ ticks: [10, -10, 0], turns: 0 }];
  const visited = new Set(['10,-10,0']);
  for (let i = 0; i < queue.length; i++) {
    const { ticks, turns } = queue[i];
    if (loss(move([0, 0, 0], ticks, 0.1)) < targetLoss) return turns;
    for (const axis of axes) {
      const next = move(ticks, axis, 1);
      const key = next.join(',');
      if (!visited.has(key)) {
        visited.add(key);
        queue.push({ ticks: next, turns: turns + 1 });
      }
    }
  }
  throw new Error('The grid has no solution');
}

export function fixedStep(rate: number, direction: (p: Point) => Point = (p) => move([0, 0, 0], gradient(p), -1)) {
  let point = start;
  for (let turns = 1; turns <= 100; turns++) {
    point = move(point, unit(direction(point)), rate);
    if (loss(point) < targetLoss) return turns;
  }
  return Infinity;
}
export function sweep() {
  return Array.from({ length: 200 }, (_, i) => (i + 1) / 100).map((rate) => ({ rate, turns: fixedStep(rate) }));
}

export function contourError(from: Point, to: Point) {
  const direction = move(to, from, -1);
  const derivative = gradient(from).reduce((sum, value, i) => sum + value * direction[i], 0);
  const curvature = 2 * loss(direction);
  const t = curvature > 0 ? Math.max(0, Math.min(1, -derivative / curvature)) : 0;
  return Math.max(...[from, to, move(from, direction, t)].map((p) => Math.abs(loss(p) - loss(start))));
}

export function keyboardTrial(a: number, b: number) {
  return loss(move(start, [a, b, 0], 0.1)) < targetLoss;
}
