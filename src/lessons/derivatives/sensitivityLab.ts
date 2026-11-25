// authored-by: Codex Meridian 6
const f32 = Math.fround;
export const experiments = [
  { title: 'Smooth cubic', at: 1.3, tolerance: 0.01, budget: 8, run: (x: number) => 0.5 * x ** 3 - x, slopes: [1.535] },
  {
    title: 'Two probes',
    at: 0.7,
    tolerance: 0.001,
    budget: 2,
    run: (x: number) => (x - 2) ** 2 + 0.2 * x ** 3,
    slopes: [-2.306],
  },
  {
    title: 'The corner',
    at: 0,
    tolerance: 0.01,
    budget: 8,
    run: (x: number) => (x < 0 ? -0.6 * x : 1.7 * x),
    slopes: [-0.6, 1.7],
  },
  {
    title: 'Float32',
    at: 1.3,
    tolerance: 0.0001,
    budget: 2,
    run: (x: number) => {
      const input = f32(x);
      return f32(f32(0.5 * f32(f32(input * input) * input)) - input);
    },
    slopes: [1.535],
  },
];

export type Probe = { x: number; y: number };
export function secant(a: Probe, b: Probe) {
  return a.x === b.x ? undefined : (a.y - b.y) / (a.x - b.x);
}

export function survey(index: number) {
  const { at, run, slopes, tolerance } = experiments[index];
  const probe = (x: number) => ({ x, y: run(x) });
  return Array.from({ length: 10 }, (_, i) => {
    const h = 10 ** (-i - 1);
    return {
      h,
      methods: [
        [secant(probe(at), probe(at + h))!],
        [secant(probe(at - h), probe(at + h))!],
        [secant(probe(at - h), probe(at))!],
      ].map(([value], method) => {
        const side = index === 2 && method !== 2 ? 1 : 0;
        const error = Math.abs(value - slopes[side]);
        return { value, error, passes: error <= tolerance };
      }),
    };
  });
}

export function accepts(index: number, estimates: (number | undefined)[]) {
  const { slopes, tolerance } = experiments[index];
  return slopes.every((slope, i) => estimates[i] !== undefined && Math.abs(estimates[i]! - slope) <= tolerance);
}
