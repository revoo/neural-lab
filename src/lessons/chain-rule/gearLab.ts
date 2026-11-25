// authored-by: Codex Meridian 6
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
export const initial = [1, 1, 0.8, 1.2, 1.6];
export const saturated = [1, 1, 2, 8, 1.6];
export function pipeline(knobs: number[], x = 0.7) {
  const [a, b, c, d, e] = knobs;
  const z1 = a * x;
  const z2 = Math.tanh(b * z1);
  const z3 = c * z2 + 1;
  const z4 = sigmoid(d * z3);
  const z5 = e * z4;
  return {
    outputs: [z1, z2, z3, z4, z5],
    inputSlopes: [a, b * (1 - z2 * z2), c, d * z4 * (1 - z4), e],
    knobSlopes: [x, z1 * (1 - z2 * z2), z2, z3 * z4 * (1 - z4), z4],
  };
}
export function sensitivities(knobs: number[]) {
  const { inputSlopes, knobSlopes } = pipeline(knobs);
  return knobSlopes.map((slope, i) => slope * inputSlopes.slice(i + 1).reduce((a, b) => a * b, 1));
}
export function ranked(order: number[], knobs: number[]) {
  const impact = sensitivities(knobs).map(Math.abs);
  return order.every((stage, i) => !i || impact[order[i - 1]] + 1e-12 >= impact[stage]);
}
export function turn(knobs: number[], stage: number, delta: number) {
  return knobs.map((value, i) => (i === stage ? value + delta : value));
}
export const aim = pipeline(initial).outputs[4] + 0.035;
export const revived = 0.03;

// A derivative is its knob slope times a suffix of downstream input slopes.
// A shared suffix is counted once, whichever report first computes it.
export function report(knobs: number[], stage: number, cached: (number | null)[], reuse: boolean) {
  const { inputSlopes, knobSlopes } = pipeline(knobs);
  let work = 0;
  if (!reuse) {
    return {
      value: inputSlopes.slice(stage + 1).reduce((total, slope) => {
        work++;
        return total * slope;
      }, knobSlopes[stage]),
      work,
      costs: cached,
    };
  }
  const costs = [...cached];
  costs[3] = inputSlopes[4];
  for (let i = 2; i >= stage; i--) {
    if (costs[i] != null) continue;
    costs[i] = inputSlopes[i + 1] * costs[i + 1]!;
    work++;
  }
  if (stage === 4) return { value: knobSlopes[stage], work, costs };
  return { value: knobSlopes[stage] * costs[stage]!, work: work + 1, costs };
}

export function* permutations(values: number[]): Generator<number[]> {
  if (!values.length) {
    yield [];
    return;
  }
  for (const value of values) for (const rest of permutations(values.filter((v) => v !== value))) yield [value, ...rest];
}

export function searchWalks() {
  let minimum = Infinity,
    solutions = 0,
    tried = 0;
  for (const order of permutations([0, 1, 2, 3, 4])) {
    for (let reuse = 0; reuse < 32; reuse++) {
      let cache: (number | null)[] = [],
        products = 0;
      for (const stage of order) {
        const result = report(initial, stage, cache, Boolean(reuse & (1 << stage)));
        cache = result.costs;
        products += result.work;
      }
      tried++;
      if (products < minimum) {
        minimum = products;
        solutions = 0;
      }
      if (products === minimum) solutions++;
    }
  }
  return { minimum, solutions, tried };
}
