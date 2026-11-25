// authored-by: Codex Meridian 6
export const parts = [
  [784, 128],
  [784, 64],
  [784, 10],
  [128, 128],
  [128, 64],
  [128, 10],
  [64, 128],
  [64, 10],
  [10, 128],
  [10, 64],
] as const;
export const batch = 32;
export const maxLayers = 3;

export function inspect(chain: number[]) {
  let width = 784,
    parameters = 0,
    hidden = 0;
  const widths: number[] = [];
  for (const id of chain) {
    const [input, output] = parts[id];
    if (input !== width) break;
    parameters += input * output;
    hidden += Number(output >= 64);
    width = output;
    widths.push(width);
  }
  return {
    widths,
    parameters,
    multiplyAdds: batch * parameters,
    hidden,
    complete: widths.length === chain.length && width === 10,
  };
}

type Matrix = readonly [number, number, number, number];
export const transforms = [
  { name: 'Rotate 90°', matrix: [0, -1, 1, 0] },
  { name: 'Shear right', matrix: [1, 1, 0, 1] },
  { name: 'Flip left–right', matrix: [-1, 0, 0, 1] },
  { name: 'Double size', matrix: [2, 0, 0, 2] },
] as const;

export function compose(chain: number[]): Matrix {
  return chain.reduce<Matrix>(
    ([a, b, c, d], id) => {
      const [e, f, g, h] = transforms[id].matrix;
      return [e * a + f * c, e * b + f * d, g * a + h * c, g * b + h * d];
    },
    [1, 0, 0, 1],
  );
}

export const target = compose([0, 1, 3]);
export const matches = (chain: number[]) => compose(chain).every((entry, i) => entry === target[i]);
export function swapLast(chain: number[]) {
  return [...chain.slice(0, -2), ...chain.slice(-2).reverse()];
}

function* sequences(options: number, limit: number, prefix: number[] = []): Generator<number[]> {
  if (prefix.length) yield prefix;
  if (prefix.length === limit) return;
  for (let id = 0; id < options; id++) yield* sequences(options, limit, [...prefix, id]);
}

export function solvePipeline() {
  const chains = [...sequences(parts.length, maxLayers)].map((chain) => ({ chain, ...inspect(chain) }));
  const valid = chains.filter((row) => row.complete);
  const geometry = [...sequences(transforms.length, 4)];
  const exact = geometry.filter(matches);
  return {
    trials: chains.length,
    valid,
    parameters: Math.min(...valid.filter((row) => row.hidden >= 1).map((row) => row.parameters)),
    multiplyAdds: Math.min(...valid.filter((row) => row.hidden >= 2).map((row) => row.multiplyAdds)),
    geometryTrials: geometry.length,
    shortest: Math.min(...exact.map((chain) => chain.length)),
    exact,
  };
}
