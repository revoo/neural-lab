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
/** A hidden layer counts as wide at this width. */
export const wide = 64;

export function inspect(chain: number[]) {
  let width = 784,
    parameters = 0;
  const widths: number[] = [];
  for (const id of chain) {
    const [input, output] = parts[id];
    if (input !== width) break;
    parameters += input * output;
    width = output;
    widths.push(width);
  }
  // Every layer between the input and the last one is hidden, however narrow.
  const hiddenWidths = widths.slice(0, -1);
  return {
    widths,
    parameters,
    multiplyAdds: batch * parameters,
    hidden: hiddenWidths.length,
    narrowest: hiddenWidths.length ? Math.min(...hiddenWidths) : Infinity,
    complete: widths.length === chain.length && width === 10,
  };
}

/** Brief rule: a complete chain with at least `layers` hidden layers, every one of them at least `wide` wide. */
export const meets = (row: ReturnType<typeof inspect>, layers: number) =>
  row.complete && row.hidden >= layers && (layers === 0 || row.narrowest >= wide);

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
    parameters: Math.min(...valid.filter((row) => meets(row, 1)).map((row) => row.parameters)),
    multiplyAdds: Math.min(...valid.filter((row) => meets(row, 2)).map((row) => row.multiplyAdds)),
    geometryTrials: geometry.length,
    shortest: Math.min(...exact.map((chain) => chain.length)),
    exact,
  };
}
