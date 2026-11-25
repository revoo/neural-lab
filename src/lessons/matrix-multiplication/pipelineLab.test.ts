// authored-by: Codex Meridian 6
import { expect, it } from 'vitest';
import { compose, inspect, matches, meets, parts, solvePipeline, swapLast } from './pipelineLab';

it('checks shapes left to right and stops at the first mismatch', () => {
  expect(inspect([0, 7, 5]).widths).toEqual([128]);
  expect(inspect([0, 7, 5]).complete).toBe(false);
  expect(inspect([1, 7]).widths).toEqual([64, 10]);
});

it('counts a narrow intermediate layer as a hidden layer', () => {
  const id = (a: number, b: number) => parts.findIndex(([i, o]) => i === a && o === b);
  const bottleneck = inspect([id(784, 10), id(10, 64), id(64, 10)]);
  expect(bottleneck).toMatchObject({ complete: true, hidden: 2, narrowest: 10, parameters: 9120 });
  expect(meets(bottleneck, 1)).toBe(false);
  expect(meets(bottleneck, 2)).toBe(false);
  expect(meets(bottleneck, 0)).toBe(true);
  const one = inspect([id(784, 64), id(64, 10)]);
  expect(one).toMatchObject({ hidden: 1, narrowest: 64, parameters: 50816 });
  expect(meets(one, 1)).toBe(true);
  expect(meets(one, 2)).toBe(false);
  expect(inspect([id(784, 10)])).toMatchObject({ hidden: 0, complete: true });
});

it('exhaustively prices every chain and every grid recipe', () => {
  const result = solvePipeline();
  expect(result.trials).toBe(1110);
  expect(result.valid).toHaveLength(8);
  expect(result.parameters).toBe(50816);
  expect(result.multiplyAdds).toBe(1908736);
  // each cost brief's minimum is unique: the cost goal is optional, after a working chain
  expect(result.valid.filter((row) => meets(row, 1) && row.parameters === result.parameters)).toHaveLength(1);
  expect(result.valid.filter((row) => meets(row, 2) && row.multiplyAdds === result.multiplyAdds)).toHaveLength(1);
  expect([0, 1, 2].map((layers) => result.valid.filter((row) => meets(row, layers)).length)).toEqual([8, 5, 3]);
  for (const hidden of [0, 1, 2]) {
    const passes = result.valid.filter((row) => meets(row, hidden)).length;
    expect(passes).toBeGreaterThan(1);
    expect(passes / result.trials).toBeLessThan(0.5);
  }
  expect(result.shortest).toBe(3);
  expect(result.exact).toHaveLength(3);
  expect(result.exact.length / result.geometryTrials).toBeLessThan(0.5);
  expect(compose([0, 1, 3])).toEqual([2, -2, 2, 0]);
  expect(matches(swapLast([3, 0, 1]))).toBe(false);
});
