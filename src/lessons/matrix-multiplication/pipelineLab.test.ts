// authored-by: Codex Meridian 6
import { expect, it } from 'vitest';
import { compose, inspect, matches, solvePipeline, swapLast } from './pipelineLab';

it('checks shapes left to right and stops at the first mismatch', () => {
  expect(inspect([0, 7, 5]).widths).toEqual([128]);
  expect(inspect([0, 7, 5]).complete).toBe(false);
  expect(inspect([1, 7]).widths).toEqual([64, 10]);
});

it('exhaustively prices every chain and every grid recipe', () => {
  const result = solvePipeline();
  expect(result.trials).toBe(1110);
  expect(result.valid).toHaveLength(8);
  expect(result.parameters).toBe(9120);
  expect(result.multiplyAdds).toBe(1908736);
  for (const hidden of [0, 1, 2]) {
    const passes = result.valid.filter((row) => row.hidden >= hidden).length;
    expect(passes).toBeGreaterThan(1);
    expect(passes / result.trials).toBeLessThan(0.5);
  }
  expect(result.shortest).toBe(3);
  expect(result.exact).toHaveLength(3);
  expect(result.exact.length / result.geometryTrials).toBeLessThan(0.5);
  expect(compose([0, 1, 3])).toEqual([2, -2, 2, 0]);
  expect(matches(swapLast([3, 0, 1]))).toBe(false);
});
