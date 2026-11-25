// authored-by: Codex Meridian 6
import { expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import {
  allTrees,
  average,
  bounds,
  depths,
  distributions,
  extra,
  huffman,
  leaves,
  minima,
  split,
  wrongCost,
} from './compressorLab';

it('pins Huffman costs, entropies and the cost of using uniform beliefs', () => {
  expect(minima.map((value) => Number(value.toFixed(8)))).toEqual([1.75, 1.984375, 1.9]);
  expect(bounds.map((value) => Number(value.toFixed(4)))).toEqual([1.75, 1.9844, 1.8464]);
  expect(wrongCost).toBe(2);
  expect(extra).toBeCloseTo(0.25, 12);
  expect(wrongCost - bounds[0]).toBeCloseTo(extra, 12);
});
it('finds multiple optimal trees, with most trees failing the target', () => {
  for (const index of [0, 2]) {
    const p = distributions[index];
    const costs = [...allTrees(leaves(p))].map((tree) => average(tree, p)!);
    expect(Math.min(...costs)).toBeCloseTo(minima[index], 12);
    const wins = costs.filter((value) => Math.abs(value - minima[index]) < 1e-12);
    expect(wins.length).toBeGreaterThan(1);
    expect(wins.length / costs.length).toBeLessThan(0.5);
  }
  const p = distributions[1],
    tree = huffman(p);
  if (Array.isArray(tree)) throw Error('Expected a split');
  expect(average({ yes: tree.no, no: tree.yes }, p)).toBe(minima[1]);
  const rng = createRng(17);
  const randomTree = (items: number[]): ReturnType<typeof huffman> => {
    if (items.length === 1) return items;
    const mask = 1 + rng.int((1 << items.length) - 2);
    return {
      yes: randomTree(items.filter((_, i) => mask & (1 << i))),
      no: randomTree(items.filter((_, i) => !(mask & (1 << i)))),
    };
  };
  const sample = Array.from({ length: 500 }, () => average(randomTree(leaves(p)), p)!);
  expect(sample.filter((value) => Math.abs(value - minima[1]) < 1e-12).length / sample.length).toBeLessThan(0.5);
});
it('splits a pending group without losing or duplicating items', () => {
  const tree = split(split(split([0, 1, 2, 3], '', [0]), 'n', [1]), 'nn', [2]);
  expect(depths(tree)).toEqual([
    { item: 0, depth: 1 },
    { item: 1, depth: 2 },
    { item: 2, depth: 3 },
    { item: 3, depth: 3 },
  ]);
});
