// authored-by: Codex Meridian 6
import { entropy, klDivergence } from '../../lib/prob';

export type Tree = number[] | { yes: Tree; no: Tree };
export const distributions = [
  [1 / 2, 1 / 4, 1 / 8, 1 / 8],
  [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 128],
  [0.4, 0.3, 0.2, 0.1],
];
export const balanced: Tree = { yes: { yes: [0], no: [1] }, no: { yes: [2], no: [3] } };
export const leaves = (p: number[]) => p.map((_, i) => i);
export function depths(tree: Tree, depth = 0): { item: number; depth: number | null }[] {
  if (Array.isArray(tree)) return tree.map((item) => ({ item, depth: tree.length === 1 ? depth : null }));
  return [...depths(tree.yes, depth + 1), ...depths(tree.no, depth + 1)];
}
export function average(tree: Tree, p: number[]) {
  const rows = depths(tree);
  if (rows.some((row) => row.depth === null)) return null;
  return rows.reduce((sum, row) => sum + p[row.item] * row.depth!, 0);
}
export function huffman(p: number[]): Tree {
  const queue: { weight: number; tree: Tree }[] = p.map((weight, i) => ({ weight, tree: [i] }));
  while (queue.length > 1) {
    queue.sort((a, b) => a.weight - b.weight);
    const [yes, no] = queue.splice(0, 2);
    queue.push({ weight: yes.weight + no.weight, tree: { yes: yes.tree, no: no.tree } });
  }
  return queue[0].tree;
}
export function split(tree: Tree, path: string, selected: number[]): Tree {
  if (!path.length) {
    if (!Array.isArray(tree)) return tree;
    const yes = tree.filter((item) => selected.includes(item)),
      no = tree.filter((item) => !selected.includes(item));
    return yes.length && no.length ? { yes, no } : tree;
  }
  if (Array.isArray(tree)) return tree;
  const branch = path[0] === 'y' ? 'yes' : 'no';
  return { ...tree, [branch]: split(tree[branch], path.slice(1), selected) };
}
export function* allTrees(items: number[]): Generator<Tree> {
  if (items.length === 1) {
    yield items;
    return;
  }
  for (let mask = 1; mask < (1 << items.length) - 1; mask++) {
    const yes = items.filter((_, i) => mask & (1 << i));
    const no = items.filter((_, i) => !(mask & (1 << i)));
    for (const y of allTrees(yes)) for (const n of allTrees(no)) yield { yes: y, no: n };
  }
}
export const minima = distributions.map((p) => average(huffman(p), p)!);
export const bounds = distributions.map((p) => entropy(p));
export const wrongCost = average(balanced, distributions[0])!;
export const extra = klDivergence(distributions[0], [0.25, 0.25, 0.25, 0.25]);
