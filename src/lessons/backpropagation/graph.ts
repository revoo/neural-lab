/**
 * The computational graph explored in the backprop lesson: one tanh neuron with two inputs and a squared-error loss.
 * Built from the shared autodiff engine, so every number shown is computed, not scripted.
 */
import { Value } from '../../lib/autodiff';

export type Params = { x1: number; w1: number; x2: number; w2: number; b: number; y: number };

export type NodeKey = 'x1' | 'w1' | 'x2' | 'w2' | 'b' | 'y' | 'm1' | 'm2' | 's' | 'z' | 'a' | 'd' | 'L';

export type NodeInfo = {
  key: NodeKey;
  name: string; // plain-English name
  op?: string; // how it is computed, e.g. "x₁ × w₁"
  v: Value;
  col: number;
  row: number;
  leaf: boolean;
  param?: boolean; // a learnable weight or bias
};

export type EdgeInfo = { from: NodeKey; to: NodeKey; local: number; why: string };

export function buildGraph(p: Params) {
  const x1 = new Value(p.x1, 'x₁');
  const w1 = new Value(p.w1, 'w₁');
  const x2 = new Value(p.x2, 'x₂');
  const w2 = new Value(p.w2, 'w₂');
  const b = new Value(p.b, 'b');
  const y = new Value(p.y, 'y');
  const m1 = x1.mul(w1).named('m₁');
  const m2 = x2.mul(w2).named('m₂');
  const s = m1.add(m2).named('s');
  const z = s.add(b).named('z');
  const a = z.tanh().named('a');
  const d = a.sub(y).named('d');
  const L = d.mul(d).named('L'); // d used twice: its gradient accumulates from two paths
  L.backward();

  const nodes: Record<NodeKey, NodeInfo> = {
    x1: { key: 'x1', name: 'input 1', v: x1, col: 0, row: 0, leaf: true },
    w1: { key: 'w1', name: 'weight 1', v: w1, col: 0, row: 1, leaf: true, param: true },
    x2: { key: 'x2', name: 'input 2', v: x2, col: 0, row: 2.4, leaf: true },
    w2: { key: 'w2', name: 'weight 2', v: w2, col: 0, row: 3.4, leaf: true, param: true },
    b: { key: 'b', name: 'bias', v: b, col: 2, row: 3.9, leaf: true, param: true },
    y: { key: 'y', name: 'target', v: y, col: 4, row: 4.1, leaf: true },
    m1: { key: 'm1', name: 'evidence 1', op: 'x₁ × w₁', v: m1, col: 1, row: 0.5, leaf: false },
    m2: { key: 'm2', name: 'evidence 2', op: 'x₂ × w₂', v: m2, col: 1, row: 2.9, leaf: false },
    s: { key: 's', name: 'total evidence', op: 'm₁ + m₂', v: s, col: 2, row: 1.7, leaf: false },
    z: { key: 'z', name: 'tally', op: 's + b', v: z, col: 3, row: 2.6, leaf: false },
    a: { key: 'a', name: 'output', op: 'tanh(z)', v: a, col: 4, row: 2.6, leaf: false },
    d: { key: 'd', name: 'error', op: 'a − y', v: d, col: 5, row: 3.35, leaf: false },
    L: { key: 'L', name: 'loss', op: 'd × d', v: L, col: 6, row: 3.35, leaf: false },
  };

  const keyOf = new Map<Value, NodeKey>(Object.values(nodes).map((n) => [n.v, n.key]));
  const why: Record<string, string> = {
    '×': 'the other input',
    '+': '1 (adding passes changes straight through)',
    '−': '+1 for the first input, −1 for the second',
    tanh: '1 − a² (tanh’s slope)',
  };
  const edges: EdgeInfo[] = [];
  for (const n of Object.values(nodes))
    for (const e of n.v.parents) edges.push({ from: keyOf.get(e.node)!, to: n.key, local: e.local, why: why[n.v.op] ?? '' });

  const forwardOrder: NodeKey[] = ['m1', 'm2', 's', 'z', 'a', 'd', 'L'];
  const backwardOrder: NodeKey[] = ['L', 'd', 'a', 'z', 'b', 's', 'm1', 'm2', 'w1', 'x1', 'w2', 'x2'];
  return { nodes, edges, forwardOrder, backwardOrder };
}

/** Loss after one gradient-descent update of the three learnable parameters. */
export function lossAfterUpdate(p: Params, lr: number) {
  const g = buildGraph(p);
  const next = { ...p, w1: p.w1 - lr * g.nodes.w1.v.grad, w2: p.w2 - lr * g.nodes.w2.v.grad, b: p.b - lr * g.nodes.b.v.grad };
  return { next, before: g.nodes.L.v.data, after: buildGraph(next).nodes.L.v.data };
}
