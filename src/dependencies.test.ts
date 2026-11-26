// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { ALL_LESSONS } from './curriculum';
import { ALL_NODES, allEdges, DESTINATIONS, downstream, PREREQS, prereqStatus, SHORT, topoSort, upstream } from './dependencies';

type Fs = { readFileSync: (p: string, enc: string) => string };
const loadFs = async () => (await import(/* @vite-ignore */ 'node:' + 'fs')) as Fs;

describe('V2-WP2 dependency graph', () => {
  const lessonIds = new Set(ALL_LESSONS.map((l) => l.id));

  it('every lesson is a node with a short label, and every edge endpoint exists', () => {
    expect(Object.keys(PREREQS).sort()).toEqual([...lessonIds].sort());
    for (const id of lessonIds) expect(SHORT[id], id).toBeTruthy();
    const nodes = new Set(ALL_NODES);
    for (const [a, b] of allEdges()) {
      expect(lessonIds.has(a), `${a} → ${b}`).toBe(true);
      expect(nodes.has(b), `${a} → ${b}`).toBe(true);
    }
  });

  it('is strictly acyclic, and every prerequisite comes earlier in the course', () => {
    expect(topoSort()).not.toBeNull();
    expect(topoSort()!.length).toBe(ALL_NODES.length);
    const order = ALL_LESSONS.map((l) => l.id);
    for (const [to, from] of Object.entries(PREREQS)) for (const f of from) expect(order.indexOf(f), `${f} → ${to}`).toBeLessThan(order.indexOf(to));
    // a cycle would be caught
    expect(topoSort(['a', 'b'], [['a', 'b'], ['b', 'a']])).toBeNull();
  });

  it('has no orphans: every lesson has at least one edge, every destination at least one need', () => {
    const touched = new Set(allEdges().flat());
    for (const id of lessonIds) expect(touched.has(id), id).toBe(true);
    for (const d of DESTINATIONS) expect(d.needs.length, d.id).toBeGreaterThan(0);
  });

  it('every <Bridge to> in the Maths Toolkit appears as an edge', async () => {
    const fs = await loadFs();
    const edges = new Set(allEdges().map(([a, b]) => `${a}→${b}`));
    const toolkit = ALL_LESSONS.filter((l) => l.module.id === 'foundations');
    expect(toolkit.length).toBe(9);
    for (const l of toolkit) {
      const src = fs.readFileSync(`src/lessons/${l.id}/index.mdx`, 'utf8');
      for (const m of src.matchAll(/<Bridge to="([^"]+)"/g)) expect(edges.has(`${l.id}→${m[1]}`), `${l.id} → ${m[1]}`).toBe(true);
    }
  });

  it('upstream and downstream are closures, in course order', () => {
    expect(upstream('vectors')).toEqual(['prologue']);
    expect(upstream('neuron')).toEqual(['prologue', 'vectors', 'dot-product']);
    const up = upstream('dest:llms');
    expect(up).toContain('attention');
    expect(up).toContain('backpropagation');
    expect(up).not.toContain('gans');
    expect(downstream('chain-rule')).toContain('backpropagation');
    expect(downstream('chain-rule')).toContain('dest:llms');
    expect(downstream('open-questions')).toEqual([]);
  });

  it('prerequisite progress names the earliest unfinished prerequisite', () => {
    expect(prereqStatus('neuron', {})).toEqual({ total: 3, done: 0, next: 'prologue' });
    expect(prereqStatus('neuron', { prologue: 1, vectors: 1 })).toEqual({ total: 3, done: 2, next: 'dot-product' });
    expect(prereqStatus('neuron', { prologue: 1, vectors: 1, 'dot-product': 1 }).next).toBeUndefined();
  });
});
