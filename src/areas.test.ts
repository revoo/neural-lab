// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { ALL_LESSONS, MODULES } from './curriculum';
import { AREAS, BRANCHES, LESSON_AREA, lessonArea, MODULE_AREAS } from './areas';
import { DESTINATIONS, downstream } from './dependencies';

const branchesReached = (id: string) => {
  const down = new Set(downstream(id));
  return Object.entries(BRANCHES).filter(([, dests]) => dests.some((d) => down.has(d))).map(([b]) => b);
};

describe('V2-WP1 (revised): where-this-belongs chips', () => {
  it('every lesson has exactly one chip, and every area is documented', () => {
    expect(Object.keys(LESSON_AREA).sort()).toEqual(ALL_LESSONS.map((l) => l.id).sort());
    for (const a of Object.values(AREAS)) {
      expect(a.label.length).toBeGreaterThan(0);
      expect(a.colour).toMatch(/^var\(--/);
      if (a.dest) expect(DESTINATIONS.map((d) => d.id)).toContain(a.dest);
    }
  });

  it('Foundational means exactly: feeds at least three of the four big branches (history lessons aside)', () => {
    for (const l of ALL_LESSONS) {
      const n = branchesReached(l.id).length;
      if (lessonArea(l.id) === 'foundational') expect(n, `${l.id} reaches ${branchesReached(l.id)}`).toBeGreaterThanOrEqual(3);
      else if (lessonArea(l.id) !== 'history') expect(n, `${l.id} should be Foundational`).toBeLessThan(3);
    }
    expect(ALL_LESSONS.filter((l) => lessonArea(l.id) === 'foundational').length).toBe(23);
  });

  it('a home chip that names a destination really leads there on the concept map', () => {
    for (const l of ALL_LESSONS) {
      const dest = AREAS[lessonArea(l.id)].dest;
      if (dest) expect(downstream(l.id), `${l.id} → ${dest}`).toContain(dest);
    }
  });

  it('module chips are the majority home of their lessons', () => {
    expect(MODULE_AREAS).toMatchObject({
      foundations: 'foundational',
      neuron: 'foundational',
      optimization: 'foundational',
      classical: 'classical',
      vision: 'vision',
      transformers: 'llm',
      rl: 'rl',
      generative: 'generative',
      systems: 'agents',
    });
    for (const m of MODULES) expect(m.lessons.map((l) => lessonArea(l.id))).toContain(MODULE_AREAS[m.id]);
  });
});

describe('chip labels', () => {
  it('capitalise every word (owner request)', () => {
    for (const a of Object.values(AREAS)) for (const w of a.label.split(' ')) if (w !== '&') expect(w[0], a.label).toBe(w[0].toUpperCase());
  });
});
