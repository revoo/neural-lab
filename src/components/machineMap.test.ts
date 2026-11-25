// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createElement as h } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MACHINE_STAGES, MachineMap } from './MachineMap';
import { MODULES } from '../curriculum';

describe('MachineMap', () => {
  it('has one stage per Maths Toolkit lesson, each linking to it', () => {
    const toolkit = MODULES.find((m) => m.id === 'foundations')!.lessons.map((l) => l.id);
    expect(MACHINE_STAGES.map((s) => s.lesson).sort()).toEqual([...toolkit].sort());
    const html = renderToStaticMarkup(h(MachineMap, { highlight: 'gradient' }));
    for (const s of MACHINE_STAGES) expect(html).toContain(`href="#/learn/${s.lesson}"`);
  });
  it('highlights the given stage and says "You are here"', () => {
    const html = renderToStaticMarkup(h(MachineMap, { highlight: 'gradient' }));
    expect(html.match(/class="mm-box on"/g)?.length).toBe(1);
    expect(html).toContain('You are here: <strong>every knob’s slope</strong>');
  });
});
