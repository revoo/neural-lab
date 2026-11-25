// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createElement as h } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ModuleRef } from './Term';
import { Src } from './Src';
import { MODULES } from '../curriculum';

describe('ModuleRef', () => {
  it('renders the live module number and title, linked to its first lesson', () => {
    const rl = MODULES.find((m) => m.id === 'rl')!;
    const html = renderToStaticMarkup(h(ModuleRef, { id: 'rl' }));
    expect(html).toBe(`<a href="#/learn/${rl.lessons[0].id}">Module ${rl.number} · Reinforcement Learning</a>`);
  });
  it('short form, and custom children', () => {
    const n = MODULES.find((m) => m.id === 'backprop')!.number;
    expect(renderToStaticMarkup(h(ModuleRef, { id: 'backprop', short: true }))).toContain(`>Module ${n}</a>`);
    expect(renderToStaticMarkup(h(ModuleRef, { id: 'backprop' }, 'the backprop module'))).toContain('>the backprop module</a>');
  });
  it('Src renders a small source link', () => {
    expect(renderToStaticMarkup(h(Src, { href: 'https://example.org' }))).toContain('class="src-link"');
  });
});
