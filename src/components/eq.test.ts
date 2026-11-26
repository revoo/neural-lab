// authored-by: Gemini Antigravity
import { afterEach, describe, expect, it } from 'vitest';
import { createElement as h } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Eq } from './Eq';
import { getMathMode, setMathMode, subscribeMathMode } from '../mathMode';

afterEach(() => {
  setMathMode('full');
});

describe('Eq and mathMode', () => {
  it('defaults to full mode and renders both word equation and symbolic TeX', () => {
    expect(getMathMode()).toBe('full');
    const html = renderToStaticMarkup(
      h(Eq, { words: '\\text{output} = \\text{score}', tex: 'a = f(z)' }, h('p', null, 'Legend text')),
    );
    expect(html).toContain('class="eq-words"');
    expect(html).toContain('class="eq-tex"');
    expect(html).toContain('class="eq-legend"');
    expect(html).not.toContain('Show symbols');
  });

  it('in words-first mode hides symbolic TeX behind a Show symbols button when words is supplied', () => {
    setMathMode('words-first');
    const html = renderToStaticMarkup(
      h(Eq, { words: '\\text{output} = \\text{score}', tex: 'a = f(z)' }, h('p', null, 'Legend text')),
    );
    expect(html).toContain('eq-words-first');
    expect(html).toContain('class="eq-words"');
    expect(html).toContain('Show symbols');
    expect(html).toContain('class="eq-legend"');
    expect(html).not.toContain('class="eq-tex"');
  });

  it('always shows symbolic TeX when words is omitted, even in words-first mode', () => {
    setMathMode('words-first');
    const html = renderToStaticMarkup(h(Eq, { tex: 'a = f(z)' }, h('p', null, 'Legend text')));
    expect(html).toContain('class="eq-tex"');
    expect(html).not.toContain('Show symbols');
    expect(html).not.toContain('eq-words-first');
  });

  it('notifies subscribers when mode changes', () => {
    let calls = 0;
    const unsub = subscribeMathMode(() => {
      calls += 1;
    });
    setMathMode('words-first');
    setMathMode('words-first'); // idempotent
    setMathMode('full');
    unsub();
    setMathMode('words-first');
    expect(calls).toBe(2);
  });
});
