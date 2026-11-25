// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { createElement as h } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { EraStrip, eraStripItems } from './EraStrip';

describe('EraStrip', () => {
  it('selects the exact hand-computed eras, milestones and history entries for 1969–1980 in date order', () => {
    const items = eraStripItems(1969, 1980);
    expect(items.map((i) => ({ kind: i.kind, year: i.year, id: i.id ?? null, label: i.label, href: i.href }))).toEqual([
      { kind: 'era', year: 1956, id: null, label: 'The founding years', href: '#/timeline' },
      { kind: 'milestone', year: 1969, id: null, label: 'Minsky and Papert’s book Perceptrons', href: '#/timeline' },
      { kind: 'history', year: 1969, id: 'pipeline', label: 'Faster matrix multiplication', href: '#/timeline/pipeline' },
      { kind: 'history', year: 1970, id: 'slipping-gear', label: 'Local slopes and the backward walk', href: '#/timeline/slipping-gear' },
      { kind: 'milestone', year: 1973, id: null, label: 'The Lighthill report on AI research', href: '#/timeline' },
      { kind: 'era', year: 1974, id: null, label: 'First AI winter', href: '#/timeline' },
      { kind: 'history', year: 1975, id: 'reward-hacking', label: 'Reward shaping and specification gaming', href: '#/timeline/reward-hacking' },
      { kind: 'history', year: 1975, id: 'spam-scorer', label: 'Documents as vectors, compared by cosine', href: '#/timeline/spam-scorer' },
    ]);
  });

  it('renders links to the timeline for every selected item', () => {
    const html = renderToStaticMarkup(h(EraStrip, { from: 1969, to: 1980 }));
    expect(html).toContain('1969–1980');
    expect(html).toContain('First AI winter');
    expect(html).toContain('href="#/timeline/pipeline"');
    expect(html).toContain('href="#/timeline/slipping-gear"');
    expect(html).toContain('href="#/timeline"');
  });
});
