// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { ERAS, HISTORY, MILESTONES, claimsOf } from './history';

const isUrl = (u: string) => /^https?:\/\/\S+$/.test(u);

describe('history data', () => {
  it('has one entry per reviewed lab, with unique ids', () => {
    expect(HISTORY.length).toBe(51);
    expect(new Set(HISTORY.map((h) => h.id)).size).toBe(HISTORY.length);
  });

  it('every claim, era mood and milestone has a source URL', () => {
    for (const h of HISTORY) for (const c of claimsOf(h)) {
      expect(isUrl(c.source), `${h.id}: ${c.text}`).toBe(true);
      expect(c.text.length).toBeGreaterThan(10);
    }
    for (const e of ERAS) expect(isUrl(e.mood.source)).toBe(true);
    for (const m of MILESTONES) expect(isUrl(m.source)).toBe(true);
  });

  it('dates are plausible, ascending and match the display string', () => {
    for (const h of HISTORY) {
      expect(h.years.length).toBeGreaterThan(0);
      for (let i = 0; i < h.years.length; i++) {
        expect(h.years[i]).toBeGreaterThanOrEqual(1700);
        expect(h.years[i]).toBeLessThanOrEqual(2026);
        if (i) expect(h.years[i]).toBeGreaterThan(h.years[i - 1]);
        expect(h.when).toContain(String(h.years[i]));
      }
    }
    for (const e of ERAS) expect(e.to).toBeGreaterThan(e.from);
  });

  it('pins key dates from the sources', () => {
    const y = (id: string) => HISTORY.find((h) => h.id === id)!.years;
    expect(y('logic')).toEqual([1943, 1958, 1969]);
    expect(y('backward-pass')).toEqual([1986]);
    expect(y('rescue')).toEqual([1991, 2010, 2015]);
    expect(y('descent-golf')).toEqual([1847]);
    expect(y('shannon')).toEqual([1948, 1951]);
    expect(ERAS.filter((e) => e.kind === 'winter').map((e) => [e.from, e.to])).toEqual([[1974, 1980], [1987, 2000]]);
  });

  it('each entry points at a real lesson whose MDX shows its card', async () => {
    // node:fs through a computed specifier: the app's tsconfig has no Node types.
    const fs = (await import(/* @vite-ignore */ 'node:' + 'fs')) as {
      readdirSync: (p: string) => string[];
      existsSync: (p: string) => boolean;
      readFileSync: (p: string, enc: string) => string;
    };
    const lessons = new Set<string>();
    const cards = new Set<string>();
    for (const dir of fs.readdirSync('src/lessons')) {
      const path = `src/lessons/${dir}/index.mdx`;
      if (!fs.existsSync(path)) continue;
      const src = fs.readFileSync(path, 'utf8');
      lessons.add(dir);
      for (const m of src.matchAll(/<HistoryCard id="([^"]+)"/g)) cards.add(`${dir}:${m[1]}`);
    }
    for (const h of HISTORY) {
      expect(lessons.has(h.lessonId), h.lessonId).toBe(true);
      expect(cards.has(`${h.lessonId}:${h.id}`), `card for ${h.id} in ${h.lessonId}`).toBe(true);
    }
    expect(cards.size).toBe(HISTORY.length);
  });
});

describe('timeline layout', () => {
  it('maps years left to right, with pre-1940 ideas in the stub', async () => {
    const { timelineX, TIMELINE } = await import('./history');
    expect(timelineX(1733)).toBeLessThan(timelineX(1847));
    expect(timelineX(1847)).toBeLessThan(TIMELINE.stubWidth);
    expect(timelineX(1940)).toBe(TIMELINE.stubWidth);
    expect(timelineX(1986)).toBeLessThan(timelineX(2017));
    expect(timelineX(2026)).toBeCloseTo(TIMELINE.width - 12, 9);
  });
  it('lanes keep same-lane dots at least the gap apart', async () => {
    const { assignLanes, timelineX } = await import('./history');
    const xs = HISTORY.map((h) => timelineX(h.years[0]));
    const lanes = assignLanes(xs, 60);
    for (let i = 0; i < xs.length; i++)
      for (let j = i + 1; j < xs.length; j++) if (lanes[i] === lanes[j]) expect(Math.abs(xs[i] - xs[j])).toBeGreaterThanOrEqual(60);
  });
});
