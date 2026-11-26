// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { ALL_LESSONS, MODULES } from './curriculum';
import { HISTORY } from './history';

type Fs = { readdirSync: (p: string) => string[]; existsSync: (p: string) => boolean; readFileSync: (p: string, enc: string) => string };
// node:fs through a computed specifier: the app's tsconfig has no Node types.
const loadFs = async () => (await import(/* @vite-ignore */ 'node:' + 'fs')) as Fs;

/** Lesson MDX with fenced code blocks and {/* … *\/} comments removed (what a reader sees as prose). */
async function lessonSources() {
  const fs = await loadFs();
  const out: { id: string; raw: string; prose: string }[] = [];
  for (const dir of fs.readdirSync('src/lessons')) {
    const path = `src/lessons/${dir}/index.mdx`;
    if (!fs.existsSync(path)) continue;
    const raw = fs.readFileSync(path, 'utf8');
    const prose = raw.replace(/```[\s\S]*?```/g, '').replace(/\{\/\*[\s\S]*?\*\/\}/g, '');
    out.push({ id: dir, raw, prose });
  }
  return out;
}


describe('curriculum guard rails', () => {
  it('no hard-coded module numbers in lesson prose', async () => {
    for (const l of await lessonSources()) {
      const m = l.prose.match(/\bModules? \d+\b/);
      expect(m?.[0], `${l.id}: use <ModuleRef id="…" />`).toBeUndefined();
    }
  });

  it('no relative lesson references in lesson prose', async () => {
    for (const l of await lessonSources()) {
      const m = l.prose.match(/\b(last|previous|next|very next|following) (lesson|module)\b/i);
      expect(m?.[0], `${l.id}: name the lesson with <LessonLink to="…">`).toBeUndefined();
    }
  });

  it('every LessonLink and ModuleRef points at something real', async () => {
    const lessons = new Set(ALL_LESSONS.map((l) => l.id));
    const modules = new Set(MODULES.map((m) => m.id));
    for (const l of await lessonSources()) {
      for (const m of l.raw.matchAll(/<LessonLink\s+to="([^"]+)"/g)) expect(lessons.has(m[1]), `${l.id} → lesson ${m[1]}`).toBe(true);
      for (const m of l.raw.matchAll(/<ModuleRef\s+id="([^"]+)"/g)) expect(modules.has(m[1]), `${l.id} → module ${m[1]}`).toBe(true);
    }
  });

  it('modules are numbered consecutively from 0 or 1, and every lesson appears exactly once', async () => {
    const start = MODULES[0].number;
    expect([0, 1]).toContain(start);
    MODULES.forEach((m, i) => expect(m.number, m.id).toBe(start + i));
    const ids = ALL_LESSONS.map((l) => l.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(MODULES.map((m) => m.id)).size).toBe(MODULES.length);
    // every registered lesson has its folder, and every lesson folder is registered
    const dirs = (await lessonSources()).map((l) => l.id).sort();
    expect([...ids].sort()).toEqual(dirs);
  });

  it('every Maths Toolkit lesson bridges to a later module and asks "Why this, and not something else?"', async () => {
    const order = ALL_LESSONS.map((l) => l.id);
    const src = new Map((await lessonSources()).map((l) => [l.id, l.raw]));
    for (const l of MODULES.find((m) => m.id === 'foundations')!.lessons) {
      const m = src.get(l.id)!.match(/<Bridge to="([^"]+)"/);
      expect(m, `${l.id} needs a <Bridge to="…">`).toBeTruthy();
      const target = ALL_LESSONS.find((x) => x.id === m![1]);
      expect(target, `${l.id} → ${m![1]}`).toBeTruthy();
      expect(target!.module.id, `${l.id} bridges inside its own module`).not.toBe('foundations');
      expect(order.indexOf(target!.id)).toBeGreaterThan(order.indexOf(l.id));
      // WP1.10: the comparison section, placed before the lab
      const raw = src.get(l.id)!;
      const why = raw.indexOf('\n## Why this, and not something else?\n');
      expect(why, `${l.id} needs "## Why this, and not something else?"`).toBeGreaterThan(-1);
      const lab = raw.indexOf('\n## Lab:');
      if (lab > -1) expect(why, `${l.id}: the comparison comes before the lab`).toBeLessThan(lab);
      expect(raw.slice(why), `${l.id}: the comparison refers to the three tests`).toMatch(/three tests/);
    }
  });

  it('summaries are plain: at most 30 words, no symbols', () => {
    const symbols = /[\u0370-\u03ff\u1d00-\u1dbf\u2070-\u209f=^_~²³]/;
    const bad = ALL_LESSONS.filter((l) => l.summary.split(/\s+/).length > 30 || symbols.test(l.summary)).map((l) => `${l.id}: ${l.summary}`);
    expect(bad).toEqual([]);
  });
  it('every module has an era, and every history card of its lessons falls inside it', () => {
    const THIS_YEAR = 2026;
    /** "1960s to 2018, then …" → [[1960, 2018], …]; a decade starts at its first year and ends at its last. */
    const spans = (era: string) =>
      era.split(', then ').map((seg) => {
        const ends = seg.split(' to ').map((t, k) => {
          if (t === 'today') return THIS_YEAR;
          const m = /^(\d{4})(s?)$/.exec(t);
          if (!m) throw new Error(`bad era segment "${seg}"`);
          return Number(m[1]) + (m[2] && k === 1 ? 9 : 0);
        });
        return [ends[0], ends[ends.length - 1]];
      });
    for (const m of MODULES) {
      expect(m.era, m.id).toBeTruthy();
      const ss = spans(m.era!);
      for (const [a, b] of ss) expect(b, m.id).toBeGreaterThanOrEqual(a);
      const ids = new Set(m.lessons.map((l) => l.id));
      for (const h of HISTORY.filter((x) => ids.has(x.lessonId)))
        for (const y of h.years) expect(ss.some(([a, b]) => y >= a && y <= b), `${m.id} (${m.era}) vs ${h.id} ${y}`).toBe(true);
    }
  });
});

