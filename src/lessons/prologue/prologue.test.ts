// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { FIRST_WINTER, MAX_STEPS, SECOND_WINTER, accuracy, createDemo, lossOf, makeCircles, stepsToPerfect, train } from './prologue';

describe('prologue demo', () => {
  it('uses 120 points, half in each group, with a gap between the disc and the ring', () => {
    const d = makeCircles();
    expect(d.X.length).toBe(120);
    expect(d.y.filter((v) => v === 1).length).toBe(60);
    const r = (p: number[]) => Math.hypot(p[0], p[1]);
    const inner = d.X.filter((_, i) => d.y[i] === 0).map(r);
    const ring = d.X.filter((_, i) => d.y[i] === 1).map(r);
    expect(Math.max(...inner)).toBeLessThan(0.6);
    expect(Math.min(...ring)).toBeGreaterThan(0.85);
  });

  it('has 25 adjustable numbers, as the prose says', () => {
    expect(createDemo(1).net.paramCount).toBe(25);
  });

  it('starts near chance and the loss falls as it trains (seed 1)', () => {
    const d = createDemo(1);
    const losses = [lossOf(d)];
    expect(accuracy(d)).toBeLessThan(0.6);
    for (const k of [100, 200, 400, 800]) {
      train(d, k - d.steps);
      losses.push(lossOf(d));
    }
    for (let i = 1; i < losses.length; i++) expect(losses[i]).toBeLessThan(losses[i - 1]);
    expect(accuracy(d)).toBe(1);
  });

  it('separates every point for every random start the Reset button can reach early on (seeds 1-40)', () => {
    const steps = Array.from({ length: 40 }, (_, i) => stepsToPerfect(i + 1));
    expect(steps.every((s) => s !== null && s <= 600)).toBe(true);
    expect(MAX_STEPS).toBeGreaterThan(600);
    expect(steps.slice(0, 3)).toEqual([258, 142, 264]);
  });

  it('takes its winter dates from the timeline data', () => {
    expect(FIRST_WINTER).toBe('1974–1980');
    expect(SECOND_WINTER).toBe('1987–2000');
  });
});
