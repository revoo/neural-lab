import { describe, expect, it } from 'vitest';
import { Clinic, HEALTHY, buildData, duplicateFraction, inputRanges, neighbourAgreement, paramCount } from './clinic';
import { CASES, singleChanges } from './cases';
import { FIXES } from './fixes';
import { fixesFor } from './ModelClinic';

const caseRun = (id: string) => new Clinic(CASES.find((c) => c.id === id)!.cfg).run();

describe('clinic engine', () => {
  it('is deterministic', () => {
    const a = new Clinic(HEALTHY).run();
    const b = new Clinic(HEALTHY).run();
    expect(a.last).toEqual(b.last);
  });
  it('the healthy configuration works', () => {
    expect(new Clinic(HEALTHY).run().last.freshAcc).toBeGreaterThan(0.97);
  });
  it('unshuffled data is sorted by class; leak copies training points into the test set', () => {
    const d = buildData({ ...HEALTHY, shuffle: false });
    const firstOne = d.train.findIndex((p) => p.label === 1);
    expect(d.train.slice(firstOne).every((p) => p.label === 1)).toBe(true);
    const c = new Clinic({ ...HEALTHY, leak: 0.5 });
    expect(duplicateFraction(c)).toBeCloseTo(0.5, 2);
  });
});

/** Each case fails the way its brief and explanation say. */
describe('the case files', () => {
  it('flatline: loss over 4 within five steps, 94% saturated by step 50, all by the end, stuck at chance', () => {
    const c = new Clinic(CASES[0].cfg);
    expect(c.initialTrouble).toBe(0);
    c.step(5);
    c.record();
    expect(c.last.trainLoss).toBeGreaterThan(4);
    c.step(45);
    expect(c.troubleFraction()).toBeCloseTo(0.94, 2);
    const r = caseRun('flatline');
    expect(r.troubleFraction()).toBe(1);
    expect(r.last.trainLoss).toBeGreaterThan(8);
    expect(Math.max(...r.layerStats().map((s) => s.gradRms))).toBeLessThan(1e-7);
    expect(r.last.freshAcc).toBeCloseTo(0.5, 1);
  });
  it('tortoise: 87% fresh, loss still falling at the end, nothing dead', () => {
    const r = caseRun('tortoise');
    expect(r.last.freshAcc).toBeCloseTo(0.87, 2);
    const h = r.history;
    expect(h[h.length - 1].trainLoss).toBeLessThan(h[h.length - 11].trainLoss);
    expect(r.troubleFraction()).toBe(0);
  });
  it('dead on arrival: every ReLU dead at step 0, loss ln 2 throughout, zero gradient', () => {
    const r = caseRun('doa');
    expect(r.initialTrouble).toBe(1);
    for (const h of r.history) expect(h.trainLoss).toBeCloseTo(Math.LN2, 2);
    expect(r.layerStats().every((s) => s.gradRms === 0)).toBe(true);
  });
  it('too stiff: training and test both around 40%', () => {
    const r = caseRun('stiff');
    expect(r.last.trainAcc).toBeLessThan(0.5);
    expect(r.last.testAcc).toBeLessThan(0.5);
  });
  it('memoriser: 609 parameters, train 100%, test 70%, fresh 75%; no duplicates', () => {
    const c = CASES.find((k) => k.id === 'memoriser')!;
    expect(paramCount(c.cfg)).toBe(609);
    const r = caseRun('memoriser');
    expect(r.last.trainAcc).toBe(1);
    expect(r.last.testAcc).toBeCloseTo(0.7, 2);
    expect(r.last.freshAcc).toBeCloseTo(0.75, 2);
    expect(duplicateFraction(r)).toBe(0);
  });
  it('too good to be true: 90% duplicates, test 93%, fresh 75%', () => {
    const r = caseRun('too-good');
    expect(duplicateFraction(r)).toBeCloseTo(0.9, 2);
    expect(r.last.testAcc).toBeCloseTo(0.93, 2);
    expect(r.last.freshAcc).toBeCloseTo(0.75, 2);
  });
  it('nothing to learn: about half the labels agree with their neighbours; everything near 50%', () => {
    const r = caseRun('random');
    expect(neighbourAgreement(r)).toBeCloseTo(0.5, 1);
    expect(neighbourAgreement(new Clinic(HEALTHY))).toBeGreaterThan(0.95);
    expect(r.last.freshAcc).toBeCloseTo(0.5, 1);
    expect(r.last.trainLoss).toBeCloseTo(Math.LN2, 1);
  });
  it('lost in translation: x spans ±160, y about ±1; all 8 first-layer neurons saturated; about two-thirds accuracy', () => {
    const r = caseRun('units');
    const [x, y] = inputRanges(r);
    expect(x.max).toBeCloseTo(160, 0);
    expect(x.min).toBeCloseTo(-160, 0);
    expect(Math.max(Math.abs(y.min), Math.abs(y.max))).toBeLessThan(1.2);
    expect(r.layerStats()[0].saturated).toBe(8);
    expect(r.last.freshAcc).toBeCloseTo(0.66, 2);
  });
  it('see-saw: batches of a single class; accuracy alternates; ends barely better than chance', () => {
    const r = caseRun('seesaw');
    expect(r.batchMix.every((m) => m === 0 || m === 1 || (m > 0 && m < 1))).toBe(true);
    expect(r.batchMix.filter((m) => m === 0).length).toBeGreaterThan(40);
    expect(r.batchMix.filter((m) => m === 1).length).toBeGreaterThan(40);
    expect(r.last.freshAcc).toBeLessThan(0.65);
    expect(new Clinic({ ...CASES.find((k) => k.id === 'seesaw')!.cfg, shuffle: true }).run().last.freshAcc).toBeGreaterThan(0.95);
  });
  it('quiet death: alive at step 0, about two-thirds dead at the end, around 65% fresh', () => {
    const r = caseRun('quiet');
    expect(r.initialTrouble).toBe(0);
    expect(r.troubleFraction()).toBeCloseTo(0.69, 2);
    expect(r.last.freshAcc).toBeCloseTo(0.645, 3);
    expect(r.diverged).toBe(false);
  });
});

describe('recorded one-change fixes', () => {
  it('cover exactly the single-change menu of each case', () => {
    for (const c of CASES) expect(FIXES[c.id].map((r) => r.label)).toEqual(singleChanges(c.cfg).map((ch) => ch.label));
  });
  it('match the claims in the debrief', () => {
    const fixes = (id: string) => fixesFor(CASES.find((c) => c.id === id)!).map((r) => r.label);
    expect(fixes('random')).toEqual(['fix the labels']);
    expect(fixes('doa').sort()).toEqual(['activation tanh', 'standard initialisation']);
    expect(fixes('memoriser').sort()).toEqual(['activation relu', 'collect 4× more data']);
    expect(fixes('units').sort()).toEqual(['normalise the inputs', 'optimizer adam']);
    expect(fixes('seesaw').sort()).toEqual(['8000 steps', 'activation relu', 'batch size 50', 'shuffle the training data']);
    expect(fixes('tortoise').every((l) => l.startsWith('step size'))).toBe(true);
    expect(fixes('tortoise')).not.toContain('8000 steps');
    expect(fixes('stiff').every((l) => l.startsWith('hidden'))).toBe(true);
    expect(fixes('stiff').length).toBe(4);
    // "a bigger network fixes exactly one case; longer training fixes one"
    const withHidden = CASES.filter((c) => fixes(c.id).some((l) => l.startsWith('hidden')));
    expect(withHidden.map((c) => c.id)).toEqual(['stiff']);
    const withSteps = CASES.filter((c) => fixes(c.id).includes('8000 steps'));
    expect(withSteps.map((c) => c.id)).toEqual(['seesaw']);
  });
  it('are genuine: re-training a sample reproduces them', () => {
    const sample: [string, string][] = [
      ['random', 'fix the labels'],
      ['units', 'optimizer adam'],
      ['doa', 'activation tanh'],
      ['tortoise', '8000 steps'],
    ];
    for (const [id, label] of sample) {
      const c = CASES.find((k) => k.id === id)!;
      const ch = singleChanges(c.cfg).find((x) => x.label === label)!;
      const fresh = new Clinic(ch.apply(c.cfg)).run(100).last.freshAcc;
      expect(fresh).toBeCloseTo(FIXES[id].find((r) => r.label === label)!.fresh, 4);
    }
  }, 30000);
  it('the ReLU fix for the memoriser is luck; more data mostly is not (seeds 2–5, as the debrief says)', () => {
    const c = CASES.find((k) => k.id === 'memoriser')!;
    const fresh = (p: object, seed: number) => new Clinic({ ...c.cfg, ...p, seed }).run(500).last.freshAcc;
    const relu = [2, 3, 4, 5].map((s) => Math.round(fresh({ activation: 'relu' }, s) * 100));
    expect(relu).toEqual([79, 73, 84, 81]);
    const better = [2, 3, 4, 5].filter((s) => fresh({ n: 240 }, s) > fresh({}, s)).length;
    expect(better).toBe(3);
  }, 60000);
});
