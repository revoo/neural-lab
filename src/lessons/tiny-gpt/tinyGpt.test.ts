// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import text from '../../lib/corpora/grimm.txt?raw';
import { evalWindows, prepareCorpus, VOCAB } from '../../lib/corpora/grimm';
import { trainBpe } from '../../lib/corpora/bpe';
import { Gpt, gptParamCount } from '../../lib/compute/gpt';
import { CPU_RUN, GPU_RUN } from '../../lib/compute/gptReference';
import { CPU_REFERENCE } from '../../lib/compute/gptReferenceResults';
import { nodeFs } from '../../lib/nodeIo';
import { MERGES } from './bpeMerges';
import { runTraining } from '../../lib/compute/gptRun';
import { InlineCpuTrainer } from '../../lib/compute/gptTrainerCore';
import { designSpec } from './shipLab';
import { births, continueText, headLocality, landmarks, parameterGroups, topNext } from './tinyGpt';
import { allDesigns, designKey, designParams, designSteps, gridSummary, meanVal, shipHint, TARGET, type Recorded } from './shipLab';
import { SHIP_RESULTS } from './results';

const corpus = prepareCorpus(text);

async function finalModel() {
  const fs = await nodeFs();
  const b = fs.readFileSync(new URL('../../lib/compute/tinyGpt.cpu.bin', import.meta.url));
  const m = new Gpt(CPU_RUN.cfg);
  m.load(new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  return m;
}

describe('assemble: parameter counts quoted in the lesson', () => {
  it('tiny: 29,696 parameters, 12,704 per block, 86% in the blocks; MLP twice attention', () => {
    const g = parameterGroups(CPU_RUN.cfg);
    const per = g.filter((x) => x.perBlock).reduce((s, x) => s + x.count, 0);
    expect(gptParamCount(CPU_RUN.cfg)).toBe(29696);
    expect(per).toBe(12704);
    expect(Math.round((100 * per * 2) / 29696)).toBe(86);
    expect(g.find((x) => x.id === 'att')!.count).toBe(4224);
    expect(g.find((x) => x.id === 'mlp')!.count).toBe(8352);
    expect(VOCAB * CPU_RUN.cfg.d).toBe(2176);
  });
  it('larger: 348,384 parameters, 111,840 per block (nearly twelve times the tiny one)', () => {
    const g = parameterGroups(GPU_RUN.cfg);
    expect(gptParamCount(GPU_RUN.cfg)).toBe(348384);
    expect(g.filter((x) => x.perBlock).reduce((s, x) => s + x.count, 0)).toBe(111840);
    expect(348384 / 29696).toBeGreaterThan(11.7);
  });
});

describe('tokenisation', () => {
  it('the shipped 500 merges are what BPE learns on the training tales', () => {
    expect(MERGES).toEqual(trainBpe(corpus.text.slice(0, corpus.split), 500, VOCAB).merges);
    expect(568 * 32).toBe(18176);
  }, 30_000);
});

describe('the recorded run, as the prose describes it', () => {
  const ref = CPU_REFERENCE;
  const at = (s: number) => ref.checkpoints.find((c) => c.step === s)!;
  it('milestones at steps 77, 464 and 1,668; never four in five, never six in a row', () => {
    expect(ref.stages).toEqual({ letters: 77, spacing: 464, 'half-real': 1668, 'most-real': null, phrase: null });
    expect(births(ref.checkpoints, ref.calibration).map((b) => b.step)).toEqual([77, 464, 1668]);
    expect(landmarks(ref.checkpoints.map((c) => c.step), births(ref.checkpoints, ref.calibration))).toEqual([0, 77, 464, 1668, 2000]);
  });
  it('thresholds from real passages: KL ≤ 0.059, word length 3.90 ± 0.30, at least 148 real words in a row', () => {
    expect(ref.calibration.klP95).toBeCloseTo(0.059, 3);
    expect(ref.calibration.wordLen).toBeCloseTo(3.9, 2);
    expect(ref.calibration.wordLenP95).toBeCloseTo(0.3, 2);
    expect(ref.calibration.runP05).toBe(148);
  });
  it('real words: one in seven at step 77, 37% at 1,000, 51% at 2,000; loss 1.869 → 1.712', () => {
    expect(at(77).stats.realShare).toBeCloseTo(0.147, 3);
    expect(at(1000).stats.realShare).toBeCloseTo(0.373, 3);
    expect(at(2000).stats.realShare).toBeCloseTo(0.51, 3);
    expect(at(1000).valLoss.toFixed(3)).toBe('1.869');
    expect(at(2000).valLoss.toFixed(3)).toBe('1.712');
    expect(at(0).valLoss.toFixed(3)).toBe('4.227');
    expect(ref.baselines.unigram.toFixed(2)).toBe('2.99');
  });
});

describe('open it up: what the trained model does', () => {
  it('next-character bets quoted in the lesson', async () => {
    const m = await finalModel();
    const gret = topNext(m, 'Hansel and Gret', 4);
    expect(gret[0].ch).toBe('e');
    expect((100 * gret[0].p).toFixed(1)).toBe('39.1');
    const wol = topNext(m, 'and the wol', 68);
    expect(wol[0].ch).toBe('d');
    expect((100 * wol[0].p).toFixed(1)).toBe('79.1');
    expect((100 * wol.find((x) => x.ch === 'f')!.p).toFixed(1)).toBe('5.8');
  });
  it('greedy decoding loops', async () => {
    const m = await finalModel();
    expect(continueText(m, 'There was once', 80, 7, 0).startsWith(' to the said the said the was the was the was the had the said')).toBe(true);
  });
  it('first-block heads put 54% to 58% on the current and previous character; second-block heads about 15%', async () => {
    const m = await finalModel();
    const loc = headLocality(m, evalWindows(corpus.val, 32, 64), 32, 64);
    for (const [s, p] of loc[0]) {
      expect(s + p).toBeGreaterThan(0.54);
      expect(s + p).toBeLessThan(0.58);
    }
    for (const [s, p] of loc[1]) {
      expect(s + p).toBeGreaterThan(0.14);
      expect(s + p).toBeLessThan(0.18);
    }
  }, 30_000);
});

describe('Ship the tiny GPT: budget and hints', () => {
  it('every design spends the budget: the smallest gets 12,584 steps, the largest 160', () => {
    const ds = allDesigns();
    expect(ds.length).toBe(108);
    const small = ds.find((x) => x.layers === 1 && x.d === 16 && x.T === 16)!;
    const big = ds.find((x) => x.layers === 3 && x.d === 48 && x.T === 64)!;
    expect([designParams(small), designSteps(small)]).toEqual([4656, 12584]);
    expect([designParams(big), designSteps(big)]).toEqual([91248, 160]);
    // the big design sees fewer characters than one pass over the training tales
    expect(designSteps(big) * 16 * 64).toBeLessThan(corpus.train.length);
  });
  it('hints describe evidence, never a setting to choose', () => {
    const x = { layers: 2, d: 32, heads: 4, T: 64, lr: 0.001 };
    const r: Recorded = { params: designParams(x), steps: designSteps(x), val: [2.3, 2.31], curve: [[[0, 4.2], [10, 4.19], [100, 3], [200, 2.6], [300, 2.4], [400, 2.3]]] };
    const h = shipHint(x, r);
    expect(h).toMatch(/still falling/);
    for (const banned of ['learning rate', 'width', 'head', 'block', 'context', 'try ', 'should'])
      expect(h.toLowerCase()).not.toContain(banned);
  });

  it('the recorded grid: 79 beat pair counts, 10 reach 1.79 (all context 16), best 1.723, fewest parameters 7,936 (three designs)', () => {
    const r = SHIP_RESULTS as unknown as Record<string, Recorded>;
    expect(Object.keys(r).length).toBe(108);
    expect(TARGET).toBe(1.79);
    const s = gridSummary(r, TARGET);
    expect([s.total, s.beatBigram, s.hitTarget]).toEqual([108, 79, 10]);
    expect(designKey(s.best.x)).toBe('L2-d32-h4-T16-lr0.01');
    expect(meanVal(s.best.r).toFixed(3)).toBe('1.723');
    expect(s.best.r.params).toBe(28160);
    expect(s.leanest!.r.params).toBe(7936);
    expect(s.leanestCount).toBe(3);
    const hit = Object.entries(r).filter(([, v]) => meanVal(v) <= TARGET).map(([k]) => k);
    expect(hit.every((k) => k.includes('-T16-'))).toBe(true);
    const mean = (pred: (k: string) => boolean) => {
      const v = Object.entries(r).filter(([k]) => pred(k)).map(([, x]) => meanVal(x));
      return v.reduce((a, b) => a + b, 0) / v.length;
    };
    expect(mean((k) => k.includes('-T16-')).toFixed(3)).toBe('1.921');
    expect(mean((k) => k.includes('-T64-')).toFixed(3)).toBe('2.220');
    expect(Math.max(...Object.values(r).map((x) => Math.abs(x.val[0] - x.val[1]))).toFixed(2)).toBe('0.25');
  });
});

describe("Ship the tiny GPT: the recorded optimums re-train exactly (seed 1; seed 2 comes from the same deterministic script)", () => {
  for (const key of ['L2-d32-h4-T16-lr0.01', 'L2-d16-h4-T16-lr0.01'])
    it(`${key}, seed 1`, async () => {
      const x = allDesigns().find((d) => designKey(d) === key)!;
      const rec = (SHIP_RESULTS as unknown as Record<string, Recorded>)[key];
      for (const [i, seed] of [1].entries()) {
        let last = NaN;
        const spec = designSpec(x, seed);
        await runTraining(spec, corpus, new InlineCpuTrainer(spec.cfg), (c) => void (last = c.valLoss));
        expect(+last.toFixed(4)).toBe(rec.val[i]);
      }
    }, 300_000);
});
