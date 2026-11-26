// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import text from '../../lib/corpora/grimm.txt?raw';
import { evalWindows, prepareCorpus } from '../../lib/corpora/grimm';
import { Gpt } from '../../lib/compute/gpt';
import { CPU_RUN } from '../../lib/compute/gptReference';
import { CPU_REFERENCE } from '../../lib/compute/gptReferenceResults';
import { GPU_REFERENCE } from '../../lib/compute/gptReferenceGpuResults';
import { CHAR_GROUP_IDS, embeddings, headScores, looCentroidAccuracy, PROBE_L, repeatBatch, uniformShares } from '../../lib/compute/gptProbes';
import { nodeFs } from '../../lib/nodeIo';
import { IND_BAR, PREV_BAR, speakMoments, type SpeakCheckpoint } from './speak';
import { searchSteer, steer, STEER_PROMPT, WINDOW } from './steer';
import { STEER_GREEDY, STEER_RESULTS } from './steerResults';

const REF = CPU_REFERENCE.checkpoints as unknown as SpeakCheckpoint[];
async function finalModel() {
  const fs = await nodeFs();
  const b = fs.readFileSync(new URL('../../lib/compute/tinyGpt.cpu.bin', import.meta.url));
  const m = new Gpt(CPU_RUN.cfg);
  m.load(new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  return m;
}

describe('learning to speak: the measured timeline', () => {
  it('moments at steps 36, 77, 464, 599, 774, 1,668; no induction head; never four in five or six in a row', () => {
    expect(speakMoments(REF, CPU_REFERENCE.calibration).map((m) => [m.id, m.step])).toEqual([
      ['embed', 36],
      ['letters', 77],
      ['spacing', 464],
      ['prev-head', 599],
      ['prev-heads', 774],
      ['half-real', 1668],
    ]);
  });
  it('bars: even attention scores 0.074 (previous) and 0.028 (induction); heads count at twice that', () => {
    const [p, q] = uniformShares(PROBE_L);
    expect(p.toFixed(3)).toBe('0.074');
    expect(q.toFixed(3)).toBe('0.028');
    expect(PREV_BAR.toFixed(3)).toBe('0.147');
    expect(IND_BAR.toFixed(3)).toBe('0.057');
  });
  it('heads at step 2,000: first block 23–27% on the previous character, second block about 7%; induction stays ≈ 0.03', () => {
    const last = REF.at(-1)!.probe;
    for (const v of last.prev[0]) expect(v).toBeGreaterThan(0.23), expect(v).toBeLessThan(0.28);
    for (const v of last.prev[1]) expect(v).toBeGreaterThan(0.06), expect(v).toBeLessThan(0.09);
    for (const c of REF) expect(Math.max(...c.probe.induction.flat())).toBeLessThan(0.032);
    expect(last.lossSecond.toFixed(2)).toBe('6.09');
    expect(last.lossFirst.toFixed(2)).toBe('6.08');
  });
  it('validation loss falls at every checkpoint (no bump)', () => {
    for (let i = 1; i < REF.length; i++) expect(REF[i].valLoss).toBeLessThan(REF[i - 1].valLoss);
  });
  it('embeddings: 24% at step 0, 82% at step 2,000, against a shuffled bar of 34%', async () => {
    expect(REF[0].probe.emb.toFixed(2)).toBe('0.24');
    expect(REF.at(-1)!.probe.emb.toFixed(2)).toBe('0.82');
    expect(REF.at(-1)!.probe.embBar.toFixed(2)).toBe('0.34');
    const m = await finalModel();
    expect(looCentroidAccuracy(embeddings(m), CHAR_GROUP_IDS).toFixed(4)).toBe(REF.at(-1)!.probe.emb.toFixed(4));
  });
  it('a repeated real passage is no easier the second time either', async () => {
    const m = await finalModel();
    const c = prepareCorpus(text);
    const w = evalWindows(c.val, 16, 31);
    const seqs = Array.from({ length: 16 }, (_, r) => {
      const half = Array.from(w.subarray(r * 32, r * 32 + 32));
      return Int32Array.from([...half, ...half]);
    });
    const h = headScores(m, seqs);
    expect(h.lossSecond).toBeGreaterThan(h.lossFirst);
    expect(h.lossFirst.toFixed(3)).toBe('1.856');
    expect(h.lossSecond.toFixed(3)).toBe('1.918');
    // and the recorded probes match a fresh computation on the shipped weights
    const r = headScores(m, repeatBatch(16, PROBE_L));
    expect(r.prev.map((x) => x.map((v) => +v.toFixed(4)))).toEqual(REF.at(-1)!.probe.prev);
  });
});

describe('the larger model', () => {
  it('has no induction head at any checkpoint of its 3,000 steps, and its second copy is not easier', () => {
    const cps = GPU_REFERENCE.checkpoints as unknown as SpeakCheckpoint[];
    expect(cps.at(-1)!.step).toBe(3000);
    for (const c of cps) expect(Math.max(...c.probe.induction.flat())).toBeLessThan(IND_BAR);
    expect(cps.at(-1)!.probe.lossSecond).toBeGreaterThan(cps.at(-1)!.probe.lossFirst);
  });
});

describe('Steer the sentence', () => {
  it('greedy text and every recorded plan reproduce; the search re-finds " king" in one forcing', async () => {
    const m = await finalModel();
    expect(steer(m, STEER_PROMPT, [])).toBe(STEER_GREEDY);
    expect(STEER_GREEDY).toBe(' to the said the said the was ');
    for (const r of STEER_RESULTS) {
      const out = steer(m, STEER_PROMPT, r.plan as unknown as { pos: number; ch: string }[]);
      expect(out.length).toBe(WINDOW);
      expect(out.includes(r.target)).toBe(true);
    }
    expect(STEER_RESULTS.map((r) => [r.target, r.cost, r.exact])).toEqual([
      [' king', 1, true],
      [' daughter', 4, false],
      [' wolf', 3, true],
    ]);
    const k = searchSteer(m, STEER_PROMPT, ' king');
    expect([k.cost, k.plan]).toEqual([1, [{ pos: 4, ch: 'k' }]]);
    expect(steer(m, STEER_PROMPT, k.plan)).toBe(' to king the said the said the');
  });
});
