// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import text from '../corpora/grimm.txt?raw';
import { charFrequencies, dictionary, prepareCorpus } from '../corpora/grimm';
import { calibrate, sampleStats, stageSteps } from '../corpora/stages';
import { gptParamCount } from './gpt';
import { CPU_RUN, samplesAt } from './gptReference';
import { CPU_REFERENCE } from './gptReferenceResults';
import { GPU_REFERENCE } from './gptReferenceGpuResults';
import { GPU_RUN } from './gptReference';
import { runTraining } from './gptRun';
import { InlineCpuTrainer } from './gptTrainerCore';
import { paramHash } from './gptPinned';
import { nodeFs } from '../nodeIo';

const corpus = prepareCorpus(text);
const trainText = corpus.text.slice(0, corpus.split);
const dict = dictionary(trainText);
const freq = charFrequencies(corpus.train);

describe('the recorded CPU reference run', () => {
  it('re-trains bit for bit: the first checkpoints (to step 100) match the record exactly', async () => {
    const recorded = CPU_REFERENCE.checkpoints;
    let i = 0;
    await runTraining(CPU_RUN, corpus, new InlineCpuTrainer(CPU_RUN.cfg), (c) => {
      const r = recorded[i++];
      expect(c.step).toBe(r.step);
      expect(c.valLoss).toBe(r.valLoss);
      expect(samplesAt(CPU_RUN, c.params)).toEqual(r.samples);
      return c.step < 100;
    });
    expect(recorded[i - 1].step).toBe(100);
  }, 120_000);

  it('its statistics and stage steps follow from the recorded samples', () => {
    const cal = calibrate(trainText, 800, 400, dict);
    expect(cal).toEqual(CPU_REFERENCE.calibration);
    const hist = CPU_REFERENCE.checkpoints.map((c) => ({ step: c.step, stats: sampleStats(c.samples.join(' '), dict, freq) }));
    hist.forEach((h, k) => expect(h.stats).toEqual(CPU_REFERENCE.checkpoints[k].stats));
    expect(stageSteps(hist, cal)).toEqual(CPU_REFERENCE.stages);
    expect(CPU_REFERENCE.params).toBe(gptParamCount(CPU_RUN.cfg));
  });

  it('the shipped final weights are the step-2000 parameters of the run', async () => {
    const fs = await nodeFs();
    const bytes = fs.readFileSync(new URL('./tinyGpt.cpu.bin', import.meta.url));
    const p = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    expect(p.length).toBe(CPU_REFERENCE.params);
    expect(paramHash(p)).toBe(CPU_REFERENCE.finalHash);
    const last = CPU_REFERENCE.checkpoints.at(-1)!;
    expect(last.step).toBe(2000);
    expect(samplesAt(CPU_RUN, p)).toEqual(last.samples);
  });
});

describe('the recorded larger (GPU-size) run', () => {
  it('its statistics and stages follow from its samples; final 1.423, 78% real words, six in a row from step 774', () => {
    const cal = calibrate(trainText, 800, 400, dict);
    const hist = GPU_REFERENCE.checkpoints.map((c) => ({ step: c.step, stats: sampleStats(c.samples.join(' '), dict, freq) }));
    hist.forEach((h, k) => expect(h.stats).toEqual(GPU_REFERENCE.checkpoints[k].stats));
    expect(stageSteps(hist, cal)).toEqual(GPU_REFERENCE.stages);
    expect(GPU_REFERENCE.stages).toEqual({ letters: 1668, spacing: 2154, 'half-real': 774, 'most-real': null, phrase: 774 });
    const last = GPU_REFERENCE.checkpoints.at(-1)!;
    expect(last.step).toBe(3000);
    expect(last.valLoss.toFixed(3)).toBe('1.423');
    expect((100 * last.stats.realShare).toFixed(0)).toBe('78');
    expect(GPU_REFERENCE.params).toBe(gptParamCount(GPU_RUN.cfg));
  });
});
