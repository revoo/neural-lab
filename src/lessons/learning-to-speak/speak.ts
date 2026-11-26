// authored-by: Claude Opus 5.5
/**
 * Lesson logic for "learning to speak" (V1, B2): every moment on the timeline is a measured event,
 * from the recorded run's samples (stages.ts) or from its probes (gptProbes.ts).
 */
import { STAGES, type Calibration, type SampleStats } from '../../lib/corpora/stages';
import { HEAD_BAR_FACTOR, PROBE_L, uniformShares } from '../../lib/compute/gptProbes';

export type Probe = { prev: readonly (readonly number[])[]; induction: readonly (readonly number[])[]; lossFirst: number; lossSecond: number; emb: number; embBar: number; embPca: readonly (readonly [number, number])[] };
export type SpeakCheckpoint = { step: number; valLoss: number; samples: readonly string[]; stats: SampleStats; probe: Probe };

export const [UNIFORM_PREV, UNIFORM_IND] = uniformShares(PROBE_L);
export const PREV_BAR = HEAD_BAR_FACTOR * UNIFORM_PREV;
export const IND_BAR = HEAD_BAR_FACTOR * UNIFORM_IND;

export type MomentId = 'embed' | 'letters' | 'spacing' | 'half-real' | 'most-real' | 'phrase' | 'prev-head' | 'prev-heads' | 'induction';
export type Moment = { id: MomentId; label: string; step: number };

const count = (m: readonly (readonly number[])[], bar: number) => m.flat().filter((v) => v > bar).length;

/** Every test, in the order the lesson introduces them. */
export function speakTests(cal: Calibration): { id: MomentId; label: string; test: (c: SpeakCheckpoint) => boolean }[] {
  return [
    { id: 'embed', label: 'its characters sort into kinds', test: (c) => c.probe.emb > c.probe.embBar },
    ...STAGES.map((s) => ({ id: s.id as MomentId, label: s.label, test: (c: SpeakCheckpoint) => s.test(c.stats, cal) })),
    { id: 'prev-head', label: 'its first previous-character head', test: (c) => count(c.probe.prev, PREV_BAR) >= 1 },
    { id: 'prev-heads', label: 'four previous-character heads', test: (c) => count(c.probe.prev, PREV_BAR) >= 4 },
    { id: 'induction', label: 'an induction head', test: (c) => count(c.probe.induction, IND_BAR) >= 1 },
  ];
}

/** Moments that hold "for good" (from that checkpoint to the last one so far), sorted by step. */
export function speakMoments(hist: readonly SpeakCheckpoint[], cal: Calibration): Moment[] {
  const out: Moment[] = [];
  for (const t of speakTests(cal)) {
    let from: number | null = null;
    for (const h of hist) {
      if (t.test(h)) from ??= h.step;
      else from = null;
    }
    if (from !== null) out.push({ id: t.id, label: t.label, step: from });
  }
  return out.sort((a, b) => a.step - b.step);
}

/** The groups the embedding map colours by, in CHAR_GROUPS order. */
export const GROUP_LABELS = ['vowels', 'consonants', 'capitals', 'digits', 'punctuation', 'space and newline'];
