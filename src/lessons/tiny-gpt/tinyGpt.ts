// authored-by: Claude Opus 5.5
/**
 * Lesson logic for the capstone (pure functions, tested in tinyGpt.test.ts).
 */
import { Gpt, gptParamCount, sampleDistribution, type GptConfig } from '../../lib/compute/gpt';
import { CHARS, decode, encode } from '../../lib/corpora/grimm';
import { sampleStats, STAGES, type Calibration, type SampleStats, type StageId } from '../../lib/corpora/stages';

export type ParamGroup = { id: string; label: string; formula: string; count: number; perBlock?: boolean };

/** Where the parameters live, in the order data flows through the model. */
export function parameterGroups(cfg: GptConfig): ParamGroup[] {
  const { vocab: V, ctx: T, d, layers: L } = cfg;
  const F = (cfg.ff ?? 4) * d;
  const groups: ParamGroup[] = [
    { id: 'wte', label: 'token embedding', formula: `${V} × ${d}`, count: V * d },
    { id: 'wpe', label: 'position embedding', formula: `${T} × ${d}`, count: T * d },
    { id: 'ln1', label: 'layer norm (before attention)', formula: `2 × ${d}`, count: 2 * d, perBlock: true },
    { id: 'att', label: 'attention: Q, K, V and output', formula: `4 × ${d} × ${d} + 4 × ${d}`, count: 4 * d * d + 4 * d, perBlock: true },
    { id: 'ln2', label: 'layer norm (before the MLP)', formula: `2 × ${d}`, count: 2 * d, perBlock: true },
    { id: 'mlp', label: `MLP: ${d} → ${F} → ${d}`, formula: `2 × ${d} × ${F} + ${F} + ${d}`, count: 2 * d * F + F + d, perBlock: true },
    { id: 'lnf', label: 'final layer norm', formula: `2 × ${d}`, count: 2 * d },
    { id: 'out', label: 'output layer (tied to the token embedding)', formula: 'reuses the embedding', count: 0 },
  ];
  const total = groups.reduce((s, g) => s + g.count * (g.perBlock ? L : 1), 0);
  if (total !== gptParamCount(cfg)) throw new Error('parameter groups do not add up');
  return groups;
}

/** Everything the timeline keeps for one checkpoint of a live run. */
export type LiveCheckpoint = {
  step: number;
  params: Float32Array;
  trainLoss: number;
  valLoss: number;
  samples: string[];
  stats: SampleStats;
  sec: number;
};

export type Birth = { id: StageId; label: string; step: number };

/**
 * The moments of birth so far: for each stage, the first checkpoint from which it has held at every
 * checkpoint since. (A stage that later breaks again loses its moment until it settles.)
 */
export function births(history: readonly { step: number; stats: SampleStats }[], cal: Calibration): Birth[] {
  const out: Birth[] = [];
  for (const st of STAGES) {
    let from: number | null = null;
    for (const h of history) {
      if (st.test(h.stats, cal)) from ??= h.step;
      else from = null;
    }
    if (from !== null) out.push({ id: st.id, label: st.label, step: from });
  }
  return out;
}

/** Landmark checkpoints for the "same prompt, same seed" stack: step 0, each birth, and the last. */
export function landmarks(steps: readonly number[], born: readonly Birth[]): number[] {
  const set = new Set<number>([steps[0], ...born.map((b) => b.step), steps.at(-1)!]);
  return [...set].filter((s) => steps.includes(s)).sort((a, b) => a - b);
}

export function statsOf(samples: string[], dict: Set<string>, freq: Float64Array) {
  return sampleStats(samples.join(' '), dict, freq);
}

/** The k most likely next characters after `prompt`, with probabilities (temperature 1). */
export function topNext(model: Gpt, prompt: string, k: number): { ch: string; p: number }[] {
  const p = sampleDistribution(model.nextLogits(encode(prompt)), 1, 1);
  return Array.from(p, (v, i) => ({ ch: CHARS[i], p: v }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k);
}

/** A continuation of `prompt` from a model, deterministic for a seed. */
export function continueText(model: Gpt, prompt: string, n: number, seed: number, temperature = 1, topP = 1): string {
  return decode(model.generate(encode(prompt), n, { seed, temperature, topP }));
}

/** Show a character legibly in a widget (spaces and newlines made visible). */
export const showChar = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c);

/** Attention weights for a prompt (last `ctx` characters): [layer][head] → T×T. */
export function attentionFor(model: Gpt, prompt: string): { tokens: string[]; att: Float64Array[][] } {
  const ids = encode(prompt).slice(-model.cfg.ctx);
  model.forward(ids, 1, ids.length);
  return { tokens: [...decode(ids)], att: model.attention() };
}

/**
 * Mean attention each head puts on the current character and the one before it, over `rows`
 * windows of T tokens laid out [rows × (T+1)] (the first T of each row are read): [layer][head] → [self, previous].
 */
export function headLocality(model: Gpt, windows: Int32Array, rows: number, T: number): [number, number][][] {
  const L = model.cfg.layers;
  const H = model.cfg.heads;
  const acc = Array.from({ length: L }, () => Array.from({ length: H }, () => [0, 0] as [number, number]));
  for (let r = 0; r < rows; r++) {
    model.forward(windows.subarray(r * (T + 1), r * (T + 1) + T), 1, T);
    model.attention().forEach((layer, l) =>
      layer.forEach((a, h) => {
        for (let i = 1; i < T; i++) {
          acc[l][h][0] += a[i * T + i];
          acc[l][h][1] += a[i * T + i - 1];
        }
      }),
    );
  }
  const n = rows * (T - 1);
  return acc.map((layer) => layer.map(([s, p]) => [s / n, p / n] as [number, number]));
}
