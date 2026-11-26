// authored-by: Gemini Antigravity
/**
 * Speculative decoding engine (S2) on real trained character-level models:
 *   - Target model: `tinyGpt.cpu.bin` (d = 32, 4 heads, 2 layers, 29,696 params, val loss 1.712),
 *     the exact model built in the tiny-GPT capstone.
 *   - Neural draft model: `draftGpt.bin` (d = 16, 2 heads, 1 layer, 5,424 params — 5.5× smaller),
 *     trained on the exact same Grimm's Fairy Tales training split.
 *   - Zero-cost rival draft: character bigram counting table on the training split (add-0.5 smoothing).
 *
 * Uses `Decoder` from `src/lib/compute/gpt.ts`: rolling back rejected draft tokens is a single
 * assignment (`decoder.len = savedLen`), because later KV slots are never read and are overwritten
 * by subsequent steps.
 */
import { createRng } from '../../lib/rng';
import { Decoder, drawFrom, Gpt, gptParamCount, sampleDistribution, type GptConfig } from '../../lib/compute/gpt';
import { CPU_RUN } from '../../lib/compute/gptReference';
import { CHARS, decode, encode, taleStarts, VOCAB } from '../../lib/corpora/grimm';

export interface ModelSpec {
  id: string;
  name: string;
  shortName: string;
  paramsB: number;
  layers: number;
  qHeads: number;
  kvHeads: number;
  dHead: number;
}

export interface GpuSpec {
  id: string;
  name: string;
  shortName: string;
  vramGiB: number;
  bwTBps: number;
  peakTflops: number;
  overheadGiB: number;
}

/** Duplicated from kv-cache / llm-serving so lessons stay independent. */
export const LLAMA3_8B: ModelSpec = {
  id: 'llama3-8b',
  name: 'Llama-3 8B (GQA 4:1)',
  shortName: 'Llama-3 8B',
  paramsB: 8.03,
  layers: 32,
  qHeads: 32,
  kvHeads: 8,
  dHead: 128,
};

export const RTX_4090: GpuSpec = {
  id: 'rtx-4090',
  name: 'NVIDIA RTX 4090 (24 GiB)',
  shortName: 'RTX 4090 24G',
  vramGiB: 24,
  bwTBps: 1.008,
  peakTflops: 165.2,
  overheadGiB: 1.5,
};

function stepTime(
  items: { kind: 'decode' | 'prefill'; cachedTokens: number; newTokens: number }[],
  gpu: GpuSpec,
  model: ModelSpec,
) {
  const N = model.paramsB * 1e9;
  const kvTokBytes = 2 * model.layers * model.kvHeads * model.dHead * 2;
  const attnPerPair = 4 * model.layers * model.qHeads * model.dHead;
  let newToks = 0;
  let pairs = 0;
  let kvToks = 0;
  for (const it of items) {
    newToks += it.newTokens;
    pairs += it.newTokens * it.cachedTokens + (it.newTokens * (it.newTokens + 1)) / 2;
    kvToks += it.cachedTokens + it.newTokens;
  }
  const flops = 2 * N * newToks + attnPerPair * pairs;
  const bytes = N * 2 + kvToks * kvTokBytes;
  const stepTimeMs = Math.max(bytes / (gpu.bwTBps * 1e12), flops / (gpu.peakTflops * 1e12)) * 1000;
  return { stepTimeMs };
}

export const TARGET_CFG: GptConfig = CPU_RUN.cfg;
export const DRAFT_CFG: GptConfig = { vocab: VOCAB, ctx: 64, d: 16, heads: 2, layers: 1 };

export const TARGET_PARAMS = gptParamCount(TARGET_CFG); // 29,696
export const DRAFT_PARAMS = gptParamCount(DRAFT_CFG); // 5,424

/**
 * Production-scale draft model spec used ONLY for the modelled speed-up calculation:
 * a 1.0B-parameter FP16 draft paired with Llama-3 8B FP16 on an RTX 4090 at 2k context.
 */
export const PROD_DRAFT_1B: ModelSpec = {
  id: 'draft-1b',
  name: '1B-class FP16 draft',
  shortName: '1B draft',
  paramsB: 1.0,
  layers: 16,
  qHeads: 16,
  kvHeads: 4,
  dHead: 128,
};

/**
 * Exact draft-to-target single-step cost ratio `c` from the RTX 4090 roofline at 2k context.
 * For the bigram table lookup (host/SRAM table), we use a stated 1% overhead (c = 0.01).
 */
export function productionCostRatio(
  drafterId: DrafterId,
  gpu: GpuSpec = RTX_4090,
  target: ModelSpec = LLAMA3_8B,
  ctxLen = 2048,
): { c: number; draftStepMs: number; targetStepMs: number } {
  const targetStepMs = stepTime([{ kind: 'decode', cachedTokens: ctxLen - 1, newTokens: 1 }], gpu, target).stepTimeMs;
  if (drafterId === 'bigram') {
    const c = 0.01;
    return { c, draftStepMs: targetStepMs * c, targetStepMs };
  }
  const draftStepMs = stepTime([{ kind: 'decode', cachedTokens: ctxLen - 1, newTokens: 1 }], gpu, PROD_DRAFT_1B).stepTimeMs;
  return { c: draftStepMs / targetStepMs, draftStepMs, targetStepMs };
}

/**
 * Ratio of verifying (k + 1) tokens in one parallel target pass vs decoding 1 token,
 * computed from the RTX 4090 roofline at 2k context.
 */
export function verifyCostFactor(k: number, gpu: GpuSpec = RTX_4090, target: ModelSpec = LLAMA3_8B, ctxLen = 2048): number {
  const single = stepTime([{ kind: 'decode', cachedTokens: ctxLen - 1, newTokens: 1 }], gpu, target).stepTimeMs;
  const verify = stepTime([{ kind: 'prefill', cachedTokens: ctxLen - 1, newTokens: k + 1 }], gpu, target).stepTimeMs;
  return verify / single;
}

/** Textbook i.i.d. formula for expected tokens emitted per target pass. */
export function expectedTokensFormula(alpha: number, k: number): number {
  if (Math.abs(1 - alpha) < 1e-9) return k + 1;
  return (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
}

/** Modelled production-scale speed-up given measured tokens/pass E, drafter, and lookahead k. */
export function modelledSpeedup(
  tokensPerTargetPass: number,
  drafterId: DrafterId,
  k: number,
  gpu: GpuSpec = RTX_4090,
): { speedup: number; formulaSpeedup: number; c: number; verifyFactor: number } {
  const { c } = productionCostRatio(drafterId, gpu);
  const vf = verifyCostFactor(k, gpu);
  return {
    speedup: tokensPerTargetPass / (k * c + vf),
    formulaSpeedup: tokensPerTargetPass / (k * c + 1),
    c,
    verifyFactor: vf,
  };
}

// ------------------------------------------------------------------ 1. Drafter interface

export type DrafterId = 'bigram' | 'small-gpt';

export interface Drafter {
  readonly id: DrafterId;
  readonly name: string;
  readonly params: number;
  prefill(tokens: ArrayLike<number>): Float64Array;
  step(tok: number): Float64Array;
  rollback(len: number): void;
  getLen(): number;
}

/**
 * Character bigram log-probability table counted on the Grimm training split
 * with add-0.5 smoothing (matching `ngramLoss` in `grimm.ts`).
 */
export function buildBigramLogProbs(trainIds: Int32Array, smoothing = 0.5): Float64Array {
  const V = VOCAB;
  const counts = new Float64Array(V * V).fill(smoothing);
  const rowSum = new Float64Array(V).fill(smoothing * V);
  for (let i = 0; i + 1 < trainIds.length; i++) {
    const a = trainIds[i];
    const b = trainIds[i + 1];
    counts[a * V + b]++;
    rowSum[a]++;
  }
  const logProbs = new Float64Array(V * V);
  for (let a = 0; a < V; a++) {
    const denom = rowSum[a];
    for (let b = 0; b < V; b++) {
      logProbs[a * V + b] = Math.log(counts[a * V + b] / denom);
    }
  }
  return logProbs;
}

export class BigramDrafter implements Drafter {
  readonly id: DrafterId = 'bigram';
  readonly name = 'Bigram table';
  readonly params = VOCAB * VOCAB;
  private logProbs: Float64Array;
  private history: number[] = [];

  constructor(logProbs: Float64Array) {
    this.logProbs = logProbs;
  }

  private row(tok: number): Float64Array {
    const V = VOCAB;
    return Float64Array.from(this.logProbs.subarray(tok * V, (tok + 1) * V));
  }

  prefill(tokens: ArrayLike<number>): Float64Array {
    this.history = Array.from(tokens);
    return this.row(this.history[this.history.length - 1]);
  }

  step(tok: number): Float64Array {
    this.history.push(tok);
    return this.row(tok);
  }

  rollback(len: number) {
    this.history.length = len;
  }

  getLen(): number {
    return this.history.length;
  }
}

export class GptDrafter implements Drafter {
  readonly id: DrafterId = 'small-gpt';
  readonly name = 'Small GPT (5.4k)';
  readonly params: number;
  readonly decoder: Decoder;

  constructor(model: Gpt) {
    this.decoder = new Decoder(model);
    this.params = gptParamCount(model.cfg);
  }

  prefill(tokens: ArrayLike<number>): Float64Array {
    return this.decoder.prefill(tokens);
  }

  step(tok: number): Float64Array {
    return this.decoder.step(tok);
  }

  /** Rolling back rejected draft tokens is literally one assignment on Decoder.len. */
  rollback(len: number) {
    this.decoder.len = len;
  }

  getLen(): number {
    return this.decoder.len;
  }
}

// ------------------------------------------------------------------ 2. Exact speculative sampling math

/**
 * Computes the normalised residual distribution norm(max(0, p - q)).
 * Used when a drafted token is rejected; preserves the exact target distribution p.
 */
export function residualDistribution(p: ArrayLike<number>, q: ArrayLike<number>): Float64Array {
  const V = p.length;
  const r = new Float64Array(V);
  let sum = 0;
  for (let i = 0; i < V; i++) {
    const diff = p[i] - q[i];
    if (diff > 0) {
      r[i] = diff;
      sum += diff;
    }
  }
  if (sum > 0) {
    for (let i = 0; i < V; i++) r[i] /= sum;
  } else {
    // Only happens when p === q everywhere (cannot reject, but guard numerically)
    for (let i = 0; i < V; i++) r[i] = p[i];
  }
  return r;
}

/**
 * Exact marginal distribution of the emitted token in a single speculative accept/reject step,
 * computed analytically by summing over every possible draft proposal x ∈ 0..V-1.
 * Used in exactness tests to verify equality to p(y) within 1e-12.
 */
export function exactSingleStepMarginal(p: ArrayLike<number>, q: ArrayLike<number>): Float64Array {
  const V = p.length;
  const out = new Float64Array(V);
  const res = residualDistribution(p, q);
  let rejectMass = 0;
  for (let x = 0; x < V; x++) {
    const acceptProb = q[x] > 0 ? Math.min(1, p[x] / q[x]) : 0;
    out[x] += q[x] * acceptProb;
    rejectMass += q[x] * (1 - acceptProb);
  }
  for (let y = 0; y < V; y++) {
    out[y] += rejectMass * res[y];
  }
  return out;
}

// ------------------------------------------------------------------ 3. Multi-round speculative runner

export interface DraftDecision {
  index: number;
  token: number;
  char: string;
  qProb: number;
  pProb: number;
  acceptProb: number;
  u: number;
  accepted: boolean;
  /** True when the preceding character was a non-letter (space, newline, punctuation). */
  isWordStart: boolean;
}

export interface SpecRoundTrace {
  round: number;
  prefixBefore: string;
  drafted: DraftDecision[];
  acceptedCount: number;
  rejectedAt: number; // -1 if all drafted tokens accepted
  /** Top entries of the residual distribution norm(max(0, p - q)) on rejection, or bonus dist. */
  correctionDistTop: { token: number; char: string; prob: number; pProb: number; qProb: number }[];
  correctionToken: number;
  correctionChar: string;
  wasBonus: boolean;
  emittedTokens: number[];
  emittedText: string;
  /** How many drafted KV slots were rolled back on the target Decoder. */
  rolledBack: number;
}

export interface SpecRunSummary {
  prompt: string;
  generatedText: string;
  generatedTokens: number[];
  rounds: SpecRoundTrace[];
  targetPasses: number;
  draftSteps: number;
  totalEmitted: number;
  totalDrafted: number;
  totalAccepted: number;
  acceptanceRate: number;
  wordStartDrafted: number;
  wordStartAccepted: number;
  wordStartAcceptance: number;
  midWordDrafted: number;
  midWordAccepted: number;
  midWordAcceptance: number;
  tokensPerTargetPass: number;
  wallMs: number;
}

const isLetter = (ch: string) => (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');

function topDistribution(dist: Float64Array, p: Float64Array, q: Float64Array | null, limit = 6) {
  const idx = Array.from({ length: dist.length }, (_, i) => i)
    .filter((i) => dist[i] > 1e-6)
    .sort((a, b) => dist[b] - dist[a] || a - b)
    .slice(0, limit);
  return idx.map((token) => ({
    token,
    char: CHARS[token],
    prob: dist[token],
    pProb: p[token],
    qProb: q ? q[token] : 0,
  }));
}

/**
 * Runs speculative decoding from `prompt` for `maxNewTokens` characters using a real target
 * `Gpt` and a `Drafter` (`BigramDrafter` or `GptDrafter`).
 *
 * Keeps prompt + generated ≤ target.cfg.ctx (64 characters) so no sliding-window re-prefill
 * blurs the pass count.
 */
export function runSpeculative(params: {
  targetModel: Gpt;
  drafter: Drafter;
  prompt: string;
  maxNewTokens: number;
  k: number;
  temperature: number;
  seed: number;
}): SpecRunSummary {
  const { targetModel, drafter, prompt, maxNewTokens, k, temperature, seed } = params;
  const promptIds = encode(prompt);
  const ctx = targetModel.cfg.ctx;
  const limitNew = Math.min(maxNewTokens, ctx - promptIds.length);
  if (limitNew <= 0) throw new Error(`prompt length ${promptIds.length} fills context ${ctx}`);

  const rng = createRng(seed);
  const targetDec = new Decoder(targetModel);

  const t0 = performance.now();
  let curTargetLogits = targetDec.prefill(promptIds);
  let curDraftLogits = drafter.prefill(promptIds);

  const emittedAll: number[] = [];
  const rounds: SpecRoundTrace[] = [];

  let targetPasses = 0;
  let draftSteps = 0;
  let totalDrafted = 0;
  let totalAccepted = 0;
  let wordStartDrafted = 0;
  let wordStartAccepted = 0;
  let midWordDrafted = 0;
  let midWordAccepted = 0;

  let lastChar = prompt[prompt.length - 1] ?? ' ';
  let roundIdx = 0;

  while (emittedAll.length < limitNew) {
    const remaining = limitNew - emittedAll.length;
    const baseLen = targetDec.len;
    // Draft up to k tokens, never exceeding remaining budget or context - 1
    const m = Math.min(k, remaining, ctx - 1 - baseLen);
    const prefixBefore = prompt + decode(emittedAll);

    if (m <= 0) {
      // Only room for 1 final token directly from target
      const p0 = sampleDistribution(curTargetLogits, temperature, 1);
      const tok = drawFrom(p0, rng.next());
      emittedAll.push(tok);
      targetPasses++;
      rounds.push({
        round: roundIdx++,
        prefixBefore,
        drafted: [],
        acceptedCount: 0,
        rejectedAt: -1,
        correctionDistTop: topDistribution(p0, p0, null),
        correctionToken: tok,
        correctionChar: CHARS[tok],
        wasBonus: true,
        emittedTokens: [tok],
        emittedText: CHARS[tok],
        rolledBack: 0,
      });
      break;
    }

    // 1. Draft m tokens sequentially
    const draftedToks: number[] = [];
    const draftProbs: Float64Array[] = [];
    let dLogits = curDraftLogits;
    for (let i = 0; i < m; i++) {
      const q = sampleDistribution(dLogits, temperature, 1);
      const x = drawFrom(q, rng.next());
      draftedToks.push(x);
      draftProbs.push(q);
      dLogits = drafter.step(x);
      draftSteps++;
    }

    // 2. Verify all m drafted tokens in ONE conceptual parallel target forward pass
    const targetProbs: Float64Array[] = [sampleDistribution(curTargetLogits, temperature, 1)];
    const targetLogitsAfter: Float64Array[] = [];
    for (let i = 0; i < m; i++) {
      const nextL = targetDec.step(draftedToks[i]);
      targetLogitsAfter.push(nextL);
      targetProbs.push(sampleDistribution(nextL, temperature, 1));
    }
    targetPasses++;

    // 3. Accept/reject left to right
    const decisions: DraftDecision[] = [];
    let acceptedCount = 0;
    let rejectedAt = -1;
    let prevCh = lastChar;

    for (let i = 0; i < m; i++) {
      const x = draftedToks[i];
      const q = draftProbs[i];
      const p = targetProbs[i];
      const qVal = q[x];
      const pVal = p[x];
      const acceptProb = qVal > 0 ? Math.min(1, pVal / qVal) : (pVal > 0 ? 1 : 0);
      const u = rng.next();
      const accepted = u < acceptProb;
      const wordStart = !isLetter(prevCh);

      decisions.push({
        index: i,
        token: x,
        char: CHARS[x],
        qProb: qVal,
        pProb: pVal,
        acceptProb,
        u,
        accepted,
        isWordStart: wordStart,
      });

      totalDrafted++;
      if (wordStart) wordStartDrafted++;
      else midWordDrafted++;

      if (accepted) {
        acceptedCount++;
        totalAccepted++;
        if (wordStart) wordStartAccepted++;
        else midWordAccepted++;
        prevCh = CHARS[x];
      } else {
        rejectedAt = i;
        break;
      }
    }

    const roundEmitted: number[] = draftedToks.slice(0, acceptedCount);
    let correctionToken: number;
    let correctionDistTop: SpecRoundTrace['correctionDistTop'];
    let wasBonus = false;
    let rolledBack = 0;

    if (rejectedAt >= 0) {
      // Rejection at index rejectedAt: draw correction from residual norm(max(0, p - q))
      const p = targetProbs[rejectedAt];
      const q = draftProbs[rejectedAt];
      const res = residualDistribution(p, q);
      correctionToken = drawFrom(res, rng.next());
      correctionDistTop = topDistribution(res, p, q);
      roundEmitted.push(correctionToken);

      // Roll back target & draft KV caches to baseLen + acceptedCount (one assignment!)
      const keepLen = baseLen + acceptedCount;
      rolledBack = m - acceptedCount;
      targetDec.len = keepLen;
      drafter.rollback(keepLen);

      if (emittedAll.length + roundEmitted.length < limitNew && keepLen < ctx) {
        curTargetLogits = targetDec.step(correctionToken);
        curDraftLogits = drafter.step(correctionToken);
      }
    } else {
      // All m drafted tokens accepted! Emit bonus token if we still need more tokens
      wasBonus = true;
      const bonusP = targetProbs[m];
      if (emittedAll.length + roundEmitted.length < limitNew) {
        correctionToken = drawFrom(bonusP, rng.next());
        correctionDistTop = topDistribution(bonusP, bonusP, null);
        roundEmitted.push(correctionToken);
        if (emittedAll.length + roundEmitted.length < limitNew && targetDec.len < ctx) {
          curTargetLogits = targetDec.step(correctionToken);
          curDraftLogits = drafter.step(correctionToken);
        }
      } else {
        // Reached limitNew exactly on the m-th accepted token
        correctionToken = roundEmitted[roundEmitted.length - 1];
        correctionDistTop = topDistribution(bonusP, bonusP, null);
      }
    }

    for (const t of roundEmitted) emittedAll.push(t);
    lastChar = CHARS[emittedAll[emittedAll.length - 1]];

    rounds.push({
      round: roundIdx++,
      prefixBefore,
      drafted: decisions,
      acceptedCount,
      rejectedAt,
      correctionDistTop,
      correctionToken,
      correctionChar: CHARS[correctionToken],
      wasBonus,
      emittedTokens: roundEmitted,
      emittedText: decode(roundEmitted),
      rolledBack,
    });
  }

  const wallMs = performance.now() - t0;

  return {
    prompt,
    generatedText: decode(emittedAll),
    generatedTokens: emittedAll,
    rounds,
    targetPasses,
    draftSteps,
    totalEmitted: emittedAll.length,
    totalDrafted,
    totalAccepted,
    acceptanceRate: totalDrafted > 0 ? totalAccepted / totalDrafted : 0,
    wordStartDrafted,
    wordStartAccepted,
    wordStartAcceptance: wordStartDrafted > 0 ? wordStartAccepted / wordStartDrafted : 0,
    midWordDrafted,
    midWordAccepted,
    midWordAcceptance: midWordDrafted > 0 ? midWordAccepted / midWordDrafted : 0,
    tokensPerTargetPass: targetPasses > 0 ? emittedAll.length / targetPasses : 0,
    wallMs,
  };
}

/** Plain autoregressive decoding on `targetModel` (baseline for exact greedy & wall-clock checks). */
export function runPlainTarget(params: {
  targetModel: Gpt;
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  seed: number;
}): { generatedText: string; generatedTokens: number[]; wallMs: number } {
  const { targetModel, prompt, maxNewTokens, temperature, seed } = params;
  const promptIds = encode(prompt);
  const limitNew = Math.min(maxNewTokens, targetModel.cfg.ctx - promptIds.length);
  const rng = createRng(seed);
  const dec = new Decoder(targetModel);
  const t0 = performance.now();
  let logits = dec.prefill(promptIds);
  const out: number[] = [];
  for (let i = 0; i < limitNew; i++) {
    const p = sampleDistribution(logits, temperature, 1);
    const tok = drawFrom(p, rng.next());
    out.push(tok);
    if (i + 1 < limitNew && dec.len < targetModel.cfg.ctx) {
      logits = dec.step(tok);
    }
  }
  const wallMs = performance.now() - t0;
  return { generatedText: decode(out), generatedTokens: out, wallMs };
}

/**
 * Extract 12 fixed 24-character prompts from the 4 held-out validation tales of `grimm.txt`
 * (3 non-overlapping sentence openings per tale, chosen mechanically before measuring).
 */
export function heldOutPrompts(corpusText: string, splitOffset: number): { title: string; prompt: string }[] {
  const valText = corpusText.slice(splitOffset);
  const starts = taleStarts(valText);
  const out: { title: string; prompt: string }[] = [];
  for (let i = 0; i < starts.length; i++) {
    const s = starts[i];
    const end = i + 1 < starts.length ? starts[i + 1].at : valText.length;
    const body = valText.slice(s.at + s.title.length, end).trim();
    const lines = body
      .split(/\n+/)
      .map((p) => p.replace(/\s+/g, ' ').trim())
      .filter((p) => p.length >= 30);
    // Pick 3 evenly spaced distinct paragraphs from each held-out tale
    const picks = [0, Math.floor(lines.length / 3), Math.floor((2 * lines.length) / 3)];
    for (const pi of picks) {
      out.push({ title: s.title, prompt: lines[pi].slice(0, 24) });
    }
  }
  return out;
}
