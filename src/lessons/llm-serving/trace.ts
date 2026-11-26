// authored-by: Gemini Antigravity
/**
 * Seeded workload trace generators and committed fixed traces for `llm-serving`.
 *
 * Every seed below was chosen before running any simulation or grid search.
 * Distributions are stated explicitly in comments and prose:
 *   - Arrivals: Poisson process (exponential inter-arrival gaps).
 *   - Prompt & reply lengths: clipped log-normal (heavy-tailed replies, mimicking a mix of
 *     short answers and long explanations; we do not claim these parameters match any specific
 *     production service).
 */
import { createRng } from '../../lib/rng';
import type { TraceRequest, WorkloadTrace } from './serve';

/** Standard normal draw via Box–Muller on the seeded RNG. */
function normal(rng: ReturnType<typeof createRng>): number {
  const u1 = Math.max(1e-12, rng.next());
  const u2 = rng.next();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Clipped integer log-normal draw with median exp(mu) and log-sd sigma. */
function logNormalInt(rng: ReturnType<typeof createRng>, mu: number, sigma: number, min: number, max: number): number {
  const v = Math.round(Math.exp(mu + sigma * normal(rng)));
  return Math.max(min, Math.min(max, v));
}

/** Exponential draw with mean 1 / ratePerMs (in ms). */
function expMs(rng: ReturnType<typeof createRng>, ratePerSec: number): number {
  const u = Math.max(1e-12, 1 - rng.next());
  return (-Math.log(u) / ratePerSec) * 1000;
}

export interface TraceSpec {
  id: string;
  name: string;
  description: string;
  seed: number;
  count: number;
  arrivalRatePerSec: number;
  maxModelLen: number;
  sharedPrefixLen: number;
  /** Prompt user-suffix log-normal params (added on top of sharedPrefixLen). */
  promptMu: number;
  promptSigma: number;
  promptMin: number;
  promptMax: number;
  /** Reply length log-normal params. */
  outputMu: number;
  outputSigma: number;
  outputMin: number;
  outputMax: number;
  /** Optional fraction of requests that are long documents (8k–16k prompt, concise summary reply). */
  longDocFraction?: number;
}

export function generateTrace(spec: TraceSpec): WorkloadTrace {
  const rng = createRng(spec.seed);
  const requests: TraceRequest[] = [];
  let tMs = 0;

  for (let i = 0; i < spec.count; i++) {
    if (i > 0) tMs += expMs(rng, spec.arrivalRatePerSec);
    const isDoc = (spec.longDocFraction ?? 0) > 0 && i >= 3 && rng.next() < (spec.longDocFraction ?? 0);
    if (isDoc) {
      const promptLen = Math.min(
        spec.maxModelLen - 128,
        Math.round((8192 + rng.next() * 8192) / 64) * 64,
      );
      const outputLen = logNormalInt(rng, Math.log(48), 0.35, 24, 96);
      requests.push({
        id: i + 1,
        arrivalMs: Math.round(tMs),
        promptLen,
        outputLen,
        sharedPrefixLen: spec.sharedPrefixLen,
        kind: 'doc',
      });
    } else {
      const userPrompt = logNormalInt(rng, spec.promptMu, spec.promptSigma, spec.promptMin, spec.promptMax);
      const promptLen = spec.sharedPrefixLen + userPrompt;
      const outputLen = logNormalInt(rng, spec.outputMu, spec.outputSigma, spec.outputMin, spec.outputMax);
      requests.push({
        id: i + 1,
        arrivalMs: Math.round(tMs),
        promptLen,
        outputLen,
        sharedPrefixLen: spec.sharedPrefixLen,
        kind: 'chat',
      });
    }
  }

  return {
    id: spec.id,
    name: spec.name,
    description: spec.description,
    maxModelLen: spec.maxModelLen,
    requests,
  };
}

/**
 * Opening launch-day burst (§3.1 Predict): 200 chat messages arriving within 1 second.
 * Seed 202600 fixed before measuring.
 */
export const LAUNCH_BURST_TRACE: WorkloadTrace = (() => {
  const rng = createRng(202600);
  const requests: TraceRequest[] = [];
  for (let i = 0; i < 200; i++) {
    requests.push({
      id: i + 1,
      arrivalMs: Math.round(rng.next() * 1000),
      promptLen: logNormalInt(rng, Math.log(320), 0.45, 64, 1024),
      outputLen: logNormalInt(rng, Math.log(160), 0.55, 32, 512),
      sharedPrefixLen: 0,
      kind: 'chat',
    });
  }
  requests.sort((a, b) => a.arrivalMs - b.arrivalMs || a.id - b.id);
  requests.forEach((r, idx) => { r.id = idx + 1; });
  return {
    id: 'launch-burst',
    name: 'Launch-day 200-user burst',
    description: '200 users send a chat message within the same second (seed 202600).',
    maxModelLen: 4096,
    requests,
  };
})();

/**
 * Widget B (BatchingRace): 24 chat requests with heavy-tailed replies.
 * Seed 202601 fixed before measuring.
 */
export const RACE_TRACE: WorkloadTrace = generateTrace({
  id: 'batching-race',
  name: '24-request chat wave',
  description: '24 Poisson chat arrivals (12 req/s), heavy-tailed replies (median ~55, tail up to 320 tokens).',
  seed: 202601,
  count: 24,
  arrivalRatePerSec: 12,
  maxModelLen: 4096,
  sharedPrefixLen: 0,
  promptMu: Math.log(240),
  promptSigma: 0.45,
  promptMin: 64,
  promptMax: 768,
  outputMu: Math.log(55),
  outputSigma: 0.85,
  outputMin: 12,
  outputMax: 320,
});

/**
 * Widget C (ChunkedPrefill): 10 chat users decoding steadily while a 16,384-token document arrives
 * mid-conversation at t = 350 ms. Seed 202602 fixed before measuring.
 */
export const TROLLEY_TRACE: WorkloadTrace = (() => {
  const rng = createRng(202602);
  const requests: TraceRequest[] = [];
  for (let i = 0; i < 10; i++) {
    requests.push({
      id: i + 1,
      arrivalMs: i < 6 ? i * 35 : 380 + (i - 6) * 140,
      promptLen: logNormalInt(rng, Math.log(256), 0.35, 96, 512),
      outputLen: logNormalInt(rng, Math.log(90), 0.3, 55, 150),
      sharedPrefixLen: 0,
      kind: 'chat',
    });
  }
  // Insert the 16k document right while the first 6 chatters are streaming!
  requests.push({
    id: 11,
    arrivalMs: 350,
    promptLen: 16384,
    outputLen: 48,
    sharedPrefixLen: 0,
    kind: 'doc',
  });
  requests.sort((a, b) => a.arrivalMs - b.arrivalMs || a.id - b.id);
  requests.forEach((r, idx) => { r.id = idx + 1; });
  return {
    id: 'trolley-16k',
    name: 'Chat stream + 16k document',
    description: '10 short chats plus one 16,384-token document arriving at t = 350 ms (seed 202602).',
    maxModelLen: 16896,
    requests,
  };
})();

/**
 * Widget D (KvMemoryMap): 64 live conversations sharing a 1,024-token system prompt.
 * Seed 202603 fixed before measuring.
 */
export const MEMORY_MAP_SEQUENCES: { id: number; promptLen: number; currentLen: number; sharedPrefixLen: number }[] = (() => {
  const rng = createRng(202603);
  const shared = 1024;
  return Array.from({ length: 64 }, (_, i) => {
    const userPrompt = logNormalInt(rng, Math.log(220), 0.5, 48, 768);
    const generated = logNormalInt(rng, Math.log(140), 0.65, 20, 640);
    return {
      id: i + 1,
      promptLen: shared + userPrompt,
      currentLen: shared + userPrompt + generated,
      sharedPrefixLen: shared,
    };
  });
})();

/**
 * Lab 1 Brief 1 ("Nobody waits forever"): 40 chat requests, heavy-tailed replies.
 * Seed 202611 fixed before measuring.
 */
export const LAB_TRACE_CHAT: WorkloadTrace = generateTrace({
  id: 'lab-chat',
  name: 'Morning chat burst',
  description: '40 Poisson chat requests at 8 req/s, heavy-tailed replies (seed 202611).',
  seed: 202611,
  count: 40,
  arrivalRatePerSec: 8,
  maxModelLen: 4096,
  sharedPrefixLen: 0,
  promptMu: Math.log(280),
  promptSigma: 0.45,
  promptMin: 64,
  promptMax: 768,
  outputMu: Math.log(70),
  outputSigma: 0.8,
  outputMin: 16,
  outputMax: 340,
});

/**
 * Lab 1 Brief 2 ("The long document"): chat traffic mixed with 8k–16k documents (~10%).
 * Seed 202612 fixed before measuring.
 */
export const LAB_TRACE_LONGDOC: WorkloadTrace = generateTrace({
  id: 'lab-longdoc',
  name: 'Chat + long documents',
  description: '36 Poisson requests at 7 req/s with ~10% 8k–16k documents (seed 202612).',
  seed: 202612,
  count: 36,
  arrivalRatePerSec: 7,
  maxModelLen: 16896,
  sharedPrefixLen: 0,
  promptMu: Math.log(260),
  promptSigma: 0.45,
  promptMin: 64,
  promptMax: 768,
  outputMu: Math.log(75),
  outputSigma: 0.65,
  outputMin: 20,
  outputMax: 280,
  longDocFraction: 0.12,
});

/**
 * Lab 1 Brief 3 ("Fit the crowd"): 64 busy requests sharing a 1,024-token system prompt
 * on the memory-tight RTX 4090 (maxModelLen 8,192). Seed 202613 fixed before measuring.
 */
export const LAB_TRACE_CROWD: WorkloadTrace = generateTrace({
  id: 'lab-crowd',
  name: 'Shared-prompt crowd',
  description: '64 Poisson requests at 16 req/s, all sharing a 1,024-token system prompt (seed 202613).',
  seed: 202613,
  count: 64,
  arrivalRatePerSec: 16,
  maxModelLen: 8192,
  sharedPrefixLen: 1024,
  promptMu: Math.log(260),
  promptSigma: 0.5,
  promptMin: 64,
  promptMax: 896,
  outputMu: Math.log(120),
  outputSigma: 0.55,
  outputMin: 32,
  outputMax: 420,
});

export function summarizeTrace(trace: WorkloadTrace) {
  const prompts = trace.requests.map((r) => r.promptLen);
  const outputs = trace.requests.map((r) => r.outputLen);
  const docs = trace.requests.filter((r) => r.kind === 'doc').length;
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  return {
    count: trace.requests.length,
    docs,
    durationMs: trace.requests.length ? trace.requests[trace.requests.length - 1].arrivalMs - trace.requests[0].arrivalMs : 0,
    meanPrompt: mean(prompts),
    maxPrompt: Math.max(...prompts),
    meanOutput: mean(outputs),
    maxOutput: Math.max(...outputs),
    totalOutputTokens: outputs.reduce((a, b) => a + b, 0),
    sharedPrefixLen: trace.requests[0]?.sharedPrefixLen ?? 0,
  };
}
