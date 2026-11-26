// authored-by: Gemini Antigravity
/**
 * Lab 1 (The Serving Desk) definitions, evaluation, and evidence-only hint ladder
 * (tested in `hints.test.ts` and `search.test.ts`).
 */
import {
  LLAMA3_8B,
  RTX_4090,
  simulate,
  type AdmissionPolicy,
  type BatchingMode,
  type ServingConfig,
  type SimResult,
  type WorkloadTrace,
} from './serve';
import { LAB_TRACE_CHAT, LAB_TRACE_CROWD, LAB_TRACE_LONGDOC } from './trace';
import type { DeskConfig, LabMemoryMode } from './realCode';

// ------------------------------------------------------------------ The Serving Desk

export const DESK_BATCHINGS: readonly BatchingMode[] = ['static', 'continuous'];
export const DESK_MAX_RUNS = [4, 8, 16, 32, 64] as const;
export const DESK_TOKEN_BUDGETS: readonly (number | null)[] = [null, 256, 512, 1024, 2048];
export const DESK_MEM_MODES: readonly LabMemoryMode[] = ['reserve-max', 'paged', 'paged-prefix'];
export const DESK_ADMISSIONS: readonly AdmissionPolicy[] = ['fcfs', 'sjf'];

export function allDeskConfigs(): DeskConfig[] {
  const out: DeskConfig[] = [];
  for (const batching of DESK_BATCHINGS) {
    for (const maxRunning of DESK_MAX_RUNS) {
      for (const tokenBudget of DESK_TOKEN_BUDGETS) {
        for (const memoryMode of DESK_MEM_MODES) {
          for (const admission of DESK_ADMISSIONS) {
            out.push({ batching, maxRunning, tokenBudget, memoryMode, admission });
          }
        }
      }
    }
  }
  return out;
}

export function deskConfigKey(c: DeskConfig): string {
  return `${c.batching}|m${c.maxRunning}|b${c.tokenBudget ?? 'off'}|${c.memoryMode}|${c.admission}`;
}

export function deskToServingConfig(
  c: DeskConfig,
  slo: { ttftMs: number; tpotMs: number },
  recordSnapshots = false,
): ServingConfig {
  return {
    batching: c.batching,
    staticBatchSize: c.maxRunning,
    staticMaxWaitMs: 100,
    maxRunning: c.maxRunning,
    tokenBudget: c.tokenBudget,
    allocator: c.memoryMode === 'reserve-max' ? 'reserve-max' : 'paged',
    blockSize: 16,
    prefixSharing: c.memoryMode === 'paged-prefix',
    admission: c.admission,
    gpu: RTX_4090,
    model: LLAMA3_8B,
    slo,
    recordSnapshots,
  };
}

export interface DeskBrief {
  id: 'chat' | 'longdoc' | 'crowd';
  tab: string;
  title: string;
  trace: WorkloadTrace;
  slo: { ttftMs: number; tpotMs: number };
  minAttainment: number;
  minGoodput: number;
  defaultConfig: DeskConfig;
  description: string;
  optGoal: string;
}

export const DESK_BRIEFS: readonly DeskBrief[] = [
  {
    id: 'chat',
    tab: '1 · Launch chat',
    title: 'Brief 1 — Nobody waits forever',
    trace: LAB_TRACE_CHAT,
    slo: { ttftMs: 1500, tpotMs: 100 },
    minAttainment: 0.9,
    minGoodput: 0,
    defaultConfig: {
      batching: 'static',
      maxRunning: 8,
      tokenBudget: null,
      memoryMode: 'reserve-max',
      admission: 'fcfs',
    },
    description:
      '40 chat requests arrive on an RTX 4090 (24 GiB) running Llama-3 8B FP16 (max context 4,096). Meet the SLO — TTFT ≤ 1,500 ms and worst TPOT ≤ 100 ms — for at least 90% of requests.',
    optGoal: 'Maximise goodput (SLO-meeting requests/s).',
  },
  {
    id: 'longdoc',
    tab: '2 · Long docs',
    title: 'Brief 2 — The long document',
    trace: LAB_TRACE_LONGDOC,
    slo: { ttftMs: 3000, tpotMs: 120 },
    minAttainment: 0.9,
    minGoodput: 0,
    defaultConfig: {
      batching: 'continuous',
      maxRunning: 16,
      tokenBudget: null,
      memoryMode: 'paged',
      admission: 'fcfs',
    },
    description:
      '36 mixed requests arrive, including 8k–16k documents (max context 16,896). Keep TTFT ≤ 3,000 ms and worst TPOT ≤ 120 ms for at least 90% of requests.',
    optGoal: 'Maximise goodput (SLO-meeting requests/s).',
  },
  {
    id: 'crowd',
    tab: '3 · Shared crowd',
    title: 'Brief 3 — Fit the crowd',
    trace: LAB_TRACE_CROWD,
    slo: { ttftMs: 2500, tpotMs: 120 },
    minAttainment: 0.85,
    minGoodput: 4.0,
    defaultConfig: {
      batching: 'continuous',
      maxRunning: 16,
      tokenBudget: 512,
      memoryMode: 'reserve-max',
      admission: 'fcfs',
    },
    description:
      '64 busy users arrive at 16 req/s, every one carrying the same 1,024-token system prompt (max context 8,192). Keep at least 85% within TTFT ≤ 2,500 ms and worst TPOT ≤ 120 ms while reaching goodput ≥ 4.0 req/s.',
    optGoal: 'Maximise goodput (with zero preemptions).',
  },
];

export function evaluateDesk(brief: DeskBrief, cfg: DeskConfig, recordSnapshots = false) {
  const sim = simulate(brief.trace, deskToServingConfig(cfg, brief.slo, recordSnapshots));
  const passes =
    sim.metrics.sloAttainment >= brief.minAttainment - 1e-9 &&
    sim.metrics.goodputReqPerSec >= brief.minGoodput - 1e-9;
  return { sim, passes };
}

/**
 * Evidence-only hint for Lab 1: points strictly at what is visible in the Gantt, KV map,
 * latency distribution, and machine log — NEVER names a control or setting to pick.
 */
export function deskHint(brief: DeskBrief, sim: SimResult): string {
  const m = sim.metrics;
  const ttftFailCount = sim.requests.filter((r) => r.ttftMs > brief.slo.ttftMs + 1e-9).length;
  const tpotFailCount = sim.requests.filter((r) => r.outputLen > 1 && r.maxTpotMs > brief.slo.tpotMs + 1e-9).length;

  if (m.idleSlotShare > 0.25) {
    return `Look at the Gantt: ${(m.idleSlotShare * 100).toFixed(0)}% of slot-steps are hatched empty while queued users wait outside. Watch what happens to a slot after a short reply finishes early.`;
  }
  if (tpotFailCount > 0 && m.worstTpotMs > brief.slo.tpotMs * 2) {
    return `Inspect the tallest spike in the TPOT trace (${m.worstTpotMs.toFixed(0)} ms against your ${brief.slo.tpotMs} ms target). Click that step in the Gantt: how many tokens entered that single iteration, and what happened to every decoding user during it?`;
  }
  if (m.preemptions > 0) {
    return `Open the machine log: ${m.preemptions} sequences were evicted mid-generation and had to recompute because physical KV space ran dry. Check how much of that KV memory holds identical prompt tokens.`;
  }
  if (m.peakKvWasteTokens > m.peakKvUsedTokens && ttftFailCount > 0) {
    return `Check the KV memory readout: reserved space (${m.peakKvReservedTokens.toLocaleString('en-GB')} tokens) dwarfs tokens actually stored (${m.peakKvUsedTokens.toLocaleString('en-GB')}). Look at how few sequences can sit in memory at once.`;
  }
  if (ttftFailCount > 0 && brief.trace.requests[0]?.sharedPrefixLen > 0 && m.totalPrefixHitTokens === 0) {
    return `Every request in this trace starts with the same ${brief.trace.requests[0].sharedPrefixLen}-token system prompt, yet the engine recomputed and stored it from scratch ${brief.trace.requests.length} times. Watch what that does to both queue wait and memory occupancy.`;
  }
  if (ttftFailCount > 0) {
    return `${ttftFailCount} of ${sim.requests.length} requests missed the ${brief.slo.ttftMs} ms TTFT bar (p95 is ${m.ttftP95Ms.toFixed(0)} ms). Click a late request in the table to see whether it spent its wait in the queue or inside prefill.`;
  }
  if (tpotFailCount > 0) {
    return `${tpotFailCount} requests saw a token gap above ${brief.slo.tpotMs} ms (worst ${m.worstTpotMs.toFixed(0)} ms). Click the slowest step in the Gantt to see how much work it packed into one step.`;
  }
  if (m.goodputReqPerSec < brief.minGoodput) {
    return `SLO attainment is ${(m.sloAttainment * 100).toFixed(0)}%, but goodput is ${m.goodputReqPerSec.toFixed(2)} req/s against the ${brief.minGoodput.toFixed(1)} req/s bar. Look at how many sequences are running together during the busy middle of the Gantt.`;
  }
  return `This configuration meets the brief (${(m.sloAttainment * 100).toFixed(0)}% SLO, ${m.goodputReqPerSec.toFixed(2)} req/s goodput). Compare its point on the throughput-versus-latency map to see whether another trade-off pushes goodput higher.`;
}
