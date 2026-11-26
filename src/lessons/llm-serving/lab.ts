// authored-by: Gemini Antigravity
/**
 * Lab 1 (The Serving Desk) and Lab 2 (Tune the draft) definitions, evaluation, and
 * evidence-only hint ladders (tested in `hints.test.ts` and `search.test.ts`).
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
import type { DrafterId } from './spec';
import { SPEC_GRID } from './specResults';

// ------------------------------------------------------------------ Lab 1: The Serving Desk

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

// ------------------------------------------------------------------ Lab 2: Tune the draft

export interface SpecBrief {
  id: 'greedy' | 'sampling' | 'cpu-impossible';
  tab: string;
  title: string;
  temperature: 0 | 1.0;
  targetSpeedup: number;
  impossible?: boolean;
  description: string;
  optGoal: string;
}

export const SPEC_BRIEFS: readonly SpecBrief[] = [
  {
    id: 'greedy',
    tab: '1 · Greedy (T = 0)',
    title: 'Brief 1 — Exact greedy speedup',
    temperature: 0,
    targetSpeedup: 1.95,
    description:
      'At temperature 0 (greedy), reach a production-scale modelled speed-up of at least 1.95× on the RTX 4090 roofline while keeping every generated character identical to plain target decoding.',
    optGoal: 'Find the highest modelled speed-up across all 16 greedy settings.',
  },
  {
    id: 'sampling',
    tab: '2 · Sampling (T = 1)',
    title: 'Brief 2 — Stochastic sampling',
    temperature: 1.0,
    targetSpeedup: 2.0,
    description:
      'At temperature 1.0, reach a production-scale modelled speed-up of at least 2.00× across the 12 held-out Grimm prompts. Watch how draft cost k·c competes with acceptance rate.',
    optGoal: 'Find the highest modelled speed-up across all 16 sampling settings.',
  },
  {
    id: 'cpu-impossible',
    tab: '3 · Your CPU clock',
    title: 'Brief 3 — Beat plain decoding on your own CPU',
    temperature: 0,
    targetSpeedup: 1.0,
    impossible: true,
    description:
      'Measure real wall-clock speed on the 29.7k-parameter tiny GPT running right here on your CPU. Find a (drafter, k) setting that runs faster than plain target decoding (> 1.00× real wall time) — or prove from the roofline why none can, and declare it impossible.',
    optGoal: 'Recognise the hardware bottleneck directly from the timing readout.',
  },
];

export function specCellFor(drafter: DrafterId, temperature: number, k: number) {
  return SPEC_GRID.find((c) => c.drafter === drafter && c.temperature === temperature && c.k === k)!;
}

export function specBriefSummary(brief: SpecBrief) {
  const cells = SPEC_GRID.filter((c) => c.temperature === brief.temperature);
  if (brief.impossible) {
    const bestCpu = [...SPEC_GRID].sort((a, b) => b.cpuWallSpeedup - a.cpuWallSpeedup)[0];
    return {
      total: SPEC_GRID.length,
      passingCount: 0,
      bestSpeedup: bestCpu.cpuWallSpeedup,
      bestCells: [bestCpu],
    };
  }
  const passing = cells.filter((c) => c.modelledSpeedup >= brief.targetSpeedup - 1e-9);
  const bestVal = Math.max(...cells.map((c) => c.modelledSpeedup));
  const bestCells = cells.filter((c) => Math.abs(c.modelledSpeedup - bestVal) < 0.005);
  return {
    total: cells.length,
    passingCount: passing.length,
    bestSpeedup: bestVal,
    bestCells,
  };
}

/** Evidence-only hint for Lab 2: points only at observed metrics, never names a drafter or k value. */
export function specHint(brief: SpecBrief, drafter: DrafterId, k: number): string {
  const cell = specCellFor(drafter, brief.temperature, k);
  if (brief.impossible) {
    return `Look at the two numbers side by side: modelled GPU speed-up is ${cell.modelledSpeedup.toFixed(2)}×, while real CPU wall-clock ratio is ${cell.cpuWallSpeedup.toFixed(2)}×. Ask yourself whether a 116 KiB model sitting inside CPU cache is memory-bandwidth bound or compute-bound.`;
  }
  const draftTax = k * cell.costRatio;
  if (draftTax > 0.55) {
    return `Notice the denominator of the speed-up formula: running the draft that many steps per round adds ${(draftTax * 100).toFixed(0)}% of a target pass in overhead, eating the extra tokens per pass (E = ${cell.tokensPerPass.toFixed(2)}).`;
  }
  if (cell.tokensPerPass < brief.targetSpeedup) {
    return `At this lookahead the round emits only ${cell.tokensPerPass.toFixed(2)} tokens per target pass on average (acceptance ${(cell.alpha * 100).toFixed(1)}%), which caps speed-up below ${brief.targetSpeedup.toFixed(2)}× even before counting draft overhead.`;
  }
  return `Look at the trade between acceptance rate (${(cell.alpha * 100).toFixed(1)}%) and per-step draft cost (${(cell.costRatio * 100).toFixed(1)}% of a target pass): a cheaper draft keeps the denominator near 1 even when acceptance is lower.`;
}
