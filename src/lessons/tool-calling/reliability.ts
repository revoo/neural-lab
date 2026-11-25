// authored-by: Gemini Antigravity
/**
 * Exact probability and expected-cost maths for multi-step agent reliability, verification,
 * retries, silent corruption, and the 4-stage Pipeline Hardening Lab.
 *
 * Every quantity is computed in closed form (geometric series over caught retries + sequential
 * early-abort chain) and verified against seeded Monte Carlo simulation in `agents.test.ts`.
 */
import { createRng } from '../../lib/rng';

export type StepResult = {
  /** Probability the step succeeds within (retries + 1) attempts. */
  pOk: number;
  /** Probability the step fails silently (escapes verifier, corrupting downstream steps). */
  pSilent: number;
  /** Probability every attempt fails loudly (caught by verifier -> pipeline aborts cleanly). */
  pAbort: number;
  /** Exact expected number of attempts executed at this step. */
  expectedAttempts: number;
};

/**
 * Single-step exact reliability with per-attempt success `p`, verifier detection rate `d`
 * (probability a failed attempt is caught and triggers a retry), and `retries` extra tries
 * (so `m = retries + 1` total attempts; if `d = 0`, retries can never trigger).
 */
export function stepMetrics(p: number, d: number, retries: number): StepResult {
  const m = d > 0 ? retries + 1 : 1;
  const qc = (1 - p) * d; // fail and caught
  const qs = (1 - p) * (1 - d); // fail silently
  // Geometric sum 1 + qc + qc^2 + ... + qc^(m-1)
  const geom = qc === 1 ? m : (1 - qc ** m) / (1 - qc);
  const pOk = p * geom;
  const pSilent = qs * geom;
  const pAbort = qc ** m;
  return { pOk, pSilent, pAbort, expectedAttempts: geom };
}

/** Uniform n-step pipeline where every step shares (p, d, retries). */
export function uniformPipeline(n: number, p: number, d: number, retries: number) {
  const s = stepMetrics(p, d, retries);
  const pTask = s.pOk ** n;
  const survive = 1 - s.pAbort; // probability a step does not abort
  // Probability of reaching step k (0-indexed) is survive^k; sum_{k=0}^{n-1} survive^k
  const reachSum = survive === 1 ? n : (1 - survive ** n) / (1 - survive);
  const expectedCalls = reachSum * s.expectedAttempts;
  const pAbortAny = 1 - survive ** n;
  const pSilentAny = 1 - pTask - pAbortAny;
  return {
    stepOk: s.pOk,
    stepSilent: s.pSilent,
    stepAbort: s.pAbort,
    pTask,
    pSilentAny,
    pAbortAny,
    expectedCalls,
    callsMultiplier: expectedCalls / n,
    /** Theoretical ceiling for a single step as retries -> infinity. */
    stepCeiling: p / (1 - (1 - p) * d),
  };
}

// ------------------------------------------------------------------ Lab: Harden the Agent Pipeline

export type ModelChoice = 'fast' | 'strong';

export type StageSpec = {
  id: 'lookup' | 'policy' | 'math' | 'action';
  name: string;
  shortName: string;
  desc: string;
  pFast: number;
  pStrong: number;
  costFast: number;
  costStrong: number;
  checkLabel: string;
  checkDetect: number;
  checkCost: number;
  deterministic: boolean;
};

export const LAB_STAGES: StageSpec[] = [
  {
    id: 'lookup',
    name: '1. Parse & Lookup',
    shortName: 'Lookup',
    desc: 'Extract order ID and emit lookup_order JSON.',
    pFast: 0.88,
    pStrong: 0.96,
    costFast: 1.0,
    costStrong: 3.0,
    checkLabel: 'Schema + ID check',
    checkDetect: 1.0,
    checkCost: 0.1,
    deterministic: true,
  },
  {
    id: 'policy',
    name: '2. Policy Judgement',
    shortName: 'Policy',
    desc: 'Decide return vs warranty eligibility from free-text ticket.',
    pFast: 0.82,
    pStrong: 0.94,
    costFast: 1.0,
    costStrong: 3.0,
    checkLabel: 'LLM verifier (75% catch)',
    checkDetect: 0.75,
    checkCost: 1.0,
    deterministic: false,
  },
  {
    id: 'math',
    name: '3. Refund Arithmetic',
    shortName: 'Math',
    desc: 'Compute refund minus €25 cleaning fee where required.',
    pFast: 0.80,
    pStrong: 0.92,
    costFast: 1.0,
    costStrong: 3.0,
    checkLabel: 'Code assert',
    checkDetect: 1.0,
    checkCost: 0.1,
    deterministic: true,
  },
  {
    id: 'action',
    name: '4. Execute Action',
    shortName: 'Action',
    desc: 'Call issue_refund or create_repair_ticket (irreversible).',
    pFast: 0.90,
    pStrong: 0.97,
    costFast: 1.0,
    costStrong: 3.0,
    checkLabel: 'Dry-run guard (95% catch)',
    checkDetect: 0.95,
    checkCost: 0.2,
    deterministic: false,
  },
];

export type StageConfig = {
  model: ModelChoice;
  check: boolean;
  retries: 0 | 1 | 2;
};

export type PipelineConfig = [StageConfig, StageConfig, StageConfig, StageConfig];

export type PipelineEval = {
  pTask: number;
  pSilentAny: number;
  pAbortAny: number;
  /** Probability of reaching Stage 4 and committing a silent bad action (wrong refund/ticket). */
  pBadAction: number;
  expectedCost: number;
  stageStats: {
    id: StageSpec['id'];
    pOk: number;
    pSilent: number;
    pAbort: number;
    reachProb: number;
    expectedCost: number;
  }[];
};

/** Exact closed-form evaluation of a 4-stage pipeline config. */
export function evaluatePipeline(cfg: PipelineConfig, stages: StageSpec[] = LAB_STAGES): PipelineEval {
  let pTask = 1;
  let reach = 1;
  let expectedCost = 0;
  const stageStats: PipelineEval['stageStats'] = [];

  for (let i = 0; i < stages.length; i++) {
    const st = stages[i];
    const c = cfg[i];
    const p = c.model === 'strong' ? st.pStrong : st.pFast;
    const callCost = c.model === 'strong' ? st.costStrong : st.costFast;
    const d = c.check ? st.checkDetect : 0;
    const checkCost = c.check ? st.checkCost : 0;
    const retries = c.check ? c.retries : 0;

    const m = stepMetrics(p, d, retries);
    const stageCost = reach * m.expectedAttempts * (callCost + checkCost);
    expectedCost += stageCost;
    pTask *= m.pOk;
    stageStats.push({
      id: st.id,
      pOk: m.pOk,
      pSilent: m.pSilent,
      pAbort: m.pAbort,
      reachProb: reach,
      expectedCost: stageCost,
    });
    reach *= 1 - m.pAbort;
  }

  const pAbortAny = 1 - reach;
  const pSilentAny = 1 - pTask - pAbortAny;
  // Stage 4 silent corruption: either Stage 4 itself fails silently, or an earlier stage failed
  // silently and Stage 4 did not abort — wait: here we measure BOTH total silent failure across
  // the whole pipeline (pSilentAny) AND specifically Stage 4's own silent execution error (reach_4 * pSilent_4).
  // Even better: if an earlier step failed silently (say wrong amount calculated in Stage 3),
  // Stage 4 commits that wrong amount! That's `pSilentAny` (whole-pipeline silent failure).
  // And Stage 4's own silent error rate is `stageStats[3].reachProb * stageStats[3].pSilent`.
  const pBadAction = stageStats[3].reachProb * stageStats[3].pSilent;

  return { pTask, pSilentAny, pAbortAny, pBadAction, expectedCost, stageStats };
}

/** Seeded Monte Carlo check of `evaluatePipeline` (used by unit tests to verify closed-form maths). */
export function simulatePipeline(cfg: PipelineConfig, trials: number, seed = 42, stages: StageSpec[] = LAB_STAGES) {
  const rng = createRng(seed);
  let okCount = 0;
  let abortCount = 0;
  let silentCount = 0;
  let badActionCount = 0;
  let totalCost = 0;

  for (let t = 0; t < trials; t++) {
    let anySilent = false;
    let aborted = false;
    for (let i = 0; i < stages.length; i++) {
      const st = stages[i];
      const c = cfg[i];
      const p = c.model === 'strong' ? st.pStrong : st.pFast;
      const callCost = c.model === 'strong' ? st.costStrong : st.costFast;
      const d = c.check ? st.checkDetect : 0;
      const checkCost = c.check ? st.checkCost : 0;
      const attempts = c.check ? c.retries + 1 : 1;

      let stepResolved = false;
      for (let a = 0; a < attempts; a++) {
        totalCost += callCost + checkCost;
        if (rng.next() < p) {
          stepResolved = true;
          break;
        }
        if (rng.next() < d) {
          // caught! retry if attempts remain
          continue;
        } else {
          // silent failure! escapes check, moves on
          anySilent = true;
          if (i === 3) badActionCount++;
          stepResolved = true;
          break;
        }
      }
      if (!stepResolved) {
        aborted = true;
        break;
      }
    }
    if (aborted) abortCount++;
    else if (anySilent) silentCount++;
    else okCount++;
  }

  return {
    pTask: okCount / trials,
    pAbortAny: abortCount / trials,
    pSilentAny: silentCount / trials,
    pBadAction: badActionCount / trials,
    expectedCost: totalCost / trials,
  };
}

// ------------------------------------------------------------------ Lab briefs & exhaustive search

export type LabBrief = {
  id: 'reliable' | 'safe-refund' | 'strict';
  title: string;
  story: string;
  minSuccess: number;
  maxSilent: number;
  maxBadAction?: number;
};

export const LAB_BRIEFS: LabBrief[] = [
  {
    id: 'reliable',
    title: 'Brief 1 · Reliable Support Bot',
    story: 'Ship the 4-stage Kestrel support pipeline: at least 90% of tickets must succeed end-to-end, with at most 5% failing silently. (Notice that upgrading every stage to the Strong model with no checks only reaches 80.5% for 12.0 credits!)',
    minSuccess: 0.9,
    maxSilent: 0.05,
  },
  {
    id: 'safe-refund',
    title: 'Brief 2 · Guard the Irreversible Action',
    story: 'Finance forbids silent payment bugs. Reach at least 85% end-to-end success while keeping Stage 4’s own silent action error at or below 0.3% (clean aborts that escalate to a human are allowed).',
    minSuccess: 0.85,
    maxSilent: 0.15,
    maxBadAction: 0.003,
  },
  {
    id: 'strict',
    title: 'Brief 3 · 95% Enterprise SLA',
    story: 'Enterprise contract: at least 95% end-to-end success and at most 2% silent failures across the whole pipeline. Combine model quality, checks and retries as cheaply as possible.',
    minSuccess: 0.95,
    maxSilent: 0.02,
  },
];

export const STAGE_OPTIONS: StageConfig[] = [
  { model: 'fast', check: false, retries: 0 },
  { model: 'fast', check: true, retries: 0 },
  { model: 'fast', check: true, retries: 1 },
  { model: 'fast', check: true, retries: 2 },
  { model: 'strong', check: false, retries: 0 },
  { model: 'strong', check: true, retries: 0 },
  { model: 'strong', check: true, retries: 1 },
  { model: 'strong', check: true, retries: 2 },
];

export function briefPasses(b: LabBrief, r: PipelineEval): boolean {
  if (r.pTask < b.minSuccess - 1e-9) return false;
  if (r.pSilentAny > b.maxSilent + 1e-9) return false;
  if (b.maxBadAction !== undefined && r.pBadAction > b.maxBadAction + 1e-9) return false;
  return true;
}

export function formatStageConfig(c: StageConfig): string {
  const m = c.model === 'strong' ? 'Strong' : 'Fast';
  if (!c.check) return `${m} (no check)`;
  return `${m} + check + ${c.retries}r`;
}

export function labExhaustive(b: LabBrief) {
  let total = 0;
  let passing = 0;
  let bestCost = Infinity;
  let best: { cfg: PipelineConfig; eval: PipelineEval }[] = [];
  const passingPts: { cost: number; pTask: number; pSilent: number }[] = [];

  for (const s0 of STAGE_OPTIONS)
    for (const s1 of STAGE_OPTIONS)
      for (const s2 of STAGE_OPTIONS)
        for (const s3 of STAGE_OPTIONS) {
          total++;
          const cfg: PipelineConfig = [s0, s1, s2, s3];
          const ev = evaluatePipeline(cfg);
          if (!briefPasses(b, ev)) continue;
          passing++;
          const rounded = Math.round(ev.expectedCost * 100) / 100;
          passingPts.push({ cost: rounded, pTask: ev.pTask, pSilent: ev.pSilentAny });
          if (rounded < bestCost - 1e-9) {
            bestCost = rounded;
            best = [{ cfg, eval: ev }];
          } else if (Math.abs(rounded - bestCost) < 1e-9) {
            best.push({ cfg, eval: ev });
          }
        }

  return { total, passing, bestCost, best, passingPts };
}

export function pipelineLabHint(b: LabBrief, cfg: PipelineConfig): string {
  const ev = evaluatePipeline(cfg);
  if (briefPasses(b, ev)) {
    return `Meets all targets at ${ev.expectedCost.toFixed(2)} credits/ticket (${(100 * ev.pTask).toFixed(1)}% success, ${(100 * ev.pSilentAny).toFixed(1)}% silent). Optional: inspect each stage's credit cost and see where a cheaper model + verifier still holds the brief.`;
  }
  const okBadAct = b.maxBadAction === undefined || ev.pBadAction <= b.maxBadAction + 1e-9;
  const okSilent = ev.pSilentAny <= b.maxSilent + 1e-9;
  if (!okBadAct) {
    const s4 = ev.stageStats[3];
    return `Stage 4 commits ${(100 * ev.pBadAction).toFixed(2)}% silent bad actions (limit ${(100 * b.maxBadAction!).toFixed(2)}%). Because its verifier only catches 95% of mistakes, reducing Stage 4's silent leak (${(100 * s4.pSilent).toFixed(2)}%) requires lowering how often errors slip past the guard.`;
  }
  if (!okSilent) {
    let worstIdx = 0;
    for (let i = 1; i < 4; i++) {
      if (ev.stageStats[i].pSilent > ev.stageStats[worstIdx].pSilent) worstIdx = i;
    }
    const st = LAB_STAGES[worstIdx];
    return `Whole-pipeline silent failures hit ${(100 * ev.pSilentAny).toFixed(1)}% (limit ${(100 * b.maxSilent).toFixed(1)}%). The biggest silent leak is ${st.name} (${(100 * ev.stageStats[worstIdx].pSilent).toFixed(2)}%); any unchecked stage passes 100% of its errors downstream silently.`;
  }
  let weakIdx = 0;
  for (let i = 1; i < 4; i++) {
    if (ev.stageStats[i].pOk < ev.stageStats[weakIdx].pOk) weakIdx = i;
  }
  const st = LAB_STAGES[weakIdx];
  return `End-to-end success is ${(100 * ev.pTask).toFixed(1)}% (need ≥ ${(100 * b.minSuccess).toFixed(1)}%). The weakest stage in the waterfall is ${st.name} (${(100 * ev.stageStats[weakIdx].pOk).toFixed(1)}% ok); check whether a deterministic verifier + retry or higher base accuracy lifts that bottleneck most cheaply.`;
}

