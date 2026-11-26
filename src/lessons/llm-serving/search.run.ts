// authored-by: Gemini Antigravity
/**
 * Exhaustive search over all 300 Lab 1 (The Serving Desk) configurations across all 3 briefs.
 * Also runs a 56-config static-batching (batchSize × maxWaitMs) sweep on Brief 1 to confirm
 * static batching fails even at its own best wait timeout.
 * Writes `results.ts`.
 *
 * Run with:
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs "$(pwd)/src/lessons/llm-serving/search.run.ts"
 */
import { nodeFs } from '../../lib/nodeIo';
import { LLAMA3_8B, RTX_4090, simulate } from './serve';
import { LAB_TRACE_CHAT } from './trace';
import {
  DESK_BRIEFS,
  allDeskConfigs,
  deskConfigKey,
  evaluateDesk,
  type DeskBrief,
} from './lab';
import type { DeskConfig } from './realCode';

export interface ScatterRow {
  key: string;
  cfg: DeskConfig;
  throughputTokPerSec: number;
  ttftP95Ms: number;
  tpotP95Ms: number;
  worstTpotMs: number;
  sloAttainment: number;
  goodputReqPerSec: number;
  preemptions: number;
  passes: boolean;
}

export interface BriefSearchResult {
  briefId: DeskBrief['id'];
  totalConfigs: number;
  passingCount: number;
  bestGoodput: number;
  bestKeys: string[];
  bestConfig: DeskConfig;
  bestStats: {
    goodputReqPerSec: number;
    throughputTokPerSec: number;
    ttftP95Ms: number;
    tpotP95Ms: number;
    worstTpotMs: number;
    sloAttainment: number;
    preemptions: number;
  };
  failingCheck: {
    staticPassing: number;
    bestStaticAttainment: number;
    noChunkPassing: number;
    reserveMaxPassing: number;
    pagedNoPrefixPassing: number;
  };
  paretoKeys: string[];
  rows: ScatterRow[];
}

export function searchBrief(brief: DeskBrief): BriefSearchResult {
  const configs = allDeskConfigs();
  const rows: ScatterRow[] = [];

  let staticPassing = 0;
  let bestStaticAttainment = 0;
  let noChunkPassing = 0;
  let reserveMaxPassing = 0;
  let pagedNoPrefixPassing = 0;

  for (const cfg of configs) {
    const { sim, passes } = evaluateDesk(brief, cfg, false);
    const m = sim.metrics;
    if (cfg.batching === 'static') {
      bestStaticAttainment = Math.max(bestStaticAttainment, m.sloAttainment);
      if (passes) staticPassing++;
    }
    if (passes) {
      if (cfg.tokenBudget === null) noChunkPassing++;
      if (cfg.memoryMode === 'reserve-max') reserveMaxPassing++;
      if (cfg.memoryMode === 'paged') pagedNoPrefixPassing++;
    }
    rows.push({
      key: deskConfigKey(cfg),
      cfg,
      throughputTokPerSec: +m.throughputTokPerSec.toFixed(2),
      ttftP95Ms: +m.ttftP95Ms.toFixed(1),
      tpotP95Ms: +m.tpotP95Ms.toFixed(2),
      worstTpotMs: +m.worstTpotMs.toFixed(2),
      sloAttainment: +m.sloAttainment.toFixed(4),
      goodputReqPerSec: +m.goodputReqPerSec.toFixed(4),
      preemptions: m.preemptions,
      passes,
    });
  }

  const passingRows = rows.filter((r) => r.passes);
  const bestGoodput = passingRows.length ? Math.max(...passingRows.map((r) => r.goodputReqPerSec)) : 0;
  const bestTies = passingRows.filter((r) => Math.abs(r.goodputReqPerSec - bestGoodput) < 0.005);
  const bestRow = bestTies.sort((a, b) => a.preemptions - b.preemptions || a.ttftP95Ms - b.ttftP95Ms)[0];

  // Pareto front over ALL 300 configs: maximise throughputTokPerSec, minimise ttftP95Ms
  const sorted = [...rows].sort((a, b) => b.throughputTokPerSec - a.throughputTokPerSec || a.ttftP95Ms - b.ttftP95Ms);
  const paretoKeys: string[] = [];
  let minTtft = Infinity;
  for (const r of sorted) {
    if (r.ttftP95Ms < minTtft - 1e-6) {
      paretoKeys.push(r.key);
      minTtft = r.ttftP95Ms;
    }
  }

  return {
    briefId: brief.id,
    totalConfigs: rows.length,
    passingCount: passingRows.length,
    bestGoodput: +bestGoodput.toFixed(4),
    bestKeys: bestTies.map((r) => r.key),
    bestConfig: bestRow.cfg,
    bestStats: {
      goodputReqPerSec: bestRow.goodputReqPerSec,
      throughputTokPerSec: bestRow.throughputTokPerSec,
      ttftP95Ms: bestRow.ttftP95Ms,
      tpotP95Ms: bestRow.tpotP95Ms,
      worstTpotMs: bestRow.worstTpotMs,
      sloAttainment: bestRow.sloAttainment,
      preemptions: bestRow.preemptions,
    },
    failingCheck: {
      staticPassing,
      bestStaticAttainment: +bestStaticAttainment.toFixed(4),
      noChunkPassing,
      reserveMaxPassing,
      pagedNoPrefixPassing,
    },
    paretoKeys,
    rows,
  };
}

export function brief1StaticWaitSweep() {
  const batches = [2, 4, 8, 12, 16, 24, 32, 64];
  const waits = [0, 25, 50, 100, 200, 400, 800];
  let bestAttainment = 0;
  let bestPair = { batchSize: 2, maxWaitMs: 0 };
  for (const batchSize of batches) {
    for (const maxWaitMs of waits) {
      const res = simulate(LAB_TRACE_CHAT, {
        batching: 'static',
        staticBatchSize: batchSize,
        staticMaxWaitMs: maxWaitMs,
        maxRunning: batchSize,
        tokenBudget: 512,
        allocator: 'paged',
        blockSize: 16,
        prefixSharing: false,
        admission: 'fcfs',
        gpu: RTX_4090,
        model: LLAMA3_8B,
        slo: DESK_BRIEFS[0].slo,
      });
      if (res.metrics.sloAttainment > bestAttainment) {
        bestAttainment = res.metrics.sloAttainment;
        bestPair = { batchSize, maxWaitMs };
      }
    }
  }
  return {
    gridSize: batches.length * waits.length,
    bestAttainment: +bestAttainment.toFixed(4),
    bestPair,
  };
}

const procArgv = (globalThis as unknown as { process?: { argv?: string[] } }).process?.argv;
const isMain = Boolean(procArgv?.[1] && import.meta.url === `file://${procArgv[1]}`);
if (isMain) {
  const fs = await nodeFs();
  const byBrief: Record<string, BriefSearchResult> = {};
  for (const b of DESK_BRIEFS) {
    const res = searchBrief(b);
    byBrief[b.id] = res;
    console.log(
      `${b.id.padEnd(8)}: ${res.passingCount}/${res.totalConfigs} pass | best goodput ${res.bestGoodput.toFixed(2)} req/s (${res.bestKeys.length} ties, best=${res.bestKeys[0]}) | pareto=${res.paretoKeys.length} | failCheck=`,
      res.failingCheck,
    );
  }
  const staticSweep = brief1StaticWaitSweep();
  console.log('Brief 1 static wait sweep (56 configs):', staticSweep);

  // Store compact scatter tuples so results.ts stays small & fast to bundle
  const compact = Object.fromEntries(
    Object.entries(byBrief).map(([id, r]) => [
      id,
      {
        briefId: r.briefId,
        totalConfigs: r.totalConfigs,
        passingCount: r.passingCount,
        bestGoodput: r.bestGoodput,
        bestKeys: r.bestKeys,
        bestConfig: r.bestConfig,
        bestStats: r.bestStats,
        failingCheck: r.failingCheck,
        paretoKeys: r.paretoKeys,
        // [throughputTokPerSec, ttftP95Ms, passes (0|1), goodputReqPerSec, key]
        points: r.rows.map((row) => [
          row.throughputTokPerSec,
          row.ttftP95Ms,
          row.passes ? 1 : 0,
          row.goodputReqPerSec,
          row.key,
        ] as const),
      },
    ]),
  );

  const src = `// authored-by: Gemini Antigravity
// Generated by search.run.ts: exhaustive 300-config Lab 1 search across all 3 briefs.
export const DESK_RESULTS = ${JSON.stringify(compact, null, 2)} as const;
export const BRIEF1_STATIC_SWEEP = ${JSON.stringify(staticSweep, null, 2)} as const;
`;
  fs.writeFileSync(new URL('./results.ts', import.meta.url), src);
  console.log('Wrote results.ts.');
}
