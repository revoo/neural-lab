// authored-by: Gemini Antigravity
/**
 * Lab 1: The Serving Desk — Zachtronics-style serving-engine configuration lab on LabShell.
 * Three launch-day briefs on an RTX 4090 (24 GiB) running Llama-3 8B FP16, with clickable
 * step Gantt, per-step batch inspector, KV block strip, latency histograms with SLO lines,
 * clickable per-request table, live vLLM command, MachineLog, and post-solve 300-config
 * throughput-vs-TTFT solution map.
 */
import { useEffect, useMemo, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Segmented } from '../../components/controls';
import {
  DESK_ADMISSIONS,
  DESK_BATCHINGS,
  DESK_BRIEFS,
  DESK_MAX_RUNS,
  DESK_MEM_MODES,
  DESK_TOKEN_BUDGETS,
  deskConfigKey,
  deskHint,
  evaluateDesk,
} from './lab';
import { DESK_RESULTS } from './results';
import { vllmDeskCommand, type DeskConfig, type LabMemoryMode } from './realCode';
import type { AdmissionPolicy, BatchingMode, SimResult } from './serve';

const STORAGE_KEY = 'llm-serving-desk-cfg-v1';
const CLS_VAR = (id: number) => `var(--cls-${Math.abs(id) % 10})`;

function fmtMs(ms: number): string {
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1_000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 100) return `${ms.toFixed(0)} ms`;
  return `${ms.toFixed(1)} ms`;
}

function loadSavedConfigs(): Record<string, DeskConfig> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return {};
}

function LatencyHistogram({
  title,
  values,
  sloMs,
  unit = 'ms',
}: {
  title: string;
  values: number[];
  sloMs: number;
  unit?: string;
}) {
  const maxVal = Math.max(sloMs * 1.4, ...values, 1);
  const bins = 14;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const bi = Math.min(bins - 1, Math.floor((v / maxVal) * bins));
    counts[bi]++;
  }
  const maxCount = Math.max(1, ...counts);
  const sloX = 12 + (Math.min(sloMs, maxVal) / maxVal) * 296;

  return (
    <div style={{ padding: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, marginBottom: 4 }}>
        <strong>{title}</strong>
        <span className="mono" style={{ color: 'var(--text-muted)' }}>SLO ≤ {fmtMs(sloMs)}</span>
      </div>
      <svg viewBox="0 0 320 96" style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label={title}>
        <line x1={12} y1={74} x2={308} y2={74} stroke="var(--c-axis)" />
        {counts.map((c, i) => {
          const bw = 296 / bins - 2;
          const bx = 12 + i * (296 / bins) + 1;
          const bh = (c / maxCount) * 58;
          const binStart = (i / bins) * maxVal;
          const overSlo = binStart >= sloMs;
          return (
            <rect
              key={i}
              x={bx}
              y={74 - bh}
              width={bw}
              height={Math.max(c > 0 ? 2 : 0, bh)}
              rx={1.5}
              fill={overSlo ? 'var(--c-bad)' : 'var(--c-good)'}
              opacity={0.82}
            />
          );
        })}
        <line x1={sloX} y1={10} x2={sloX} y2={74} stroke="var(--c-warn)" strokeDasharray="3 2" strokeWidth={1.5} />
        <text x={12} y={88} fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">0</text>
        <text x={sloX} y={88} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--c-warn)">
          {sloMs}{unit}
        </text>
        <text x={308} y={88} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">
          {fmtMs(maxVal)}
        </text>
      </svg>
    </div>
  );
}

function SolutionMap({
  briefId,
  currentKey,
}: {
  briefId: 'chat' | 'longdoc' | 'crowd';
  currentKey: string;
}) {
  const rec = DESK_RESULTS[briefId];
  const pts = rec.points;
  const maxThr = Math.max(...pts.map((p) => p[0]), 1);
  const minThr = Math.min(...pts.map((p) => p[0]), 0);
  const maxTtft = Math.max(...pts.map((p) => p[1]), 1);
  const paretoSet = new Set<string>(rec.paretoKeys as readonly string[]);

  const padL = 48;
  const padR = 16;
  const padT = 16;
  const padB = 30;
  const W = 540;
  const H = 200;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xOf = (thr: number) => padL + ((thr - minThr * 0.9) / Math.max(1, maxThr * 1.05 - minThr * 0.9)) * plotW;
  const yOf = (ttft: number) => padT + plotH - (Math.log10(Math.max(10, ttft)) - 1) / (Math.log10(Math.max(100, maxTtft * 1.15)) - 1) * plotH;

  const paretoPts = pts
    .filter((p) => paretoSet.has(p[4]))
    .sort((a, b) => a[0] - b[0]);

  const curPt = pts.find((p) => p[4] === currentKey);

  return (
    <div>
      <p style={{ margin: '0 0 8px', fontSize: 13 }}>
        Exhaustive map across all <strong>{rec.totalConfigs}</strong> configurations: <strong>{rec.passingCount}</strong> pass this brief.
        Best goodput found: <strong className="mono">{rec.bestGoodput.toFixed(2)} req/s</strong> ({rec.bestKeys.length}{' '}
        {(rec.bestKeys.length as number) === 1 ? 'config' : 'tied configs'}, e.g.{' '}
        <code className="mono">
          {rec.bestConfig.batching}, max {rec.bestConfig.maxRunning}, budget {rec.bestConfig.tokenBudget ?? 'off'}, {rec.bestConfig.memoryMode}, {rec.bestConfig.admission}
        </code>
        ).
      </p>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 360, height: 'auto', display: 'block', background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6 }} role="img" aria-label="Throughput vs TTFT p95 solution map">
          <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--c-axis)" />
          <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--c-axis)" />
          {/* Pareto frontier polyline */}
          {paretoPts.length > 1 && (
            <polyline
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.6}
              strokeDasharray="4 3"
              points={paretoPts.map((p) => `${xOf(p[0]).toFixed(1)},${yOf(p[1]).toFixed(1)}`).join(' ')}
            />
          )}
          {pts.map((p) => {
            const cx = xOf(p[0]);
            const cy = yOf(p[1]);
            const pass = p[2] === 1;
            return (
              <circle
                key={p[4]}
                cx={cx}
                cy={cy}
                r={pass ? 3.2 : 2.3}
                fill={pass ? 'var(--c-good)' : 'var(--c-bad)'}
                opacity={pass ? 0.85 : 0.35}
              />
            );
          })}
          {curPt && (
            <g>
              <circle cx={xOf(curPt[0])} cy={yOf(curPt[1])} r={7} fill="none" stroke="var(--text)" strokeWidth={2} />
              <circle cx={xOf(curPt[0])} cy={yOf(curPt[1])} r={3} fill="var(--accent)" />
            </g>
          )}
          <text x={padL + plotW / 2} y={H - 6} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--text-muted)">
            Throughput (output tok/s → higher is better)
          </text>
          <text x={padL - 6} y={padT + 8} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">
            {fmtMs(maxTtft)}
          </text>
          <text x={padL - 6} y={padT + plotH} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">
            10 ms
          </text>
        </svg>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 6 }}>
        Lower-right is better (high throughput, low TTFT p95 on log scale). Ringed dot: your current setting. Dashed line: Pareto frontier.
      </div>
    </div>
  );
}

export function ServingDeskLab() {
  const progress = useLabProgress('llm-serving-desk-v1');
  const log = useMachineLog();
  const [active, setActive] = useState(0);
  const brief = DESK_BRIEFS[active];

  const [savedByBrief, setSavedByBrief] = useState<Record<string, DeskConfig>>(loadSavedConfigs);
  const cfg: DeskConfig = savedByBrief[brief.id] ?? brief.defaultConfig;

  const setCfg = (patch: Partial<DeskConfig>) => {
    setSavedByBrief((prev) => {
      const next = { ...prev, [brief.id]: { ...(prev[brief.id] ?? brief.defaultConfig), ...patch } };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  };

  // Run state: runs immediately on mount/brief switch, and re-runs when learner presses "Run the morning"
  const [lastRun, setLastRun] = useState<{ briefId: string; key: string; sim: SimResult; passes: boolean }>(() => {
    const init = evaluateDesk(DESK_BRIEFS[0], loadSavedConfigs()[DESK_BRIEFS[0].id] ?? DESK_BRIEFS[0].defaultConfig, false);
    return { briefId: DESK_BRIEFS[0].id, key: deskConfigKey(loadSavedConfigs()[DESK_BRIEFS[0].id] ?? DESK_BRIEFS[0].defaultConfig), ...init };
  });

  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(0);
  const [selectedReqId, setSelectedReqId] = useState<number | null>(null);

  const curKey = deskConfigKey(cfg);
  const isStale = lastRun.briefId !== brief.id || lastRun.key !== curKey;

  const runMorning = (targetBrief = brief, targetCfg = cfg, quiet = false) => {
    const res = evaluateDesk(targetBrief, targetCfg, false);
    const key = deskConfigKey(targetCfg);
    setLastRun({ briefId: targetBrief.id, key, ...res });
    // Default selected step to the peak-TPOT step so stalls jump right out!
    let worstIdx = 0;
    for (let i = 1; i < res.sim.steps.length; i++) {
      if (res.sim.steps[i].durationMs > res.sim.steps[worstIdx].durationMs) worstIdx = i;
    }
    setSelectedStepIdx(worstIdx);
    setSelectedReqId(null);

    if (!quiet) {
      const m = res.sim.metrics;
      log.push(
        `[${targetBrief.tab}] ${targetCfg.batching}, max ${targetCfg.maxRunning}, budget ${targetCfg.tokenBudget ?? 'off'}, ${targetCfg.memoryMode}, ${targetCfg.admission} → SLO ${(m.sloAttainment * 100).toFixed(0)}%, goodput ${m.goodputReqPerSec.toFixed(2)} req/s, TTFT p95 ${fmtMs(m.ttftP95Ms)}, worst TPOT ${fmtMs(m.worstTpotMs)}, preemptions ${m.preemptions}`,
        res.passes ? 'good' : 'bad',
      );
      const preemptEvents = res.sim.events.filter((e) => e.type === 'preempt').slice(0, 3);
      for (const ev of preemptEvents) {
        log.push(`  step ${ev.step} (${fmtMs(ev.timeMs)}): ${ev.message}`, 'bad');
      }
      if (m.preemptions > 3) {
        log.push(`  …and ${m.preemptions - 3} more preemptions during this morning.`, 'bad');
      }
    }
  };

  // Keep brief switches fresh without polluting log
  useEffect(() => {
    if (lastRun.briefId !== brief.id) {
      runMorning(brief, savedByBrief[brief.id] ?? brief.defaultConfig, true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brief.id]);

  const rec = DESK_RESULTS[brief.id];
  const sim = lastRun.sim;
  const passing = !isStale && lastRun.passes;
  const optimal = passing && sim.metrics.goodputReqPerSec >= rec.bestGoodput - 0.015;

  const briefs: LabBrief[] = DESK_BRIEFS.map((b) => ({
    id: b.id,
    tab: b.tab,
    title: b.title,
    text: (
      <>
        {b.description} <strong>Optional goal:</strong> {b.optGoal}
      </>
    ),
  }));

  // Interactive Gantt binning (up to 160 clickable columns)
  const ganttCols = useMemo(() => {
    const n = sim.steps.length;
    const cols = Math.min(n, 160);
    const out: { col: number; stepIdx: number }[] = [];
    for (let c = 0; c < cols; c++) {
      out.push({ col: c, stepIdx: Math.floor((c * n) / cols) });
    }
    return out;
  }, [sim.steps.length]);

  const inspectedStep = sim.steps[Math.min(selectedStepIdx, Math.max(0, sim.steps.length - 1))];
  const ganttRows = Math.min(20, Math.max(4, inspectedStep?.slotCapacity ?? 8));

  const statusText = isStale
    ? 'Settings changed — press “Run the morning” to replay the traffic trace.'
    : passing
      ? optimal
        ? `Optimal! SLO ${(sim.metrics.sloAttainment * 100).toFixed(0)}%, goodput ${sim.metrics.goodputReqPerSec.toFixed(2)} req/s (matches the exhaustive optimum of ${rec.bestGoodput.toFixed(2)} req/s across all 300 configs).`
        : `Brief passed! SLO ${(sim.metrics.sloAttainment * 100).toFixed(0)}%, goodput ${sim.metrics.goodputReqPerSec.toFixed(2)} req/s. Optional goal: push goodput to ${rec.bestGoodput.toFixed(2)} req/s.`
      : `Not yet passing: SLO ${(sim.metrics.sloAttainment * 100).toFixed(0)}% (need ≥ ${(brief.minAttainment * 100).toFixed(0)}%), goodput ${sim.metrics.goodputReqPerSec.toFixed(2)} req/s${brief.minGoodput > 0 ? ` (need ≥ ${brief.minGoodput.toFixed(1)} req/s)` : ''}, TTFT p95 ${fmtMs(sim.metrics.ttftP95Ms)}, worst TPOT ${fmtMs(sim.metrics.worstTpotMs)}.`;

  return (
    <LabShell
      title="Lab 1 · The Serving Desk"
      caption="Roofline model of Llama-3 8B FP16 on an RTX 4090 (24 GiB). Configure the scheduler and KV allocator, press “Run the morning”, then click any Gantt step or request row to diagnose bottlenecks."
      briefs={briefs}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={statusText}
      statusTone={isStale ? 'neutral' : passing ? 'good' : 'bad'}
      hint={deskHint(brief, sim)}
      reveal={{
        label: 'Show all 300 configurations & Pareto frontier',
        content: <SolutionMap briefId={brief.id} currentKey={curKey} />,
      }}
      log={{ lines: log.lines, onClear: log.clear, title: 'Serving desk log', empty: 'Press “Run the morning” to trace admissions, retirements and preemptions.' }}
      actions={
        <>
          <button type="button" className="btn primary" onClick={() => runMorning(brief, cfg, false)}>
            Run the morning
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setCfg(brief.defaultConfig);
              runMorning(brief, brief.defaultConfig, false);
            }}
          >
            Reset brief
          </button>
        </>
      }
    >
      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'flex-end', marginBottom: 14 }}>
        <Segmented
          label="Batching"
          value={cfg.batching}
          onChange={(v) => setCfg({ batching: v as BatchingMode })}
          options={DESK_BATCHINGS.map((b) => ({ value: b, label: b === 'static' ? 'Static' : 'Continuous' }))}
        />
        <Segmented
          label="Max concurrent"
          value={String(cfg.maxRunning)}
          onChange={(v) => setCfg({ maxRunning: Number(v) })}
          options={DESK_MAX_RUNS.map((n) => ({ value: String(n), label: String(n) }))}
        />
        <Segmented
          label="Token budget"
          value={cfg.tokenBudget === null ? 'off' : String(cfg.tokenBudget)}
          onChange={(v) => setCfg({ tokenBudget: v === 'off' ? null : Number(v) })}
          options={DESK_TOKEN_BUDGETS.map((b) => ({
            value: b === null ? 'off' : String(b),
            label: b === null ? 'Off' : b.toLocaleString('en-GB'),
          }))}
        />
        <Segmented
          label="KV memory"
          value={cfg.memoryMode}
          onChange={(v) => setCfg({ memoryMode: v as LabMemoryMode })}
          options={DESK_MEM_MODES.map((m) => ({
            value: m,
            label: m === 'reserve-max' ? 'Reserve max' : m === 'paged' ? 'Paged' : 'Paged + prefix',
          }))}
        />
        <Segmented
          label="Queue order"
          value={cfg.admission}
          onChange={(v) => setCfg({ admission: v as AdmissionPolicy })}
          options={DESK_ADMISSIONS.map((a) => ({ value: a, label: a === 'fcfs' ? 'FCFS' : 'Shortest prompt' }))}
        />
      </div>

      {/* Headline KPI strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(130px, 100%), 1fr))', gap: 8, marginBottom: 12, opacity: isStale ? 0.6 : 1 }}>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>SLO met</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: sim.metrics.sloAttainment >= brief.minAttainment ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {(sim.metrics.sloAttainment * 100).toFixed(0)}%
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Goodput</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: sim.metrics.goodputReqPerSec >= brief.minGoodput ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {sim.metrics.goodputReqPerSec.toFixed(2)} req/s
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>TTFT p95</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: sim.metrics.ttftP95Ms <= brief.slo.ttftMs ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {fmtMs(sim.metrics.ttftP95Ms)}
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Worst TPOT</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: sim.metrics.worstTpotMs <= brief.slo.tpotMs ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {fmtMs(sim.metrics.worstTpotMs)}
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Throughput</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--c-a)' }}>
            {sim.metrics.throughputTokPerSec.toFixed(0)} tok/s
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Preemptions</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: sim.metrics.preemptions > 0 ? 'var(--c-warn)' : 'var(--c-c)' }}>
            {sim.metrics.preemptions}
          </div>
        </div>
      </div>

      {/* Clickable Step Gantt */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontSize: 12, marginBottom: 4, flexWrap: 'wrap', gap: 6 }}>
          <strong>Engine iterations (click any column to inspect batch & KV)</strong>
          <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {sim.steps.length} steps · idle slots {(sim.metrics.idleSlotShare * 100).toFixed(1)}%
            {selectedReqId !== null ? ` · highlighting Req #${selectedReqId}` : ''}
          </span>
        </div>
        <div style={{ overflowX: 'auto', border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg-sunken)' }}>
          <svg
            viewBox={`0 0 ${24 + ganttCols.length * 4 + 10} ${12 + ganttRows * 7 + 8}`}
            style={{ width: '100%', minWidth: 500, height: 'auto', display: 'block' }}
            role="img"
            aria-label="Serving Desk step Gantt chart"
          >
            <defs>
              <pattern id="desk-hatch" width="5" height="5" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
                <line x1="0" y1="0" x2="0" y2="5" stroke="var(--text-faint)" strokeWidth="1.1" opacity="0.5" />
              </pattern>
            </defs>
            {ganttCols.map(({ col, stepIdx }) => {
              const st = sim.steps[stepIdx];
              const x = 24 + col * 4;
              const isSelected = stepIdx === selectedStepIdx;
              const isSpike = st.durationMs > brief.slo.tpotMs;
              return (
                <g key={col} style={{ cursor: 'pointer' }} onClick={() => setSelectedStepIdx(stepIdx)}>
                  {isSelected && (
                    <rect x={x - 0.5} y={8} width={4.5} height={ganttRows * 7 + 8} fill="var(--accent)" opacity={0.22} />
                  )}
                  {isSpike && (
                    <rect x={x} y={6} width={3.4} height={4} fill="var(--c-bad)" />
                  )}
                  {Array.from({ length: ganttRows }, (_, row) => {
                    const slot = st.slots[row];
                    if (!slot) return null;
                    const y = 12 + row * 7;
                    if (slot.phase === 'idle' || slot.reqId === null) {
                      if (row < st.slotCapacity && cfg.batching === 'static') {
                        return <rect key={row} x={x} y={y} width={3.4} height={5.8} fill="url(#desk-hatch)" />;
                      }
                      return null;
                    }
                    const dim = selectedReqId !== null && selectedReqId !== slot.reqId;
                    return (
                      <rect
                        key={row}
                        x={x}
                        y={y}
                        width={3.4}
                        height={5.8}
                        fill={CLS_VAR(slot.reqId)}
                        opacity={dim ? 0.16 : slot.phase === 'prefill' ? 1 : 0.72}
                      />
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Selected Step Inspector + KV Occupancy Bar */}
      {inspectedStep && (
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', marginBottom: 12, fontSize: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
            <strong className="mono">
              Step #{inspectedStep.step} (t = {fmtMs(inspectedStep.startMs)} · duration {fmtMs(inspectedStep.durationMs)})
            </strong>
            <span className="mono" style={{ color: inspectedStep.bottleneck === 'compute' ? 'var(--c-a)' : 'var(--c-warn)' }}>
              {inspectedStep.bottleneck}-bound · {inspectedStep.computeUtilPct.toFixed(1)}% Tensor Cores
            </span>
          </div>
          <div className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 6 }}>
            Batch: {inspectedStep.decodeTokens} decode tok + {inspectedStep.prefillTokens} prefill tok ·{' '}
            {(inspectedStep.flops / 1e12).toFixed(2)} TFLOP · {(inspectedStep.bytes / 1e9).toFixed(2)} GB HBM ·{' '}
            KV used {inspectedStep.kvUsedTokens.toLocaleString('en-GB')} / reserved {inspectedStep.kvReservedTokens.toLocaleString('en-GB')} tok
            {inspectedStep.sharedBlocks > 0 ? ` (${inspectedStep.sharedBlocks} shared prefix blocks)` : ''}
          </div>
          {/* KV bar */}
          <div style={{ height: 10, background: 'var(--bg-elev)', borderRadius: 4, border: '1px solid var(--border)', overflow: 'hidden', position: 'relative' }}>
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${Math.min(100, (inspectedStep.kvReservedTokens / (inspectedStep.kvTotalBlocks * 16)) * 100)}%`,
                background: 'var(--c-warn)',
                opacity: 0.35,
              }}
            />
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${Math.min(100, (inspectedStep.kvUsedTokens / (inspectedStep.kvTotalBlocks * 16)) * 100)}%`,
                background: 'var(--c-a)',
              }}
            />
          </div>
        </div>
      )}

      {/* Latency Histograms */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12, marginBottom: 12 }}>
        <LatencyHistogram title="TTFT distribution" values={sim.requests.map((r) => r.ttftMs)} sloMs={brief.slo.ttftMs} />
        <LatencyHistogram title="Worst TPOT per request" values={sim.requests.map((r) => r.maxTpotMs)} sloMs={brief.slo.tpotMs} />
      </div>

      {/* Per-Request Table (collapsible / scrollable) */}
      <details style={{ marginBottom: 12 }}>
        <summary style={{ cursor: 'pointer', fontSize: 12.5, fontWeight: 600, color: 'var(--text-muted)' }}>
          Per-request table ({sim.requests.length} requests — click any row to highlight in Gantt)
        </summary>
        <div style={{ overflowX: 'auto', maxHeight: 220, overflowY: 'auto', marginTop: 6 }}>
          <table className="num-table">
            <thead>
              <tr>
                <th>Req</th>
                <th>Kind</th>
                <th>Prompt</th>
                <th>Reply</th>
                <th>TTFT</th>
                <th>Worst TPOT</th>
                <th>Prefix hit</th>
                <th>SLO</th>
              </tr>
            </thead>
            <tbody>
              {sim.requests.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => setSelectedReqId(selectedReqId === r.id ? null : r.id)}
                  style={{
                    cursor: 'pointer',
                    background: selectedReqId === r.id ? 'var(--accent-soft)' : undefined,
                  }}
                >
                  <td className="mono">#{r.id}</td>
                  <td>{r.kind}</td>
                  <td className="mono">{r.promptLen.toLocaleString('en-GB')}</td>
                  <td className="mono">{r.outputLen}</td>
                  <td className="mono" style={{ color: r.ttftMs <= brief.slo.ttftMs ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {fmtMs(r.ttftMs)}
                  </td>
                  <td className="mono" style={{ color: r.maxTpotMs <= brief.slo.tpotMs ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {fmtMs(r.maxTpotMs)}
                  </td>
                  <td className="mono">{r.prefixHitTokens > 0 ? `${r.prefixHitTokens}` : '—'}</td>
                  <td className="mono" style={{ color: r.meetsSlo ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {r.meetsSlo ? 'PASS' : 'FAIL'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      {/* Live vLLM launch command */}
      <LiveCode
        code={vllmDeskCommand(cfg, brief.trace.maxModelLen)}
        note="vLLM CLI flags verified against docs.vllm.ai/en/stable (Nov 2026)"
      />
    </LabShell>
  );
}
