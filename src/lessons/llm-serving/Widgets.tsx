// authored-by: Gemini Antigravity
/**
 * Interactive teaching widgets A–D for `llm-serving`:
 *   A. LatencyAnatomy  — queue wait vs prefill vs decode ticks + Tensor Core idle bar
 *   B. BatchingRace    — static (grid-searched best) vs continuous batching Gantt race with ReplayControls
 *   C. ChunkedPrefill  — 16k-token trolley problem: TPOT spike + RTX 4090 roofline knee
 *   D. KvMemoryMap     — physical KV VRAM grid, fragmentation, block-table inspector & prefix sharing
 */
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import {
  H100_80G,
  LLAMA3_8B,
  RTX_4090,
  evaluateKvMemoryMap,
  ridgePoint,
  simulate,
  singleRequestLatency,
  type AllocatorMode,
  type SimStep,
} from './serve';
import { MEMORY_MAP_SEQUENCES, RACE_TRACE, TROLLEY_TRACE } from './trace';
import { STATIC_RACE_SEARCH } from './staticBest';

const CLS_VAR = (id: number) => `var(--cls-${Math.abs(id) % 10})`;

function fmtMs(ms: number): string {
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1_000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 100) return `${ms.toFixed(0)} ms`;
  return `${ms.toFixed(1)} ms`;
}

// ================================================================== Widget A: LatencyAnatomy

export function LatencyAnatomy() {
  const [promptLen, setPromptLen] = useState(512);
  const [replyLen, setReplyLen] = useState(160);
  const [usersAhead, setUsersAhead] = useState(0);

  const a = useMemo(
    () => singleRequestLatency({ promptLen, replyLen, usersAhead, gpu: H100_80G, model: LLAMA3_8B }),
    [promptLen, replyLen, usersAhead],
  );

  // Timeline bar widths (minimum visible sliver when > 0)
  const total = Math.max(1e-3, a.totalLatencyMs);
  const qPct = (a.queueWaitMs / total) * 100;
  const pPct = (a.prefillMs / total) * 100;

  // Draw up to 48 evenly spaced decode ticks inside the decode span
  const tickCount = Math.min(48, Math.max(1, replyLen - 1));
  const queueSharePct = (a.queueShareOfTtft * 100).toFixed(1);
  const decodeIdlePct = (100 - a.meanDecodeUtilPct).toFixed(2);

  let liveNote = '';
  if (usersAhead === 0) {
    liveNote = `Just you on an H100 (roofline model): prefill reads your ${promptLen}-token prompt in ${fmtMs(a.prefillMs)} (${a.prefillUtilPct.toFixed(0)}% Tensor Cores busy), then ${replyLen - 1} decode steps stream at ${a.decodeTokPerSec.toFixed(0)} tok/s (${a.meanTpotMs.toFixed(1)} ms/token) — while ${decodeIdlePct}% of the GPU’s arithmetic sits idle.`;
  } else {
    liveNote = `TTFT is ${queueSharePct}% queueing (${fmtMs(a.queueWaitMs)} waiting for ${usersAhead} earlier ${usersAhead === 1 ? 'user' : 'users'} + ${fmtMs(a.prefillMs)} for your own prefill). Your streaming speed once started stays ${a.decodeTokPerSec.toFixed(0)} tok/s, yet the GPU is still ${decodeIdlePct}% idle during every decode step.`;
  }

  return (
    <Widget
      title="Anatomy of one request: queue, prefill, and decode"
      wide
      caption="Roofline model of Llama-3 8B FP16 on an H100 SXM (one request at a time, no batching). Slide “Users ahead of you” from 0 to 20 to watch queueing swallow TTFT while decode leaves over 99% of the Tensor Cores idle."
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(200px, 100%), 1fr))', gap: 14, marginBottom: 14 }}>
        <Slider label="Prompt length" min={64} max={4096} step={64} value={promptLen} onChange={setPromptLen} format={(v) => `${v} tok`} />
        <Slider label="Reply length" min={16} max={512} step={16} value={replyLen} onChange={setReplyLen} format={(v) => `${v} tok`} />
        <Slider label="Users ahead of you" min={0} max={50} step={1} value={usersAhead} onChange={setUsersAhead} format={(v) => (v === 0 ? '0 (alone)' : `${v} waiting`)} />
      </div>

      <div className="btn-row" style={{ marginBottom: 12 }}>
        <button type="button" className="btn small" onClick={() => { setPromptLen(512); setReplyLen(160); setUsersAhead(0); }}>
          Quiet morning (0 ahead)
        </button>
        <button type="button" className="btn small" onClick={() => { setPromptLen(512); setReplyLen(160); setUsersAhead(20); }}>
          Launch-day queue (20 ahead)
        </button>
        <button type="button" className="btn small" onClick={() => { setPromptLen(4096); setReplyLen(160); setUsersAhead(0); }}>
          Long prompt alone (4k)
        </button>
      </div>

      {/* Timeline SVG */}
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox="0 0 720 112" style={{ width: '100%', minWidth: 460, height: 'auto', display: 'block' }} role="img" aria-label="Request latency timeline showing queue wait, prefill block, and decode ticks">
          {/* Axis bar */}
          <rect x={16} y={32} width={688} height={34} rx={5} fill="var(--bg-sunken)" stroke="var(--border)" />
          {/* Queue wait */}
          {qPct > 0 && (
            <rect
              x={16}
              y={32}
              width={Math.max(4, (qPct / 100) * 688)}
              height={34}
              rx={4}
              fill="var(--c-warn)"
              opacity={0.85}
            />
          )}
          {/* Prefill */}
          {(() => {
            const qW = qPct > 0 ? Math.max(4, (qPct / 100) * 688) : 0;
            const pW = Math.max(6, (pPct / 100) * 688);
            const dX = 16 + qW + pW;
            const dW = Math.max(12, 688 - qW - pW);
            const ttftX = 16 + qW + pW;
            return (
              <>
                <rect x={16 + qW} y={32} width={pW} height={34} fill="var(--c-a)" opacity={0.92} />
                <rect x={dX} y={32} width={dW} height={34} rx={4} fill="var(--c-c)" opacity={0.18} />
                {Array.from({ length: tickCount }, (_, i) => {
                  const tx = dX + ((i + 0.5) / tickCount) * dW;
                  return <line key={i} x1={tx} y1={35} x2={tx} y2={63} stroke="var(--c-c)" strokeWidth={1.6} opacity={0.9} />;
                })}
                {/* TTFT bracket */}
                <line x1={16} y1={18} x2={ttftX} y2={18} stroke="var(--c-a)" strokeWidth={1.5} />
                <line x1={16} y1={13} x2={16} y2={23} stroke="var(--c-a)" strokeWidth={1.5} />
                <line x1={ttftX} y1={13} x2={ttftX} y2={23} stroke="var(--c-a)" strokeWidth={1.5} />
                <text x={Math.max(60, Math.min(640, (16 + ttftX) / 2))} y={12} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--c-a)" fontWeight={600}>
                  TTFT: {fmtMs(a.ttftMs)}
                </text>
                {/* Total latency bracket */}
                <line x1={16} y1={82} x2={704} y2={82} stroke="var(--text-muted)" strokeWidth={1.2} />
                <line x1={16} y1={77} x2={16} y2={87} stroke="var(--text-muted)" strokeWidth={1.2} />
                <line x1={704} y1={77} x2={704} y2={87} stroke="var(--text-muted)" strokeWidth={1.2} />
                <text x={360} y={100} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                  Total latency = {fmtMs(a.ttftMs)} + {replyLen - 1} × {a.meanTpotMs.toFixed(2)} ms = {fmtMs(a.totalLatencyMs)}
                </text>
              </>
            );
          })()}
        </svg>
      </div>

      {/* Key + Metric cards */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, color: 'var(--text-muted)', marginTop: 4, marginBottom: 12 }}>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: 'var(--c-warn)', marginRight: 5 }} />Queue wait ({fmtMs(a.queueWaitMs)})</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: 'var(--c-a)', marginRight: 5 }} />Prefill ({fmtMs(a.prefillMs)})</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: 'var(--c-c)', marginRight: 5 }} />Decode ({replyLen - 1} tokens, {fmtMs(a.decodeTotalMs)})</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(150px, 100%), 1fr))', gap: 10, marginBottom: 10 }}>
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>TTFT</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-a)' }}>{fmtMs(a.ttftMs)}</div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>TPOT</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-c)' }}>{a.meanTpotMs.toFixed(2)} ms</div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Decode compute busy</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-bad)' }}>{a.meanDecodeUtilPct.toFixed(2)}%</div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>GPU arithmetic idle</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-warn)' }}>{decodeIdlePct}%</div>
        </div>
      </div>

      <div className="live-note">{liveNote}</div>
    </Widget>
  );
}

// ================================================================== Widget B: BatchingRace

function GanttStrip({
  label,
  steps,
  slotCount,
  cursor,
  maxSteps,
  selectedReq,
  onSelectReq,
  patternId,
}: {
  label: string;
  steps: SimStep[];
  slotCount: number;
  cursor: number;
  maxSteps: number;
  selectedReq: number | null;
  onSelectReq: (id: number | null) => void;
  patternId: string;
}) {
  // Bin steps into at most 180 visual columns so 600+ steps stay crisp & fast
  const cols = Math.min(maxSteps, 180);
  const stepPerCol = maxSteps / cols;
  const cellW = 3.6;
  const cellH = 8;
  const padL = 34;
  const padT = 16;
  const svgW = padL + cols * cellW + 10;
  const svgH = padT + slotCount * cellH + 10;
  const cursorX = padL + (Math.min(cursor, maxSteps) / maxSteps) * (cols * cellW);

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4, flexWrap: 'wrap', gap: 6 }}>
        <strong style={{ fontSize: 13 }}>{label}</strong>
        <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {steps.length} iterations · finishes at {fmtMs(steps[steps.length - 1]?.endMs ?? 0)}
        </span>
      </div>
      <div style={{ overflowX: 'auto', border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg-sunken)' }}>
        <svg viewBox={`0 0 ${svgW} ${svgH}`} style={{ width: '100%', minWidth: 520, height: 'auto', display: 'block' }} role="img" aria-label={label}>
          <defs>
            <pattern id={patternId} width="5" height="5" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
              <line x1="0" y1="0" x2="0" y2="5" stroke="var(--text-faint)" strokeWidth="1.2" opacity="0.55" />
            </pattern>
          </defs>
          {/* Slot labels */}
          {[0, Math.floor(slotCount / 2), slotCount - 1].map((s) => (
            <text key={s} x={padL - 5} y={padT + s * cellH + cellH - 1} textAnchor="end" fontSize={8.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
              S{s + 1}
            </text>
          ))}
          {Array.from({ length: cols }, (_, c) => {
            const si = Math.floor(c * stepPerCol);
            const st = steps[si];
            if (!st) return null;
            const pastCursor = si > cursor;
            return (
              <g key={c} opacity={pastCursor ? 0.22 : 1}>
                {Array.from({ length: slotCount }, (_, slotIdx) => {
                  const slot = st.slots[slotIdx];
                  if (!slot) return null;
                  const x = padL + c * cellW;
                  const y = padT + slotIdx * cellH;
                  if (slot.phase === 'idle' || slot.reqId === null) {
                    // Only hatch slots that belong to an active static batch (slotIdx < st.slotCapacity)
                    if (slotIdx < st.slotCapacity) {
                      return <rect key={slotIdx} x={x} y={y} width={cellW - 0.3} height={cellH - 1} fill={`url(#${patternId})`} />;
                    }
                    return null;
                  }
                  const dimmed = selectedReq !== null && selectedReq !== slot.reqId;
                  const isPrefill = slot.phase === 'prefill';
                  return (
                    <rect
                      key={slotIdx}
                      x={x}
                      y={y}
                      width={cellW - 0.3}
                      height={cellH - 1}
                      fill={CLS_VAR(slot.reqId)}
                      opacity={dimmed ? 0.18 : isPrefill ? 1 : 0.72}
                      stroke={isPrefill ? 'var(--text)' : 'none'}
                      strokeWidth={isPrefill ? 0.5 : 0}
                      style={{ cursor: 'pointer' }}
                      onClick={() => onSelectReq(selectedReq === slot.reqId ? null : slot.reqId)}
                    />
                  );
                })}
              </g>
            );
          })}
          {/* Step cursor */}
          <line x1={cursorX} y1={padT - 4} x2={cursorX} y2={svgH - 4} stroke="var(--accent)" strokeWidth={1.8} />
        </svg>
      </div>
    </div>
  );
}

export function BatchingRace() {
  const bestStatic = STATIC_RACE_SEARCH.bestByThroughput;
  const { staticSim, contSim, maxSteps } = useMemo(() => {
    const s = simulate(RACE_TRACE, {
      batching: 'static',
      staticBatchSize: bestStatic.batchSize,
      staticMaxWaitMs: bestStatic.maxWaitMs,
      maxRunning: bestStatic.batchSize,
      tokenBudget: null,
      allocator: 'paged',
      blockSize: 16,
      prefixSharing: false,
      admission: 'fcfs',
      gpu: RTX_4090,
      model: LLAMA3_8B,
    });
    const c = simulate(RACE_TRACE, {
      batching: 'continuous',
      maxRunning: bestStatic.batchSize,
      tokenBudget: null,
      allocator: 'paged',
      blockSize: 16,
      prefixSharing: false,
      admission: 'fcfs',
      gpu: RTX_4090,
      model: LLAMA3_8B,
    });
    return { staticSim: s, contSim: c, maxSteps: Math.max(s.steps.length, c.steps.length) };
  }, [bestStatic.batchSize, bestStatic.maxWaitMs]);

  const replay = useReplay(maxSteps, 28, 'race', false);
  const [selectedReq, setSelectedReq] = useState<number | null>(null);

  const staticStepAt = staticSim.steps[Math.min(replay.i, staticSim.steps.length - 1)];
  const contStepAt = contSim.steps[Math.min(replay.i, contSim.steps.length - 1)];
  const staticDone = staticSim.requests.filter((r) => r.finishMs <= (staticStepAt?.endMs ?? 0) + 1e-6).length;
  const contDone = contSim.requests.filter((r) => r.finishMs <= (contStepAt?.endMs ?? 0) + 1e-6).length;

  const selStatic = selectedReq !== null ? staticSim.requests.find((r) => r.id === selectedReq) : null;
  const selCont = selectedReq !== null ? contSim.requests.find((r) => r.id === selectedReq) : null;

  return (
    <Widget
      title="Static vs continuous batching: the same 24 requests"
      wide
      caption={`Roofline model of Llama-3 8B FP16 on an RTX 4090 (24 GiB). Static batching runs at its best settings found by searching all ${STATIC_RACE_SEARCH.gridSize} (batch size × wait timeout) pairs: batch ${bestStatic.batchSize}, wait ${bestStatic.maxWaitMs} ms. Press Play to watch empty hatched slots pile up in static batching while continuous batching refills every freed slot on the very next step. Click any bar to trace one request.`}
    >
      <ReplayControls r={replay} label="iter" />

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, color: 'var(--text-muted)', margin: '8px 0 10px' }}>
        <span>Coloured blocks: active requests (outlined = prefill, solid = decode)</span>
        <span>Hatched cells: slot held empty by a finished request</span>
        {selectedReq !== null && (
          <button type="button" className="btn small" onClick={() => setSelectedReq(null)}>
            Clear Req #{selectedReq} highlight
          </button>
        )}
      </div>

      <GanttStrip
        label={`Static batching (best searched: batch ${bestStatic.batchSize}, wait ${bestStatic.maxWaitMs} ms) — ${staticDone}/24 done`}
        steps={staticSim.steps}
        slotCount={bestStatic.batchSize}
        cursor={replay.i}
        maxSteps={maxSteps}
        selectedReq={selectedReq}
        onSelectReq={setSelectedReq}
        patternId="hatch-static-race"
      />
      <GanttStrip
        label={`Continuous batching (${bestStatic.batchSize} slots) — ${contDone}/24 done`}
        steps={contSim.steps}
        slotCount={bestStatic.batchSize}
        cursor={replay.i}
        maxSteps={maxSteps}
        selectedReq={selectedReq}
        onSelectReq={setSelectedReq}
        patternId="hatch-cont-race"
      />

      <div style={{ overflowX: 'auto', marginTop: 10 }}>
        <table className="num-table" style={{ whiteSpace: 'nowrap' }}>
          <thead>
            <tr>
              <th>Scheduler</th>
              <th>Throughput</th>
              <th>TTFT p50</th>
              <th>TTFT p95</th>
              <th>TPOT p50</th>
              <th>TPOT p95</th>
              <th>Idle slot-steps</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Static (best of 36)</td>
              <td className="mono">{staticSim.metrics.throughputTokPerSec.toFixed(1)} tok/s</td>
              <td className="mono">{fmtMs(staticSim.metrics.ttftP50Ms)}</td>
              <td className="mono" style={{ color: 'var(--c-bad)' }}>{fmtMs(staticSim.metrics.ttftP95Ms)}</td>
              <td className="mono">{staticSim.metrics.tpotP50Ms.toFixed(1)} ms</td>
              <td className="mono">{staticSim.metrics.tpotP95Ms.toFixed(1)} ms</td>
              <td className="mono" style={{ color: 'var(--c-warn)' }}>{(staticSim.metrics.idleSlotShare * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td><strong>Continuous (16 slots)</strong></td>
              <td className="mono" style={{ color: 'var(--c-good)' }}><strong>{contSim.metrics.throughputTokPerSec.toFixed(1)} tok/s</strong></td>
              <td className="mono" style={{ color: 'var(--c-good)' }}>{fmtMs(contSim.metrics.ttftP50Ms)}</td>
              <td className="mono" style={{ color: 'var(--c-good)' }}><strong>{fmtMs(contSim.metrics.ttftP95Ms)}</strong></td>
              <td className="mono">{contSim.metrics.tpotP50Ms.toFixed(1)} ms</td>
              <td className="mono" style={{ color: 'var(--c-warn)' }}>{contSim.metrics.tpotP95Ms.toFixed(1)} ms</td>
              <td className="mono" style={{ color: 'var(--c-good)' }}>{(contSim.metrics.idleSlotShare * 100).toFixed(1)}%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {selStatic && selCont
          ? `Req #${selStatic.id} (${selStatic.promptLen} prompt → ${selStatic.outputLen} output tokens): TTFT drops from ${fmtMs(selStatic.ttftMs)} in static batching to ${fmtMs(selCont.ttftMs)} in continuous batching, finishing at ${fmtMs(selCont.finishMs)} instead of ${fmtMs(selStatic.finishMs)}.`
          : `Continuous batching cuts TTFT p95 by ${(staticSim.metrics.ttftP95Ms / contSim.metrics.ttftP95Ms).toFixed(0)}× (${fmtMs(staticSim.metrics.ttftP95Ms)} → ${fmtMs(contSim.metrics.ttftP95Ms)}) and boosts throughput by ${(((contSim.metrics.throughputTokPerSec / staticSim.metrics.throughputTokPerSec) - 1) * 100).toFixed(0)}% by eliminating ${(staticSim.metrics.idleSlotShare * 100).toFixed(1)}% idle slot-steps. Notice the one catch already peeking out: continuous TPOT p95 rose to ${contSim.metrics.tpotP95Ms.toFixed(1)} ms because newly arriving prefills interrupt active decodes.`}
      </div>
    </Widget>
  );
}

// ================================================================== Widget C: ChunkedPrefill

const BUDGET_CHOICES: (number | null)[] = [null, 64, 128, 256, 512, 1024, 2048, 4096];

export function ChunkedPrefill() {
  const [budgetIdx, setBudgetIdx] = useState(0); // start at 'off' so the learner feels the 2-second freeze!
  const budget = BUDGET_CHOICES[budgetIdx];

  // Precompute all 8 budgets once so switching is instant and we can plot the roofline knee
  const allRuns = useMemo(() => {
    return BUDGET_CHOICES.map((b) => {
      const sim = simulate(TROLLEY_TRACE, {
        batching: 'continuous',
        maxRunning: 16,
        tokenBudget: b,
        allocator: 'paged',
        blockSize: 16,
        prefixSharing: false,
        admission: 'fcfs',
        gpu: RTX_4090,
        model: LLAMA3_8B,
      });
      const doc = sim.requests.find((r) => r.kind === 'doc')!;
      return { budget: b, sim, docTtftMs: doc.ttftMs, worstTpotMs: sim.metrics.worstTpotMs };
    });
  }, []);

  const cur = allRuns[budgetIdx];
  const sim = cur.sim;
  const ridge = ridgePoint(RTX_4090);

  // Build TPOT-over-time curve across steps (steps where at least 1 decode happened)
  const decodeSteps = sim.steps.filter((s) => s.decodeTokens > 0);
  const maxT = Math.max(1, sim.metrics.totalTimeMs);
  const yCap = Math.max(250, Math.min(2200, Math.ceil(cur.worstTpotMs / 100) * 100 + 100));

  const kneePoints = allRuns.filter((r) => r.budget !== null);
  const maxDocTtft = Math.max(...allRuns.map((r) => r.docTtftMs));

  const isBelowRidge = budget !== null && budget < ridge;
  const liveNote =
    budget === null
      ? `No chunking: the 16,384-token document monopolises one ${cur.worstTpotMs.toFixed(0)} ms prefill step — freezing every chatting user for over two full seconds! Doc TTFT is ${fmtMs(cur.docTtftMs)}.`
      : isBelowRidge
        ? `Token budget ${budget} sits BELOW the RTX 4090 ridge point (~${ridge.toFixed(0)} tokens/step): every chunk falls back into the memory-bound regime and re-reads all 16.06 GB of weights. Worst TPOT is silky (${cur.worstTpotMs.toFixed(1)} ms), but doc TTFT stretches to ${fmtMs(cur.docTtftMs)}.`
        : `Token budget ${budget} sits ABOVE the RTX 4090 ridge point (~${ridge.toFixed(0)} tokens/step): each chunk is compute-bound, so active decodes piggyback nearly for free! Worst TPOT drops to ${cur.worstTpotMs.toFixed(1)} ms while doc TTFT stays ${fmtMs(cur.docTtftMs)} (${sim.metrics.throughputTokPerSec.toFixed(0)} tok/s overall).`;

  return (
    <Widget
      title="The 16k-token trolley problem: chunked prefill and the roofline knee"
      wide
      caption="Roofline model of Llama-3 8B FP16 on an RTX 4090 (10 chatters + one 16,384-token document arriving at 350 ms). Start at “Off” to see the 2,019 ms TPOT freeze, then step the token budget down through 512, 256, 128, and 64 to feel both sides of the GPU’s 164-token ridge point."
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="Per-step token budget"
          value={String(budgetIdx)}
          onChange={(v) => setBudgetIdx(Number(v))}
          options={BUDGET_CHOICES.map((b, i) => ({
            value: String(i),
            label: b === null ? 'Off' : b.toLocaleString('en-GB'),
          }))}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 14 }}>
        {/* Left: TPOT over time trace */}
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6, padding: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
            Inter-token latency (TPOT) over time
          </div>
          <svg viewBox="0 0 360 165" style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="TPOT over time sparkline">
            {/* 100 ms interactive target line */}
            {(() => {
              const y100 = 135 - (Math.min(100, yCap) / yCap) * 115;
              return (
                <>
                  <line x1={38} y1={y100} x2={348} y2={y100} stroke="var(--c-good)" strokeDasharray="4 3" strokeWidth={1.2} />
                  <text x={346} y={y100 - 3} textAnchor="end" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--c-good)">
                    100 ms target
                  </text>
                </>
              );
            })()}
            {/* Axes */}
            <line x1={38} y1={20} x2={38} y2={135} stroke="var(--c-axis)" />
            <line x1={38} y1={135} x2={348} y2={135} stroke="var(--c-axis)" />
            <text x={34} y={24} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">{yCap}</text>
            <text x={34} y={138} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">0</text>
            <text x={193} y={156} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
              wall clock (0 → {fmtMs(maxT)})
            </text>
            {/* Polyline of step max TPOT */}
            <polyline
              fill="none"
              stroke={cur.worstTpotMs > 100 ? 'var(--c-bad)' : 'var(--c-c)'}
              strokeWidth={1.8}
              points={decodeSteps
                .map((s) => {
                  const x = 38 + (s.endMs / maxT) * 310;
                  const y = 135 - (Math.min(s.maxTpotMs, yCap) / yCap) * 115;
                  return `${x.toFixed(1)},${y.toFixed(1)}`;
                })
                .join(' ')}
            />
          </svg>
        </div>

        {/* Right: Roofline knee across budgets */}
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6, padding: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
            Doc TTFT vs token budget (RTX 4090 ridge ≈ 164 tok)
          </div>
          <svg viewBox="0 0 360 165" style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="Document TTFT across chunk sizes showing the roofline knee">
            <line x1={42} y1={20} x2={42} y2={135} stroke="var(--c-axis)" />
            <line x1={42} y1={135} x2={348} y2={135} stroke="var(--c-axis)" />
            {/* Ridge vertical marker between 128 and 256 */}
            {(() => {
              const logMin = Math.log2(64);
              const logMax = Math.log2(4096);
              const xFor = (b: number) => 52 + ((Math.log2(b) - logMin) / (logMax - logMin)) * 284;
              const yFor = (ms: number) => 135 - (ms / (maxDocTtft * 1.08)) * 112;
              const ridgeX = xFor(ridge);
              return (
                <>
                  <line x1={ridgeX} y1={20} x2={ridgeX} y2={135} stroke="var(--c-warn)" strokeDasharray="3 3" strokeWidth={1.2} />
                  <text x={ridgeX + 4} y={30} fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--c-warn)">
                    ridge (~164)
                  </text>
                  <polyline
                    fill="none"
                    stroke="var(--c-a)"
                    strokeWidth={2}
                    points={kneePoints.map((p) => `${xFor(p.budget!).toFixed(1)},${yFor(p.docTtftMs).toFixed(1)}`).join(' ')}
                  />
                  {kneePoints.map((p) => {
                    const cx = xFor(p.budget!);
                    const cy = yFor(p.docTtftMs);
                    const active = p.budget === budget;
                    return (
                      <g key={p.budget}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={active ? 5.5 : 3.2}
                          fill={active ? 'var(--accent)' : 'var(--c-a)'}
                          stroke="var(--bg-elev)"
                          strokeWidth={1.5}
                        />
                        <text x={cx} y={149} textAnchor="middle" fontSize={8.5} fontFamily="var(--font-mono)" fill={active ? 'var(--accent-text)' : 'var(--text-muted)'}>
                          {p.budget! >= 1024 ? `${p.budget! / 1024}k` : p.budget}
                        </text>
                      </g>
                    );
                  })}
                  <text x={38} y={24} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                    {(maxDocTtft / 1000).toFixed(1)}s
                  </text>
                  <text x={38} y={138} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">0</text>
                </>
              );
            })()}
          </svg>
        </div>
      </div>

      {/* Summary metrics row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(150px, 100%), 1fr))', gap: 10, marginTop: 12, marginBottom: 10 }}>
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Worst chatter TPOT</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: cur.worstTpotMs > 100 ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {fmtMs(cur.worstTpotMs)}
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>16k document TTFT</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-a)' }}>
            {fmtMs(cur.docTtftMs)}
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Overall throughput</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-c)' }}>
            {sim.metrics.throughputTokPerSec.toFixed(1)} tok/s
          </div>
        </div>
      </div>

      <div className="live-note">{liveNote}</div>
    </Widget>
  );
}

// ================================================================== Widget D: KvMemoryMap

type MapModeOption = 'reserve-max' | 'reserve-grow' | 'paged' | 'paged-prefix';

export function KvMemoryMap() {
  const [mapMode, setMapMode] = useState<MapModeOption>('reserve-max');
  const [blockSize, setBlockSize] = useState<8 | 16 | 32>(16);
  const [selectedSeqId, setSelectedSeqId] = useState<number | null>(1);

  const report = useMemo(() => {
    const mode: AllocatorMode = mapMode === 'paged' || mapMode === 'paged-prefix' ? 'paged' : mapMode;
    return evaluateKvMemoryMap({
      sequences: MEMORY_MAP_SEQUENCES,
      mode,
      blockSize,
      prefixSharing: mapMode === 'paged-prefix',
      maxModelLen: 8192,
      gpu: RTX_4090,
      model: LLAMA3_8B,
    });
  }, [mapMode, blockSize]);

  const activeSeq = report.fittedSeqs.find((s) => s.id === selectedSeqId) ?? report.fittedSeqs[0] ?? null;

  const liveNote =
    mapMode === 'reserve-max'
      ? `Reserving a full 8,192-token slab per user fits only ${report.sequencesFit} of 64 conversations in ${report.kvBudgetGiB.toFixed(2)} GiB — ${report.wastePct.toFixed(1)}% of reserved VRAM sits empty, capping roofline decode throughput at ${report.decodeThroughputTokPerSec.toFixed(0)} tok/s.`
      : mapMode === 'reserve-grow'
        ? `Growing contiguous slabs doubles capacity to ${report.sequencesFit} conversations, but as earlier replies finish and survivors relocate, ${report.externalFragTokens.toLocaleString('en-GB')} tokens of memory are stranded in external fragmentation holes (total waste ${report.wastePct.toFixed(1)}%).`
        : mapMode === 'paged'
          ? `PagedAttention (${blockSize}-token blocks) packs ${report.sequencesFit} of 64 conversations with only ${report.wastePct.toFixed(1)}% internal waste (< 1 block per sequence) — lifting decode throughput to ${report.decodeThroughputTokPerSec.toFixed(0)} tok/s. Notice smaller blocks waste less here; real engines pick ~16 tokens for kernel efficiency.`
          : `Prefix sharing stores the 1,024-token system prompt once (${report.sharedBlocks} physical blocks, ref count ${report.prefixRefCount}) instead of ${report.sequencesFit} times. All ${report.sequencesFit} of 64 conversations fit simultaneously, pushing decode throughput to ${report.decodeThroughputTokPerSec.toFixed(0)} tok/s (${report.decodeComputeUtilPct.toFixed(1)}% Tensor Core utilisation).`;

  return (
    <Widget
      title="Inside the KV pool: contiguous slabs, fragmentation, paging and prefix sharing"
      wide
      caption="Roofline model of Llama-3 8B FP16 on an RTX 4090 (7.54 GiB KV budget = 61,791 tokens, max context 8,192) facing 64 conversations that share a 1,024-token system prompt. Switch allocators to watch VRAM waste collapse and batch throughput climb; in paged modes, click a conversation to inspect its block table."
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 14 }}>
        <Segmented
          label="KV memory allocator"
          value={mapMode}
          onChange={(v) => setMapMode(v as MapModeOption)}
          options={[
            { value: 'reserve-max', label: 'Reserve max' },
            { value: 'reserve-grow', label: 'Grow slabs' },
            { value: 'paged', label: 'Paged' },
            { value: 'paged-prefix', label: 'Paged + prefix' },
          ]}
        />
        {(mapMode === 'paged' || mapMode === 'paged-prefix') && (
          <Segmented
            label="Block size"
            value={String(blockSize)}
            onChange={(v) => setBlockSize(Number(v) as 8 | 16 | 32)}
            options={[
              { value: '8', label: '8 tok' },
              { value: '16', label: '16 tok' },
              { value: '32', label: '32 tok' },
            ]}
          />
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        {/* Left: 16x16 Physical KV Pool Map */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
            Physical KV VRAM ({report.kvBudgetGiB.toFixed(2)} GiB · 256 slices ≈ {Math.round(report.kvCapTokens / 256)} tok each)
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(16, 1fr)',
              gap: 2,
              padding: 8,
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              borderRadius: 6,
            }}
            role="img"
            aria-label="Physical KV cache block map"
          >
            {report.cells.map((cell, idx) => {
              const isSelected = activeSeq && cell.owner === activeSeq.id;
              const isSharedHit = activeSeq && cell.kind === 'shared' && activeSeq.sharedPrefix > 0;
              let bg = 'var(--bg-elev)';
              let opacity = 0.45;
              let border = '1px solid var(--border)';
              if (cell.kind === 'shared') {
                bg = 'var(--accent)';
                opacity = 1;
                border = isSharedHit ? '1px solid var(--text)' : '1px solid var(--accent)';
              } else if (cell.kind === 'used') {
                bg = CLS_VAR(cell.owner);
                opacity = activeSeq ? (isSelected ? 1 : 0.42) : 0.85;
                border = isSelected ? '1px solid var(--text)' : '1px solid transparent';
              } else if (cell.kind === 'reserved-waste') {
                bg = CLS_VAR(cell.owner);
                opacity = 0.16;
                border = '1px dashed var(--border-strong)';
              } else if (cell.kind === 'frag') {
                bg = 'var(--c-bad)';
                opacity = 0.25;
                border = '1px dashed var(--c-bad)';
              }
              return (
                <div
                  key={idx}
                  onClick={() => cell.owner >= 0 && setSelectedSeqId(cell.owner)}
                  title={
                    cell.kind === 'shared'
                      ? `Shared system prompt (ref=${report.prefixRefCount})`
                      : cell.kind === 'used'
                        ? `Req #${cell.owner} (used)`
                        : cell.kind === 'reserved-waste'
                          ? `Req #${cell.owner} (reserved but empty)`
                          : cell.kind === 'frag'
                            ? 'External fragmentation hole'
                            : 'Free VRAM'
                  }
                  style={{
                    aspectRatio: '1 / 1',
                    borderRadius: 2,
                    background: bg,
                    opacity,
                    border,
                    cursor: cell.owner >= 0 ? 'pointer' : 'default',
                  }}
                />
              );
            })}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
            <span><span style={{ display: 'inline-block', width: 9, height: 9, background: 'var(--c-a)', marginRight: 4 }} />Used KV</span>
            {mapMode === 'paged-prefix' && (
              <span><span style={{ display: 'inline-block', width: 9, height: 9, background: 'var(--accent)', marginRight: 4 }} />Shared prefix (ref={report.prefixRefCount})</span>
            )}
            {mapMode !== 'paged' && mapMode !== 'paged-prefix' && (
              <span><span style={{ display: 'inline-block', width: 9, height: 9, background: 'var(--c-a)', opacity: 0.2, border: '1px dashed var(--border-strong)', marginRight: 4 }} />Reserved empty</span>
            )}
            {mapMode === 'reserve-grow' && (
              <span><span style={{ display: 'inline-block', width: 9, height: 9, background: 'var(--c-bad)', opacity: 0.35, marginRight: 4 }} />Frag hole</span>
            )}
            <span><span style={{ display: 'inline-block', width: 9, height: 9, background: 'var(--bg-elev)', border: '1px solid var(--border)', marginRight: 4 }} />Free</span>
          </div>
        </div>

        {/* Right: Metrics + Block Table / Slab Inspector */}
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10 }}>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Users resident</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-a)' }}>
                {report.sequencesFit} / {report.totalRequested}
              </div>
            </div>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Decode throughput</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-good)' }}>
                {report.decodeThroughputTokPerSec.toFixed(0)} tok/s
              </div>
            </div>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Memory waste</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: report.wastePct > 20 ? 'var(--c-bad)' : 'var(--c-c)' }}>
                {report.wastePct.toFixed(1)}%
              </div>
            </div>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Compute busy</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-text)' }}>
                {report.decodeComputeUtilPct.toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Waste breakdown */}
          <div style={{ fontSize: 12, padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', marginBottom: 10 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Token accounting</div>
            <div className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Logical tokens stored: {report.usedTokens.toLocaleString('en-GB')}<br />
              Physical slots held: {report.reservedTokens.toLocaleString('en-GB')}<br />
              {mapMode === 'paged' || mapMode === 'paged-prefix'
                ? `Internal tail waste: ${report.internalWasteTokens.toLocaleString('en-GB')} tok (< ${blockSize}/seq)`
                : `Reserved unused: ${report.reservedUnusedTokens.toLocaleString('en-GB')} tok · Frag holes: ${report.externalFragTokens.toLocaleString('en-GB')} tok`}
            </div>
          </div>

          {/* Sequence selector & block table */}
          {activeSeq && (
            <div style={{ fontSize: 12, padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8, flexWrap: 'wrap' }}>
                <strong>
                  {mapMode === 'paged' || mapMode === 'paged-prefix' ? `Block table: Req #${activeSeq.id}` : `Slab: Req #${activeSeq.id}`}
                </strong>
                <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {activeSeq.tokens} tokens
                </span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 6 }}>
                {report.fittedSeqs.slice(0, 12).map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className={`btn small ${activeSeq.id === s.id ? 'primary' : ''}`}
                    style={{ padding: '2px 6px', fontSize: 11 }}
                    onClick={() => setSelectedSeqId(s.id)}
                  >
                    #{s.id}
                  </button>
                ))}
              </div>
              {activeSeq.blocks ? (
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.55 }}>
                  {activeSeq.blocks.length} physical blocks ({blockSize} tok each):{' '}
                  {mapMode === 'paged-prefix' && report.sharedBlocks > 0
                    ? `logical 0…${report.sharedBlocks - 1} → shared [#${activeSeq.blocks[0]}…#${activeSeq.blocks[report.sharedBlocks - 1]}], private → [#${activeSeq.blocks.slice(report.sharedBlocks, report.sharedBlocks + 4).join(', #')}${activeSeq.blocks.length > report.sharedBlocks + 4 ? ', …' : ''}]`
                    : `logical 0…5 → [#${activeSeq.blocks.slice(0, 6).join(', #')}${activeSeq.blocks.length > 6 ? ', …' : ''}]`}
                </div>
              ) : activeSeq.slab ? (
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Contiguous range [{activeSeq.slab.start.toLocaleString('en-GB')}, {(activeSeq.slab.start + activeSeq.slab.size).toLocaleString('en-GB')}) — {activeSeq.tokens} used of {activeSeq.slab.size} reserved
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>{liveNote}</div>
    </Widget>
  );
}
