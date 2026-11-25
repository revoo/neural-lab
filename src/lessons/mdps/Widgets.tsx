// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { Widget } from '../../components/Widget';
import { StuckButton, useStuck } from '../../components/AnswerReveal';
import { Segmented, Slider } from '../../components/controls';
import {
  ACTION_ARROWS,
  ACTION_NAMES,
  POLICY_LAB_LEVELS,
  TWO_TREASURE_CLIFF,
  cellIndex,
  evaluateLabLevel,
  evaluatePolicyExact,
  inspectBellmanBackup,
  isTerminal,
  sampleEpisodes,
  solveValueIteration,
  type SampledEpisode,
  type Action,
  type GridMdp,
} from './mdp';

// ------------------------------------------------------------------ Shared SVG Grid Renderer

function GridSvg({
  grid,
  V,
  policy,
  greedyPath,
  selectedIdx,
  onCellClick,
  showValues = true,
  interactive = false,
  ariaLabel,
  overlay,
  flash,
}: {
  /** Extra drawing on top, given the centre of a cell (used for sampled episodes). */
  overlay?: (center: (idx: number) => [number, number]) => ReactNode;
  /** Cells to outline briefly, keyed by cell index, with the size of their value change. */
  flash?: Map<number, number>;
  grid: GridMdp;
  V: number[];
  policy: (Action | null)[];
  greedyPath?: number[];
  selectedIdx?: number;
  onCellClick?: (idx: number) => void;
  showValues?: boolean;
  interactive?: boolean;
  ariaLabel: string;
}) {
  const cellW = 78;
  const cellH = 68;
  const pad = 6;
  const W = grid.cols * cellW + pad * 2;
  const H = grid.rows * cellH + pad * 2;

  const pathSet = useMemo(() => new Set(greedyPath ?? []), [greedyPath]);
  const maxAbsV = useMemo(() => {
    let m = 1;
    for (let i = 0; i < V.length; i++) {
      const a = Math.abs(V[i]);
      if (a > m) m = a;
    }
    return m;
  }, [V]);

  // Intended-direction path polyline coordinates
  const pathPoints = useMemo(() => {
    if (!greedyPath || greedyPath.length < 2) return '';
    return greedyPath
      .map((idx) => {
        const r = Math.floor(idx / grid.cols);
        const c = idx % grid.cols;
        return `${pad + c * cellW + cellW / 2},${pad + r * cellH + cellH / 2}`;
      })
      .join(' ');
  }, [greedyPath, grid.cols]);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={ariaLabel}
      style={{ width: '100%', height: 'auto', display: 'block', maxWidth: W }}
    >
      {grid.cells.map((row, r) =>
        row.map((cell, c) => {
          const idx = cellIndex(grid, r, c);
          const x = pad + c * cellW;
          const y = pad + r * cellH;
          const isStart = r === grid.start[0] && c === grid.start[1];
          const isSelected = selectedIdx === idx;
          const onPath = pathSet.has(idx);
          const v = V[idx] ?? 0;
          const act = policy[idx];
          const clickable = Boolean(onCellClick && cell.kind === 'empty');

          // Value-based tint opacity for non-terminal cells
          const norm = Math.min(1, Math.abs(v) / Math.max(maxAbsV, 6));
          const tintFill =
            cell.kind === 'wall'
              ? 'var(--border-strong)'
              : cell.kind === 'goal'
                ? 'var(--c-good)'
                : cell.kind === 'near'
                  ? 'var(--c-a)'
                  : cell.kind === 'hazard'
                    ? 'var(--c-bad)'
                    : v >= 0
                      ? 'var(--c-good)'
                      : 'var(--c-bad)';
          const tintOpacity =
            cell.kind === 'wall'
              ? 0.55
              : cell.kind === 'goal' || cell.kind === 'near' || cell.kind === 'hazard'
                ? 0.22
                : 0.04 + 0.26 * norm;

          return (
            <g
              key={idx}
              onClick={clickable ? () => onCellClick?.(idx) : undefined}
              style={{ cursor: clickable ? 'pointer' : 'default' }}
            >
              <rect
                x={x + 2}
                y={y + 2}
                width={cellW - 4}
                height={cellH - 4}
                rx={7}
                fill="var(--bg)"
                stroke={
                  isSelected
                    ? 'var(--accent)'
                    : onPath && cell.kind === 'empty'
                      ? 'var(--accent)'
                      : cell.kind === 'goal'
                        ? 'var(--c-good)'
                        : cell.kind === 'hazard'
                          ? 'var(--c-bad)'
                          : cell.kind === 'near'
                            ? 'var(--c-a)'
                            : 'var(--border)'
                }
                strokeWidth={isSelected ? 2.4 : onPath && cell.kind === 'empty' ? 1.6 : 1.1}
                strokeDasharray={onPath && !isSelected && cell.kind === 'empty' ? '4 2' : undefined}
              />
              <rect
                x={x + 2}
                y={y + 2}
                width={cellW - 4}
                height={cellH - 4}
                rx={7}
                fill={tintFill}
                fillOpacity={tintOpacity}
                pointerEvents="none"
              />

              {/* Start badge */}
              {isStart && (
                <g pointerEvents="none">
                  <rect x={x + 6} y={y + 6} width={16} height={15} rx={4} fill="var(--accent)" />
                  <text
                    x={x + 14}
                    y={y + 17}
                    textAnchor="middle"
                    fill="var(--bg)"
                    fontSize={10}
                    fontWeight={700}
                    fontFamily="var(--font-mono)"
                  >
                    S
                  </text>
                </g>
              )}

              {/* Wall hatching label */}
              {cell.kind === 'wall' && (
                <text
                  x={x + cellW / 2}
                  y={y + cellH / 2 + 4}
                  textAnchor="middle"
                  fill="var(--text-faint)"
                  fontSize={12}
                  fontWeight={700}
                  fontFamily="var(--font-mono)"
                  pointerEvents="none"
                >
                  WALL
                </text>
              )}

              {/* Terminal cell content */}
              {isTerminal(cell) && (
                <g pointerEvents="none">
                  <text
                    x={x + cellW / 2}
                    y={y + 28}
                    textAnchor="middle"
                    fill={
                      cell.kind === 'goal'
                        ? 'var(--c-good)'
                        : cell.kind === 'near'
                          ? 'var(--c-a)'
                          : 'var(--c-bad)'
                    }
                    fontSize={16}
                    fontWeight={800}
                    fontFamily="var(--font-mono)"
                  >
                    {cell.reward > 0 ? `+${cell.reward}` : cell.reward}
                  </text>
                  <text
                    x={x + cellW / 2}
                    y={y + 46}
                    textAnchor="middle"
                    fill="var(--text-muted)"
                    fontSize={10.5}
                    fontWeight={600}
                    fontFamily="var(--font-ui)"
                  >
                    {cell.label ?? cell.kind}
                  </text>
                </g>
              )}

              {/* Empty cell: policy arrow + value */}
              {cell.kind === 'empty' && (
                <g pointerEvents="none">
                  {act !== null && act !== undefined && (
                    <text
                      x={x + cellW / 2}
                      y={showValues ? y + 32 : y + cellH / 2 + 8}
                      textAnchor="middle"
                      fill={onPath ? 'var(--accent-text)' : 'var(--text)'}
                      fontSize={interactive ? 22 : 20}
                      fontWeight={700}
                      fontFamily="var(--font-mono)"
                    >
                      {ACTION_ARROWS[act]}
                    </text>
                  )}
                  {showValues && (
                    <text
                      x={x + cellW / 2}
                      y={y + 53}
                      textAnchor="middle"
                      fill="var(--text-muted)"
                      fontSize={11}
                      fontFamily="var(--font-mono)"
                    >
                      {v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2)}
                    </text>
                  )}
                </g>
              )}
            </g>
          );
        }),
      )}

      {/* Subtle intended-direction polyline from Start */}
      {pathPoints && (
        <polyline
          points={pathPoints}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeOpacity={0.35}
          strokeLinecap="round"
          strokeLinejoin="round"
          pointerEvents="none"
        />
      )}
      {flash &&
        [...flash.entries()].map(([idx, d]) => {
          const r = Math.floor(idx / grid.cols);
          const c = idx % grid.cols;
          return (
            <rect key={`f${idx}`} className="value-flash" x={pad + c * cellW + 2} y={pad + r * cellH + 2} width={cellW - 4} height={cellH - 4} rx={7} fill="none" stroke="var(--c-warn)" strokeWidth={1.5 + Math.min(4, d * 3)} pointerEvents="none" />
          );
        })}
      {overlay?.((idx) => [pad + (idx % grid.cols) * cellW + cellW / 2, pad + Math.floor(idx / grid.cols) * cellH + cellH / 2])}
    </svg>
  );
}

// ------------------------------------------------------------------ 1. Discount Horizon & Risk Explorer

export function HorizonAndRiskExplorer() {
  const [gamma, setGamma] = useState(0.9);
  const [slipPct, setSlipPct] = useState(0);
  const slip = slipPct / 100;

  const sol = useMemo(() => solveValueIteration(TWO_TREASURE_CLIFF, gamma, slip), [gamma, slip]);
  const ev = useMemo(() => evaluatePolicyExact(TWO_TREASURE_CLIFF, sol.policy, gamma, slip), [sol.policy, gamma, slip]);

  const horizon = gamma >= 0.995 ? '200+' : (1 / (1 - gamma)).toFixed(1);

  const regimeNote = useMemo(() => {
    if (ev.probNear > 0.5) {
      return `Myopic regime (γ = ${gamma.toFixed(2)}): future rewards shrink too fast to justify the 5-step walk to +10. The optimal policy grabs the nearby +2 Gem in 2 steps (V*(S) = ${ev.startValue.toFixed(2)}).`;
    }
    if (slip === 0) {
      return `Patient & deterministic (γ = ${gamma.toFixed(2)}, 0% slip): the agent marches straight along the Row 2 cliff edge in 5 steps to collect the +10 Vault with 100% reliability (V*(S) = ${ev.startValue.toFixed(2)}).`;
    }
    if (slip <= 0.25) {
      return `Stochastic detour (${slipPct}% slip): hugging the Row 2 cliff now risks slipping into the −10 pits, so exact Value Iteration detours inland onto Row 1 — reaching the +10 Vault ${(ev.probGoal * 100).toFixed(1)}% of the time with only ${(ev.probHazard * 100).toFixed(1)}% pit risk (V*(S) = ${ev.startValue.toFixed(2)}).`;
    }
    return `High ice risk (${slipPct}% slip): every move near the cliff is dangerous. Value Iteration routes well inland, trading extra step costs (and even an occasional slip into the +2 Gem) to keep pit falls at ${(ev.probHazard * 100).toFixed(1)}% (V*(S) = ${ev.startValue.toFixed(2)}).`;
  }, [ev, gamma, slip, slipPct]);

  return (
    <Widget
      title="Discount horizon γ & slippery transitions: watch the optimal policy flip"
      wide
      caption={
        <>
          Exact Value Iteration on a 6×4 grid (step cost −0.10). Drag <strong>γ</strong> down past <strong>0.60</strong> to watch the optimal policy flip from the distant <strong>+10 Vault</strong> to the nearby <strong>+2 Gem</strong>. Then return to <strong>γ = 0.90</strong> and turn <strong>slip</strong> up to <strong>20%</strong>: the arrows detour inland away from the −10 pits.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 12, alignItems: 'flex-end' }}>
        <div style={{ flex: '1 1 200px' }}>
          <Slider
            label="Discount factor γ"
            min={0.1}
            max={0.99}
            step={0.01}
            value={gamma}
            onChange={setGamma}
            format={(v) => v.toFixed(2)}
            color="c-a"
          />
        </div>
        <div style={{ flex: '1 1 200px' }}>
          <Slider
            label="Slip probability (half left, half right)"
            min={0}
            max={40}
            step={2}
            value={slipPct}
            onChange={setSlipPct}
            format={(v) => `${v}%`}
            color="c-b"
          />
        </div>
      </div>

      <div className="btn-row" style={{ marginBottom: 12 }}>
        <button type="button" className="btn" onClick={() => { setGamma(0.5); setSlipPct(0); }}>
          Myopic (γ=0.50, 0%)
        </button>
        <button type="button" className="btn primary" onClick={() => { setGamma(0.9); setSlipPct(0); }}>
          Cliff sprint (γ=0.90, 0%)
        </button>
        <button type="button" className="btn" onClick={() => { setGamma(0.9); setSlipPct(20); }}>
          Slippery detour (γ=0.90, 20%)
        </button>
        <button type="button" className="btn" onClick={() => { setGamma(0.9); setSlipPct(40); }}>
          Extreme ice (γ=0.90, 40%)
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, alignItems: 'start', marginBottom: 12 }}>
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 8 }}>
          <GridSvg
            grid={TWO_TREASURE_CLIFF}
            V={sol.V}
            policy={sol.policy}
            greedyPath={ev.greedyPath}
            ariaLabel="Two-Treasure Cliff optimal value and policy grid"
          />
        </div>

        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Optimal value V*(S)</div>
              <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: 'var(--c-good)' }}>
                {ev.startValue >= 0 ? `+${ev.startValue.toFixed(2)}` : ev.startValue.toFixed(2)}
              </div>
            </div>
            <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Effective horizon 1/(1−γ)</div>
              <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: 'var(--c-a)' }}>
                {horizon} steps
              </div>
            </div>
          </div>

          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>Exact episode outcomes from S</div>
          {[
            { label: 'Reach +10 Vault', pct: ev.probGoal * 100, color: 'var(--c-good)' },
            { label: 'Take +2 Near Gem', pct: ev.probNear * 100, color: 'var(--c-a)' },
            { label: 'Fall into −10 Pit', pct: ev.probHazard * 100, color: 'var(--c-bad)' },
          ].map((row) => (
            <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '115px 1fr 44px', gap: 8, alignItems: 'center', fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>{row.label}</span>
              <div style={{ height: 10, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ width: `${Math.min(100, row.pct)}%`, height: '100%', background: row.color, transition: 'width 150ms ease' }} />
              </div>
              <span className="mono" style={{ textAlign: 'right', fontSize: 11.5 }}>{row.pct.toFixed(1)}%</span>
            </div>
          ))}

          <div style={{ fontSize: 11.5, color: 'var(--text-faint)', borderTop: '1px solid var(--border)', paddingTop: 8 }}>
            Solved in <span className="mono">{sol.iterations}</span> Bellman sweeps to tolerance 10⁻¹⁰. Dashed outline shows the intended path from S.
          </div>
        </div>
      </div>

      <div className="live-note">{regimeNote}</div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Step-by-Step Bellman Backup Inspector

export function BellmanStepper() {
  const [slipMode, setSlipMode] = useState<'det' | 'slip'>('slip');
  const gamma = 0.9;
  const slip = slipMode === 'slip' ? 0.2 : 0.0;

  const sol = useMemo(() => solveValueIteration(TWO_TREASURE_CLIFF, gamma, slip, 1e-10, 80, true), [gamma, slip]);
  const [snapIdx, setSnapIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(() => cellIndex(TWO_TREASURE_CLIFF, 2, 4)); // adjacent to Vault!

  // Clamp snapIdx when mode changes
  const safeSnapIdx = Math.min(snapIdx, sol.snapshots.length - 1);
  const curSnap = sol.snapshots[safeSnapIdx];
  const prevSnap = safeSnapIdx > 0 ? sol.snapshots[safeSnapIdx - 1] : null;

  useEffect(() => {
    if (!playing) return;
    if (safeSnapIdx >= sol.snapshots.length - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setSnapIdx((i) => Math.min(sol.snapshots.length - 1, i + 1)), 350);
    return () => window.clearTimeout(id);
  }, [playing, safeSnapIdx, sol.snapshots.length]);

  // Show the Bellman backup that produced curSnap (using prevSnap.V) OR the next backup from curSnap.V if at k=0
  const baseV = prevSnap ? prevSnap.V : curSnap.V;
  const baseSweep = prevSnap ? prevSnap.sweep : 0;
  const backup = useMemo(
    () => inspectBellmanBackup(TWO_TREASURE_CLIFF, baseV, selectedIdx, gamma, slip),
    [baseV, selectedIdx, gamma, slip],
  );
  const winningBackup = backup.actions[backup.bestAction];

  return (
    <Widget
      title="Watch value ripple backwards: step-by-step Value Iteration"
      wide
      caption={
        <>
          Press <strong>Step +1</strong> to apply one Bellman sweep across all cells simultaneously. Click any non-terminal square (default: <code>(2,4)</code> right next to the +10 Vault) to inspect the exact 4-action backup formula that produced its new value.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div className="btn-row">
          <button type="button" className="btn" onClick={() => { setPlaying(false); setSnapIdx(0); }}>
            Reset (k=0)
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={safeSnapIdx >= sol.snapshots.length - 1}
            onClick={() => { setPlaying(false); setSnapIdx((i) => Math.min(sol.snapshots.length - 1, i + 1)); }}
          >
            Step +1
          </button>
          <button
            type="button"
            className="btn"
            disabled={safeSnapIdx >= sol.snapshots.length - 1}
            onClick={() => { setPlaying(false); setSnapIdx((i) => Math.min(sol.snapshots.length - 1, i + 5)); }}
          >
            +5
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              if (safeSnapIdx >= sol.snapshots.length - 1) setSnapIdx(0);
              setPlaying((p) => !p);
            }}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => { setPlaying(false); setSnapIdx(sol.snapshots.length - 1); }}
          >
            Converge
          </button>
        </div>
        <Segmented
          value={slipMode}
          onChange={(v) => { setSlipMode(v); setPlaying(false); setSnapIdx(0); }}
          options={[
            { value: 'det', label: '0% slip' },
            { value: 'slip', label: '20% slip' },
          ]}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, alignItems: 'start', marginBottom: 12 }}>
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6, paddingInline: 4 }}>
            <span>
              Sweep <strong className="mono">k = {curSnap.sweep}</strong>
            </span>
            <span className="mono" style={{ color: 'var(--text-muted)' }}>
              max |ΔV| = {curSnap.sweep === 0 ? '—' : curSnap.maxDelta < 1e-4 ? '<0.0001' : curSnap.maxDelta.toFixed(4)}
            </span>
          </div>
          <GridSvg
            grid={TWO_TREASURE_CLIFF}
            V={curSnap.V}
            policy={curSnap.policy}
            selectedIdx={selectedIdx}
            onCellClick={setSelectedIdx}
            ariaLabel="Bellman Value Iteration step-through grid"
          />
        </div>

        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 12.5, fontWeight: 700, marginBottom: 8 }}>
            Bellman backup at cell <span className="mono">({backup.r},{backup.c})</span>{' '}
            <span style={{ fontWeight: 400, color: 'var(--text-muted)' }}>
              (using V<sub>{baseSweep}</sub>)
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6, marginBottom: 10 }}>
            {backup.actions.map((ab) => {
              const isBest = ab.action === backup.bestAction;
              return (
                <div
                  key={ab.action}
                  style={{
                    background: isBest ? 'color-mix(in srgb, var(--c-good) 14%, var(--bg))' : 'var(--bg)',
                    border: `1px solid ${isBest ? 'var(--c-good)' : 'var(--border)'}`,
                    borderRadius: 6,
                    padding: '6px 8px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: 12,
                  }}
                >
                  <span style={{ fontWeight: isBest ? 700 : 500 }}>
                    {ACTION_ARROWS[ab.action]} {ACTION_NAMES[ab.action]}
                  </span>
                  <span className="mono" style={{ fontWeight: 700, color: isBest ? 'var(--c-good)' : 'var(--text-muted)' }}>
                    {ab.q >= 0 ? `+${ab.q.toFixed(2)}` : ab.q.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>

          <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 6 }}>
            Winning action <strong>{ACTION_ARROWS[winningBackup.action]} {ACTION_NAMES[winningBackup.action]}</strong> breakdown:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11.5 }} className="mono">
            {winningBackup.outcomes.map((o) => (
              <div
                key={o.nextIdx}
                style={{
                  background: 'var(--bg)',
                  border: '1px solid var(--border)',
                  borderRadius: 5,
                  padding: '4px 7px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: 6,
                }}
              >
                <span style={{ color: 'var(--text-muted)' }}>
                  {(o.prob * 100).toFixed(0)}% → {o.label}:{' '}
                  {o.terminal ? `(${o.reward > 0 ? `+${o.reward}` : o.reward})` : `(${o.reward} + ${gamma}×${o.futureV.toFixed(2)})`}
                </span>
                <span style={{ fontWeight: 700, color: o.contribution >= 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {o.contribution >= 0 ? `+${o.contribution.toFixed(2)}` : o.contribution.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <div className="mono" style={{ marginTop: 8, fontSize: 12, textAlign: 'right', fontWeight: 700, color: 'var(--accent-text)' }}>
            max Q = V<sub>{baseSweep + 1}</sub>({backup.r},{backup.c}) = {backup.bestQ >= 0 ? `+${backup.bestQ.toFixed(2)}` : backup.bestQ.toFixed(2)}
          </div>
        </div>
      </div>

      <div className="live-note">
        {curSnap.sweep === 0
          ? 'Sweep 0: every non-terminal square starts with V₀ = 0.00. Press Step +1 — only cells immediately adjacent to a terminal square (+10, +2, or −10) feel any reward on the first sweep.'
          : curSnap.maxDelta < 1e-4
            ? `Converged at sweep k = ${curSnap.sweep}! Every square’s value now satisfies V*(s) = maxₐ Σ P(s′|s,a)[R + γV*(s′)] to within 0.0001.`
            : `Sweep k = ${curSnap.sweep}: the reward wave has travelled ${curSnap.sweep} step${curSnap.sweep === 1 ? '' : 's'} outward from the terminals. Notice how Start (2,0) currently has V(S) = ${curSnap.V[cellIndex(TWO_TREASURE_CLIFF, 2, 0)].toFixed(2)}.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Lab: Wire the Policy

const LAB_STORAGE_KEY = 'nl-mdp-lab-v1';

type SavedLabState = {
  policies: Record<string, (Action | null)[]>;
};

function loadLabState(): SavedLabState {
  try {
    const raw = localStorage.getItem(LAB_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as SavedLabState;
      if (parsed && typeof parsed.policies === 'object') return parsed;
    }
  } catch {
    // ignore storage errors
  }
  return { policies: {} };
}

export function PolicyArchitectLab() {
  const [levelIdx, setLevelIdx] = useState(0);
  const [saved, setSaved] = useState<SavedLabState>(() => loadLabState());
  const [showHint, setShowHint] = useState(false);
  const [showOptimal, setShowOptimal] = useState(false);

  const level = POLICY_LAB_LEVELS[levelIdx];
  const currentPolicy = saved.policies[level.id] ?? level.initialPolicy;

  const evalsByLevel = useMemo(
    () =>
      POLICY_LAB_LEVELS.map((lvl) => {
        const pol = saved.policies[lvl.id] ?? lvl.initialPolicy;
        return evaluateLabLevel(lvl, pol);
      }),
    [saved.policies],
  );
  const curEval = evalsByLevel[levelIdx];

  const updatePolicy = (nextPol: (Action | null)[]) => {
    const next: SavedLabState = {
      policies: { ...saved.policies, [level.id]: nextPol },
    };
    setSaved(next);
    try {
      localStorage.setItem(LAB_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };

  const log = useMachineLog();
  const [flash, setFlash] = useState<Map<number, number> | undefined>(undefined);
  const flashTimer = useRef<number | null>(null);
  const handleCellClick = (idx: number) => {
    const nextPol = currentPolicy.slice();
    const cur = nextPol[idx];
    nextPol[idx] = cur === null || cur === undefined ? 0 : (((cur + 1) % 4) as Action);
    updatePolicy(nextPol);
    // Which squares' values moved, and by how much: the ripple of one arrow.
    const before = curEval.evalResult.V;
    const after = evaluatePolicyExact(level.grid, nextPol, level.gamma, level.slip);
    const changed = new Map<number, number>();
    after.V.forEach((v, i) => {
      const d = Math.abs(v - before[i]);
      if (d > 0.005) changed.set(i, d);
    });
    setFlash(changed);
    if (flashTimer.current !== null) window.clearTimeout(flashTimer.current);
    flashTimer.current = window.setTimeout(() => setFlash(undefined), 1100);
    const [r, c] = [Math.floor(idx / level.grid.cols), idx % level.grid.cols];
    log.push(
      `(${r},${c}) → ${ACTION_ARROWS[nextPol[idx] as Action]} · V(S) ${curEval.evalResult.startValue.toFixed(3)} → ${after.startValue.toFixed(3)} · ${changed.size} square${changed.size === 1 ? '' : 's'} changed value`,
      after.startValue > curEval.evalResult.startValue + 1e-9 ? 'good' : after.startValue < curEval.evalResult.startValue - 1e-9 ? 'bad' : 'info',
    );
  };

  // "Run 20 episodes": sample real slips under the current policy and replay them together.
  const [episodes, setEpisodes] = useState<{ key: string; eps: SampledEpisode[]; run: number } | null>(null);
  const runCount = useRef(0);
  const policyKey = `${level.id}|${currentPolicy.join(',')}`;
  const liveEps = episodes && episodes.key === policyKey && !showOptimal ? episodes : null;
  const maxLen = liveEps ? Math.max(...liveEps.eps.map((e) => e.cells.length - 1)) : 0;
  const replay = useReplay(maxLen, 230, liveEps?.run);
  const runEpisodes = () => {
    runCount.current++;
    const eps = sampleEpisodes(level.grid, currentPolicy, level.slip, 20, 1000 + runCount.current);
    setEpisodes({ key: policyKey, eps, run: runCount.current });
    const count = (k: SampledEpisode['outcome']) => eps.filter((e) => e.outcome === k).length;
    const slips = eps.reduce((a, e) => a + e.slipped.filter(Boolean).length, 0);
    log.push(
      `run ${runCount.current}: 20 episodes · vault ${count('goal')}, gem ${count('near')}, pit ${count('hazard')}, still walking ${count('timeout')} · ${slips} slips in all · exact odds: vault ${(100 * curEval.evalResult.probGoal).toFixed(1)}%, pit ${(100 * curEval.evalResult.probHazard).toFixed(1)}%`,
      count('hazard') > 0 ? 'bad' : 'good',
    );
  };

  const [stuck, markStuck] = useStuck(`wire-policy:${level.id}`);

  const displayPolicy = showOptimal ? curEval.optimal.policy : currentPolicy;
  const displayV = showOptimal ? curEval.optimal.V : curEval.evalResult.V;
  const displayEv = useMemo(
    () =>
      showOptimal
        ? evaluatePolicyExact(level.grid, curEval.optimal.policy, level.gamma, level.slip)
        : curEval.evalResult,
    [showOptimal, level.grid, level.gamma, level.slip, curEval.optimal.policy, curEval.evalResult],
  );

  return (
    <Widget
      title="Lab: Wire the Policy"
      wide
      caption={
        <>
          Click any non-terminal square to rotate its policy arrow (<code>↑ → ↓ ←</code>). Every click runs exact linear-system policy evaluation <code>(I − γP<sup>π</sup>)V<sup>π</sup> = R<sup>π</sup></code> and compares your expected return against the exact Bellman optimum <code>V*(S)</code>. Progress saves automatically.
        </>
      }
    >
      {/* Level selector */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {POLICY_LAB_LEVELS.map((lvl, i) => {
          const ev = evalsByLevel[i];
          const active = i === levelIdx;
          return (
            <button
              key={lvl.id}
              type="button"
              className={`btn${active ? ' primary' : ''}`}
              onClick={() => {
                setLevelIdx(i);
                setShowHint(false);
                setShowOptimal(false);
              }}
            >
              {ev.isOptimal ? '★ ' : ev.passed ? '✓ ' : ''}
              {lvl.title}
            </button>
          );
        })}
      </div>

      <p style={{ margin: '0 0 12px', fontSize: 13.5, color: 'var(--text-muted)' }}>{level.brief}</p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, alignItems: 'start', marginBottom: 12 }}>
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 6, paddingInline: 4 }}>
            <span>{showOptimal ? 'Viewing: Exact Bellman Optimum π*' : 'Click any open cell to rotate arrow'}</span>
            <span className="mono">γ={level.gamma}, slip={Math.round(level.slip * 100)}%</span>
          </div>
          <GridSvg
            grid={level.grid}
            V={displayV}
            policy={displayPolicy}
            greedyPath={liveEps ? undefined : displayEv.greedyPath}
            flash={showOptimal ? undefined : flash}
            overlay={
              liveEps
                ? (center) =>
                    liveEps.eps.map((e, k) => {
                      const upto = Math.min(replay.i, e.cells.length - 1);
                      const jitter = ((k % 5) - 2) * 3;
                      const pts = e.cells.slice(0, upto + 1).map((idx) => center(idx));
                      const col = e.outcome === 'goal' ? 'var(--c-good)' : e.outcome === 'hazard' ? 'var(--c-bad)' : e.outcome === 'near' ? 'var(--c-a)' : 'var(--text-faint)';
                      const [hx, hy] = pts[pts.length - 1];
                      return (
                        <g key={k} pointerEvents="none">
                          <polyline points={pts.map(([x, y]) => `${x + jitter},${y + jitter}`).join(' ')} fill="none" stroke={col} strokeWidth={1.4} opacity={0.55} />
                          {e.slipped.slice(0, upto).map((sl, t) => (sl ? <circle key={t} cx={pts[t + 1][0] + jitter} cy={pts[t + 1][1] + jitter} r={2.6} fill="var(--c-warn)" /> : null))}
                          <circle cx={hx + jitter} cy={hy + jitter} r={4} fill={upto >= e.cells.length - 1 ? col : 'var(--text)'} stroke="var(--bg)" strokeWidth={1} />
                        </g>
                      );
                    })
                : undefined
            }
            onCellClick={showOptimal ? undefined : handleCellClick}
            interactive={!showOptimal}
            ariaLabel={`Policy lab grid for ${level.title}`}
          />
          <div className="btn-row" style={{ marginTop: 6 }}>
            <button type="button" className="btn small primary" onClick={runEpisodes} disabled={showOptimal}>
              Run 20 episodes
            </button>
            {liveEps && (
              <button type="button" className="btn small" onClick={() => setEpisodes(null)}>
                Clear runs
              </button>
            )}
          </div>
          {liveEps && (
            <>
              <ReplayControls r={replay} />
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {(['goal', 'near', 'hazard', 'timeout'] as const)
                  .map((k) => [k, liveEps.eps.filter((e) => e.outcome === k && e.cells.length - 1 <= replay.i).length] as const)
                  .filter(([, v]) => v > 0)
                  .map(([k, v]) => `${v} ${k === 'goal' ? 'at the vault' : k === 'near' ? 'at the gem' : k === 'hazard' ? 'in the pit' : 'still walking after 40 steps'}`)
                  .join(' · ') || 'walking…'}
                . Orange dots are slips: the floor sent the robot sideways.
              </div>
            </>
          )}
        </div>

        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
            <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '7px 8px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                Your V<sup>π</sup>(S)
              </div>
              <div
                className="mono"
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: curEval.passed ? 'var(--c-good)' : 'var(--c-b)',
                }}
              >
                {curEval.evalResult.startValue >= 0
                  ? `+${curEval.evalResult.startValue.toFixed(3)}`
                  : curEval.evalResult.startValue.toFixed(3)}
              </div>
            </div>
            <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '7px 8px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>Goal target</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>
                +{level.targetReturn.toFixed(2)}
              </div>
            </div>
            <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '7px 8px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>Bellman V*(S)</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-text)' }}>
                +{curEval.optStartValue.toFixed(3)}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            <span
              className="mono"
              style={{
                fontSize: 11.5,
                padding: '3px 8px',
                borderRadius: 999,
                border: `1px solid ${curEval.passed ? 'var(--c-good)' : 'var(--border)'}`,
                color: curEval.passed ? 'var(--c-good)' : 'var(--text-muted)',
                background: curEval.passed ? 'color-mix(in srgb, var(--c-good) 12%, var(--bg))' : 'var(--bg)',
              }}
            >
              {curEval.passed ? '✓ Target reached' : '○ Target not yet met'}
            </span>
            <span
              className="mono"
              style={{
                fontSize: 11.5,
                padding: '3px 8px',
                borderRadius: 999,
                border: `1px solid ${curEval.isOptimal ? 'var(--accent)' : 'var(--border)'}`,
                color: curEval.isOptimal ? 'var(--accent-text)' : 'var(--text-muted)',
                background: curEval.isOptimal ? 'color-mix(in srgb, var(--accent) 14%, var(--bg))' : 'var(--bg)',
              }}
            >
              {curEval.isOptimal ? '★ Bellman optimum matched' : '☆ Optional: match V*(S)'}
            </span>
          </div>

          <div style={{ fontSize: 12, fontWeight: 700 }}>Outcome probabilities ({showOptimal ? 'Bellman π*' : 'your policy'})</div>
          {[
            { label: 'Vault (+10)', pct: displayEv.probGoal * 100, color: 'var(--c-good)' },
            { label: 'Pit hazard', pct: displayEv.probHazard * 100, color: 'var(--c-bad)' },
            ...(displayEv.probNear > 0.001 ? [{ label: 'Near Gem (+2)', pct: displayEv.probNear * 100, color: 'var(--c-a)' }] : []),
            ...(displayEv.probLoop > 0.001 ? [{ label: 'Stuck in loop', pct: displayEv.probLoop * 100, color: 'var(--c-warn)' }] : []),
          ].map((row) => (
            <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '105px 1fr 44px', gap: 8, alignItems: 'center', fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>{row.label}</span>
              <div style={{ height: 9, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ width: `${Math.min(100, row.pct)}%`, height: '100%', background: row.color, transition: 'width 120ms ease' }} />
              </div>
              <span className="mono" style={{ textAlign: 'right', fontSize: 11.5 }}>{row.pct.toFixed(1)}%</span>
            </div>
          ))}

          <div className="btn-row" style={{ marginTop: 2 }}>
            <button type="button" className="btn" onClick={() => { setShowOptimal(false); updatePolicy(level.initialPolicy); }}>
              Reset level
            </button>
            <button type="button" className="btn" onClick={() => setShowHint((h) => !h)}>
              {showHint ? 'Hide hint' : 'Hint'}
            </button>
            {curEval.passed || stuck ? (
              <button type="button" className="btn" onClick={() => setShowOptimal((s) => !s)}>
                {showOptimal ? 'Back to mine' : 'Inspect π*'}
              </button>
            ) : (
              <StuckButton
                stuckKey={`wire-policy:${level.id}`}
                label="the optimal policy π*"
                onStuck={() => {
                  markStuck();
                  setShowOptimal(true);
                }}
              />
            )}
          </div>

          {showHint && (
            <div style={{ fontSize: 12.5, background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px', color: 'var(--text-muted)' }}>
              {level.hint}
            </div>
          )}
        </div>
      </div>

      <div
        className="live-note"
        style={{
          borderColor: curEval.isOptimal ? 'var(--c-good)' : curEval.passed ? 'var(--c-a)' : undefined,
        }}
      >
        {curEval.isOptimal
          ? `Exact Bellman optimum matched! Your hand-wired policy achieves V(S) = +${curEval.evalResult.startValue.toFixed(3)} — no stationary policy on this grid can beat it.`
          : curEval.passed
            ? `Goal cleared (V(S) = +${curEval.evalResult.startValue.toFixed(3)} ≥ +${level.targetReturn.toFixed(2)})! You are within ${(curEval.optStartValue - curEval.evalResult.startValue).toFixed(3)} of the exact Bellman optimum V*(S) = +${curEval.optStartValue.toFixed(3)} — see if you can tighten a few recovery arrows.`
            : curEval.evalResult.probLoop > 0.5
              ? `Heads up: your arrows currently trap the agent in a cycle (${(curEval.evalResult.probLoop * 100).toFixed(0)}% loop), paying step cost forever. Break the loop to route toward the +10 Vault.`
              : `Current expected return from S is ${curEval.evalResult.startValue.toFixed(3)} (target ≥ +${level.targetReturn.toFixed(2)}). Notice that in stochastic grids, even squares off your main path matter because a slip can drop you there!`}
      </div>
      <MachineLog lines={log.lines} onClear={log.clear} empty="Rotate an arrow or run episodes: each change in value, and each batch of runs, is logged here." />
    </Widget>
  );
}
