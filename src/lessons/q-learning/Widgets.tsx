// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Segmented, Slider } from '../../components/controls';
import { createRng } from '../../lib/rng';
import {
  ACTION_ARROWS,
  ACTION_NAMES,
  BANDIT_MEANS,
  BANDIT_OPTIMAL_ARM,
  CLIFF_COLS,
  CLIFF_GOAL,
  CLIFF_ROWS,
  CLIFF_START,
  Q_MAZE_GRID,
  QLAB_ALGOS,
  QLAB_ALPHAS,
  QLAB_BRIEFS,
  QLAB_EPS,
  QLAB_QINITS,
  createQSession,
  evaluateGreedyMazeReturn,
  evaluateQLabConfig,
  isCliffCell,
  runCliffWalkingRace,
  runFairBanditRace,
  runQEpisodes,
  solveQLabBriefExhaustive,
  stepQSession,
  type Action,
  type LabEpsSchedule,
  type QLabConfig,
  type TdAlgo,
} from './qlearn';

// ------------------------------------------------------------------ 1. Bandit Explorer & Fair Race

const BANDIT_COLORS: Record<string, string> = {
  greedy: 'var(--c-bad)',
  'eps-greedy': 'var(--c-b)',
  'eps-decay': 'var(--c-a)',
  optimistic: 'var(--accent)',
  ucb1: 'var(--c-good)',
};

export function BanditExplorer() {
  const [tab, setTab] = useState<'play' | 'race'>('race');

  // Play mode state (25 pulls)
  const [playSeed, setPlaySeed] = useState(11);
  const [counts, setCounts] = useState<number[]>([0, 0, 0, 0, 0]);
  const [sums, setSums] = useState<number[]>([0, 0, 0, 0, 0]);
  const [lastPull, setLastPull] = useState<{ arm: number; reward: number } | null>(null);
  const [revealed, setRevealed] = useState(false);

  const totalPulls = counts.reduce((a, b) => a + b, 0);
  const totalReward = sums.reduce((a, b) => a + b, 0);
  const maxPulls = 25;

  const pullArm = (arm: number) => {
    if (totalPulls >= maxPulls) return;
    const rng = createRng(playSeed * 997 + totalPulls * 53 + arm * 17 + 3);
    const reward = rng.normal(BANDIT_MEANS[arm], 1.0);
    setCounts((prev) => prev.map((c, i) => (i === arm ? c + 1 : c)));
    setSums((prev) => prev.map((s, i) => (i === arm ? s + reward : s)));
    setLastPull({ arm, reward });
    if (totalPulls + 1 >= maxPulls) setRevealed(true);
  };

  const resetPlay = () => {
    setPlaySeed((s) => s + 1);
    setCounts([0, 0, 0, 0, 0]);
    setSums([0, 0, 0, 0, 0]);
    setLastPull(null);
    setRevealed(false);
  };

  // Fair 40-seed race
  const race = useMemo(() => runFairBanditRace(200, 40), []);

  // SVG Cumulative Regret chart
  const W = 440;
  const H = 205;
  const pad = { l: 44, r: 16, t: 16, b: 32 };
  const maxRegret = Math.max(...race.map((r) => r.finalRegret), 100);
  const sx = (t: number) => pad.l + (t / 199) * (W - pad.l - pad.r);
  const sy = (val: number) => H - pad.b - (Math.min(maxRegret, Math.max(0, val)) / maxRegret) * (H - pad.t - pad.b);

  return (
    <Widget
      title="Exploration vs exploitation: 5-armed bandit (play & fair 40-seed race)"
      wide
      caption={
        <>
          Five slot machines with noisy Gaussian payouts (σ = 1.0) and hidden means. In the <strong>Fair 40-seed race</strong>, every algorithm uses its own grid-searched best parameter over 200 pulls: <strong>Pure greedy</strong> gets trapped on mediocre arms (regret 94.9), while <strong>UCB1</strong> cuts regret nearly 5× (20.2). Switch to <strong>Play 25 pulls</strong> to feel the dilemma yourself!
        </>
      }
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { value: 'race', label: 'Fair 40-seed race (200 pulls)' },
            { value: 'play', label: 'Play 25 pulls yourself' },
          ]}
        />
        {tab === 'play' && (
          <div className="btn-row">
            <button type="button" className="btn" onClick={() => setRevealed((r) => !r)}>
              {revealed ? 'Hide true means' : 'Reveal true means'}
            </button>
            <button type="button" className="btn" onClick={resetPlay}>
              New 25 pulls
            </button>
          </div>
        )}
      </div>

      {tab === 'race' ? (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(265px, 1fr))', gap: 12, marginBottom: 12 }}>
            <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>Cumulative regret over 200 pulls (lower is better)</div>
              <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="Cumulative regret chart for five bandit algorithms">
                {[0, 25, 50, 75, 100].map((v) => (
                  <g key={v}>
                    <line x1={pad.l} y1={sy(v)} x2={W - pad.r} y2={sy(v)} stroke="var(--border)" strokeDasharray="3 3" />
                    <text x={pad.l - 6} y={sy(v) + 3.5} textAnchor="end" fill="var(--text-faint)" fontSize={9.5} fontFamily="var(--font-mono)">
                      {v}
                    </text>
                  </g>
                ))}
                {[0, 50, 100, 150, 200].map((t) => (
                  <text key={t} x={sx(Math.min(199, t))} y={H - 12} textAnchor="middle" fill="var(--text-faint)" fontSize={9.5} fontFamily="var(--font-mono)">
                    {t}
                  </text>
                ))}
                <text x={(pad.l + W - pad.r) / 2} y={H - 1} textAnchor="middle" fill="var(--text-muted)" fontSize={9.5}>
                  Pull number t →
                </text>

                {race.map((algo) => {
                  const pts = algo.cumRegretCurve
                    .filter((_, i) => i % 2 === 0 || i === 199)
                    .map((val, idx, arr) => {
                      const t = idx === arr.length - 1 ? 199 : idx * 2;
                      return `${sx(t).toFixed(1)},${sy(val).toFixed(1)}`;
                    })
                    .join(' ');
                  return (
                    <polyline
                      key={algo.id}
                      points={pts}
                      fill="none"
                      stroke={BANDIT_COLORS[algo.id]}
                      strokeWidth={algo.id === 'ucb1' || algo.id === 'greedy' ? 2.5 : 1.9}
                      strokeDasharray={algo.id === 'eps-greedy' ? '5 3' : undefined}
                    />
                  );
                })}
              </svg>
            </div>

            <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>40-seed average (each at its searched best)</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {race.map((algo) => (
                  <div
                    key={algo.id}
                    style={{
                      background: 'var(--bg)',
                      border: '1px solid var(--border)',
                      borderLeft: `3px solid ${BANDIT_COLORS[algo.id]}`,
                      borderRadius: 6,
                      padding: '6px 9px',
                      display: 'grid',
                      gridTemplateColumns: '1fr auto auto',
                      gap: 8,
                      alignItems: 'center',
                      fontSize: 12,
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 700 }}>{algo.label}</div>
                      <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-faint)' }}>{algo.paramLabel}</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className="mono" style={{ fontWeight: 700, color: BANDIT_COLORS[algo.id] }}>
                        {algo.finalRegret.toFixed(1)}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-faint)' }}>regret</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className="mono" style={{ fontWeight: 600 }}>{algo.optimalPullPct.toFixed(0)}%</div>
                      <div style={{ fontSize: 10, color: 'var(--text-faint)' }}>best arm</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="live-note">
            <strong>Pure greedy</strong> suffers linear regret (94.9) because it picks the best arm only <strong>29.8%</strong> of the time — whichever machine gets lucky on pull 1 locks it in forever. <strong>UCB1</strong> directs exploration toward under-tested arms via its confidence bonus, finding the best arm <strong>82.5%</strong> of the time with only <strong>20.2</strong> total regret.
          </div>
        </>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 8, marginBottom: 12 }}>
            {BANDIT_MEANS.map((trueMean, i) => {
              const est = counts[i] > 0 ? sums[i] / counts[i] : 0;
              const isBest = i === BANDIT_OPTIMAL_ARM;
              const isLast = lastPull?.arm === i;
              return (
                <button
                  key={i}
                  type="button"
                  disabled={totalPulls >= maxPulls}
                  onClick={() => pullArm(i)}
                  style={{
                    background: isLast ? 'color-mix(in srgb, var(--accent) 12%, var(--bg))' : 'var(--bg-sunken)',
                    border: `1.5px solid ${revealed && isBest ? 'var(--c-good)' : isLast ? 'var(--accent)' : 'var(--border)'}`,
                    borderRadius: 8,
                    padding: '10px 8px',
                    textAlign: 'center',
                    cursor: totalPulls >= maxPulls ? 'default' : 'pointer',
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>Machine {i + 1}</div>
                  <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-text)' }}>
                    {counts[i] > 0 ? (est >= 0 ? `+${est.toFixed(2)}` : est.toFixed(2)) : '—'}
                  </div>
                  <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                    pulls: {counts[i]}
                  </div>
                  {revealed && (
                    <div className="mono" style={{ fontSize: 10.5, marginTop: 4, color: isBest ? 'var(--c-good)' : 'var(--text-faint)' }}>
                      true μ = {trueMean.toFixed(1)} {isBest ? '★' : ''}
                    </div>
                  )}
                </button>
              );
            })}
          </div>

          <div className="live-note">
            Pulls used: <strong className="mono">{totalPulls}/{maxPulls}</strong> · Total reward:{' '}
            <strong className="mono">{totalReward.toFixed(2)}</strong>
            {lastPull && (
              <>
                {' '}· Last pull on Machine {lastPull.arm + 1} paid{' '}
                <strong className="mono">{lastPull.reward >= 0 ? `+${lastPull.reward.toFixed(2)}` : lastPull.reward.toFixed(2)}</strong>.
              </>
            )}{' '}
            {totalPulls === 0
              ? 'Click any machine to start pulling — remember each payout has σ = 1.0 noise!'
              : totalPulls >= maxPulls
                ? `Done! An oracle pulling Machine 3 every time averages ${(maxPulls * 1.5).toFixed(1)}.`
                : 'Do you stick with your current leader or test an under-sampled machine?'}
          </div>
        </>
      )}
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Live Q-Table & TD Backup Stepper

export function QTableStepper() {
  const [alpha, setAlpha] = useState(0.3);
  const [eps, setEps] = useState(0.15);
  const [session, setSession] = useState(() => createQSession(Q_MAZE_GRID, 0, 42));

  const greedyEval = useMemo(() => evaluateGreedyMazeReturn(Q_MAZE_GRID, session.Q), [session.Q]);

  // SVG 4x5 Q-table with 4 directional triangles per empty cell
  const cellW = 86;
  const cellH = 74;
  const pad = 6;
  const W = Q_MAZE_GRID.cols * cellW + pad * 2;
  const H = Q_MAZE_GRID.rows * cellH + pad * 2;

  const qColor = (val: number) => (val >= 0 ? 'var(--c-good)' : 'var(--c-bad)');
  const qOpacity = (val: number) => Math.min(0.45, Math.abs(val) / 14);

  const u = session.lastUpdate;

  return (
    <Widget
      title="Watch Q(s, a) learn from scratch: live TD backup inspector"
      wide
      caption={
        <>
          Each open square is split into four directional triangles showing <code>Q(s, ↑)</code>, <code>Q(s, →)</code>, <code>Q(s, ↓)</code>, and <code>Q(s, ←)</code>. Press <strong>Step 1 move</strong> to watch individual Temporal-Difference updates, then <strong>+20 eps</strong> to watch reward values propagate all the way back to Start without any transition model!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 10, alignItems: 'flex-end' }}>
        <div style={{ flex: '1 1 180px' }}>
          <Slider label="Learning rate α" min={0.05} max={0.8} step={0.05} value={alpha} onChange={setAlpha} color="c-a" />
        </div>
        <div style={{ flex: '1 1 180px' }}>
          <Slider label="Exploration ε" min={0} max={0.5} step={0.05} value={eps} onChange={setEps} color="c-b" />
        </div>
      </div>

      <div className="btn-row" style={{ marginBottom: 12 }}>
        <button type="button" className="btn primary" onClick={() => setSession((s) => stepQSession(s, alpha, eps))}>
          Step 1 move
        </button>
        <button type="button" className="btn" onClick={() => setSession((s) => runQEpisodes(s, 1, alpha, eps))}>
          1 episode
        </button>
        <button type="button" className="btn" onClick={() => setSession((s) => runQEpisodes(s, 20, alpha, eps))}>
          +20 eps
        </button>
        <button type="button" className="btn" onClick={() => setSession((s) => runQEpisodes(s, 100, alpha, eps))}>
          +100 eps
        </button>
        <button type="button" className="btn" onClick={() => setSession(createQSession(Q_MAZE_GRID, 0, 42))}>
          Reset
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(265px, 1fr))', gap: 12, alignItems: 'start', marginBottom: 12 }}>
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 6, paddingInline: 4 }}>
            <span>Episodes: <strong className="mono">{session.episodes}</strong> · Steps: <strong className="mono">{session.totalSteps}</strong></span>
            <span>Greedy V(S): <strong className="mono" style={{ color: greedyEval.reachedGoal ? 'var(--c-good)' : 'var(--text-muted)' }}>{greedyEval.discountedReturn.toFixed(2)}</strong></span>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block', maxWidth: W }} role="img" aria-label="4 by 5 Q-table triangle grid">
            {Q_MAZE_GRID.cells.map((row, r) =>
              row.map((cell, c) => {
                const idx = r * Q_MAZE_GRID.cols + c;
                const x = pad + c * cellW;
                const y = pad + r * cellH;
                const cx = x + cellW / 2;
                const cy = y + cellH / 2;
                const isAgent = session.pos[0] === r && session.pos[1] === c;
                const isLastS = u && u.s[0] === r && u.s[1] === c;
                const q = session.Q[idx];

                const x0 = x + 2;
                const y0 = y + 2;
                const x1 = x + cellW - 2;
                const y1 = y + cellH - 2;

                // 4 triangles: 0=Up, 1=Right, 2=Down, 3=Left
                const tris = [
                  { a: 0 as Action, pts: `${x0},${y0} ${x1},${y0} ${cx},${cy}`, tx: cx, ty: y0 + 13 },
                  { a: 1 as Action, pts: `${x1},${y0} ${x1},${y1} ${cx},${cy}`, tx: x1 - 15, ty: cy + 3 },
                  { a: 2 as Action, pts: `${x0},${y1} ${x1},${y1} ${cx},${cy}`, tx: cx, ty: y1 - 5 },
                  { a: 3 as Action, pts: `${x0},${y0} ${x0},${y1} ${cx},${cy}`, tx: x0 + 15, ty: cy + 3 },
                ];

                return (
                  <g key={idx}>
                    <rect
                      x={x0}
                      y={y0}
                      width={cellW - 4}
                      height={cellH - 4}
                      rx={5}
                      fill={
                        cell.kind === 'wall'
                          ? 'var(--border-strong)'
                          : cell.kind === 'goal'
                            ? 'color-mix(in srgb, var(--c-good) 20%, var(--bg))'
                            : cell.kind === 'hazard'
                              ? 'color-mix(in srgb, var(--c-bad) 20%, var(--bg))'
                              : 'var(--bg)'
                      }
                      stroke={isLastS ? 'var(--accent)' : cell.kind === 'goal' ? 'var(--c-good)' : cell.kind === 'hazard' ? 'var(--c-bad)' : 'var(--border)'}
                      strokeWidth={isLastS ? 2.2 : 1.1}
                    />

                    {cell.kind === 'empty' &&
                      tris.map((tr) => {
                        const val = q[tr.a];
                        const isUpdatedTri = isLastS && u?.a === tr.a;
                        return (
                          <g key={tr.a}>
                            <polygon
                              points={tr.pts}
                              fill={qColor(val)}
                              fillOpacity={Math.abs(val) < 0.01 ? 0 : qOpacity(val)}
                              stroke={isUpdatedTri ? 'var(--accent)' : 'var(--border)'}
                              strokeWidth={isUpdatedTri ? 1.8 : 0.6}
                            />
                            <text
                              x={tr.tx}
                              y={tr.ty}
                              textAnchor="middle"
                              fill={Math.abs(val) < 0.01 ? 'var(--text-faint)' : 'var(--text)'}
                              fontSize={8.5}
                              fontWeight={isUpdatedTri ? 700 : 500}
                              fontFamily="var(--font-mono)"
                            >
                              {val.toFixed(1)}
                            </text>
                          </g>
                        );
                      })}

                    {cell.kind === 'wall' && (
                      <text x={cx} y={cy + 4} textAnchor="middle" fill="var(--text-faint)" fontSize={11} fontWeight={700} fontFamily="var(--font-mono)">
                        WALL
                      </text>
                    )}

                    {(cell.kind === 'goal' || cell.kind === 'hazard') && (
                      <g>
                        <text
                          x={cx}
                          y={cy - 2}
                          textAnchor="middle"
                          fill={cell.kind === 'goal' ? 'var(--c-good)' : 'var(--c-bad)'}
                          fontSize={15}
                          fontWeight={800}
                          fontFamily="var(--font-mono)"
                        >
                          {cell.reward > 0 ? `+${cell.reward}` : cell.reward}
                        </text>
                        <text x={cx} y={cy + 14} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>
                          {cell.label}
                        </text>
                      </g>
                    )}

                    {isAgent && (
                      <circle cx={cx} cy={cy} r={7} fill="var(--accent)" stroke="var(--bg)" strokeWidth={2} />
                    )}
                  </g>
                );
              }),
            )}
          </svg>
        </div>

        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 12.5, fontWeight: 700, marginBottom: 8 }}>Last Temporal-Difference update</div>
          {u ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12 }}>
              <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '6px 9px' }}>
                <span className="mono">
                  ({u.s[0]},{u.s[1]}) {ACTION_ARROWS[u.a]} {ACTION_NAMES[u.a]} → ({u.sNext[0]},{u.sNext[1]})
                </span>
                <span
                  className="mono"
                  style={{
                    float: 'right',
                    fontSize: 10.5,
                    color: u.wasExplore ? 'var(--c-b)' : 'var(--c-a)',
                    fontWeight: 700,
                  }}
                >
                  {u.wasExplore ? 'EXPLORE' : 'GREEDY'}
                </span>
              </div>

              <div className="mono" style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '7px 9px', display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11.5 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Old estimate Q(s, a):</span>
                  <span>{u.oldQ.toFixed(3)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Reward r + γ·max Q(s′, a′):</span>
                  <span style={{ color: 'var(--c-a)', fontWeight: 700 }}>
                    {u.r.toFixed(1)} + 0.9×{u.maxNextQ.toFixed(2)} = {u.target.toFixed(3)}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>TD error δ = target − old:</span>
                  <span style={{ color: u.tdError >= 0 ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
                    {u.tdError >= 0 ? `+${u.tdError.toFixed(3)}` : u.tdError.toFixed(3)}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border)', paddingTop: 4 }}>
                  <span style={{ color: 'var(--text-muted)' }}>New Q ← old + α·δ:</span>
                  <span style={{ color: 'var(--accent-text)', fontWeight: 700 }}>{u.newQ.toFixed(3)}</span>
                </div>
              </div>

              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Current greedy policy from Start:{' '}
                <strong style={{ color: greedyEval.reachedGoal ? 'var(--c-good)' : 'var(--text)' }}>
                  {greedyEval.reachedGoal
                    ? `reaches +10 Goal in ${greedyEval.steps} steps (V = +${greedyEval.discountedReturn.toFixed(2)})`
                    : 'has not found a full path to the Goal yet'}
                </strong>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              All Q-values start at <code>0.0</code>. Press <strong>Step 1 move</strong> or <strong>+20 eps</strong> to watch the agent wander, hit the +10 Goal (or −10 Pit), and propagate values backwards purely from experience.
            </div>
          )}
        </div>
      </div>

      <div className="live-note">
        {session.episodes === 0
          ? 'No transition table P(s′|s,a) anywhere — the agent only sees (s, a, r, s′) one transition at a time.'
          : greedyEval.reachedGoal && greedyEval.steps === 7
            ? `Solved after ${session.episodes} episodes! Notice the glowing green triangles tracing the exact 7-step optimal corridor around the wall and −10 Pit (greedy discounted return +${greedyEval.discountedReturn.toFixed(2)}).`
            : `After ${session.episodes} episodes (${session.totalSteps} moves), Q-values near the Goal are warming up. Keep stepping or click +20 eps to watch the wave reach Start at (3,0).`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Cliff Walking: Q-Learning vs SARSA

function CliffMiniGrid({
  title,
  policy,
  path,
  accentColor,
}: {
  title: string;
  policy: Action[];
  path: [number, number][];
  accentColor: string;
}) {
  const cw = 36;
  const ch = 30;
  const pad = 4;
  const W = CLIFF_COLS * cw + pad * 2;
  const H = CLIFF_ROWS * ch + pad * 2;
  const pathSet = new Set(path.map(([r, c]) => r * CLIFF_COLS + c));
  const pts = path.map(([r, c]) => `${pad + c * cw + cw / 2},${pad + r * ch + ch / 2}`).join(' ');

  return (
    <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: 8 }}>
      <div style={{ fontSize: 11.5, fontWeight: 700, color: accentColor, marginBottom: 4 }}>{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label={title}>
        {Array.from({ length: CLIFF_ROWS }, (_, r) =>
          Array.from({ length: CLIFF_COLS }, (__, c) => {
            const idx = r * CLIFF_COLS + c;
            const x = pad + c * cw;
            const y = pad + r * ch;
            const cliff = isCliffCell(r, c);
            const isStart = r === CLIFF_START[0] && c === CLIFF_START[1];
            const isGoal = r === CLIFF_GOAL[0] && c === CLIFF_GOAL[1];
            const onPath = pathSet.has(idx);
            return (
              <g key={idx}>
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={cw - 2}
                  height={ch - 2}
                  rx={4}
                  fill={
                    cliff
                      ? 'color-mix(in srgb, var(--c-bad) 24%, var(--bg))'
                      : isGoal
                        ? 'color-mix(in srgb, var(--c-good) 24%, var(--bg))'
                        : onPath
                          ? `color-mix(in srgb, ${accentColor} 16%, var(--bg))`
                          : 'var(--bg-sunken)'
                  }
                  stroke={cliff ? 'var(--c-bad)' : isGoal ? 'var(--c-good)' : onPath ? accentColor : 'var(--border)'}
                  strokeWidth={onPath ? 1.4 : 0.8}
                />
                {cliff ? (
                  <text x={x + cw / 2} y={y + ch / 2 + 3.5} textAnchor="middle" fill="var(--c-bad)" fontSize={8.5} fontWeight={700} fontFamily="var(--font-mono)">
                    −50
                  </text>
                ) : isGoal ? (
                  <text x={x + cw / 2} y={y + ch / 2 + 4} textAnchor="middle" fill="var(--c-good)" fontSize={10} fontWeight={800} fontFamily="var(--font-mono)">
                    G
                  </text>
                ) : (
                  <text
                    x={x + cw / 2}
                    y={y + ch / 2 + 4}
                    textAnchor="middle"
                    fill={onPath ? accentColor : 'var(--text-faint)'}
                    fontSize={12}
                    fontWeight={onPath ? 700 : 400}
                    fontFamily="var(--font-mono)"
                  >
                    {isStart ? 'S' : ACTION_ARROWS[policy[idx] ?? 0]}
                  </text>
                )}
              </g>
            );
          }),
        )}
        {pts && <polyline points={pts} fill="none" stroke={accentColor} strokeWidth={2} strokeOpacity={0.5} strokeLinejoin="round" />}
      </svg>
    </div>
  );
}

export function CliffWalkingRace() {
  const race = useMemo(() => runCliffWalkingRace(0.1, 400, 20), []);

  // Moving average smoothing (window = 15 episodes) for clean chart
  const smooth = (arr: number[], win = 15) =>
    arr.map((_, i) => {
      const lo = Math.max(0, i - win + 1);
      let s = 0;
      for (let j = lo; j <= i; j++) s += arr[j];
      return s / (i - lo + 1);
    });

  const qCurve = useMemo(() => smooth(race.qLearning.onlineCurve), [race.qLearning.onlineCurve]);
  const sCurve = useMemo(() => smooth(race.sarsa.onlineCurve), [race.sarsa.onlineCurve]);

  const W = 440;
  const H = 182;
  const pad = { l: 42, r: 14, t: 14, b: 34 };
  const yMin = -65;
  const yMax = -8;
  const sx = (ep: number) => pad.l + (ep / 399) * (W - pad.l - pad.r);
  const sy = (val: number) => H - pad.b - ((Math.min(yMax, Math.max(yMin, val)) - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);

  const toPts = (arr: number[]) =>
    arr
      .filter((_, i) => i % 4 === 0 || i === 399)
      .map((val, idx, a) => `${sx(idx === a.length - 1 ? 399 : idx * 4).toFixed(1)},${sy(val).toFixed(1)}`)
      .join(' ');

  return (
    <Widget
      title="Cliff Walking: off-policy Q-learning vs on-policy SARSA (20 seeds, ε = 0.10)"
      wide
      caption={
        <>
          Fair comparison on Sutton &amp; Barto’s Cliff Walking (4×8 grid, −1/step, −50 cliff reset to Start, ε = 0.10, each at its grid-searched best α = 0.2). <strong>Q-learning</strong> learns the optimal cliff-edge path (greedy eval <strong>−8.0</strong>) but suffers <strong>−19.9</strong> online from ε-slips. <strong>SARSA</strong> routes inland: safer online (<strong>−14.5</strong>), slightly longer greedy path (<strong>−10.4</strong>).
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(255px, 1fr))', gap: 10, marginBottom: 12 }}>
        <CliffMiniGrid
          title={`Q-learning (off-policy) · greedy ${race.qLearning.greedyEvalMean.toFixed(1)} · online ${race.qLearning.lateOnlineMean.toFixed(1)}`}
          policy={race.qLearning.samplePolicy}
          path={race.qLearning.samplePath}
          accentColor="var(--c-b)"
        />
        <CliffMiniGrid
          title={`SARSA (on-policy) · greedy ${race.sarsa.greedyEvalMean.toFixed(1)} · online ${race.sarsa.lateOnlineMean.toFixed(1)}`}
          policy={race.sarsa.samplePolicy}
          path={race.sarsa.samplePath}
          accentColor="var(--c-a)"
        />
      </div>

      <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10, marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6, fontSize: 12, marginBottom: 4 }}>
          <strong>Online episode return (15-ep moving avg across 20 seeds)</strong>
          <span className="mono" style={{ fontSize: 11 }}>
            <span style={{ color: 'var(--c-a)', fontWeight: 700 }}>── SARSA ({race.sarsa.lateOnlineMean.toFixed(1)})</span>{'  '}
            <span style={{ color: 'var(--c-b)', fontWeight: 700 }}>── Q-learning ({race.qLearning.lateOnlineMean.toFixed(1)})</span>
          </span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="Online episode return chart comparing SARSA and Q-learning">
          {[-60, -40, -20, -10].map((v) => (
            <g key={v}>
              <line x1={pad.l} y1={sy(v)} x2={W - pad.r} y2={sy(v)} stroke="var(--border)" strokeDasharray="3 3" />
              <text x={pad.l - 5} y={sy(v) + 3} textAnchor="end" fill="var(--text-faint)" fontSize={9.5} fontFamily="var(--font-mono)">
                {v}
              </text>
            </g>
          ))}
          {[0, 100, 200, 300, 400].map((e) => (
            <text key={e} x={sx(Math.min(399, e))} y={H - 16} textAnchor="middle" fill="var(--text-faint)" fontSize={9.5} fontFamily="var(--font-mono)">
              {e}
            </text>
          ))}
          <text x={(pad.l + W - pad.r) / 2} y={H - 2} textAnchor="middle" fill="var(--text-muted)" fontSize={9.5}>
            Training episode →
          </text>
          <polyline points={toPts(qCurve)} fill="none" stroke="var(--c-b)" strokeWidth={2.2} />
          <polyline points={toPts(sCurve)} fill="none" stroke="var(--c-a)" strokeWidth={2.4} />
        </svg>
      </div>

      <div className="live-note">
        <strong>Two different winners for two different jobs:</strong> if you train in a simulator and turn ε off at deployment, <strong>Q-learning</strong> wins (greedy −8.0 vs −10.4). If the agent must stay safe <em>while</em> exploring in the real world, <strong>SARSA</strong> wins by <strong>+5.4 reward per episode</strong> (−14.5 vs −19.9).
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tune the Q-Learner

const QLAB_STORAGE_KEY = 'nl-qlearn-lab-v1';
const QLAB_PROGRESS_KEY = 'nl-qlearn-lab-progress-v1';

const DEFAULT_CFG: QLabConfig = {
  algo: 'qlearning',
  alpha: 0.1,
  epsSchedule: 'const-0.02',
  qInit: 0,
};

function loadQLabConfigs(): Record<string, QLabConfig> {
  try {
    const raw = localStorage.getItem(QLAB_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Record<string, QLabConfig>;
      if (parsed && typeof parsed === 'object') return parsed;
    }
  } catch {
    // ignore storage errors
  }
  return {};
}

const fmtQ = (v: number) => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));
const cfgLabel = (c: QLabConfig) =>
  `${c.algo === 'qlearning' ? 'Q-learning' : 'SARSA'}, α ${c.alpha}, ε ${c.epsSchedule === 'decay' ? 'decaying' : c.epsSchedule.replace('const-', '')}, Q₀ ${c.qInit}`;

export function QLearnerLab() {
  const [briefIdx, setBriefIdx] = useState(0);
  const [configs, setConfigs] = useState<Record<string, QLabConfig>>(() => loadQLabConfigs());
  // Seeded from the saved settings, so learners who solved a brief before this frame existed keep their ticks.
  const progress = useLabProgress(QLAB_PROGRESS_KEY, () => ({
    solved: Object.fromEntries(QLAB_BRIEFS.map((b) => [b.id, evaluateQLabConfig(b.id, configs[b.id] ?? DEFAULT_CFG).passed])),
  }));
  const log = useMachineLog();

  const brief = QLAB_BRIEFS[briefIdx];
  const cfg = configs[brief.id] ?? DEFAULT_CFG;

  const curOpt = useMemo(() => solveQLabBriefExhaustive(brief.id), [brief.id]);
  const curEval = useMemo(() => evaluateQLabConfig(brief.id, cfg), [brief.id, cfg]);
  const isOpt = curEval.passed && curEval.score >= curOpt.bestScore - 0.01;

  const updateCfg = (patch: Partial<QLabConfig>) => {
    const nextCfg = { ...cfg, ...patch };
    const next = { ...configs, [brief.id]: nextCfg };
    setConfigs(next);
    const ev = evaluateQLabConfig(brief.id, nextCfg);
    log.push(
      `${brief.title.split(' · ')[0]} · ${cfgLabel(nextCfg)} → ${brief.metricLabel} ${fmtQ(ev.score)} (was ${fmtQ(curEval.score)}) · ${ev.secondaryLabel.toLowerCase()} ${ev.secondaryValue}${ev.passed ? ' · passes' : ''}`,
      ev.passed ? 'good' : ev.score < curEval.score ? 'bad' : 'info',
    );
    try {
      localStorage.setItem(QLAB_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };

  return (
    <LabShell
      title="Lab: Tune the Q-Learner"
      caption={
        <>
          Configure algorithm, learning rate <code>α</code>, exploration schedule <code>ε</code>, and initial optimism <code>Q₀</code> across three realistic briefs. Every setting is averaged over 10 fixed seeds; the optional goal is the best of all <strong>36 grid combinations</strong>, found by trying every one. Progress saves automatically.
        </>
      }
      briefs={QLAB_BRIEFS.map((b) => ({ id: b.id, tab: b.title.split(' · ')[1] ?? b.title, title: b.title, text: b.description }))}
      active={briefIdx}
      onSelect={setBriefIdx}
      progress={progress}
      passing={curEval.passed}
      optimal={isOpt}
      status={
        isOpt
          ? `Exhaustive grid optimum matched (${fmtQ(curEval.score)})! Out of all 36 combinations, ${curOpt.passingCount} pass and yours ties the top score.`
          : curEval.passed
            ? `Brief passed (${fmtQ(curEval.score)} vs target ${fmtQ(brief.target)})! Optional challenge: can you reach the exhaustive grid best of ${fmtQ(curOpt.bestScore)}?`
            : `Not quite yet (${fmtQ(curEval.score)} vs target ${fmtQ(brief.target)}). Read the two numbers above: they say which failure you are looking at.`
      }
      hint={brief.hint}
      actions={
        <button type="button" className="btn" onClick={() => updateCfg(DEFAULT_CFG)}>
          Reset brief
        </button>
      }
      reveal={{
        label: 'the exhaustive grid search',
        content: (
          <>
            All {curOpt.totalConfigs} settings were tried: <strong>{curOpt.passingCount}</strong> pass this brief. The best is <strong>{cfgLabel(curOpt.bestConfig)}</strong> at {fmtQ(curOpt.bestScore)}.
          </>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Change a setting: each 10-seed result is logged here, so you can compare what you tried.' }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12 }}>
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <Segmented<TdAlgo>
            label="Algorithm"
            value={cfg.algo}
            onChange={(algo) => updateCfg({ algo })}
            options={QLAB_ALGOS.map((a) => ({ value: a, label: a === 'qlearning' ? 'Q-learning (off-policy)' : 'SARSA (on-policy)' }))}
          />
          <Segmented<number>
            label="Learning rate"
            value={cfg.alpha}
            onChange={(alpha) => updateCfg({ alpha })}
            options={QLAB_ALPHAS.map((a) => ({ value: a, label: `α = ${a}` }))}
          />
          <Segmented<LabEpsSchedule>
            label="Exploration schedule"
            value={cfg.epsSchedule}
            onChange={(epsSchedule) => updateCfg({ epsSchedule })}
            options={QLAB_EPS.map((e) => ({
              value: e,
              label: e === 'const-0.02' ? 'const 0.02' : e === 'const-0.15' ? 'const 0.15' : 'decaying',
            }))}
          />
          <Segmented<number>
            label="Initial Q-table value"
            value={cfg.qInit}
            onChange={(qInit) => updateCfg({ qInit })}
            options={QLAB_QINITS.map((q) => ({ value: q, label: q === 0 ? 'Neutral (Q₀=0)' : 'Optimistic (Q₀=+5)' }))}
          />
        </div>

        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{brief.metricLabel}</div>
              <div className="mono" style={{ fontSize: 19, fontWeight: 700, color: curEval.passed ? 'var(--c-good)' : 'var(--c-bad)' }}>{fmtQ(curEval.score)}</div>
              <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-faint)' }}>target ≥ {fmtQ(brief.target)}</div>
            </div>
            <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{curEval.secondaryLabel}</div>
              <div className="mono" style={{ fontSize: 19, fontWeight: 700, color: 'var(--c-a)' }}>{curEval.secondaryValue}</div>
              <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-faint)' }}>10-seed average</div>
            </div>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {curEval.passed ? (
              <>
                Optional goal: the best of all 36 settings scores <strong>{fmtQ(curOpt.bestScore)}</strong>.
              </>
            ) : (
              <>{curOpt.passingCount} of the 36 settings pass this brief.</>
            )}
          </div>
        </div>
      </div>
    </LabShell>
  );
}
