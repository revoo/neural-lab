import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Dot, Handle, Plot, usePlot } from '../../components/plot';
import { bestFit, bestPeak, makeData, makeSchedule, meanFinalExcess, searchKnots, stats, train, type ScheduleName } from './schedules';

const DATA = makeData();
const STATS = stats(DATA);
const BEST = bestFit(STATS);
const B = 4;
const SEEDS = Array.from({ length: 20 }, (_, k) => k + 1);
const GRID = Array.from({ length: 25 }, (_, k) => Math.round(10 ** (-3 + k * 0.125) * 1e5) / 1e5);
const fmtLr = (v: number) => (v >= 0.1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));

/** A small log-scale line chart drawn directly in SVG. */
function LogChart({
  series,
  T,
  lo,
  hi,
  width = 360,
  height = 190,
  yLabel,
  marks = [],
}: {
  series: { color: string; ys: number[]; dash?: string }[];
  T: number;
  lo: number;
  hi: number;
  width?: number;
  height?: number;
  yLabel: string;
  marks?: { y: number; label: string }[];
}) {
  const pad = { l: 52, r: 8, t: 10, b: 22 };
  const X = (s: number, n: number) => pad.l + (s / Math.max(1, n - 1)) * (width - pad.l - pad.r);
  const Y = (v: number) => {
    const e = Math.log10(Math.max(v, 10 ** lo));
    return pad.t + ((hi - Math.min(e, hi)) / (hi - lo)) * (height - pad.t - pad.b);
  };
  const ticks = [];
  for (let e = Math.ceil(lo); e <= hi; e++) ticks.push(e);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={yLabel}>
      {ticks.map((e) => (
        <g key={e}>
          <line x1={pad.l} x2={width - pad.r} y1={Y(10 ** e)} y2={Y(10 ** e)} className="plot-grid" />
          <text x={pad.l - 4} y={Y(10 ** e) + 3.5} className="plot-tick" textAnchor="end">
            {e >= 0 ? 10 ** e : (10 ** e).toFixed(-e)}
          </text>
        </g>
      ))}
      {marks.map((m) => (
        <g key={m.label}>
          <line x1={pad.l} x2={width - pad.r} y1={Y(m.y)} y2={Y(m.y)} stroke="var(--text-faint)" strokeDasharray="4 3" />
          <text x={width - pad.r} y={Y(m.y) - 3} className="plot-tick" textAnchor="end">
            {m.label}
          </text>
        </g>
      ))}
      <text x={pad.l + 2} y={pad.t + 9} className="plot-label" fontSize={11}>
        {yLabel}
      </text>
      <text x={width - pad.r} y={height - 6} className="plot-tick" textAnchor="end">
        {T} steps
      </text>
      {series.map((s, i) => (
        <polyline
          key={i}
          points={s.ys.map((v, k) => `${X(k, s.ys.length).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}
          fill="none"
          stroke={s.color}
          strokeWidth={1.8}
          strokeDasharray={s.dash}
        />
      ))}
    </svg>
  );
}

/** Average of several runs' excess-loss curves (recorded every `every` steps). */
function meanCurve(schedule: (t: number, T: number) => number, T: number, seeds: number[], every: number) {
  const runs = seeds.map((seed) => train(DATA, STATS, schedule, T, B, seed, [0, 10], every).losses.map((l) => l - BEST.loss));
  const n = Math.min(...runs.map((r) => r.length));
  return Array.from({ length: n }, (_, k) => runs.reduce((s, r) => s + r[k], 0) / runs.length);
}

// ---------------------------------------------------------------- fast or precise

export function FastOrPrecise() {
  const [big, setBig] = useState(-1); // log10 step size
  const [small, setSmall] = useState(-2.3);
  const T = 3000;
  const lrBig = 10 ** big;
  const lrSmall = 10 ** small;
  const cBig = useMemo(() => meanCurve(makeSchedule('constant', lrBig), T, SEEDS.slice(0, 8), 10), [lrBig]);
  const cSmall = useMemo(() => meanCurve(makeSchedule('constant', lrSmall), T, SEEDS.slice(0, 8), 10), [lrSmall]);
  const cloud = useMemo(() => train(DATA, STATS, makeSchedule('constant', lrBig), T, B, 1, [0, 10], 1).path.slice(T / 2), [lrBig]);
  const cloudS = useMemo(() => train(DATA, STATS, makeSchedule('constant', lrSmall), T, B, 1, [0, 10], 1).path.slice(T / 2), [lrSmall]);
  const tail = (c: number[]) => c.slice(Math.floor(c.length / 2)).reduce((a, b) => a + b, 0) / Math.ceil(c.length / 2);
  const crossing = (c: number[], level: number) => {
    const k = c.findIndex((v) => v < level);
    return k < 0 ? null : k * 10;
  };
  const level = 1;
  return (
    <Widget
      title="Fast or precise?"
      wide
      caption={
        <>
          Two runs of mini-batch SGD (batches of {B}) on the delivery-time problem, each with a <strong>constant</strong> step size — one big, one small. Left: how far
          above the best possible loss each one is, averaged over 8 runs (log scale). Right: a close-up of the bottom of the bowl, with each run’s position over its last{' '}
          {T / 2} steps. Move the sliders.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            T={T}
            lo={-3}
            hi={3}
            yLabel="above the best loss"
            series={[
              { color: 'var(--c-b)', ys: cBig },
              { color: 'var(--c-a)', ys: cSmall },
            ]}
            marks={[{ y: level, label: 'within 1' }]}
          />
        </div>
        <div className="widget-col" style={{ flex: '0 1 240px' }}>
          <Plot x={[BEST.w - 1.5, BEST.w + 1.5]} y={[BEST.b - 1.5, BEST.b + 1.5]} width={230} height={230} grid={false} tickLabels={false} pad={8} ariaLabel="Where each run wanders near the minimum">
            <Cloud pts={cloud} color="var(--c-b)" />
            <Cloud pts={cloudS} color="var(--c-a)" />
            <Dot at={[BEST.w, BEST.b]} r={4} color="var(--text)" />
          </Plot>
          <div className="muted" style={{ fontSize: 12 }}>Close-up of the minimum (black dot), ±1.5 in each weight.</div>
        </div>
      </div>
      <div className="controls inline" style={{ marginTop: 8 }}>
        <Slider label="big step size" value={big} min={-2} max={-0.5} step={0.05} onChange={setBig} format={(v) => fmtLr(10 ** v)} color="c-b" />
        <Slider label="small step size" value={small} min={-3} max={-1.5} step={0.05} onChange={setSmall} format={(v) => fmtLr(10 ** v)} color="c-a" />
      </div>
      <div className="readout" style={{ marginTop: 8, fontSize: 12.5 }}>
        <span className="c-b">big   ({fmtLr(lrBig).padEnd(6)})</span> <span className="k">gets within 1 after</span> {crossing(cBig, level) ?? '—'} <span className="k">steps · leftover jitter</span>{' '}
        <strong>{tail(cBig).toFixed(3)}</strong>
        {'\n'}
        <span className="c-a">small ({fmtLr(lrSmall).padEnd(6)})</span> <span className="k">gets within 1 after</span> {crossing(cSmall, level) ?? '—'} <span className="k">steps · leftover jitter</span>{' '}
        <strong>{tail(cSmall).toFixed(3)}</strong>
      </div>
      <div className="live-note" style={{ marginTop: 8 }}>
        The big step size gets close quickly but never settles: its leftover jitter is {(tail(cBig) / Math.max(1e-9, tail(cSmall))).toFixed(0)}× the small one’s. The small
        step size settles more precisely but takes {crossing(cSmall, level) !== null && crossing(cBig, level) !== null ? `${(crossing(cSmall, level)! / Math.max(10, crossing(cBig, level)!)).toFixed(0)}× as long` : 'much longer'}{' '}
        to get there. With a constant step size, you have to choose. A schedule doesn’t.
      </div>
    </Widget>
  );
}

function Cloud({ pts, color }: { pts: [number, number][]; color: string }) {
  const { x, y } = usePlot();
  return (
    <g>
      {pts.map(([a, b], i) =>
        i % 3 === 0 ? <circle key={i} cx={x(a)} cy={y(b)} r={1.6} fill={color} opacity={0.5} /> : null,
      )}
    </g>
  );
}

// ---------------------------------------------------------------- schedule race

const SCHEDS: { key: Exclude<ScheduleName, 'custom'>; label: string; color: string }[] = [
  { key: 'constant', label: 'Constant', color: 'var(--text-muted)' },
  { key: 'step', label: 'Step decay', color: 'var(--c-b)' },
  { key: 'cosine', label: 'Cosine', color: 'var(--c-c)' },
  { key: 'warmup-cosine', label: 'Warmup + cosine', color: 'var(--c-grad)' },
];
const RACE_T = 400;

export function ScheduleRace() {
  const tuned = useMemo(() => Object.fromEntries(SCHEDS.map((s) => [s.key, bestPeak(DATA, STATS, s.key, RACE_T, B, SEEDS, GRID)])) as Record<string, { peak: number; excess: number }>, []);
  const [mode, setMode] = useState<'best' | 'same'>('best');
  const [same, setSame] = useState(-1.4);
  const peakOf = (k: string) => (mode === 'best' ? tuned[k].peak : 10 ** same);
  const curves = useMemo(
    () => SCHEDS.map((s) => ({ ...s, lr: Array.from({ length: RACE_T }, (_, t) => makeSchedule(s.key, peakOf(s.key))(t, RACE_T)), loss: meanCurve(makeSchedule(s.key, peakOf(s.key)), RACE_T, SEEDS, 4) })),
    [mode, same], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const finals = curves.map((c) => meanFinalExcess(DATA, STATS, makeSchedule(c.key, peakOf(c.key)), RACE_T, B, SEEDS));
  const bestIdx = finals.indexOf(Math.min(...finals));
  return (
    <Widget
      title="Four schedules, one budget"
      wide
      caption={
        <>
          Four schedules with {RACE_T} steps each, batches of {B}. By default each uses <strong>its own best peak step size</strong>, found by trying {GRID.length} values;
          switch to “same peak” to compare shapes at one setting. Curves are averaged over {SEEDS.length} runs with different random batches, so what you see is the
          schedule, not luck.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 8 }}>
        <Segmented label="Peak step size" value={mode} onChange={setMode} options={[{ value: 'best', label: 'each at its best' }, { value: 'same', label: 'same peak for all' }]} />
        {mode === 'same' && <Slider label="peak step size" value={same} min={-2.5} max={-0.5} step={0.05} onChange={setSame} format={(v) => fmtLr(10 ** v)} />}
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <LogChart T={RACE_T} lo={-5} hi={0} yLabel="step size" width={320} height={170} series={curves.map((c) => ({ color: c.color, ys: c.lr }))} />
        </div>
        <div className="widget-col" style={{ flex: '1 1 340px' }}>
          <LogChart T={RACE_T} lo={-2} hi={3} yLabel="above the best loss" width={360} height={170} series={curves.map((c) => ({ color: c.color, ys: c.loss }))} />
        </div>
      </div>
      <table className="num-table" style={{ width: '100%', marginTop: 8 }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left' }}>schedule</th>
            <th>peak step size</th>
            <th>final distance above best (avg of {SEEDS.length})</th>
          </tr>
        </thead>
        <tbody>
          {curves.map((c, i) => (
            <tr key={c.key} style={{ background: i === bestIdx ? 'var(--accent-soft)' : undefined }}>
              <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)', fontWeight: 650, color: c.color }}>{c.label}</td>
              <td>{fmtLr(peakOf(c.key))}</td>
              <td>{Number.isFinite(finals[i]) ? finals[i].toFixed(3) : 'diverged'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="live-note" style={{ marginTop: 8 }}>
        {mode === 'best' ? (
          <>
            Each schedule at its best: <strong>{curves[bestIdx].label}</strong> ends closest to the bottom, {(finals[0] / finals[bestIdx]).toFixed(1)}× closer than the best
            possible constant step size. Every decaying schedule beats constant, for the same reason: big steps early, small steps late. Warmup changes nothing useful
            here — this problem has no early instability for it to prevent.
          </>
        ) : (
          <>
            At one shared peak, the shapes differ in how much time they spend at small step sizes. Push the peak up until constant starts to suffer: the decaying
            schedules still finish well, because they don’t stay at the big step size.
          </>
        )}
      </div>
    </Widget>
  );
}

// ---------------------------------------------------------------- designer puzzle

const KNOT_T = [0, 0.25, 0.5, 0.75, 1];
const GOAL = 0.15;

export function ScheduleDesigner() {
  const [knots, setKnots] = useState([-1.3, -1.3, -1.3, -1.3, -1.3]); // log10 step sizes
  const [best, setBest] = useState<number | null>(null);
  const [search, setSearch] = useState<ReturnType<typeof searchKnots> | 'pending' | null>(null);
  const ref = useMemo(() => ({ constant: bestPeak(DATA, STATS, 'constant', RACE_T, B, SEEDS, GRID), cosine: bestPeak(DATA, STATS, 'cosine', RACE_T, B, SEEDS, GRID) }), []);
  const sched = useMemo(() => makeSchedule('custom', 0, knots.map((k) => 10 ** k)), [knots]);
  const score = useMemo(() => meanFinalExcess(DATA, STATS, sched, RACE_T, B, SEEDS), [sched]);
  const curve = useMemo(() => meanCurve(sched, RACE_T, SEEDS.slice(0, 10), 4), [sched]);
  const ok = score < GOAL;
  useEffect(() => {
    if (ok) setBest((b) => (b === null ? score : Math.min(b, score)));
  }, [ok, score]);
  return (
    <Widget
      title="Design a schedule"
      wide
      caption={
        <>
          Drag the five knots to shape the step size over {RACE_T} steps (log scale; straight lines between knots). <strong>Goal:</strong> finish within {GOAL} of the best
          possible loss, on average over {SEEDS.length} runs — no constant step size can do that (the best one manages {ref.constant.excess.toFixed(3)}).{' '}
          <strong>Then, optionally:</strong> beat the best cosine schedule found by search ({ref.cosine.excess.toFixed(3)}).
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <Plot x={[0, RACE_T]} y={[-3, 0]} width={400} height={240} grid={false} axes={false} pad={44} ariaLabel="Your schedule: draggable knots">
            {({ x, y }) => (
              <g>
                {[-3, -2, -1, 0].map((e) => (
                  <g key={e}>
                    <line x1={x(0)} x2={x(RACE_T)} y1={y(e)} y2={y(e)} className="plot-grid" />
                    <text x={x(0) - 4} y={y(e) + 3.5} className="plot-tick" textAnchor="end">
                      {(10 ** e).toString()}
                    </text>
                  </g>
                ))}
                {[0, 100, 200, 300, 400].map((t) => (
                  <text key={t} x={x(t)} y={y(-3) + 14} className="plot-tick" textAnchor="middle">
                    {t}
                  </text>
                ))}
                <text x={x(RACE_T)} y={y(-3) - 4} className="plot-label" textAnchor="end">
                  step
                </text>
                <text x={x(0) + 4} y={y(0) + 12} className="plot-label">
                  step size
                </text>
                <polyline
                  points={Array.from({ length: 101 }, (_, k) => {
                    const t = (k / 100) * (RACE_T - 1);
                    return `${x(t).toFixed(1)},${y(Math.log10(sched(t, RACE_T))).toFixed(1)}`;
                  }).join(' ')}
                  fill="none"
                  stroke="var(--c-grad)"
                  strokeWidth={2.5}
                />
              </g>
            )}
            {knots.map((k, i) => (
              <Handle
                key={i}
                x={KNOT_T[i] * RACE_T}
                y={k}
                onDrag={(_, b) => setKnots(knots.map((v, j) => (j === i ? Math.round(b * 20) / 20 : v)))}
                color="var(--c-grad)"
                label={`knot ${i + 1}`}
              />
            ))}
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <LogChart T={RACE_T} lo={-2} hi={3} yLabel="above the best loss" width={320} height={170} series={[{ color: 'var(--c-grad)', ys: curve }]} marks={[{ y: GOAL, label: `goal ${GOAL}` }]} />
          <div className="readout" style={{ fontSize: 12.5 }}>
            <span className="k">your schedule      </span> <strong>{Number.isFinite(score) ? score.toFixed(3) : 'diverged'}</strong>
            {'\n'}
            <span className="k">your best          </span> {best?.toFixed(3) ?? '—'}
            {'\n'}
            <span className="k">best constant      </span> {ref.constant.excess.toFixed(3)} (step size {fmtLr(ref.constant.peak)})
            {'\n'}
            <span className="k">best cosine        </span> {ref.cosine.excess.toFixed(3)} (peak {fmtLr(ref.cosine.peak)})
          </div>
          <div className={`feedback ${ok ? 'good' : 'bad'}`}>
            {!Number.isFinite(score) ? (
              <>A step size that big makes the weights fly off. Bring the early knots down.</>
            ) : ok ? (
              <>
                <div className="feedback-title">✓ Within {GOAL} of the best loss.</div>
                {score < ref.cosine.excess ? 'You beat the best cosine schedule. Nobody computed the true optimum for this problem — you may have found something better.' : 'Optional: can you beat the best cosine schedule?'}
              </>
            ) : (
              <>Not yet: {score.toFixed(3)} above the best. Think about what each phase of training needs.</>
            )}
          </div>
          {best !== null &&
            (search === null ? (
              <button className="btn small" onClick={() => { setSearch('pending'); setTimeout(() => setSearch(searchKnots(DATA, STATS, RACE_T, B, SEEDS)), 20); }}>
                Search for a better schedule
              </button>
            ) : search === 'pending' ? (
              <span className="muted">Trying hundreds of five-knot schedules…</span>
            ) : (
              <div className="muted" style={{ fontSize: 12.5 }}>
                Best of {search.tried} non-increasing five-knot schedules tried by brute force: <strong>{search.excess.toFixed(3)}</strong>.{' '}
                <button className="btn small" onClick={() => setKnots(search.knots)}>Show it</button> (A coarse search, not a proof — finer knots might do better.)
              </div>
            ))}
        </div>
      </div>
    </Widget>
  );
}
