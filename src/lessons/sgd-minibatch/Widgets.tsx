import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Arrow, Clip, Dot, Handle, Plot, usePlot } from '../../components/plot';
import { Heatmap } from '../../components/heatmap';
import { batchGradients, best, grad, lossFast, makeData, run, spread, stats } from './sgd';

const DATA = makeData();
const STATS = stats(DATA);
const BEST = best(DATA);
const N = DATA.x.length;
const W_RANGE: [number, number] = [-6, 26];
const B_RANGE: [number, number] = [2, 48];
const landscape = (w: number, b: number) => lossFast(STATS, w, b);
const START: [number, number] = [0, 10];

function Minimum() {
  const { x, y } = usePlot();
  return (
    <g>
      <circle cx={x(BEST.w)} cy={y(BEST.b)} r={6} fill="none" stroke="#fff" strokeWidth={2} />
      <circle cx={x(BEST.w)} cy={y(BEST.b)} r={2} fill="#fff" />
    </g>
  );
}

// ---------------------------------------------------------------- the poll

const BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1000];

export function GradientPoll() {
  const [p, setP] = useState<[number, number]>([2, 14]);
  const [B, setB] = useState(4);
  const [seed, setSeed] = useState(1);
  const truth = useMemo(() => grad(DATA, p[0], p[1]), [p]);
  const est = useMemo(() => batchGradients(DATA, p[0], p[1], B, 30, seed), [p, B, seed]);
  const spreadNow = useMemo(() => spread(batchGradients(DATA, p[0], p[1], B, 300, seed + 99), truth), [p, B, seed, truth]);
  const spread1 = useMemo(() => spread(batchGradients(DATA, p[0], p[1], 1, 300, seed + 7), truth), [p, seed, truth]);
  const k = 0.25; // arrow scale: data units per unit of gradient
  const tip = (g: [number, number]): [number, number] => [p[0] - k * g[0], p[1] - k * g[1]];
  const fpc = Math.sqrt((N - B) / (N - 1)); // finite-population correction: sampling without replacement
  const predicted = (spread1 / Math.sqrt(B)) * fpc;
  const len = Math.hypot(truth[0], truth[1]);
  return (
    <Widget
      title="Polling the gradient"
      wide
      caption={
        <>
          The landscape is the loss of a line fitted to 1,000 deliveries, for every slope <em>w</em> and baseline <em>b</em> (the white ring is the bottom). The thick
          arrow is the true downhill direction at the purple point, computed from all 1,000 examples. The thin arrows are 30 estimates of it, each computed from a random
          batch. Change the batch size and watch the estimates tighten. Drag the point around.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={W_RANGE} y={B_RANGE} width={460} height={360} grid={false} xLabel="slope w" yLabel="baseline b" ariaLabel="Loss landscape with gradient estimates">
            <Heatmap f={landscape} range={[20, 900]} transform={Math.sqrt} contours={14} />
            <Minimum />
            <Clip>
              {est.map((g, i) => (
                <Arrow key={i} from={p} to={tip(g)} color="#fff" width={1.2} head={6} opacity={0.6} />
              ))}
              <Arrow from={p} to={tip(truth)} color="var(--c-grad)" width={4} head={12} />
            </Clip>
            <Handle x={p[0]} y={p[1]} onDrag={(a, b) => setP([Math.round(a * 4) / 4, Math.round(b * 4) / 4])} color="var(--accent)" label="where the gradient is measured" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <Segmented label="Batch size (examples per estimate)" value={B} onChange={setB} options={BATCHES.map((v) => ({ value: v, label: v === 1000 ? 'all 1000' : String(v) }))} />
          <button className="btn small" onClick={() => setSeed(seed + 1)}>
            Draw new batches
          </button>
          <div className="readout">
            <span className="k">true gradient size    </span> {len.toFixed(1)}
            {'\n'}
            <span className="k">typical error, B = {String(B).padEnd(4)} </span> <strong>{spreadNow.toFixed(1)}</strong>
            {'\n'}
            <span className="k">predicted (B=1 ÷ √B)  </span> {predicted.toFixed(1)}
            {'\n'}
            <span className="k">cost of one estimate  </span> {B} example{B > 1 ? 's' : ''}
          </div>
          <div className="live-note">
            {B === 1000 ? (
              <>Every estimate uses the whole data set, so they’re all identical to the true gradient — and each one costs 1,000 examples’ worth of work.</>
            ) : B === 1 ? (
              <>
                One example per estimate: each arrow is one delivery’s opinion. They’re wildly scattered, yet on average they point the right way — and each costs a
                thousandth of the full gradient.
              </>
            ) : (
              <>
                With {B} examples per estimate, the typical error is {spreadNow.toFixed(1)} — {(spread1 / spreadNow).toFixed(1)}× smaller than with 1. Bigger batches help, but
                with diminishing returns: to halve the error you need <strong>four times</strong> the batch.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ---------------------------------------------------------------- the race

const RUNNERS = [
  { key: 'full', label: 'Full batch (1000)', color: 'var(--c-b)' },
  { key: 'mini', label: 'Mini-batch', color: 'var(--c-c)' },
  { key: 'sgd', label: 'One at a time (1)', color: '#fff' },
] as const;

function PathLine({ pts, color }: { pts: { w: number; b: number }[]; color: string }) {
  const { x, y } = usePlot();
  return <polyline points={pts.map((q) => `${x(q.w).toFixed(1)},${y(q.b).toFixed(1)}`).join(' ')} fill="none" stroke={color} strokeWidth={1.8} strokeLinejoin="round" opacity={0.9} />;
}

function LossChart({ series, budget }: { series: { color: string; pts: { seen: number; loss: number }[] }[]; budget: number }) {
  const W = 360;
  const H = 210;
  const pad = { l: 40, r: 8, t: 8, b: 24 };
  const lo = -2;
  const hi = 3;
  const X = (s: number) => pad.l + (s / budget) * (W - pad.l - pad.r);
  const Y = (excess: number) => {
    const v = Math.log10(Math.max(excess, 10 ** lo));
    return pad.t + ((hi - Math.min(v, hi)) / (hi - lo)) * (H - pad.t - pad.b);
  };
  const epochs = Math.floor(budget / N);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Loss above the best possible, against examples processed">
      {[-2, -1, 0, 1, 2, 3].map((e) => (
        <g key={e}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(10 ** e)} y2={Y(10 ** e)} className="plot-grid" />
          <text x={pad.l - 5} y={Y(10 ** e) + 3.5} className="plot-tick" textAnchor="end">
            {e >= 0 ? 10 ** e : (10 ** e).toFixed(-e)}
          </text>
        </g>
      ))}
      {Array.from({ length: epochs + 1 }, (_, k) => (
        <g key={k}>
          <line x1={X(k * N)} x2={X(k * N)} y1={pad.t} y2={H - pad.b} className="plot-grid" />
          <text x={X(k * N)} y={H - 8} className="plot-tick" textAnchor="middle">
            {k}
          </text>
        </g>
      ))}
      <text x={W - pad.r} y={H - pad.b - 4} className="plot-label" textAnchor="end" fontSize={11}>
        epochs
      </text>
      {series.map((s, i) => (
        <polyline key={i} points={s.pts.map((q) => `${X(q.seen).toFixed(1)},${Y(q.loss - BEST.loss).toFixed(1)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2} />
      ))}
    </svg>
  );
}

export function BudgetRace() {
  const [epochs, setEpochs] = useState(3);
  const [B, setB] = useState(32);
  const [lrFull, setLrFull] = useState(0.5);
  const [lrMini, setLrMini] = useState(0.1);
  const [lrOne, setLrOne] = useState(0.02);
  const [seed, setSeed] = useState(1);
  const budget = Math.round(epochs * N);
  const runs = useMemo(
    () => ({
      full: run(DATA, N, lrFull, budget, START, seed),
      mini: run(DATA, B, lrMini, budget, START, seed, Math.max(1, Math.floor(8 / B))),
      sgd: run(DATA, 1, lrOne, budget, START, seed, 4),
    }),
    [budget, B, lrFull, lrMini, lrOne, seed],
  );
  const excess = (k: keyof typeof runs) => runs[k].points[runs[k].points.length - 1].loss - BEST.loss;
  return (
    <Widget
      title="A race on a fixed budget"
      wide
      caption={
        <>
          Three runners start from the same point. Each may process the same total number of examples — the budget — so a bigger batch means fewer, more accurate
          steps. The chart shows how far each is above the best possible loss (log scale) against examples processed. Try a budget of <strong>1 epoch</strong>, then
          play with the batch size and step sizes.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <Plot x={W_RANGE} y={B_RANGE} width={420} height={320} grid={false} xLabel="slope w" yLabel="baseline b" ariaLabel="Paths of the three runners">
            <Heatmap f={landscape} range={[20, 900]} transform={Math.sqrt} contours={14} />
            <Minimum />
            <Clip>
              <PathLine pts={runs.sgd.points} color="#fff" />
              <PathLine pts={runs.mini.points} color="var(--c-c)" />
              <PathLine pts={runs.full.points} color="var(--c-b)" />
              {runs.full.points.map((q, i) => (
                <Dot key={i} at={[q.w, q.b]} r={4} color="var(--c-b)" stroke="#fff" />
              ))}
            </Clip>
            <Dot at={START} r={5} color="var(--accent)" stroke="#fff" />
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 340px' }}>
          <LossChart
            budget={budget}
            series={[
              { color: 'var(--text-faint)', pts: runs.sgd.points },
              { color: 'var(--c-c)', pts: runs.mini.points },
              { color: 'var(--c-b)', pts: runs.full.points },
            ]}
          />
          <div className="readout" style={{ fontSize: 12.5 }}>
            {RUNNERS.map((r) => {
              const k = r.key;
              return (
                <div key={k}>
                  <span style={{ color: k === 'sgd' ? 'var(--text)' : r.color, fontWeight: 700 }}>{k === 'mini' ? `Mini-batch (${B})`.padEnd(18) : r.label.padEnd(18)}</span>
                  <span className="k">{String(runs[k].steps).padStart(5)} steps · above best </span>
                  <strong>{excess(k) < 0.005 ? '<0.01' : excess(k).toFixed(2)}</strong>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <div className="controls inline" style={{ marginTop: 10 }}>
        <Slider label="budget (epochs = passes through the data)" value={epochs} min={0.25} max={6} step={0.25} onChange={setEpochs} format={(v) => `${v} (${Math.round(v * N).toLocaleString()} examples)`} />
        <Segmented label="mini-batch size" value={B} onChange={setB} options={[4, 16, 32, 128].map((v) => ({ value: v, label: String(v) }))} />
        <button className="btn small" onClick={() => setSeed(seed + 1)}>
          Reshuffle
        </button>
      </div>
      <div className="controls inline" style={{ marginTop: 6 }}>
        <Slider label="step size, full batch" value={lrFull} min={0.05} max={1.2} step={0.05} onChange={setLrFull} color="c-b" />
        <Slider label="step size, mini-batch" value={lrMini} min={0.01} max={0.6} step={0.01} onChange={setLrMini} color="c-c" />
        <Slider label="step size, one at a time" value={lrOne} min={0.002} max={0.1} step={0.002} onChange={setLrOne} format={(v) => v.toFixed(3)} />
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        {epochs <= 1 ? (
          <>
            In {epochs === 1 ? 'one pass' : 'less than one pass'} through the data, full-batch descent has taken {runs.full.steps === 0 ? 'no steps at all' : `${runs.full.steps} step${runs.full.steps > 1 ? 's' : ''}`}
            . The mini-batch runner has taken {runs.mini.steps}, each a little noisy, and is already close to the bottom.
          </>
        ) : (
          <>
            Look at the end of the paths. The one-at-a-time runner never quite settles: every step uses a single noisy opinion, so it keeps jittering around the bottom.
            Shrinking its step size calms the jitter but slows the approach. That trade-off is the subject of learning-rate schedules.
          </>
        )}
      </div>
    </Widget>
  );
}
