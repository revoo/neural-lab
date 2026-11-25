import { useEffect, useMemo, useRef, useState } from 'react';
import { LiveCode } from '../../components/LabJob';
import { torchOptimLine } from './realCode';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Clip, Dot, Plot, usePlot } from '../../components/plot';
import { Heatmap } from '../../components/heatmap';
import type { OptimizerName } from '../../lib/optimizers';
import { DEFAULT_BETAS, LANDSCAPES, LR_GRID, golfPar, runOptimizer, tuneLR, type Config, type Landscape } from './optim';

type Opt = 'sgd' | 'momentum' | 'rmsprop' | 'adam';
const OPTS: { key: Opt; label: string; color: string }[] = [
  { key: 'sgd', label: 'Plain GD', color: '#ffffff' },
  { key: 'momentum', label: 'Momentum', color: 'var(--c-b)' },
  { key: 'rmsprop', label: 'RMSProp', color: 'var(--c-c)' },
  { key: 'adam', label: 'Adam', color: 'var(--c-grad)' },
];
const fmtLr = (v: number) => (v >= 0.1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));

function Path({ pts, color, upto }: { pts: [number, number][]; color: string; upto: number }) {
  const { x, y } = usePlot();
  const shown = pts.slice(0, upto + 1).map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ');
  return (
    <g>
      <polyline points={shown} fill="none" stroke="rgba(0,0,0,0.45)" strokeWidth={4} strokeLinejoin="round" />
      <polyline points={shown} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
    </g>
  );
}

function Minima({ L }: { L: Landscape }) {
  const { x, y } = usePlot();
  return (
    <g>
      {L.minimum.map(([a, b], i) => (
        <circle key={i} cx={x(a)} cy={y(b)} r={7} fill="none" stroke="#fff" strokeWidth={2} strokeDasharray="3 2" />
      ))}
    </g>
  );
}

function LossChart({ runs, t, T, target }: { runs: { color: string; losses: number[] }[]; t: number; T: number; target: number }) {
  const W = 340;
  const H = 200;
  const pad = { l: 38, r: 8, t: 8, b: 22 };
  const lo = -4;
  const hi = 2;
  const X = (s: number) => pad.l + (s / Math.max(1, T)) * (W - pad.l - pad.r);
  const Y = (v: number) => {
    const e = Math.log10(Math.max(v, 10 ** lo));
    return pad.t + ((hi - Math.min(e, hi)) / (hi - lo)) * (H - pad.t - pad.b);
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Loss against steps, log scale">
      {[-4, -3, -2, -1, 0, 1, 2].map((e) => (
        <g key={e}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(10 ** e)} y2={Y(10 ** e)} className="plot-grid" />
          <text x={pad.l - 4} y={Y(10 ** e) + 3.5} className="plot-tick" textAnchor="end">
            {e >= 0 ? 10 ** e : `1e${e}`}
          </text>
        </g>
      ))}
      <line x1={pad.l} x2={W - pad.r} y1={Y(target)} y2={Y(target)} stroke="var(--text-faint)" strokeDasharray="4 3" />
      <text x={W - pad.r} y={Y(target) - 3} className="plot-tick" textAnchor="end">
        target
      </text>
      <line x1={X(t)} x2={X(t)} y1={pad.t} y2={H - pad.b} stroke="var(--accent)" strokeWidth={1} />
      <text x={pad.l} y={H - 6} className="plot-tick">0</text>
      <text x={W - pad.r} y={H - 6} className="plot-tick" textAnchor="end">
        {T} steps
      </text>
      {runs.map((r, i) => (
        <polyline
          key={i}
          points={r.losses
            .slice(0, t + 1)
            .map((v, s) => `${X(s).toFixed(1)},${Y(v).toFixed(1)}`)
            .join(' ')}
          fill="none"
          stroke={r.color === '#ffffff' ? 'var(--text-muted)' : r.color}
          strokeWidth={2}
        />
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------- the race

export function OptimizerRace() {
  const [lid, setLid] = useState('valley');
  const L = LANDSCAPES.find((l) => l.id === lid)!;
  const tuned = useMemo(() => Object.fromEntries(OPTS.map((o) => [o.key, tuneLR(L, o.key)])) as Record<Opt, { lr: number; steps: number }>, [L]);
  const [own, setOwn] = useState(false);
  const [lrs, setLrs] = useState<Record<Opt, number>>({ sgd: 0.1, momentum: 0.1, rmsprop: 0.1, adam: 0.1 });
  const lrOf = (k: Opt) => (own ? lrs[k] : tuned[k].lr);
  const runs = useMemo(() => {
    const out = {} as Record<Opt, ReturnType<typeof runOptimizer>>;
    for (const o of OPTS) out[o.key] = runOptimizer(L, { name: o.key as OptimizerName, lr: lrOf(o.key), ...(o.key === 'sgd' ? {} : DEFAULT_BETAS[o.key]) }, 300);
    return out;
  }, [L, own, lrs, tuned]); // eslint-disable-line react-hooks/exhaustive-deps
  const T = Math.min(300, Math.max(20, ...OPTS.map((o) => (runs[o.key].reachedAt ?? 300) + 5)));
  const [t, setT] = useState(T);
  const [playing, setPlaying] = useState(false);
  const raf = useRef(0);

  useEffect(() => {
    setT(T);
    setPlaying(false);
  }, [lid, own, lrs, T]);
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      if (now - last > 40) {
        last = now;
        setT((v) => {
          if (v >= T) {
            setPlaying(false);
            return v;
          }
          return v + 1;
        });
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, T]);

  /** Runners stop at the finish line: the step at which they first reach the target. */
  const stopAt = (k: Opt) => Math.min(t, runs[k].reachedAt ?? Infinity);
  const order = [...OPTS].sort((a, b) => (runs[a.key].reachedAt ?? Infinity) - (runs[b.key].reachedAt ?? Infinity));
  const winner = order[0];
  const firstAt = runs[winner.key].reachedAt;
  const winners = OPTS.filter((o) => runs[o.key].reachedAt === firstAt).map((o) => o.label);
  const winText = winners.length > 1 ? `${winners.slice(0, -1).join(', ')} and ${winners[winners.length - 1]} tie here` : `${winner.label} wins here`;

  return (
    <Widget
      title="The optimizer race"
      wide
      caption={
        <>
          Four optimizers start from the same point and race to the bottom (dashed circle); each stops when its loss drops below the target. To make the race fair, each uses <strong>the step size that works best for
          it</strong> on this landscape — found by trying {LR_GRID.length} values from 0.001 to 10 and keeping whichever reaches the target soonest. Press play, or step
          through. Then switch landscapes: the winner changes.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 8 }}>
        <Segmented label="Landscape" value={lid} onChange={setLid} options={LANDSCAPES.map((l) => ({ value: l.id, label: l.label }))} />
        <div className="btn-row">
          <button className="btn primary small" onClick={() => (t >= T ? (setT(0), setPlaying(true)) : setPlaying(!playing))}>
            {playing ? '❚❚ Pause' : t >= T ? '▶ Replay' : '▶ Play'}
          </button>
          <button className="btn small" onClick={() => setT(Math.min(T, t + 1))} disabled={t >= T}>
            Step
          </button>
          <button className="btn small" onClick={() => { setPlaying(false); setT(0); }}>
            Reset
          </button>
          <span className="mono muted" style={{ fontSize: 12.5 }}>step {t}</span>
        </div>
      </div>
      <div className="muted" style={{ fontSize: 13, marginBottom: 8 }}>{L.story}</div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={[L.domain[0], L.domain[1]]} y={[L.domain[2], L.domain[3]]} width={440} height={(440 * (L.domain[3] - L.domain[2])) / (L.domain[1] - L.domain[0])} grid={false} tickLabels={false} ariaLabel={`${L.label} landscape with optimizer paths`}>
            <Heatmap f={L.f} range={L.range} transform={Math.sqrt} contours={14} />
            <Minima L={L} />
            <Clip>
              {OPTS.map((o) => (
                <Path key={o.key} pts={runs[o.key].path} color={o.color} upto={stopAt(o.key)} />
              ))}
              {OPTS.map((o) => {
                const p = runs[o.key].path[Math.min(stopAt(o.key), runs[o.key].path.length - 1)];
                return <Dot key={o.key} at={p} r={4.5} color={o.color} stroke="rgba(0,0,0,0.6)" />;
              })}
            </Clip>
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 320px' }}>
          <LossChart runs={OPTS.map((o) => ({ color: o.color, losses: runs[o.key].losses.slice(0, stopAt(o.key) + 1) }))} t={t} T={T} target={L.target} />
          <table className="num-table" style={{ width: '100%', marginTop: 6 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>optimizer</th>
                <th>step size</th>
                <th>reached target</th>
              </tr>
            </thead>
            <tbody>
              {OPTS.map((o) => (
                <tr key={o.key}>
                  <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)', fontWeight: 650 }}>
                    <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 5, background: o.color, border: '1px solid var(--border-strong)', marginRight: 6 }} />
                    {o.label}
                  </td>
                  <td>{fmtLr(lrOf(o.key))}</td>
                  <td>{runs[o.key].diverged ? 'flew off' : runs[o.key].reachedAt === null ? 'not in 300' : `step ${runs[o.key].reachedAt}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 8 }}>
            <Toggle label="Choose the step sizes myself" checked={own} onChange={setOwn} />
          </div>
          {own &&
            OPTS.map((o) => (
              <Slider
                key={o.key}
                label={`${o.label} step size`}
                value={Math.log10(lrs[o.key])}
                min={-3}
                max={1}
                step={0.05}
                onChange={(v) => setLrs({ ...lrs, [o.key]: Math.round(10 ** v * 1e4) / 1e4 })}
                format={(v) => fmtLr(10 ** v)}
              />
            ))}
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 8 }}>
        {own ? (
          <>
            {firstAt === null ? 'With your own step sizes, nobody arrives within 300 steps.' : `With your own step sizes, ${winners.join(' and ')} arrive${winners.length > 1 ? '' : 's'} first (step ${firstAt}).`} Try
            the same value for all four: it’s a fair test of a single setting, but a step size of 0.1 means very different things to plain GD (a multiple of the
            slope) and to Adam (roughly a distance).
          </>
        ) : lid === 'valley' ? (
          <>
            <strong>{winText}.</strong> Plain gradient descent zig-zags across the steep walls; momentum’s zig-zags cancel while its push along the floor builds up. Now
            look at the step size Adam needed to keep up: {fmtLr(tuned.adam.lr)}, thousands of times its usual 0.001. RMSProp-style scaling works one weight at a time —
            it rescales x and y separately — and this valley is tilted, so its steep and gentle directions don’t line up with x and y.
          </>
        ) : lid === 'plateau' ? (
          <>
            <strong>{winText}.</strong> Out on the plain the slope is only a few thousandths, so plain gradient descent shuffles forward in tiny steps — and it can’t
            just use a huge step size, because the bowl in the middle is steep. RMSProp and Adam divide by the size of recent slopes, so on the plain they take full-sized
            steps anyway.
          </>
        ) : (
          <>
            <strong>{winText}.</strong> The curved valley floor punishes everyone: its walls force small steps, and it keeps changing direction, so a
            velocity built up in one direction soon points the wrong way. Plain gradient descent needs {tuned.sgd.steps === Infinity ? 'more than 1000' : tuned.sgd.steps} steps
            even at its best.
          </>
        )}
      </div>
    </Widget>
  );
}

// ---------------------------------------------------------------- golf

const GOLF = LANDSCAPES.find((l) => l.id === 'banana')!;

export function OptimizerGolf() {
  const [name, setName] = useState<Opt>('sgd');
  const [logLr, setLogLr] = useState(-1);
  const [b1, setB1] = useState(0.9);
  const [b2, setB2] = useState(0.999);
  const [best, setBest] = useState<number | null>(null);
  const [showPar, setShowPar] = useState(false);
  const lr = Math.round(10 ** logLr * 1e5) / 1e5;
  const cfg: Config = { name, lr, ...(name === 'momentum' || name === 'adam' ? { beta1: b1 } : {}), ...(name === 'rmsprop' || name === 'adam' ? { beta2: b2 } : {}) };
  const r = useMemo(() => runOptimizer(GOLF, cfg, 1000), [name, lr, b1, b2]); // eslint-disable-line react-hooks/exhaustive-deps
  const par = useMemo(() => golfPar(GOLF), []);
  useEffect(() => {
    if (r.reachedAt !== null) setBest((b) => (b === null ? r.reachedAt : Math.min(b, r.reachedAt!)));
  }, [r.reachedAt]);
  const shown = r.reachedAt !== null ? r.reachedAt : r.path.length - 1;

  let verdict: React.ReactNode;
  if (r.diverged) verdict = <div className="feedback bad"><div className="feedback-title">✕ Flew off the course.</div>The step size is too big for the steep walls of the valley.</div>;
  else if (r.reachedAt === null)
    verdict = (
      <div className="feedback bad">
        <div className="feedback-title">Not there after 1,000 steps.</div>
        Look at where the path ends up: usually it has found the valley floor and is crawling along the bend. What would help it keep moving along the floor?
      </div>
    );
  else
    verdict = (
      <div className="feedback good">
        <div className="feedback-title">✓ Reached the minimum in {r.reachedAt} steps.</div>
        <div className="muted">
          {r.reachedAt <= par.steps ? 'That matches the best configuration found by the search. ' : `Optional: the best configuration found by searching gets there in ${par.steps}. Can you match it?`}
        </div>
      </div>
    );

  return (
    <Widget
      title="Optimizer golf: the banana"
      wide
      caption={
        <>
          <strong>Goal:</strong> get from the start (top left) to within 0.01 of the minimum (the dashed circle) in at most 1,000 steps. Pick an optimizer and tune it.{' '}
          <strong>Then, optionally:</strong> do it in as few steps as possible. Par is computed by trying every optimizer with {LR_GRID.length} step sizes and a range of
          β settings ({(LR_GRID.length * (1 + 8 + 3 + 24)).toLocaleString()} runs in all).
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <Plot x={[GOLF.domain[0], GOLF.domain[1]]} y={[GOLF.domain[2], GOLF.domain[3]]} width={420} height={420} grid={false} tickLabels={false} ariaLabel="Banana landscape with your optimizer's path">
            <Heatmap f={GOLF.f} range={GOLF.range} transform={Math.sqrt} contours={16} />
            <Minima L={GOLF} />
            <Clip>
              <Path pts={r.path} color="#fff" upto={shown} />
            </Clip>
            <Dot at={GOLF.start} r={5} color="var(--accent)" stroke="#fff" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <Segmented label="Optimizer" value={name} onChange={setName} options={OPTS.map((o) => ({ value: o.key, label: o.label }))} />
          <Slider label="step size" value={logLr} min={-3} max={1} step={0.025} onChange={setLogLr} format={(v) => fmtLr(10 ** v)} color="c-grad" />
          {(name === 'momentum' || name === 'adam') && <Slider label="β₁ (how much velocity is kept)" value={b1} min={0} max={0.98} step={0.01} onChange={setB1} />}
          {(name === 'rmsprop' || name === 'adam') && (
            <Segmented
              label="β₂ (how long the memory of slope sizes is)"
              value={b2}
              onChange={setB2}
              options={[0.9, 0.99, 0.999].map((v) => ({ value: v, label: String(v) }))}
            />
          )}
          <div className="readout">
            <span className="k">steps     </span> <strong>{r.reachedAt ?? '—'}</strong>
            {'\n'}
            <span className="k">your best </span> {best ?? '—'}
            {'\n'}
            <span className="k">par       </span> {par.steps}
          </div>
          {verdict}
          <LiveCode code={torchOptimLine(cfg)} note="PyTorch torch.optim" />
          {best !== null && (
            <button className="btn small" onClick={() => setShowPar(!showPar)}>
              {showPar ? 'Hide' : 'Show'} the par configuration
            </button>
          )}
          {showPar && (
            <div className="muted mono" style={{ fontSize: 12.5 }}>
              {OPTS.find((o) => o.key === par.cfg.name)!.label}, step size {fmtLr(par.cfg.lr)}
              {par.cfg.beta1 !== undefined && `, β₁ = ${par.cfg.beta1}`}
              {par.cfg.beta2 !== undefined && `, β₂ = ${par.cfg.beta2}`}
            </div>
          )}
        </div>
      </div>
    </Widget>
  );
}
