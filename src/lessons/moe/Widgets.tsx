// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { CLUSTER_CENTRES, DEFAULTS, TEST, train, traceToken, type Params, type TrainSnapshot } from './moe';
import { countParams, flopsPerToken, MIXTRAL, weightGB } from './sizes';

export const EXPERT_COLOURS = ['var(--c-a)', 'var(--c-b)', 'var(--c-c)', 'var(--c-grad)'];
const LIM = 3.3;

/** The test tokens on the plane, each dot coloured by its first-choice expert. */
export function TokenMap({ params, k, size = 300, selected, onPick }: { params: Params; k: 1 | 2; size?: number; selected?: [number, number] | null; onPick?: (x: number, y: number) => void }) {
  const sx = (v: number) => ((v + LIM) / (2 * LIM)) * size;
  const sy = (v: number) => size - ((v + LIM) / (2 * LIM)) * size;
  const dots = useMemo(() => {
    const out: { x: number; y: number; e: number; y01: number }[] = [];
    for (let i = 0; i < TEST.n; i += 2) {
      const t = traceToken(params, 4, k, TEST.x[2 * i], TEST.x[2 * i + 1]);
      out.push({ x: TEST.x[2 * i], y: TEST.x[2 * i + 1], e: t.chosen[0], y01: TEST.y[i] });
    }
    return out;
  }, [params, k]);
  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      style={{ width: '100%', maxWidth: size, display: 'block', cursor: onPick ? 'crosshair' : undefined, background: 'var(--bg-sunken)', borderRadius: 8 }}
      role="img"
      aria-label="Tokens on the plane, coloured by the expert the router picks first"
      onClick={(ev) => {
        if (!onPick) return;
        const r = (ev.currentTarget as SVGSVGElement).getBoundingClientRect();
        const px = ((ev.clientX - r.left) / r.width) * size;
        const py = ((ev.clientY - r.top) / r.height) * size;
        onPick((px / size) * 2 * LIM - LIM, ((size - py) / size) * 2 * LIM - LIM);
      }}
    >
      <line x1={sx(0)} x2={sx(0)} y1={0} y2={size} stroke="var(--border)" />
      <line y1={sy(0)} y2={sy(0)} x1={0} x2={size} stroke="var(--border)" />
      {dots.map((d, i) =>
        d.y01 ? (
          <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={3} fill={EXPERT_COLOURS[d.e]} opacity={0.85} />
        ) : (
          <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={2.6} fill="none" stroke={EXPERT_COLOURS[d.e]} strokeWidth={1.3} opacity={0.85} />
        ),
      )}
      {CLUSTER_CENTRES.map(([cx, cy], c) => (
        <text key={c} className="plot-tick" x={sx(cx) + (cx > 0 ? 34 : -34)} y={sy(cy) + (cy > 0 ? -34 : 40)} textAnchor="middle">
          cluster {c + 1}
        </text>
      ))}
      {selected && <circle cx={sx(selected[0])} cy={sy(selected[1])} r={7} fill="none" stroke="var(--text)" strokeWidth={2} />}
    </svg>
  );
}

export function ExpertKey({ load }: { load?: number[] }) {
  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 12.5, margin: '6px 0' }}>
      {EXPERT_COLOURS.map((c, e) => (
        <span key={e} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 10, height: 10, borderRadius: 5, background: c, display: 'inline-block' }} />
          expert {e + 1}
          {load && <span className="mono"> {(100 * load[e]).toFixed(1)}%</span>}
        </span>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ router explorer

export function RouterExplorer() {
  const [k, setK] = useState<1 | 2>(2);
  const model = useMemo(() => train({ ...DEFAULTS, alpha: 0.1 }, 1), []);
  const [pt, setPt] = useState<[number, number]>([1.3, 1.9]);
  const t = traceToken(model.params, 4, k, pt[0], pt[1]);
  const fmt = (v: number) => (v >= 0 ? ' ' : '') + v.toFixed(2);
  const prob = 1 / (1 + Math.exp(-t.z));
  return (
    <Widget
      title="Inside the router: scores, softmax, top-k, blend"
      wide
      caption={
        <>
          A real 4-expert layer, trained on this page when it loaded (load balancing on, seed 1). Dots are test tokens, coloured by the expert the router picks first;
          filled dots have label 1, rings label 0. Click anywhere on the plane to send a token through the layer, and switch between top-1 and top-2. Try a token on the
          border between two clusters.
        </>
      }
    >
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 260px', maxWidth: 320 }}>
          <TokenMap params={model.params} k={k} selected={pt} onPick={(x, y) => setPt([x, y])} />
          <ExpertKey />
        </div>
        <div style={{ flex: '1 1 300px', minWidth: 0 }}>
          <Segmented label="Experts per token" options={[{ value: 1, label: 'top-1' }, { value: 2, label: 'top-2' }]} value={k} onChange={(v) => setK(v as 1 | 2)} />
          <div style={{ overflowX: 'auto', marginTop: 8 }}>
            <table className="num-table" style={{ fontSize: 12.5 }}>
              <thead>
                <tr>
                  <th>expert</th>
                  <th>router score</th>
                  <th>softmax</th>
                  <th>runs?</th>
                  <th>expert output</th>
                  <th>gate × output</th>
                </tr>
              </thead>
              <tbody>
                {t.scores.map((s, e) => {
                  const on = t.chosen.includes(e);
                  return (
                    <tr key={e} style={on ? { background: 'var(--accent-soft)' } : { color: 'var(--text-faint)' }}>
                      <td>
                        <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 5, background: EXPERT_COLOURS[e], marginRight: 5 }} />
                        {e + 1}
                      </td>
                      <td className="mono">{fmt(s)}</td>
                      <td className="mono">{(100 * t.probs[e]).toFixed(1)}%</td>
                      <td>{on ? '✓' : '—'}</td>
                      <td className="mono">{on ? fmt(t.expertOut[e]) : 'skipped'}</td>
                      <td className="mono">{on ? fmt(t.probs[e] * t.expertOut[e]) : ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="live-note">
            Token at ({pt[0].toFixed(2)}, {pt[1].toFixed(2)}). The router’s top {k === 1 ? 'choice is expert' : 'two choices are experts'}{' '}
            {t.chosen.map((e) => e + 1).join(' and ')}, with {t.chosen.map((e) => `${(100 * t.probs[e]).toFixed(1)}%`).join(' and ')} of the probability. Their gated
            outputs add up to {t.z.toFixed(2)}, so the layer says label 1 with probability {(100 * prob).toFixed(1)}%. {4 - k} of the 4 experts never ran for this
            token: {k === 1 ? 'three quarters' : 'half'} of the layer’s expert weights sat idle.
          </p>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ collapse, live

function LoadChart({ history, upto }: { history: TrainSnapshot[]; upto: number }) {
  const W = 420;
  const H = 150;
  const shown = history.slice(0, upto);
  const bw = W / history.length;
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} style={{ width: '100%', maxWidth: W, display: 'block' }} role="img" aria-label="Each expert's share of the tokens over training">
      {shown.map((h, i) => {
        let y0 = H;
        return (
          <g key={i}>
            {h.load.map((l, e) => {
              const hgt = l * H;
              y0 -= hgt;
              return <rect key={e} x={i * bw} y={y0} width={bw + 0.3} height={hgt} fill={EXPERT_COLOURS[e]} opacity={0.85} />;
            })}
          </g>
        );
      })}
      <line x1={0} x2={W} y1={H * 0.75} y2={H * 0.75} stroke="var(--text-faint)" strokeDasharray="3 3" />
      <line x1={0} x2={W} y1={H * 0.5} y2={H * 0.5} stroke="var(--text-faint)" strokeDasharray="3 3" />
      <line x1={0} x2={W} y1={H * 0.25} y2={H * 0.25} stroke="var(--text-faint)" strokeDasharray="3 3" />
      <text className="plot-tick" x={0} y={H + 14}>
        step 10
      </text>
      <text className="plot-tick" x={W} y={H + 14} textAnchor="end">
        step {history.at(-1)?.step ?? 0}
      </text>
      <text className="plot-tick" x={W / 2} y={H + 14} textAnchor="middle">
        dashed lines: equal shares
      </text>
    </svg>
  );
}

export function CollapseDemo() {
  const [alpha, setAlpha] = useState(0);
  const [seed, setSeed] = useState(1);
  const run = useMemo(() => train({ ...DEFAULTS, alpha }, seed, 10), [alpha, seed]);
  const [frame, setFrame] = useState(run.history.length);
  const timer = useRef<number | null>(null);
  useEffect(() => setFrame(run.history.length), [run]);
  useEffect(() => () => void (timer.current && cancelAnimationFrame(timer.current)), []);
  const play = () => {
    let f = 0;
    const tick = () => {
      f += 1;
      setFrame(f);
      if (f < run.history.length) timer.current = requestAnimationFrame(tick);
    };
    tick();
  };
  const h = run.history[Math.max(0, frame - 1)];
  const idle = h.load.filter((l) => l === 0).length;
  const busiest = Math.max(...h.load);
  return (
    <Widget
      title="Rich get richer: watch the router starve an expert"
      wide
      caption={
        <>
          The same 4-expert layer, trained live with top-1 routing (600 steps, mini-batches of 64). The bars show each expert’s share of the test tokens every 10
          steps. Train with the balancing loss off, then on, and try several seeds. Press Replay to watch it happen.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <Segmented label="Balancing loss" options={[{ value: 0, label: 'off (α = 0)' }, { value: 0.1, label: 'on (α = 0.1)' }]} value={alpha} onChange={(v) => setAlpha(v as number)} />
        <Segmented label="Seed" options={[1, 2, 3, 4, 5]} value={seed} onChange={(v) => setSeed(v as number)} />
        <button className="btn" onClick={play}>
          ▶ Replay
        </button>
      </div>
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 300px', maxWidth: 440 }}>
          <LoadChart history={run.history} upto={frame} />
          <ExpertKey load={h.load} />
        </div>
        <div style={{ flex: '1 1 220px', maxWidth: 280 }}>
          <TokenMap params={run.params} k={1} size={260} />
        </div>
      </div>
      <p className="live-note">
        Step {h.step}: the busiest expert takes {(100 * busiest).toFixed(1)}% of the tokens
        {idle ? `, and ${idle === 1 ? 'one expert gets' : `${idle} experts get`} none at all` : ', and every expert has work'}. Test accuracy {(100 * h.acc).toFixed(1)}%.
        {alpha === 0 && idle > 0 && frame >= run.history.length
          ? ' The idle expert’s weights still take up memory, but it learns nothing: no token reaches it, so no gradient does either.'
          : ''}
        {alpha > 0 && frame >= run.history.length ? ' The balancing loss charges the router for piling probability onto busy experts, so every expert ends up with a share.' : ''}
      </p>
    </Widget>
  );
}

// ------------------------------------------------------------------ memory vs compute

const fmtB = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : `${(n / 1e6).toFixed(0)}M`);

export function SizeCalculator() {
  const [experts, setExperts] = useState(8);
  const [topK, setTopK] = useState(2);
  const k = Math.min(topK, experts);
  const c = countParams({ ...MIXTRAL, experts, topK: k });
  const dense = countParams({ ...MIXTRAL, experts: 1, topK: 1 });
  const bars = [
    { label: 'weights held in memory', v: c.total, colour: 'var(--c-b)' },
    { label: 'weights used per token', v: c.active, colour: 'var(--c-a)' },
  ];
  const max = Math.max(c.total, 1);
  return (
    <Widget
      title="Memory versus compute: Mixtral’s shape, your experts"
      caption={
        <>
          The architecture of Mixtral 8x7B (32 layers of width 4,096; each expert a SwiGLU block with 176 million weights), with the number of experts and the number
          used per token under your control. Every expert must sit in GPU memory; only the chosen ones do arithmetic. Start at 8 and 2, Mixtral’s own setting.
        </>
      }
    >
      <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fit, minmax(min(240px, 100%), 1fr))' }}>
        <Slider label="Experts per layer" value={experts} min={1} max={64} step={1} onChange={(v) => setExperts(Math.round(v))} format={(v) => String(Math.round(v))} />
        <Slider label="Experts used per token" value={k} min={1} max={Math.min(8, experts)} step={1} onChange={(v) => setTopK(Math.round(v))} format={(v) => String(Math.round(v))} />
      </div>
      <div style={{ marginTop: 10 }}>
        {bars.map((b) => (
          <div key={b.label} style={{ marginBottom: 6 }}>
            <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              {b.label}: <strong className="mono">{fmtB(b.v)}</strong>
            </div>
            <div style={{ background: 'var(--bg-sunken)', borderRadius: 4, height: 14 }}>
              <div style={{ width: `${(100 * b.v) / max}%`, background: b.colour, height: 14, borderRadius: 4 }} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="num-table" style={{ fontSize: 12.5 }}>
          <tbody>
            <tr>
              <td>memory for the weights (2 bytes each)</td>
              <td className="mono">{weightGB(c.total).toFixed(1)} GB</td>
            </tr>
            <tr>
              <td>arithmetic per token (≈ 2 per active weight)</td>
              <td className="mono">{(flopsPerToken(c.active) / 1e9).toFixed(1)} GFLOP</td>
            </tr>
            <tr>
              <td>share of weights that work on a given token</td>
              <td className="mono">{((100 * c.active) / c.total).toFixed(0)}%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="live-note">
        {experts === 1
          ? `With a single expert this is an ordinary dense model of ${fmtB(dense.total)} weights: everything held is everything used.`
          : `This model holds ${fmtB(c.total)} weights (${weightGB(c.total).toFixed(1)} GB) but each token touches only ${fmtB(c.active)}, so it does the arithmetic of a ${fmtB(c.active)} dense model while needing the memory of a ${fmtB(c.total)} one. ${k === 1 ? 'Routing to one expert halves the expert arithmetic of top-2, for the same memory.' : ''}`}
      </p>
    </Widget>
  );
}
