import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Value } from '../../lib/autodiff';

function human(seconds: number) {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 120) return `${seconds.toFixed(1)} s`;
  if (seconds < 7200) return `${(seconds / 60).toFixed(0)} min`;
  if (seconds < 172800) return `${(seconds / 3600).toFixed(0)} hours`;
  if (seconds < 3.15e7 * 2) return `${(seconds / 86400).toFixed(0)} days`;
  return `${(seconds / 3.15e7).toLocaleString(undefined, { maximumFractionDigits: 0 })} years`;
}

export function BackpropCost() {
  const [logN, setLogN] = useState(6);
  const [fwd, setFwd] = useState(0.1);
  const N = 10 ** logN;
  const nudge = 2 * N * fwd; // two forward passes per parameter
  const back = 3 * fwd; // forward + backward ≈ 3 forward passes
  const maxLog = Math.log10(Math.max(nudge, back)) + 1;
  const bar = (v: number) => `${Math.max(2, (100 * (Math.log10(v) + 3)) / (maxLog + 3))}%`;
  return (
    <Widget
      title="Why going backwards matters"
      caption={
        <>
          Two ways to get every weight’s gradient. <strong>Nudging</strong>: change one weight a tiny bit, rerun the model, see how the loss moved — twice per weight.{' '}
          <strong>Backpropagation</strong>: one forward pass and one backward pass, which together cost roughly three forward passes, regardless of how many weights there
          are. Bars are on a logarithmic scale.
        </>
      }
    >
      <div className="controls inline">
        <Slider label="number of weights" value={logN} min={1} max={12} step={0.1} onChange={setLogN} format={() => N.toExponential(1).replace('e+', ' × 10^')} />
        <Slider label="time for one forward pass (s)" value={fwd} min={0.001} max={1} step={0.001} onChange={setFwd} format={(v) => v.toFixed(3)} />
      </div>
      <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '150px 1fr auto', gap: '10px 12px', alignItems: 'center', fontSize: 14 }}>
        <span>Nudging each weight</span>
        <div style={{ height: 18, background: 'var(--bg-sunken)', borderRadius: 4 }}>
          <div style={{ width: bar(nudge), height: '100%', background: 'var(--c-b)', borderRadius: 4 }} />
        </div>
        <strong className="mono">{human(nudge)}</strong>
        <span>Backpropagation</span>
        <div style={{ height: 18, background: 'var(--bg-sunken)', borderRadius: 4 }}>
          <div style={{ width: bar(back), height: '100%', background: 'var(--c-c)', borderRadius: 4 }} />
        </div>
        <strong className="mono">{human(back)}</strong>
      </div>
      <p className="muted" style={{ fontSize: 13.5, marginBottom: 0 }}>
        …and that’s the cost of <em>one</em> training step. Training takes many thousands of them. Without backpropagation, a modern model would be untrainable; with it,
        the gradient costs about the same as running the model a few times.
      </p>
    </Widget>
  );
}

type Act = 'sigmoid' | 'tanh' | 'relu';

export function fadingGradients(depth: number, act: Act, w: number): number[] {
  const ws = Array.from({ length: depth }, (_, i) => new Value(w, `w${i}`));
  let h: Value = new Value(1);
  for (const wi of ws) {
    const z = wi.mul(h);
    h = act === 'sigmoid' ? z.sigmoid() : act === 'tanh' ? z.tanh() : z.relu();
  }
  h.backward();
  return ws.map((v) => Math.abs(v.grad));
}

export function FadingSignal() {
  const [depth, setDepth] = useState(12);
  const [act, setAct] = useState<Act>('sigmoid');
  const [w, setW] = useState(1);
  const grads = useMemo(() => fadingGradients(depth, act, w), [depth, act, w]);
  const lo = -14;
  const hi = 8;
  const H = 180;
  const W = 640;
  const toY = (g: number) => {
    const l = g > 0 ? Math.log10(g) : lo;
    return H - ((Math.max(lo, Math.min(hi, l)) - lo) / (hi - lo)) * H;
  };
  const first = grads[0];
  const last = grads[grads.length - 1];
  const ratio = last > 0 ? first / last : 0;

  let note: React.ReactNode;
  if (first === 0 && last === 0) note = <>Every gradient is exactly zero: a “dead” network. With ReLU and a negative weight, every neuron outputs 0 and passes back nothing.</>;
  else if (ratio < 1e-3)
    note = (
      <>
        <strong>Vanishing.</strong> The first layer’s gradient is about {ratio.toExponential(0)} times the last layer’s. Each layer multiplies the signal by a local slope below 1,
        so the early layers barely learn.{act === 'sigmoid' && ' Sigmoid’s slope is never more than 0.25, which makes this almost unavoidable in deep sigmoid networks.'}
      </>
    );
  else if (ratio > 1e3)
    note = (
      <>
        <strong>Exploding.</strong> The first layer’s gradient is {ratio.toExponential(0)} times the last layer’s. Each layer multiplies by more than 1, and early layers get
        enormous, unstable updates.
      </>
    );
  else
    note = (
      <>
        <strong>Healthy.</strong> Gradients stay within a factor of {Math.max(ratio, 1 / ratio).toFixed(1)} from the first layer to the last, so every layer gets a useful
        learning signal.{act === 'relu' && Math.abs(w - 1) < 0.05 && ' ReLU with weight 1 passes the signal back perfectly unchanged.'}
      </>
    );

  return (
    <Widget
      title="The fading signal"
      wide
      caption={
        <>
          A chain of {depth} single neurons, each multiplying by a weight and applying an activation. The bars show how strongly the loss responds to each layer’s weight —
          the learning signal that reaches it — on a <strong>logarithmic</strong> scale (each grid line is 100× the one below). Make the chain deeper, change the activation
          and the weight, and watch the early layers starve or explode.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="Activation" value={act} onChange={setAct} options={[{ value: 'sigmoid', label: 'Sigmoid' }, { value: 'tanh', label: 'Tanh' }, { value: 'relu', label: 'ReLU' }]} />
        <Slider label="depth (layers)" value={depth} min={2} max={30} step={1} onChange={setDepth} format={(v) => String(v)} />
        <Slider label="weight in every layer" value={w} min={-1} max={3} step={0.05} onChange={setW} />
      </div>
      <svg viewBox={`0 -10 ${W} ${H + 34}`} width={W} height={H + 34} role="img" aria-label="Gradient size at each layer, log scale">
        {Array.from({ length: (hi - lo) / 2 + 1 }, (_, i) => lo + i * 2).map((l) => (
          <g key={l}>
            <line x1={40} x2={W} y1={toY(10 ** l)} y2={toY(10 ** l)} stroke="var(--c-grid)" />
            <text x={34} y={toY(10 ** l) + 4} textAnchor="end" fontSize={10} fill="var(--text-faint)" style={{ fontFamily: 'var(--font-mono)' }}>
              1e{l}
            </text>
          </g>
        ))}
        {grads.map((g, i) => {
          const bw = (W - 50) / depth;
          const y = toY(g);
          return (
            <g key={i}>
              <rect x={46 + i * bw} y={y} width={Math.max(2, bw - 3)} height={H - y} fill="var(--c-grad)" opacity={0.75} rx={2} />
              {(depth <= 15 || i % 5 === 0) && (
                <text x={46 + i * bw + bw / 2} y={H + 16} textAnchor="middle" fontSize={10} fill="var(--text-faint)" style={{ fontFamily: 'var(--font-ui)' }}>
                  {i + 1}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="muted" style={{ fontSize: 12.5, margin: '2px 0 8px' }}>layer (1 = closest to the input)</div>
      <div className="live-note">{note}</div>
    </Widget>
  );
}
