// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import {
  TINY222_PRESETS,
  cloneTiny222Params,
  stepTiny222,
  traceTiny222,
  twinGap,
  type Tiny222Params,
  type Tiny222Preset,
  type TinyAct,
} from './tinyMlp';

type Selection =
  | { kind: 'h'; j: 0 | 1 }
  | { kind: 'y'; k: 0 | 1 }
  | { kind: 'x'; i: 0 | 1 }
  | { kind: 'w1'; i: 0 | 1; j: 0 | 1 }
  | { kind: 'w2'; j: 0 | 1; k: 0 | 1 };

const PRESET_OPTIONS: Array<{ value: Tiny222Preset['id']; label: string }> = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'cancelling', label: 'Tug-of-war' },
  { value: 'twins', label: 'Identical twins' },
  { value: 'saturated', label: 'Saturated tanh' },
];

const LR = 0.25;

const fmtS = (v: number, d = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;
const fmtU = (v: number, d = 3) => (Math.abs(v) < 5e-5 ? '0.000' : v.toFixed(d));

// SVG geometry for 2 -> 2 -> 2 -> L
const X_POS = { x: 52, h: 215, y: 388, L: 508 };
const Y_ROW: [number, number] = [84, 206];
const Y_MID = 145;

export function TinyMlpBackprop() {
  const [presetId, setPresetId] = useState<Tiny222Preset['id']>('balanced');
  const [act, setAct] = useState<TinyAct>('tanh');
  const [params, setParams] = useState<Tiny222Params>(() => cloneTiny222Params(TINY222_PRESETS.balanced.params));
  const [stepCount, setStepCount] = useState(0);
  const [sel, setSel] = useState<Selection>({ kind: 'h', j: 0 });

  const preset = TINY222_PRESETS[presetId];
  const tr = useMemo(() => traceTiny222(params, preset.x, preset.t, act), [params, preset.x, preset.t, act]);
  const gap = useMemo(() => twinGap(params), [params]);

  const choosePreset = (id: Tiny222Preset['id']) => {
    setPresetId(id);
    setAct(TINY222_PRESETS[id].defaultAct);
    setParams(cloneTiny222Params(TINY222_PRESETS[id].params));
    setStepCount(0);
    setSel({ kind: 'h', j: 0 });
  };

  const takeSteps = (n: number) => {
    setParams((cur) => {
      let next = cur;
      for (let i = 0; i < n; i++) next = stepTiny222(next, preset.x, preset.t, act, LR);
      return next;
    });
    setStepCount((s) => s + n);
  };

  const reset = () => {
    setParams(cloneTiny222Params(preset.params));
    setStepCount(0);
  };

  const w2EdgeActive = (j: 0 | 1, k: 0 | 1) =>
    (sel.kind === 'h' && sel.j === j) ||
    (sel.kind === 'y' && sel.k === k) ||
    (sel.kind === 'w2' && sel.j === j && sel.k === k);

  const w1EdgeActive = (i: 0 | 1, j: 0 | 1) =>
    (sel.kind === 'h' && sel.j === j) ||
    (sel.kind === 'x' && sel.i === i) ||
    (sel.kind === 'w1' && sel.i === i && sel.j === j);

  // Live contextual note
  let liveNote: React.ReactNode;
  if (presetId === 'twins') {
    liveNote = (
      <>
        <strong>Symmetry trap (step {stepCount}):</strong> max weight gap between <span className="mono">h₁</span> and{' '}
        <span className="mono">h₂</span> is <strong className="mono">{gap.toFixed(6)}</strong>. Because both hidden
        units start identical, they compute the exact same <span className="mono">h = {fmtU(tr.h[0], 3)}</span>, receive
        the exact same blame <span className="mono">dh = {fmtS(tr.dh[0], 3)}</span>, and take identical updates forever.
        Press <strong>Take 5 steps</strong>: loss falls, yet the twins never separate.
      </>
    );
  } else if (presetId === 'saturated') {
    const ratio = Math.abs(tr.dW1[0][0]) > 1e-12 ? Math.abs(tr.dW1[0][1]) / Math.abs(tr.dW1[0][0]) : Infinity;
    liveNote =
      act === 'tanh' ? (
        <>
          <strong>Saturated tanh bottleneck:</strong> both hidden units receive identical upstream blame (
          <span className="mono">dh₁ = dh₂ = {fmtS(tr.dh[0], 3)}</span>), yet <span className="mono">z₁ = {tr.z1[0].toFixed(2)}</span>{' '}
          sits on tanh’s flat shelf where <span className="mono">1 − h₁² = {tr.actDeriv[0].toFixed(4)}</span>. Incoming
          gradients to <span className="mono">h₁</span> are starved by{' '}
          <strong className="mono">{Number.isFinite(ratio) ? `${ratio.toFixed(0)}×` : '∞'}</strong> compared with active{' '}
          <span className="mono">h₂</span>. Switch the activation to <strong>ReLU</strong> to watch{' '}
          <span className="mono">h₁</span> wake back up.
        </>
      ) : (
        <>
          <strong>ReLU rescue:</strong> with ReLU, <span className="mono">z₁ = {tr.z1[0].toFixed(2)} &gt; 0</span> has
          local slope <strong className="mono">1.000</strong> instead of <span className="mono">0.0013</span>, so{' '}
          <span className="mono">h₁</span> gets the exact same learning signal as <span className="mono">h₂</span> (
          <span className="mono">dz₁ = {fmtS(tr.dz1[0], 3)}</span> vs <span className="mono">{fmtS(tr.dz1[1], 3)}</span>).
        </>
      );
  } else if (presetId === 'cancelling') {
    liveNote = (
      <>
        <strong>Multi-path tug-of-war:</strong> output <span className="mono">y₁</span> sends{' '}
        <span className="mono">{fmtS(tr.dhPaths[0][0], 3)}</span> back into <span className="mono">h₁</span> while{' '}
        <span className="mono">y₂</span> sends <span className="mono">{fmtS(tr.dhPaths[0][1], 3)}</span> — opposing
        pulls that cancel down to <strong className="mono">dh₁ = {fmtS(tr.dh[0], 3)}</strong>. Click{' '}
        <span className="mono">h₂</span> to compare: there both paths reinforce to{' '}
        <strong className="mono">dh₂ = {fmtS(tr.dh[1], 3)}</strong>.
      </>
    );
  } else {
    liveNote = (
      <>
        <strong>Click any node or edge</strong> to inspect its backward credit. Selecting{' '}
        <span className="mono">h₁</span> or <span className="mono">h₂</span> highlights the two outgoing weights it
        collects blame along — notice those exact two entries form one row of <span className="mono">W₂</span> and one
        column of <span className="mono">W₂ᵀ</span>.
      </>
    );
  }

  return (
    <Widget
      title="Two hidden neurons, two outputs: multi-path credit & why W₂ᵀ appears"
      wide
      caption={
        <>
          A complete <span className="mono">2 → 2 → 2</span> network with squared-error loss{' '}
          <span className="mono">L = ½((y₁−t₁)² + (y₂−t₂)²)</span>. Click <span className="mono">h₁</span> or{' '}
          <span className="mono">h₂</span> to see both backward paths add up and light up the matching column of{' '}
          <span className="mono">W₂ᵀ</span>. Try <strong>Tug-of-war</strong>, <strong>Identical twins</strong> (step
          several times), and <strong>Saturated tanh</strong> (then flip to ReLU).
        </>
      }
    >
      <div className="controls inline" style={{ gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <Segmented options={PRESET_OPTIONS} value={presetId} onChange={choosePreset} />
        <Segmented
          options={[
            { value: 'tanh' as const, label: 'tanh' },
            { value: 'relu' as const, label: 'ReLU' },
          ]}
          value={act}
          onChange={setAct}
        />
        <div className="btn-row" style={{ marginLeft: 'auto' }}>
          <button type="button" className="btn primary small" onClick={() => takeSteps(1)}>
            Take 1 step (η={LR})
          </button>
          <button type="button" className="btn small" onClick={() => takeSteps(5)}>
            +5 steps
          </button>
          <button type="button" className="btn small" onClick={reset} disabled={stepCount === 0}>
            Reset
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap', marginTop: 12 }}>
        {/* Left: clickable 2 -> 2 -> 2 -> L SVG */}
        <div style={{ flex: '1 1 360px', minWidth: 280 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
            <span className="mono">
              step {stepCount} · loss L = <strong style={{ color: 'var(--text)' }}>{tr.loss.toFixed(4)}</strong>
            </span>
            <span className="mono">
              targets t = [{preset.t[0].toFixed(2)}, {preset.t[1].toFixed(2)}]
            </span>
          </div>
          <svg
            viewBox="0 0 560 284"
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
              background: 'var(--bg-sunken)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
            }}
            role="img"
            aria-label="2-2-2 multi-layer perceptron credit assignment diagram"
          >
            {/* Layer headers */}
            <text x={X_POS.x} y={24} textAnchor="middle" fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
              input x
            </text>
            <text x={X_POS.h} y={24} textAnchor="middle" fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
              hidden h ({act})
            </text>
            <text x={X_POS.y} y={24} textAnchor="middle" fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
              output y
            </text>
            <text x={X_POS.L} y={24} textAnchor="middle" fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
              loss L
            </text>

            {/* W1 edges: x_i -> h_j */}
            {([0, 1] as const).map((i) =>
              ([0, 1] as const).map((j) => {
                const x1 = X_POS.x + 28;
                const y1 = Y_ROW[i];
                const x2 = X_POS.h - 40;
                const y2 = Y_ROW[j];
                const active = w1EdgeActive(i, j);
                const w = params.W1[i][j];
                const dw = tr.dW1[i][j];
                // Offset label along the line so crossing diagonals never collide
                const t = i === j ? 0.5 : i === 0 ? 0.34 : 0.34;
                const lx = x1 + (x2 - x1) * t;
                const ly = y1 + (y2 - y1) * t + (i === j ? (i === 0 ? -9 : 14) : i === 0 ? -7 : 13);
                return (
                  <g
                    key={`w1-${i}-${j}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSel({ kind: 'w1', i, j })}
                  >
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={active ? 'var(--c-grad)' : w >= 0 ? 'var(--c-a)' : 'var(--c-b)'}
                      strokeWidth={active ? 3 : 1.6}
                      strokeOpacity={active ? 1 : 0.45}
                    />
                    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" strokeWidth={14} />
                    <text
                      x={lx}
                      y={ly}
                      textAnchor="middle"
                      fontSize={10}
                      fontWeight={active ? 700 : 500}
                      fill={active ? 'var(--c-grad)' : 'var(--text-muted)'}
                      style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-sunken)', strokeWidth: 4 }}
                    >
                      {active ? `g ${fmtS(dw, 3)}` : fmtS(w, 2)}
                    </text>
                  </g>
                );
              }),
            )}

            {/* W2 edges: h_j -> y_k */}
            {([0, 1] as const).map((j) =>
              ([0, 1] as const).map((k) => {
                const x1 = X_POS.h + 40;
                const y1 = Y_ROW[j];
                const x2 = X_POS.y - 36;
                const y2 = Y_ROW[k];
                const active = w2EdgeActive(j, k);
                const w = params.W2[j][k];
                const pathVal = tr.dhPaths[j][k];
                const t = j === k ? 0.5 : 0.36;
                const lx = x1 + (x2 - x1) * t;
                const ly = y1 + (y2 - y1) * t + (j === k ? (j === 0 ? -9 : 14) : j === 0 ? -10 : 16);
                return (
                  <g
                    key={`w2-${j}-${k}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSel({ kind: 'w2', j, k })}
                  >
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={active ? 'var(--c-grad)' : w >= 0 ? 'var(--c-a)' : 'var(--c-b)'}
                      strokeWidth={active ? 3.2 : 1.7}
                      strokeOpacity={active ? 1 : 0.5}
                    />
                    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" strokeWidth={14} />
                    <text
                      x={lx}
                      y={ly}
                      textAnchor="middle"
                      fontSize={10.5}
                      fontWeight={active ? 700 : 500}
                      fill={active ? 'var(--c-grad)' : 'var(--text-muted)'}
                      style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-sunken)', strokeWidth: 4 }}
                    >
                      {active && sel.kind === 'h' ? `← ${fmtS(pathVal, 3)}` : `w ${fmtS(w, 2)}`}
                    </text>
                  </g>
                );
              }),
            )}

            {/* y_k -> L edges */}
            {([0, 1] as const).map((k) => (
              <g key={`yl-${k}`}>
                <line
                  x1={X_POS.y + 36}
                  y1={Y_ROW[k]}
                  x2={X_POS.L - 28}
                  y2={Y_MID}
                  stroke="var(--border-strong)"
                  strokeWidth={1.6}
                />
                <text
                  x={(X_POS.y + 36 + X_POS.L - 28) / 2}
                  y={(Y_ROW[k] + Y_MID) / 2 + (k === 0 ? -8 : 14)}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--c-grad)"
                  style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-sunken)', strokeWidth: 4 }}
                >
                  {fmtS(tr.dy[k], 2)}
                </text>
              </g>
            ))}

            {/* Input nodes x1, x2 */}
            {([0, 1] as const).map((i) => {
              const isSel = sel.kind === 'x' && sel.i === i;
              return (
                <g key={`x-${i}`} style={{ cursor: 'pointer' }} onClick={() => setSel({ kind: 'x', i })} data-node={`x-${i}`}>
                  <rect
                    x={X_POS.x - 28}
                    y={Y_ROW[i] - 24}
                    width={56}
                    height={48}
                    rx={8}
                    fill="var(--bg-elev)"
                    stroke={isSel ? 'var(--accent)' : 'var(--border-strong)'}
                    strokeWidth={isSel ? 2.5 : 1.3}
                  />
                  <text x={X_POS.x} y={Y_ROW[i] - 6} textAnchor="middle" fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                    x{i + 1}
                  </text>
                  <text x={X_POS.x} y={Y_ROW[i] + 12} textAnchor="middle" fontSize={12.5} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>
                    {preset.x[i].toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Hidden nodes h1, h2 */}
            {([0, 1] as const).map((j) => {
              const isSel = sel.kind === 'h' && sel.j === j;
              const starved = tr.actDeriv[j] < 0.02;
              return (
                <g key={`h-${j}`} style={{ cursor: 'pointer' }} onClick={() => setSel({ kind: 'h', j })} data-node={`h-${j}`}>
                  <rect
                    x={X_POS.h - 40}
                    y={Y_ROW[j] - 32}
                    width={80}
                    height={64}
                    rx={10}
                    fill="var(--bg-elev)"
                    stroke={isSel ? 'var(--c-grad)' : starved ? 'var(--c-warn)' : 'var(--accent)'}
                    strokeWidth={isSel ? 3 : 1.6}
                  />
                  <text x={X_POS.h} y={Y_ROW[j] - 15} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>
                    h{j + 1} = {fmtU(tr.h[j], 3)}
                  </text>
                  <text x={X_POS.h} y={Y_ROW[j] + 1} textAnchor="middle" fontSize={10} fill={starved ? 'var(--c-warn)' : 'var(--text-muted)'} style={{ fontFamily: 'var(--font-mono)' }}>
                    φ′ = {tr.actDeriv[j].toFixed(4)}
                  </text>
                  <text x={X_POS.h} y={Y_ROW[j] + 18} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--c-grad)" style={{ fontFamily: 'var(--font-mono)' }}>
                    dh {fmtS(tr.dh[j], 3)}
                  </text>
                </g>
              );
            })}

            {/* Output nodes y1, y2 */}
            {([0, 1] as const).map((k) => {
              const isSel = sel.kind === 'y' && sel.k === k;
              return (
                <g key={`y-${k}`} style={{ cursor: 'pointer' }} onClick={() => setSel({ kind: 'y', k })} data-node={`y-${k}`}>
                  <rect
                    x={X_POS.y - 36}
                    y={Y_ROW[k] - 28}
                    width={72}
                    height={56}
                    rx={9}
                    fill="var(--bg-elev)"
                    stroke={isSel ? 'var(--c-grad)' : 'var(--c-c)'}
                    strokeWidth={isSel ? 2.8 : 1.5}
                  />
                  <text x={X_POS.y} y={Y_ROW[k] - 10} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>
                    y{k + 1} = {fmtU(tr.y[k], 2)}
                  </text>
                  <text x={X_POS.y} y={Y_ROW[k] + 5} textAnchor="middle" fontSize={9.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                    t{k + 1} = {preset.t[k].toFixed(2)}
                  </text>
                  <text x={X_POS.y} y={Y_ROW[k] + 20} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--c-grad)" style={{ fontFamily: 'var(--font-mono)' }}>
                    dy {fmtS(tr.dy[k], 2)}
                  </text>
                </g>
              );
            })}

            {/* Loss node L */}
            <g>
              <rect
                x={X_POS.L - 28}
                y={Y_MID - 26}
                width={56}
                height={52}
                rx={10}
                fill="var(--bg-elev)"
                stroke="var(--c-grad)"
                strokeWidth={2}
              />
              <text x={X_POS.L} y={Y_MID - 6} textAnchor="middle" fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                Loss
              </text>
              <text x={X_POS.L} y={Y_MID + 12} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>
                {tr.loss.toFixed(3)}
              </text>
            </g>
          </svg>
        </div>

        {/* Right: Scalar inspector + Matrix W2 / W2^T correspondence */}
        <div style={{ flex: '1 1 260px', minWidth: 250, display: 'flex', flexDirection: 'column', gap: 10 }}>
          {/* Selected credit arithmetic */}
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              fontSize: 12.5,
            }}
          >
            {sel.kind === 'h' && (
              <>
                <div className="mono" style={{ fontWeight: 700, color: 'var(--c-grad)', marginBottom: 6 }}>
                  Hidden neuron h{sel.j + 1} — fan-in sum
                </div>
                <div className="mono" style={{ fontSize: 12, lineHeight: 1.65 }}>
                  <div>
                    via y₁: ({fmtS(tr.dy[0], 3)}) × ({fmtS(params.W2[sel.j][0], 2)}) ={' '}
                    <strong>{fmtS(tr.dhPaths[sel.j][0], 4)}</strong>
                  </div>
                  <div>
                    via y₂: ({fmtS(tr.dy[1], 3)}) × ({fmtS(params.W2[sel.j][1], 2)}) ={' '}
                    <strong>{fmtS(tr.dhPaths[sel.j][1], 4)}</strong>
                  </div>
                  <div style={{ borderTop: '1px solid var(--border)', marginTop: 4, paddingTop: 4 }}>
                    ∂L/∂h{sel.j + 1} = <strong style={{ color: 'var(--c-grad)' }}>{fmtS(tr.dh[sel.j], 4)}</strong>
                  </div>
                  <div>
                    × φ′(z{sel.j + 1}={tr.z1[sel.j].toFixed(2)}) = {tr.actDeriv[sel.j].toFixed(4)} →{' '}
                    <strong>dz{sel.j + 1} = {fmtS(tr.dz1[sel.j], 4)}</strong>
                  </div>
                  <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
                    dW₁[1,{sel.j + 1}] = {fmtS(tr.dW1[0][sel.j], 4)} · dW₁[2,{sel.j + 1}] = {fmtS(tr.dW1[1][sel.j], 4)}
                  </div>
                </div>
              </>
            )}
            {sel.kind === 'y' && (
              <>
                <div className="mono" style={{ fontWeight: 700, color: 'var(--c-c)', marginBottom: 6 }}>
                  Output neuron y{sel.k + 1}
                </div>
                <div className="mono" style={{ fontSize: 12, lineHeight: 1.65 }}>
                  <div>
                    y{sel.k + 1} = h₁·W₂[1,{sel.k + 1}] + h₂·W₂[2,{sel.k + 1}] + b₂ = {tr.y[sel.k].toFixed(3)}
                  </div>
                  <div>
                    ∂L/∂y{sel.k + 1} = y{sel.k + 1} − t{sel.k + 1} ={' '}
                    <strong style={{ color: 'var(--c-grad)' }}>{fmtS(tr.dy[sel.k], 4)}</strong>
                  </div>
                  <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
                    Forward reads column {sel.k + 1} of W₂; backward sends dy{sel.k + 1} back along those two edges.
                  </div>
                </div>
              </>
            )}
            {sel.kind === 'x' && (
              <>
                <div className="mono" style={{ fontWeight: 700, color: 'var(--accent)', marginBottom: 6 }}>
                  Input x{sel.i + 1} = {preset.x[sel.i].toFixed(2)}
                </div>
                <div className="mono" style={{ fontSize: 12, lineHeight: 1.65 }}>
                  <div>
                    ∂L/∂x{sel.i + 1} = dz₁·W₁[{sel.i + 1},1] + dz₂·W₁[{sel.i + 1},2] ={' '}
                    <strong style={{ color: 'var(--c-grad)' }}>{fmtS(tr.dx[sel.i], 4)}</strong>
                  </div>
                  <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
                    Inputs aren’t updated during training, but the same fan-in rule computes input saliency.
                  </div>
                </div>
              </>
            )}
            {sel.kind === 'w1' && (
              <>
                <div className="mono" style={{ fontWeight: 700, color: 'var(--c-grad)', marginBottom: 6 }}>
                  Hidden weight W₁[{sel.i + 1},{sel.j + 1}] (x{sel.i + 1} → h{sel.j + 1})
                </div>
                <div className="mono" style={{ fontSize: 12, lineHeight: 1.65 }}>
                  <div>
                    grad = x{sel.i + 1} × dz{sel.j + 1} = ({preset.x[sel.i].toFixed(2)}) × ({fmtS(tr.dz1[sel.j], 4)}) ={' '}
                    <strong>{fmtS(tr.dW1[sel.i][sel.j], 4)}</strong>
                  </div>
                  <div>
                    next w = {fmtS(params.W1[sel.i][sel.j], 3)} − {LR}·({fmtS(tr.dW1[sel.i][sel.j], 4)}) ={' '}
                    <strong>{fmtS(params.W1[sel.i][sel.j] - LR * tr.dW1[sel.i][sel.j], 3)}</strong>
                  </div>
                </div>
              </>
            )}
            {sel.kind === 'w2' && (
              <>
                <div className="mono" style={{ fontWeight: 700, color: 'var(--c-grad)', marginBottom: 6 }}>
                  Output weight W₂[{sel.j + 1},{sel.k + 1}] (h{sel.j + 1} → y{sel.k + 1})
                </div>
                <div className="mono" style={{ fontSize: 12, lineHeight: 1.65 }}>
                  <div>
                    grad = h{sel.j + 1} × dy{sel.k + 1} = ({tr.h[sel.j].toFixed(3)}) × ({fmtS(tr.dy[sel.k], 3)}) ={' '}
                    <strong>{fmtS(tr.dW2[sel.j][sel.k], 4)}</strong>
                  </div>
                  <div>
                    blame routed to h{sel.j + 1} = dy{sel.k + 1} × w ={' '}
                    <strong>{fmtS(tr.dhPaths[sel.j][sel.k], 4)}</strong>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Matrix view: W2 (forward) vs W2^T (backward) */}
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              fontSize: 12,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 6 }}>
              Matrix view: <span className="mono">dh = dy @ W₂ᵀ</span>
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>
                  W₂ (forward: h @ W₂)
                </div>
                <table className="num-table">
                  <thead>
                    <tr>
                      <th />
                      <th style={{ textTransform: 'none' }}>→y₁</th>
                      <th style={{ textTransform: 'none' }}>→y₂</th>
                    </tr>
                  </thead>
                  <tbody>
                    {([0, 1] as const).map((j) => (
                      <tr key={j}>
                        <th className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', paddingRight: 6, textTransform: 'none' }}>
                          h{j + 1}
                        </th>
                        {([0, 1] as const).map((k) => {
                          const hl = w2EdgeActive(j, k);
                          return (
                            <td
                              key={k}
                              style={{
                                whiteSpace: 'nowrap',
                                background: hl ? 'color-mix(in srgb, var(--c-grad) 22%, transparent)' : undefined,
                                fontWeight: hl ? 700 : 400,
                              }}
                            >
                              {fmtS(params.W2[j][k], 2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--c-grad)', marginBottom: 3 }}>
                  W₂ᵀ (backward: dy @ W₂ᵀ)
                </div>
                <table className="num-table">
                  <thead>
                    <tr>
                      <th />
                      <th style={{ textTransform: 'none' }}>→h₁</th>
                      <th style={{ textTransform: 'none' }}>→h₂</th>
                    </tr>
                  </thead>
                  <tbody>
                    {([0, 1] as const).map((k) => (
                      <tr key={k}>
                        <th className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', paddingRight: 6, textTransform: 'none' }}>
                          dy{k + 1}
                        </th>
                        {([0, 1] as const).map((j) => {
                          const hl = w2EdgeActive(j, k);
                          return (
                            <td
                              key={j}
                              style={{
                                whiteSpace: 'nowrap',
                                background: hl ? 'color-mix(in srgb, var(--c-grad) 22%, transparent)' : undefined,
                                fontWeight: hl ? 700 : 400,
                              }}
                            >
                              {fmtS(params.W2[j][k], 2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 6 }}>
              Forward collects into <span className="mono">y_k</span> down a column of <span className="mono">W₂</span>;
              backward collects into <span className="mono">h_j</span> down a column of{' '}
              <span className="mono">W₂ᵀ</span> — the exact same weights, flipped from “by destination” to “by source”.
            </div>
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 12 }}>
        {liveNote}
      </div>
    </Widget>
  );
}
