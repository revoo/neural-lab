// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { LogChart } from '../../components/LogChart';
import {
  DEFAULT_STEPPER_SEQ,
  evalRnnBrief1,
  evalRnnBrief2,
  evalRnnBrief3,
  getStepperParams,
  LAB_CLIP_LRS,
  LAB_CLIP_NORMS,
  LAB_FLOW_ACTS,
  LAB_FLOW_RHOS,
  LAB_TBPTT_HS,
  LAB_TBPTT_KS,
  LAG_BENCH_LAGS,
  rnnLabExhaustive,
  rnnLossAndGrad,
  runLagBenchmark,
  runPascanuCliff,
  spectralGradientFlow,
  type RnnBriefId,
  type StepperPreset,
} from './rnn';

const fmtSci = (v: number) => {
  if (!Number.isFinite(v)) return '∞';
  const a = Math.abs(v);
  if (a === 0) return '0.000';
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  return v.toFixed(3);
};

// ------------------------------------------------------------------ 1. Unrolled RNN Stepper (Forward + BPTT)

export function UnrolledRnnWidget() {
  const [preset, setPreset] = useState<StepperPreset>('leaky');
  const [seq, setSeq] = useState<number[]>(DEFAULT_STEPPER_SEQ);
  const [target, setTarget] = useState<0 | 1>(1);

  const params = useMemo(() => getStepperParams(preset), [preset]);
  const res = useMemo(() => rnnLossAndGrad(params, seq, target), [params, seq, target]);

  const cycleToken = (idx: number) => {
    setSeq((prev) => prev.map((v, i) => (i === idx ? (v === 1 ? 0 : v === 0 ? -1 : 1) : v)));
  };

  return (
    <Widget
      title="Unroll the RNN in time: forward hidden state & backward BPTT gradient"
      wide
      caption={
        <>
          Click any input token <code>xₜ</code> to cycle <code>+1 / 0 / −1</code>, and switch between <strong>Leaky</strong> (<code>W_h = 0.55</code>) and <strong>Long Memory</strong> (<code>W_h = 0.95</code>). Notice how the exact same cell runs 6 times — and how <strong>BPTT</strong> flows right-to-left: with <code>W_h = 0.55</code>, step 1’s gradient shrinks <strong>50×</strong> (<code>−1.539 → −0.031</code>)!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Recurrent weight W_h"
          value={preset}
          onChange={setPreset}
          options={[
            { value: 'leaky', label: 'Leaky (W_h = 0.55)' },
            { value: 'long', label: 'Long memory (W_h = 0.95)' },
          ]}
        />
        <Segmented
          label="Final target y₆"
          value={String(target)}
          onChange={(v) => setTarget(v === '1' ? 1 : 0)}
          options={[
            { value: '1', label: 'y₆ = +1 (positive)' },
            { value: '0', label: 'y₆ = 0 (negative)' },
          ]}
        />
        <button className="btn" type="button" onClick={() => setSeq(DEFAULT_STEPPER_SEQ)}>
          Reset sequence
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(105px, 1fr))', gap: 8, marginBottom: 12 }}>
        {seq.map((xt, t) => {
          const ht = res.trace.hs[t][0];
          const dht = res.grads.dhTrace[t][0];
          const mag = Math.min(1, Math.abs(dht) / Math.max(0.01, Math.abs(res.grads.dhTrace[5][0])));
          return (
            <div
              key={t}
              style={{
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: '8px 10px',
                background: 'var(--bg-subtle)',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
                <strong style={{ fontFamily: 'var(--mono)' }}>t = {t + 1}</strong>
                <button
                  type="button"
                  onClick={() => cycleToken(t)}
                  title="Click to cycle input x_t"
                  style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 12,
                    fontWeight: 700,
                    padding: '2px 7px',
                    borderRadius: 6,
                    border: '1px solid var(--border)',
                    cursor: 'pointer',
                    background: xt > 0 ? 'var(--c-good)' : xt < 0 ? 'var(--c-bad)' : 'var(--bg-elev)',
                    color: xt === 0 ? 'var(--text)' : 'var(--bg)',
                  }}
                >
                  x={xt > 0 ? '+1' : xt < 0 ? '−1' : '0'}
                </button>
              </div>
              <div style={{ fontSize: 12, fontFamily: 'var(--mono)' }}>
                <span style={{ color: 'var(--text-muted)' }}>hₜ: </span>
                <span style={{ color: ht >= 0 ? 'var(--c-a)' : 'var(--c-b)', fontWeight: 600 }}>
                  {ht >= 0 ? '+' : ''}
                  {ht.toFixed(3)}
                </span>
              </div>
              <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-muted)' }}>
                ← ∂L/∂hₜ:{' '}
                <strong style={{ color: 'var(--accent)' }}>{dht.toFixed(3)}</strong>
              </div>
              <div style={{ height: 5, borderRadius: 3, background: 'var(--border)', overflow: 'hidden' }}>
                <div style={{ width: `${(100 * mag).toFixed(1)}%`, height: '100%', background: 'var(--accent)' }} />
              </div>
            </div>
          );
        })}
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
        <div className="readout">
          <div className="readout-label">Final prediction P(y=1)</div>
          <div className="readout-value">{(100 * res.trace.probs[5]).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">BCE Loss at t = 6</div>
          <div className="readout-value">{res.loss.toFixed(3)}</div>
        </div>
        <div className="readout">
          <div className="readout-label">Step-1 / Step-6 grad</div>
          <div className="readout-value" style={{ color: Math.abs(res.grads.dhTrace[0][0]) / Math.abs(res.grads.dhTrace[5][0]) < 0.1 ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {(100 * Math.abs(res.grads.dhTrace[0][0]) / Math.max(1e-9, Math.abs(res.grads.dhTrace[5][0]))).toFixed(1)}%
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Summed ∂L/∂W_h</div>
          <div className="readout-value">{res.grads.dWh[0].toFixed(3)}</div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Spectral Radius Vanishing / Exploding Flow

export function SpectralFlowWidget() {
  const [rho, setRho] = useState<number>(0.8);
  const [T, setT] = useState<number>(20);

  const linFlow = useMemo(() => spectralGradientFlow(rho, T, 'linear', 0.2), [rho, T]);
  const tanhFlow = useMemo(() => spectralGradientFlow(rho, T, 'tanh', 0.2), [rho, T]);

  const series = useMemo(
    () => [
      {
        points: linFlow.points.map((p): [number, number] => [p.stepsBack, Math.max(1e-8, Math.min(1e5, p.gradRatio))]),
        color: 'var(--c-a)',
        dashed: true,
        label: 'Linear (ρ^k exact)',
      },
      {
        points: tanhFlow.points.map((p): [number, number] => [p.stepsBack, Math.max(1e-8, Math.min(1e5, p.gradRatio))]),
        color: 'var(--accent)',
        label: 'tanh RNN',
      },
    ],
    [linFlow, tanhFlow],
  );

  const xTicks = useMemo(() => {
    const step = T <= 15 ? 3 : 5;
    const out: number[] = [];
    for (let k = 0; k <= T - 1; k += step) out.push(k);
    if (out[out.length - 1] !== T - 1) out.push(T - 1);
    return out;
  }, [T]);

  return (
    <Widget
      title="Vanishing & exploding gradients through time: the ρᵏ law"
      wide
      caption={
        <>
          Exact Jacobian norm <code>‖∂h_T / ∂h_(T−k)‖</code> walking <code>k</code> steps backwards through an orthogonal recurrent matrix scaled by spectral radius <strong>ρ</strong>. Slide <strong>ρ</strong>: below <code>1.0</code> gradients vanish exponentially (<code>0.5¹⁹ = 1.9×10⁻⁶</code>); above <code>1.0</code> linear gradients explode (<code>1.6¹⁹ = 7,556</code>); even at <code>ρ = 1.0</code>, <code>tanh</code> shrinks because <code>1 − h² &lt; 1</code>!
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 10 }}>
        <Slider label="Spectral radius ρ = ‖W_h‖₂" value={rho} min={0.5} max={1.6} step={0.05} onChange={setRho} format={(v) => `ρ = ${v.toFixed(2)}`} />
        <Slider label="Sequence length T" value={T} min={10} max={30} step={5} onChange={setT} format={(v) => `${v} steps`} />
      </div>

      <LogChart
        series={series}
        xDomain={[0, T - 1]}
        yDomain={[1e-4, 1e4]}
        width={760}
        height={230}
        xTicks={xTicks}
        xLabel="Steps backwards k = T − t"
        ariaLabel="BPTT gradient magnitude vs steps backwards on log scale"
        hlines={[{ y: 1, label: 'ideal = 1.0', color: 'var(--text-muted)' }]}
      />

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, margin: '6px 0 10px', color: 'var(--text-muted)' }}>
        <span><span style={{ color: 'var(--c-a)', fontWeight: 700 }}>---</span> Linear (exact ρᵏ)</span>
        <span><span style={{ color: 'var(--accent)', fontWeight: 700 }}>━━</span> tanh RNN (diag(1−h²)·W_hᵀ)</span>
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8, margin: 0 }}>
        <div className="readout">
          <div className="readout-label">Linear at t = 1</div>
          <div className="readout-value" style={{ color: linFlow.step1Ratio < 0.05 || linFlow.step1Ratio > 20 ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {fmtSci(linFlow.step1Ratio)}
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">tanh at t = 1</div>
          <div className="readout-value" style={{ color: tanhFlow.step1Ratio < 0.05 || tanhFlow.step1Ratio > 20 ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {fmtSci(tanhFlow.step1Ratio)}
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Regime</div>
          <div className="readout-value">
            {rho < 0.95 ? 'Vanishing' : rho > 1.05 ? 'Exploding / Saturated' : 'Critical (ρ ≈ 1)'}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Pascanu Cliff & Lag Degradation

export function CliffAndLagWidget() {
  const [mode, setMode] = useState<'cliff' | 'lag'>('cliff');
  const [clipOn, setClipOn] = useState<boolean>(true);
  const [lr, setLr] = useState<number>(0.015);

  const unclipped = useMemo(() => runPascanuCliff(lr, null, 45, 1.32), [lr]);
  const clipped = useMemo(() => runPascanuCliff(lr, 1.0, 45, 1.32), [lr]);
  const lagBench = useMemo(() => LAG_BENCH_LAGS.map((lag) => runLagBenchmark(lag)), []);

  const active = clipOn ? clipped : unclipped;

  return (
    <Widget
      title="Two faces of BPTT: rescuing exploding cliffs vs losing long-range memory"
      wide
      caption={
        mode === 'cliff' ? (
          <>
            <strong>Pascanu’s (2013) recurrent cliff:</strong> starting at <code>w = 1.32</code> on a 16-step chain, the gradient hits <strong>93.9</strong>! Without clipping, one SGD step catapults <code>w</code> off the cliff into the vanishing-gradient desert (<code>w = −0.09</code>, stuck at loss <code>0.211</code> — or exploding to <code>−∞</code> at <code>lr = 0.05</code>). Turn <strong>Gradient clipping</strong> on to glide calmly to the exact root <code>w* = 1.1865</code>!
          </>
        ) : (
          <>
            <strong>Why clipping alone isn’t enough:</strong> even with gradient clipping on, a plain <code>tanh</code> RNN trained with SGD on delayed recall (<code>noise = ±1.0</code>, averaged across 3 fixed seeds) drops from <strong>100.0%</strong> at lag 1–3 to <strong>53.3%</strong> at lag 10 and <strong>48.3%</strong> at lag 15 — motivating LSTMs & GRUs next!
          </>
        )
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Experiment"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'cliff', label: 'Exploding cliff + clipping' },
            { value: 'lag', label: 'Long-lag memory decay (3 seeds)' },
          ]}
        />
        {mode === 'cliff' && (
          <>
            <Segmented
              label="Gradient norm clipping"
              value={clipOn ? 'on' : 'off'}
              onChange={(v) => setClipOn(v === 'on')}
              options={[
                { value: 'off', label: 'Off (unclipped SGD)' },
                { value: 'on', label: 'On (maxNorm = 1.0)' },
              ]}
            />
            <Segmented
              label="Step size (lr)"
              value={lr}
              onChange={setLr}
              options={[
                { value: 0.015, label: '0.015' },
                { value: 0.025, label: '0.025' },
                { value: 0.05, label: '0.050' },
              ]}
            />
          </>
        )}
      </div>

      {mode === 'cliff' ? (
        <>
          <Plot x={[0, 44]} y={[-1.2, 1.5]} width={760} height={210} xLabel="SGD step" yLabel="Recurrent weight w">
            {({ x: sx, y: sy }) => (
              <>
                <line x1={sx(0)} x2={sx(44)} y1={sy(1.1865)} y2={sy(1.1865)} stroke="var(--c-good)" strokeDasharray="5 4" strokeWidth={1.5} />
                <text x={sx(43)} y={sy(1.1865) - 6} textAnchor="end" fontSize={11} fill="var(--c-good)" fontFamily="var(--mono)">
                  exact w* = 1.1865
                </text>
                <polyline
                  points={unclipped.trace.map((p) => `${sx(p.step)},${sy(Math.max(-1.15, Math.min(1.45, p.w)))}`).join(' ')}
                  fill="none"
                  stroke="var(--c-bad)"
                  strokeWidth={clipOn ? 1.5 : 2.5}
                  strokeDasharray={clipOn ? '4 4' : undefined}
                  opacity={clipOn ? 0.55 : 1}
                />
                <polyline
                  points={clipped.trace.map((p) => `${sx(p.step)},${sy(Math.max(-1.15, Math.min(1.45, p.w)))}`).join(' ')}
                  fill="none"
                  stroke="var(--c-good)"
                  strokeWidth={clipOn ? 2.5 : 1.5}
                  strokeDasharray={clipOn ? undefined : '4 4'}
                  opacity={clipOn ? 1 : 0.45}
                />
              </>
            )}
          </Plot>

          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: '10px 0 0' }}>
            <div className="readout">
              <div className="readout-label">Peak raw |∂L/∂w|</div>
              <div className="readout-value" style={{ color: 'var(--c-b)' }}>{active.maxRawGrad.toFixed(1)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Final weight w</div>
              <div className="readout-value">{active.exploded ? '−∞ (overflow)' : active.finalW.toFixed(4)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Final MSE loss</div>
              <div className="readout-value" style={{ color: !active.exploded && active.finalLoss < 1e-4 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {active.exploded ? 'EXPLODED' : fmtSci(active.finalLoss)}
              </div>
            </div>
          </div>
        </>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {lagBench.map((row) => (
            <div key={row.lag} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 70px', alignItems: 'center', gap: 10, fontSize: 13 }}>
              <div style={{ fontFamily: 'var(--mono)' }}>Lag L = {row.lag}</div>
              <div style={{ height: 18, borderRadius: 9, background: 'var(--bg-subtle)', border: '1px solid var(--border)', overflow: 'hidden', position: 'relative' }}>
                <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--text-muted)', opacity: 0.45 }} />
                <div
                  style={{
                    width: `${(100 * row.meanAcc).toFixed(1)}%`,
                    height: '100%',
                    background: row.meanAcc >= 0.85 ? 'var(--c-good)' : row.meanAcc >= 0.65 ? 'var(--c-b)' : 'var(--c-bad)',
                  }}
                />
              </div>
              <div style={{ fontFamily: 'var(--mono)', fontWeight: 700, textAlign: 'right' }}>{(100 * row.meanAcc).toFixed(1)}%</div>
            </div>
          ))}
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            Vertical centre mark = 50% coin-flip chance. Plain <code>tanh</code> RNN (<code>H = 4</code>, SGD + clipping, 3 seeds) nails lags 1–3, dips to 90.0% at lag 6, and collapses to coin-flip chance by lag 10–15.
          </div>
        </div>
      )}
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tame BPTT

type RnnLabState = {
  b1Act: 'tanh' | 'linear';
  b1Rho: number;
  b2Lr: number;
  b2Clip: number | null;
  b3K: number;
  b3H: number;
  solved: Record<RnnBriefId, boolean>;
};

const RNN_LAB_KEY = 'nl-rnn-lab-v1';
const RNN_LAB_DEFAULT: RnnLabState = {
  b1Act: 'tanh',
  b1Rho: 0.5,
  b2Lr: 0.025,
  b2Clip: null,
  b3K: 2,
  b3H: 4,
  solved: { flow: false, clip: false, tbptt: false },
};

function loadRnnLab(): RnnLabState {
  try {
    const raw = localStorage.getItem(RNN_LAB_KEY);
    if (!raw) return RNN_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...RNN_LAB_DEFAULT, ...p, solved: { ...RNN_LAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return RNN_LAB_DEFAULT;
  }
}

export function RnnLab() {
  const [briefId, setBriefId] = useState<RnnBriefId>('flow');
  const [st, setSt] = useState<RnnLabState>(loadRnnLab);
  const [revealed, setRevealed] = useState<Record<RnnBriefId, boolean>>({ flow: false, clip: false, tbptt: false });

  useEffect(() => {
    try {
      localStorage.setItem(RNN_LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalRnnBrief1(st.b1Rho, st.b1Act), [st.b1Rho, st.b1Act]);
  const r1Trace = useMemo(() => spectralGradientFlow(st.b1Rho, 20, st.b1Act, 0.2), [st.b1Rho, st.b1Act]);
  const r2 = useMemo(() => evalRnnBrief2(st.b2Lr, st.b2Clip), [st.b2Lr, st.b2Clip]);
  const r2Trace = useMemo(() => runPascanuCliff(st.b2Lr, st.b2Clip, 45, 1.32), [st.b2Lr, st.b2Clip]);
  const r3 = useMemo(() => evalRnnBrief3(st.b3K, st.b3H), [st.b3K, st.b3H]);
  const exhaust = useMemo(() => rnnLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'flow' ? r1.pass : briefId === 'clip' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const solvedCount = Object.values(st.solved).filter(Boolean).length;
  const canReveal = st.solved[briefId] || revealed[briefId];

  return (
    <Widget
      title="Lab: Tame BPTT"
      wide
      caption={
        <>
          Three real RNN debugging jobs — keep a 20-step gradient alive, cross Pascanu’s exploding cliff with clipping, and size a Truncated BPTT window under an activation-RAM cap. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
        {(
          [
            ['flow', 'Brief 1 · Signal Flow'],
            ['clip', 'Brief 2 · Exploding Cliff'],
            ['tbptt', 'Brief 3 · TBPTT RAM'],
          ] as [RnnBriefId, string][]
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={`btn${briefId === id ? ' primary' : ''}`}
            onClick={() => setBriefId(id)}
          >
            {label} {st.solved[id] ? '✓' : ''}
          </button>
        ))}
      </div>

      {briefId === 'flow' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> Your 20-step sensor RNN receives almost zero gradient at step 1. Pick activation and orthogonal init spectral radius <code>ρ</code> so the step-1 Jacobian ratio stays healthy in <strong>[0.10, 10.0]</strong>.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Activation"
              value={st.b1Act}
              onChange={(v) => setSt((p) => ({ ...p, b1Act: v }))}
              options={LAB_FLOW_ACTS.map((a) => ({ value: a, label: a }))}
            />
            <Segmented
              label="Init spectral radius"
              value={st.b1Rho}
              onChange={(v) => setSt((p) => ({ ...p, b1Rho: v }))}
              options={LAB_FLOW_RHOS.map((r) => ({ value: r, label: `ρ = ${r}` }))}
            />
          </div>
          <LogChart
            series={[
              {
                points: r1Trace.points.map((p): [number, number] => [p.stepsBack, Math.max(1e-4, Math.min(1e4, p.gradRatio))]),
                color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)',
              },
            ]}
            xDomain={[0, 19]}
            yDomain={[1e-4, 1e4]}
            width={760}
            xTicks={[0, 5, 10, 15, 19]}
            height={175}
            xLabel="Steps back k"
            ariaLabel="Lab 1 gradient ratio across 20 steps"
            hlines={[
              { y: 10, label: 'max 10', color: 'var(--text-muted)' },
              { y: 0.1, label: 'min 0.1', color: 'var(--text-muted)' },
            ]}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Step-1 ratio</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{fmtSci(r1.ratio)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Target band</div>
              <div className="readout-value">0.10 – 10.0</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'clip' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> A 16-step recurrent run starts on Pascanu’s cliff (<code>w = 1.32</code>, peak slope <code>93.9</code>). Unclipped SGD either gets catapulted into the flat desert or blows up to <code>−∞</code>. Configure step size and <code>clip_grad_norm_</code> to hit MSE loss <strong>≤ 0.0001</strong> with zero blowups.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Learning rate (lr)"
              value={st.b2Lr}
              onChange={(v) => setSt((p) => ({ ...p, b2Lr: v }))}
              options={LAB_CLIP_LRS.map((v) => ({ value: v, label: `${v}` }))}
            />
            <Segmented
              label="max_norm clip"
              value={st.b2Clip === null ? 'none' : String(st.b2Clip)}
              onChange={(v) => setSt((p) => ({ ...p, b2Clip: v === 'none' ? null : Number(v) }))}
              options={LAB_CLIP_NORMS.map((c) => ({ value: c === null ? 'none' : String(c), label: c === null ? 'None' : `${c}` }))}
            />
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 12, padding: '6px 10px', borderRadius: 6, background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
            {st.b2Clip === null
              ? `# No gradient clipping — SGD(lr=${st.b2Lr})`
              : `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=${st.b2Clip})  # lr=${st.b2Lr}`}
          </div>
          <Plot x={[0, 44]} y={[-1.2, 1.5]} width={760} height={175} xLabel="SGD step" yLabel="Weight w">
            {({ x: sx, y: sy }) => (
              <>
                <line x1={sx(0)} x2={sx(44)} y1={sy(1.1865)} y2={sy(1.1865)} stroke="var(--c-good)" strokeDasharray="4 4" strokeWidth={1.2} />
                <polyline
                  points={r2Trace.trace.map((p) => `${sx(p.step)},${sy(Math.max(-1.15, Math.min(1.45, p.w)))}`).join(' ')}
                  fill="none"
                  stroke={r2.pass ? 'var(--c-good)' : 'var(--c-bad)'}
                  strokeWidth={2.4}
                />
              </>
            )}
          </Plot>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Final w (target 1.1865)</div>
              <div className="readout-value">{r2.exploded ? '−∞' : r2.finalW.toFixed(4)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Final loss (≤ 1e−4)</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.exploded ? 'EXPLODED' : fmtSci(r2.finalLoss)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'tbptt' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> Your edge device only fits <strong>≤ 20</strong> stored hidden-activation slots (<code>K × H</code>) for Truncated BPTT on a 7-step delayed-recall sequence (<code>t = 0…6</code>). Pick truncation horizon <code>K</code> and hidden width <code>H</code> to reach <strong>≥ 95%</strong> test accuracy and loss <strong>≤ 0.10</strong> within the 20-slot budget!
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Truncation horizon K"
              value={st.b3K}
              onChange={(v) => setSt((p) => ({ ...p, b3K: v }))}
              options={LAB_TBPTT_KS.map((k) => ({ value: k, label: `K = ${k}` }))}
            />
            <Segmented
              label="Hidden width H"
              value={st.b3H}
              onChange={(v) => setSt((p) => ({ ...p, b3H: v }))}
              options={LAB_TBPTT_HS.map((h) => ({ value: h, label: `H = ${h}` }))}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 6 }}>
            {[0, 1, 2, 3, 4, 5, 6].map((t) => {
              const active = t >= 7 - st.b3K;
              return (
                <div
                  key={t}
                  style={{
                    textAlign: 'center',
                    padding: '6px 4px',
                    borderRadius: 6,
                    fontFamily: 'var(--mono)',
                    fontSize: 11.5,
                    border: '1px solid var(--border)',
                    background: active ? 'var(--bg-elev)' : 'var(--bg-subtle)',
                    color: active ? (t === 0 ? 'var(--c-good)' : 'var(--accent)') : 'var(--text-muted)',
                    opacity: active ? 1 : 0.45,
                  }}
                >
                  <div>t={t}</div>
                  <div style={{ fontSize: 10 }}>{t === 0 ? (active ? 'CUE ✓' : 'CUT ✗') : active ? 'grad' : '0'}</div>
                </div>
              );
            })}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Test acc (≥ 95%)</div>
              <div className="readout-value" style={{ color: r3.testAcc >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>{(100 * r3.testAcc).toFixed(1)}%</div>
            </div>
            <div className="readout">
              <div className="readout-label">Final loss (≤ 0.10)</div>
              <div className="readout-value" style={{ color: r3.finalLoss <= 0.10 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.finalLoss.toFixed(3)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">RAM slots (≤ 20)</div>
              <div className="readout-value" style={{ color: r3.memSlots <= 20 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.memSlots}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r3.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: 14, display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
        {!canReveal ? (
          <button
            type="button"
            className="btn"
            onClick={() => setRevealed((p) => ({ ...p, [briefId]: true }))}
          >
            Stuck? Reveal exhaustive search ({exhaust.total} configs)
          </button>
        ) : (
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            <strong>Exhaustive search ({exhaust.total} configs):</strong> {exhaust.passingCount}/{exhaust.total} pass · Best: <code>{exhaust.bestLabel}</code>
          </div>
        )}
      </div>
    </Widget>
  );
}
