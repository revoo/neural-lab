// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { LogChart } from '../../components/LogChart';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useReplay, useTweenedArray } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
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
  rnnLabHint,
  rnnLossAndGrad,
  runLagBenchmark,
  runPascanuCliff,
  spectralGradientFlow,
  type RnnBriefId,
  type StepperPreset,
} from './rnn';
import { rnnLabRealCode } from './realCode';

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
};

const RNN_LAB_KEY = 'nl-rnn-lab-v1';
const RNN_LAB_DEFAULT: RnnLabState = {
  b1Act: 'tanh',
  b1Rho: 0.5,
  b2Lr: 0.025,
  b2Clip: null,
  b3K: 2,
  b3H: 4,
};

function readLegacyRnnSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(RNN_LAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved && typeof p.solved === 'object' ? p.solved : {};
  } catch {
    return {};
  }
}

const RNN_BRIEFS: { id: RnnBriefId; tab: string; title: string; text: string }[] = [
  {
    id: 'flow',
    tab: 'Brief 1 · Signal Flow',
    title: 'Keep a 20-Step Gradient Alive',
    text: 'Your 20-step sensor RNN receives almost zero gradient at step 1. Pick activation and orthogonal init spectral radius ρ so the step-1 Jacobian ratio stays healthy in [0.10, 10.0].',
  },
  {
    id: 'clip',
    tab: 'Brief 2 · Exploding Cliff',
    title: 'Cross Pascanu’s Exploding Cliff',
    text: 'A 16-step recurrent run starts on Pascanu’s cliff (w = 1.32, peak slope 93.9). Unclipped SGD gets catapulted into the flat desert or blows up to −∞. Configure learning rate and clip_grad_norm_ to reach MSE loss ≤ 0.0001 with zero blowups.',
  },
  {
    id: 'tbptt',
    tab: 'Brief 3 · TBPTT RAM',
    title: 'Size Truncated BPTT Under a 20-Slot Cap',
    text: 'Your edge device holds at most 20 stored hidden-activation slots (K × H) on a 7-step delayed-recall sequence (t = 0…6). Pick truncation horizon K and hidden width H to reach ≥ 95% test accuracy and loss ≤ 0.10 within the 20-slot budget.',
  },
];

export function RnnLab() {
  const [active, setActive] = useState(0);
  const [st, setSt] = useState<RnnLabState>(RNN_LAB_DEFAULT);
  const progress = useLabProgress('nl-rnn-lab-v2', () => ({ solved: readLegacyRnnSolved() }));
  const log = useMachineLog(12);

  const briefId = RNN_BRIEFS[active].id;

  const r1 = useMemo(() => evalRnnBrief1(st.b1Rho, st.b1Act), [st.b1Rho, st.b1Act]);
  const r1Trace = useMemo(() => spectralGradientFlow(st.b1Rho, 20, st.b1Act, 0.2), [st.b1Rho, st.b1Act]);
  const r2 = useMemo(() => evalRnnBrief2(st.b2Lr, st.b2Clip), [st.b2Lr, st.b2Clip]);
  const r2Trace = useMemo(() => runPascanuCliff(st.b2Lr, st.b2Clip, 45, 1.32), [st.b2Lr, st.b2Clip]);
  const r3 = useMemo(() => evalRnnBrief3(st.b3K, st.b3H), [st.b3K, st.b3H]);
  const exhaust = useMemo(() => rnnLabExhaustive(briefId), [briefId]);

  const maxFrame = Math.max(0, r2Trace.trace.length - 1);
  const cliffReplay = useReplay(maxFrame, 90, `${st.b2Lr}-${st.b2Clip}`, false);
  const frameIdx = Math.min(cliffReplay.i, maxFrame);
  const curCliffPt = r2Trace.trace[frameIdx] ?? r2Trace.trace[0];
  const visibleCliffTrace = r2Trace.trace.slice(0, frameIdx + 1);
  const wasClippedNow =
    st.b2Clip !== null && curCliffPt && Math.abs(curCliffPt.rawGrad) > Math.abs(curCliffPt.clippedGrad) + 1e-9;

  // Tweened meters: [b1LogPos (0..100), b3AccPct (0..100), b3RamPct (0..100)]
  const b1LogClamped = Math.max(-4, Math.min(4, Math.log10(Math.max(1e-12, r1.ratio))));
  const b1GaugePct = ((b1LogClamped + 4) / 8) * 100;
  const tweened = useTweenedArray([
    b1GaugePct,
    r3.testAcc * 100,
    Math.min(100, (r3.memSlots / 40) * 100),
  ]);

  const curPass = briefId === 'flow' ? r1.pass : briefId === 'clip' ? r2.pass : r3.pass;
  const curOptimal =
    briefId === 'flow'
      ? r1.pass && r1.logDist < 0.01
      : briefId === 'clip'
        ? r2.pass && r2.finalLoss < 1e-20
        : r3.pass && r3.memSlots === 14;

  const hintText = useMemo(() => rnnLabHint(briefId, st), [briefId, st]);
  const liveCode = useMemo(() => rnnLabRealCode(briefId, st), [briefId, st]);

  const gridB1 = useMemo(
    () => LAB_FLOW_ACTS.map((act) => ({ act, cells: LAB_FLOW_RHOS.map((rho) => evalRnnBrief1(rho, act)) })),
    [],
  );
  const gridB2 = useMemo(
    () => LAB_CLIP_LRS.map((lr) => ({ lr, cells: LAB_CLIP_NORMS.map((clip) => evalRnnBrief2(lr, clip)) })),
    [],
  );
  const gridB3 = useMemo(
    () => LAB_TBPTT_KS.map((K) => ({ K, cells: LAB_TBPTT_HS.map((H) => evalRnnBrief3(K, H)) })),
    [],
  );

  const updateB1 = (nextAct: 'tanh' | 'linear', nextRho: number) => {
    setSt((p) => ({ ...p, b1Act: nextAct, b1Rho: nextRho }));
    const res = evalRnnBrief1(nextRho, nextAct);
    log.push(
      `B1: ${nextAct} rho=${nextRho} -> step-1 ratio ${fmtSci(res.ratio)} (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const updateB2 = (nextLr: number, nextClip: number | null) => {
    setSt((p) => ({ ...p, b2Lr: nextLr, b2Clip: nextClip }));
    const res = evalRnnBrief2(nextLr, nextClip);
    log.push(
      `B2: lr=${nextLr} clip=${nextClip ?? 'none'} -> ${res.exploded ? 'EXPLODED' : `loss ${fmtSci(res.finalLoss)}, w=${res.finalW.toFixed(4)}`} (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const updateB3 = (nextK: number, nextH: number) => {
    setSt((p) => ({ ...p, b3K: nextK, b3H: nextH }));
    const res = evalRnnBrief3(nextK, nextH);
    log.push(
      `B3: K=${nextK} H=${nextH} -> ${res.memSlots} slots, acc ${(100 * res.testAcc).toFixed(0)}%, loss ${res.finalLoss.toFixed(3)} (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const resetBrief = () => {
    if (briefId === 'flow') setSt((p) => ({ ...p, b1Act: RNN_LAB_DEFAULT.b1Act, b1Rho: RNN_LAB_DEFAULT.b1Rho }));
    else if (briefId === 'clip') setSt((p) => ({ ...p, b2Lr: RNN_LAB_DEFAULT.b2Lr, b2Clip: RNN_LAB_DEFAULT.b2Clip }));
    else setSt((p) => ({ ...p, b3K: RNN_LAB_DEFAULT.b3K, b3H: RNN_LAB_DEFAULT.b3H }));
    log.push(`Reset ${RNN_BRIEFS[active].tab}`, 'info');
  };

  const statusText =
    briefId === 'flow'
      ? `Step-1 Jacobian ratio ${fmtSci(r1.ratio)} (target 0.10 – 10.0) · activation ${st.b1Act}, ρ = ${st.b1Rho}.`
      : briefId === 'clip'
        ? `Final w = ${r2.exploded ? '−∞' : r2.finalW.toFixed(4)} (target 1.1865) · loss ${r2.exploded ? 'EXPLODED' : fmtSci(r2.finalLoss)} (≤ 1e−4).`
        : `Test acc ${(100 * r3.testAcc).toFixed(1)}% (≥ 95%) · loss ${r3.finalLoss.toFixed(3)} (≤ 0.10) · RAM ${r3.memSlots}/20 slots.`;

  const revealContent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
        <strong>Exhaustive search ({exhaust.total} configs):</strong> {exhaust.passingCount}/{exhaust.total} pass · Best: <code>{exhaust.bestLabel}</code>
      </div>
      {briefId === 'flow' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>act \ ρ</th>
                {LAB_FLOW_RHOS.map((rho) => (
                  <th key={rho} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>{rho}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB1.map((row) => (
                <tr key={row.act}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>{row.act}</td>
                  {row.cells.map((c) => (
                    <td
                      key={c.rho}
                      style={{
                        textAlign: 'center',
                        padding: '4px 6px',
                        borderRadius: 4,
                        color: c.pass ? 'var(--c-good)' : 'var(--text-muted)',
                        fontWeight: c.pass ? 700 : 400,
                        background: c.pass ? 'color-mix(in srgb, var(--c-good) 12%, transparent)' : 'transparent',
                      }}
                    >
                      {fmtSci(c.ratio)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {briefId === 'clip' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>lr \ clip</th>
                {LAB_CLIP_NORMS.map((c) => (
                  <th key={String(c)} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>
                    {c === null ? 'None' : c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB2.map((row) => (
                <tr key={row.lr}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>{row.lr}</td>
                  {row.cells.map((c) => (
                    <td
                      key={String(c.clipNorm)}
                      style={{
                        textAlign: 'center',
                        padding: '4px 6px',
                        borderRadius: 4,
                        color: c.pass ? 'var(--c-good)' : c.exploded ? 'var(--c-bad)' : 'var(--text-muted)',
                        fontWeight: c.pass ? 700 : 400,
                        background: c.pass ? 'color-mix(in srgb, var(--c-good) 12%, transparent)' : 'transparent',
                      }}
                    >
                      {c.exploded ? 'EXPLODED' : fmtSci(c.finalLoss)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {briefId === 'tbptt' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>K \ H</th>
                {LAB_TBPTT_HS.map((h) => (
                  <th key={h} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>H = {h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB3.map((row) => (
                <tr key={row.K}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>K = {row.K}</td>
                  {row.cells.map((c) => (
                    <td
                      key={c.H}
                      style={{
                        textAlign: 'center',
                        padding: '4px 6px',
                        borderRadius: 4,
                        color: c.pass ? 'var(--c-good)' : 'var(--text-muted)',
                        fontWeight: c.pass ? 700 : 400,
                        background: c.pass ? 'color-mix(in srgb, var(--c-good) 12%, transparent)' : 'transparent',
                      }}
                    >
                      {c.memSlots} slots · {(100 * c.testAcc).toFixed(0)}% · L={c.finalLoss.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  return (
    <LabShell
      title="Lab: Tame BPTT"
      caption="Keep a 20-step gradient alive, step through Pascanu’s exploding cliff with clipping, and size a Truncated BPTT window under a 20-slot activation-RAM cap."
      briefs={RNN_BRIEFS}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={curPass}
      optimal={curOptimal}
      status={statusText}
      statusTone={curPass ? 'good' : 'neutral'}
      hint={hintText}
      reveal={{ label: `exhaustive search (${exhaust.total} configs)`, content: revealContent }}
      log={{
        title: 'BPTT BENCH LOG',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Adjust spectral radius, gradient clipping, or TBPTT window to log runs.',
      }}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {briefId === 'flow' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Activation"
              value={st.b1Act}
              onChange={(v) => updateB1(v, st.b1Rho)}
              options={LAB_FLOW_ACTS.map((a) => ({ value: a, label: a }))}
            />
            <Segmented
              label="Init spectral radius"
              value={st.b1Rho}
              onChange={(v) => updateB1(st.b1Act, v)}
              options={LAB_FLOW_RHOS.map((r) => ({ value: r, label: `ρ = ${r}` }))}
            />
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <LogChart
              series={[
                {
                  points: r1Trace.points.map((p): [number, number] => [p.stepsBack, Math.max(1e-4, Math.min(1e4, p.gradRatio))]),
                  color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)',
                },
              ]}
              xDomain={[0, 19]}
              yDomain={[1e-4, 1e4]}
              width={740}
              xTicks={[0, 5, 10, 15, 19]}
              height={170}
              xLabel="Steps back k"
              ariaLabel="Lab 1 gradient ratio across 20 steps"
              hlines={[
                { y: 10, label: 'max 10', color: 'var(--text-muted)' },
                { y: 0.1, label: 'min 0.1', color: 'var(--text-muted)' },
              ]}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, fontFamily: 'var(--mono)', color: 'var(--text-muted)' }}>
                <span>10⁻⁴ (vanished)</span>
                <span style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
                  Step-1 ratio: {fmtSci(r1.ratio)} (healthy band: 0.10 – 10.0)
                </span>
                <span>10⁴ (exploded)</span>
              </div>
              <div style={{ position: 'relative', height: 10, borderRadius: 5, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                <div
                  style={{
                    position: 'absolute',
                    left: '37.5%',
                    width: '25%',
                    top: 0,
                    bottom: 0,
                    background: 'color-mix(in srgb, var(--c-good) 24%, transparent)',
                    borderLeft: '1px dashed var(--c-good)',
                    borderRight: '1px dashed var(--c-good)',
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    left: `calc(${Math.max(2, Math.min(98, tweened[0]))}% - 5px)`,
                    top: -2,
                    width: 10,
                    height: 12,
                    borderRadius: 3,
                    background: r1.pass ? 'var(--c-good)' : 'var(--c-bad)',
                  }}
                />
              </div>
            </div>
          </div>
          <LiveCode note="PyTorch nn.RNN orthogonal init" code={liveCode} />
        </div>
      )}

      {briefId === 'clip' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Learning rate (lr)"
              value={st.b2Lr}
              onChange={(v) => updateB2(v, st.b2Clip)}
              options={LAB_CLIP_LRS.map((v) => ({ value: v, label: `${v}` }))}
            />
            <Segmented
              label="max_norm clip"
              value={st.b2Clip === null ? 'none' : String(st.b2Clip)}
              onChange={(v) => updateB2(st.b2Lr, v === 'none' ? null : Number(v))}
              options={LAB_CLIP_NORMS.map((c) => ({ value: c === null ? 'none' : String(c), label: c === null ? 'None' : `${c}` }))}
            />
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <Plot x={[0, 44]} y={[-1.2, 1.5]} width={740} height={175} xLabel="SGD step" yLabel="Weight w">
              {({ x: sx, y: sy }) => (
                <>
                  <line x1={sx(0)} x2={sx(44)} y1={sy(1.1865)} y2={sy(1.1865)} stroke="var(--c-good)" strokeDasharray="4 4" strokeWidth={1.2} />
                  <text x={sx(43.5)} y={sy(1.1865) - 5} textAnchor="end" fontSize={10.5} fill="var(--c-good)" fontFamily="var(--mono)">
                    target w* = 1.1865
                  </text>
                  {/* Full run ghost */}
                  <polyline
                    points={r2Trace.trace.map((p) => `${sx(p.step)},${sy(Math.max(-1.15, Math.min(1.45, p.w)))}`).join(' ')}
                    fill="none"
                    stroke="var(--border)"
                    strokeWidth={1.4}
                    strokeDasharray="3 3"
                  />
                  {/* Replayed path */}
                  <polyline
                    points={visibleCliffTrace.map((p) => `${sx(p.step)},${sy(Math.max(-1.15, Math.min(1.45, p.w)))}`).join(' ')}
                    fill="none"
                    stroke={r2.pass ? 'var(--c-good)' : 'var(--c-bad)'}
                    strokeWidth={2.4}
                  />
                  {/* Mark clipped steps */}
                  {st.b2Clip !== null &&
                    visibleCliffTrace
                      .filter((p) => Math.abs(p.rawGrad) > Math.abs(p.clippedGrad) + 1e-9)
                      .map((p) => (
                        <circle
                          key={p.step}
                          cx={sx(p.step)}
                          cy={sy(Math.max(-1.15, Math.min(1.45, p.w)))}
                          r={3.2}
                          fill="var(--c-b)"
                        />
                      ))}
                  {/* Active replay marker */}
                  {curCliffPt && (
                    <circle
                      cx={sx(curCliffPt.step)}
                      cy={sy(Math.max(-1.15, Math.min(1.45, curCliffPt.w)))}
                      r={5.2}
                      fill={r2.pass ? 'var(--c-good)' : 'var(--c-bad)'}
                      stroke="var(--bg)"
                      strokeWidth={1.8}
                    />
                  )}
                </>
              )}
            </Plot>
            <ReplayControls r={cliffReplay} label="step" />
            {curCliffPt && (
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 10,
                  alignItems: 'center',
                  fontFamily: 'var(--mono)',
                  fontSize: 12,
                  padding: '6px 10px',
                  borderRadius: 6,
                  background: 'var(--bg-elev)',
                  border: '1px solid var(--border)',
                }}
              >
                <span>step {curCliffPt.step}</span>
                <span>w = {Math.abs(curCliffPt.w) >= 1e4 ? '−∞' : curCliffPt.w.toFixed(4)}</span>
                <span>|g_raw| = {curCliffPt.rawGrad >= 1e6 ? '∞' : Math.abs(curCliffPt.rawGrad).toFixed(2)}</span>
                <span style={{ color: wasClippedNow ? 'var(--c-b)' : 'var(--text-muted)', fontWeight: wasClippedNow ? 700 : 400 }}>
                  |g_used| = {curCliffPt.clippedGrad >= 1e6 ? '∞' : Math.abs(curCliffPt.clippedGrad).toFixed(2)}
                  {wasClippedNow ? ` (clipped ${(Math.abs(curCliffPt.rawGrad) / Math.max(1e-9, Math.abs(curCliffPt.clippedGrad))).toFixed(1)}×)` : ''}
                </span>
                <span style={{ marginLeft: 'auto', color: curCliffPt.loss <= 1e-4 ? 'var(--c-good)' : 'var(--text-muted)' }}>
                  loss = {curCliffPt.loss >= 1e6 ? 'EXPLODED' : fmtSci(curCliffPt.loss)}
                </span>
              </div>
            )}
          </div>
          <LiveCode note="PyTorch gradient clipping" code={liveCode} />
        </div>
      )}

      {briefId === 'tbptt' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Truncation horizon K"
              value={st.b3K}
              onChange={(v) => updateB3(v, st.b3H)}
              options={LAB_TBPTT_KS.map((k) => ({ value: k, label: `K = ${k}` }))}
            />
            <Segmented
              label="Hidden width H"
              value={st.b3H}
              onChange={(v) => updateB3(st.b3K, v)}
              options={LAB_TBPTT_HS.map((h) => ({ value: h, label: `H = ${h}` }))}
            />
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              7-step unrolled backward tape (cue arrives at <code>t=0</code>, loss read at <code>t=6</code>)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 6 }}>
              {[0, 1, 2, 3, 4, 5, 6].map((t) => {
                const inWindow = t >= 7 - st.b3K;
                return (
                  <div
                    key={t}
                    style={{
                      textAlign: 'center',
                      padding: '6px 4px',
                      borderRadius: 6,
                      fontFamily: 'var(--mono)',
                      fontSize: 11.5,
                      border: `1px solid ${inWindow ? (t === 0 ? 'var(--c-good)' : 'var(--accent)') : 'var(--border)'}`,
                      background: inWindow ? 'var(--bg-elev)' : 'var(--bg)',
                      color: inWindow ? (t === 0 ? 'var(--c-good)' : 'var(--accent)') : 'var(--text-muted)',
                      opacity: inWindow ? 1 : 0.45,
                    }}
                  >
                    <div>t={t}</div>
                    <div style={{ fontSize: 10 }}>{t === 0 ? (inWindow ? 'CUE ✓' : 'CUT ✗') : inWindow ? 'grad' : '0'}</div>
                  </div>
                );
              })}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                  <span>Delayed-recall accuracy (≥ 95%)</span>
                  <strong style={{ color: r3.testAcc >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>{(100 * r3.testAcc).toFixed(1)}%</strong>
                </div>
                <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                  <div
                    style={{
                      width: `${Math.max(0, Math.min(100, tweened[1]))}%`,
                      height: '100%',
                      borderRadius: 4,
                      background: r3.testAcc >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)',
                    }}
                  />
                  <div style={{ position: 'absolute', left: '95%', top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                  <span>Activation RAM (cap = 20 slots)</span>
                  <strong style={{ color: r3.memSlots <= 20 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {st.b3K} × {st.b3H} = {r3.memSlots} / 20
                  </strong>
                </div>
                <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                  <div
                    style={{
                      width: `${Math.max(0, Math.min(100, tweened[2]))}%`,
                      height: '100%',
                      borderRadius: 4,
                      background: r3.memSlots <= 20 ? 'var(--c-good)' : 'var(--c-bad)',
                    }}
                  />
                  <div style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
                </div>
              </div>
            </div>
          </div>
          <LiveCode note="PyTorch Truncated BPTT (h.detach())" code={liveCode} />
        </div>
      )}
    </LabShell>
  );
}
