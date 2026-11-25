// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { AnswerReveal } from '../../components/AnswerReveal';
import { Segmented, Slider } from '../../components/controls';
import {
  DEEP_RACE_PRESETS,
  INIT_BY_ID,
  INIT_SPECS,
  RESCUE_ACTS,
  RESCUE_INITS,
  RESCUE_LEVELS,
  evaluateRescue,
  measureLayerVariance,
  probeSignal,
  runDeepTrainingRace,
  solveRescueLevel,
  rescueHint,
  type ActName,
  type InitName,
  type RescueRecipe,
} from './vanish';

const fmtSci = (v: number): string => {
  if (!Number.isFinite(v)) return 'NaN / ∞';
  if (v === 0) return '0';
  if (v >= 0.01 && v < 1000) return v.toPrecision(3);
  return v.toExponential(2);
};

// Shared compact log-scale layer trace SVG
function LogTraceSvg({
  title,
  values,
  color,
  bandLo,
  bandHi,
  directionNote,
}: {
  title: string;
  values: number[];
  color: string;
  bandLo: number;
  bandHi: number;
  directionNote: string;
}) {
  const W = 280;
  const H = 165;
  const pad = { l: 42, r: 12, t: 22, b: 28 };
  const n = values.length - 1;

  // Clamp log10 range to [-10, +10] so both vanishing (1e-10) and exploding (1e+9) fit cleanly
  const logMin = -10;
  const logMax = 10;
  const clampLog = (v: number) => {
    if (!Number.isFinite(v)) return logMax;
    if (v <= 0) return logMin;
    return Math.max(logMin, Math.min(logMax, Math.log10(v)));
  };

  const sx = (i: number) => pad.l + (i / Math.max(1, n)) * (W - pad.l - pad.r);
  const sy = (lv: number) => H - pad.b - ((lv - logMin) / (logMax - logMin)) * (H - pad.t - pad.b);

  const pts = values.map((v, i) => `${sx(i).toFixed(1)},${sy(clampLog(v)).toFixed(1)}`).join(' ');
  const yTicks = [-8, -4, 0, 4, 8];

  const bandTop = sy(Math.log10(bandHi));
  const bandBot = sy(Math.log10(bandLo));

  return (
    <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: '8px 8px 4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', padding: '0 4px 4px' }}>
        <strong style={{ fontSize: 12, color }}>{title}</strong>
        <span className="mono" style={{ fontSize: 10, color: 'var(--text-faint)' }}>{directionNote}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* Safe corridor */}
        <rect
          x={pad.l}
          y={bandTop}
          width={W - pad.l - pad.r}
          height={Math.max(2, bandBot - bandTop)}
          fill="var(--c-good)"
          opacity={0.14}
        />

        {/* Horizontal grid */}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={pad.l}
              y1={sy(t)}
              x2={W - pad.r}
              y2={sy(t)}
              stroke={t === 0 ? 'var(--border-strong)' : 'var(--border)'}
              strokeDasharray={t === 0 ? undefined : '3 3'}
            />
            <text x={pad.l - 5} y={sy(t) + 3} textAnchor="end" fill="var(--text-faint)" fontSize={9} fontFamily="var(--font-mono)">
              {t === 0 ? '1' : `10${t > 0 ? `+${t}` : t}`}
            </text>
          </g>
        ))}

        {/* X axis ticks */}
        {[0, Math.round(n / 2), n].map((layer) => (
          <text key={layer} x={sx(layer)} y={H - 8} textAnchor="middle" fill="var(--text-faint)" fontSize={9} fontFamily="var(--font-mono)">
            L{layer}
          </text>
        ))}

        {/* Curve */}
        <polyline fill="none" stroke={color} strokeWidth={2.2} points={pts} />
        {/* Endpoints */}
        <circle cx={sx(0)} cy={sy(clampLog(values[0]))} r={3.5} fill={color} />
        <circle cx={sx(n)} cy={sy(clampLog(values[n]))} r={3.5} fill={color} />
      </svg>
    </div>
  );
}

// ------------------------------------------------------------------ 1. The Signal Telephone

export function SignalTelephone() {
  const [act, setAct] = useState<ActName>('sigmoid');
  const [init, setInit] = useState<InitName>('xavier');
  const [depth, setDepth] = useState(16);
  const [useLayerNorm, setUseLayerNorm] = useState(false);
  const [useResidual, setUseResidual] = useState(false);

  const trace = useMemo(
    () =>
      probeSignal({
        depth,
        width: 32,
        act,
        init,
        useLayerNorm,
        useResidual,
        seed: 1337,
        samples: 16,
      }),
    [depth, act, init, useLayerNorm, useResidual]
  );

  const status = useMemo(() => {
    if (trace.overflowed || trace.gradRatio > 20 || trace.maxActRms > 20) {
      return { kind: 'bad', label: 'Exploding!', msg: 'Weights amplify the signal at every layer — activations and gradients blow up exponentially.' };
    }
    if (trace.gradRatio < 0.05) {
      return {
        kind: 'bad',
        label: 'Vanishing!',
        msg:
          act === 'sigmoid' && !useResidual
            ? 'Sigmoid’s derivative is capped at 0.25, so backpropagating through every layer shrinks the gradient by at least 4× — layer 1 is effectively deaf.'
            : act === 'relu' && init === 'xavier'
              ? 'Xavier init forgets that ReLU zeroes half the neurons — signal shrinks by (1/√2)^L. Switch to He init (√(2/d)) to fix it!'
              : 'Gradients decay exponentially on the way back to layer 1, starving early layers of learning signal.',
      };
    }
    return {
      kind: 'good',
      label: 'Healthy signal',
      msg: 'Both forward activations and backward gradients stay near order 1 from input to output — every layer can learn together.',
    };
  }, [trace, act, init, useResidual]);

  return (
    <Widget
      title="The signal telephone: forward RMS & backward gradients across depth"
      wide
      caption={
        <>
          Send 16 unit-variance Gaussian vectors (width 32) through a deep stack and backpropagate a unit gradient from the last layer back to layer 0. Try <strong>Sigmoid + Xavier</strong> (vanishes to 10⁻¹¹), <strong>ReLU + σ=1.0</strong> (explodes to 10⁺⁹), <strong>ReLU + He</strong> (balanced!), and flip on <strong>Residual skip</strong> to rescue even Sigmoid.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="activation"
          options={[
            { value: 'linear', label: 'Linear' },
            { value: 'tanh', label: 'Tanh' },
            { value: 'sigmoid', label: 'Sigmoid' },
            { value: 'relu', label: 'ReLU' },
          ]}
          value={act}
          onChange={(v) => setAct(v as ActName)}
        />
        <Segmented
          label="weight init scale"
          options={INIT_SPECS.map((s) => ({ value: s.id, label: s.shortLabel }))}
          value={init}
          onChange={(v) => setInit(v as InitName)}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ flex: '1 1 210px' }}>
          <Slider
            label={`Depth: ${depth} layers (width 32)`}
            min={4}
            max={24}
            step={2}
            value={depth}
            onChange={setDepth}
            format={(d) => `${d}L`}
          />
        </div>
        <div className="btn-row">
          <button
            className={useLayerNorm ? 'btn primary' : 'btn'}
            onClick={() => setUseLayerNorm((v) => !v)}
          >
            {useLayerNorm ? '✓ LayerNorm ON' : 'LayerNorm OFF'}
          </button>
          <button
            className={useResidual ? 'btn primary' : 'btn'}
            onClick={() => setUseResidual((v) => !v)}
          >
            {useResidual ? '✓ Residual skip ON' : 'Residual skip OFF'}
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10, marginBottom: 12 }}>
        <LogTraceSvg
          title="Forward activation RMS"
          values={trace.actRms}
          color="var(--c-a)"
          bandLo={0.3}
          bandHi={3.0}
          directionNote="L0 → L_last"
        />
        <LogTraceSvg
          title="Backward gradient RMS"
          values={trace.gradRms}
          color="var(--accent)"
          bandLo={0.25}
          bandHi={4.0}
          directionNote="L_last → L0"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 8, marginBottom: 10 }}>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>FINAL ACT RMS (L{depth})</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: trace.actRms[depth] < 0.2 || trace.actRms[depth] > 5 ? 'var(--c-bad)' : 'var(--c-a)' }}>
            {fmtSci(trace.actRms[depth])}
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>GRAD RATIO (L0 / L{depth})</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: status.kind === 'good' ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {fmtSci(trace.gradRatio)}×
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>INIT σ (d = 32)</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>
            {INIT_BY_ID[init].sigma(32).toFixed(3)}
          </div>
        </div>
      </div>

      <div
        className="live-note"
        style={{
          borderColor: status.kind === 'good' ? 'var(--c-good)' : 'var(--c-bad)',
          background:
            status.kind === 'good'
              ? 'color-mix(in srgb, var(--c-good) 10%, var(--bg-sunken))'
              : 'color-mix(in srgb, var(--c-bad) 10%, var(--bg-sunken))',
        }}
      >
        <strong style={{ color: status.kind === 'good' ? 'var(--c-good)' : 'var(--c-bad)' }}>{status.label}</strong>{' '}
        {status.msg}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Single-layer Variance Conservation Explorer

const WIDTH_STOPS = [4, 16, 64, 256];

export function VarianceDerivation() {
  const [act, setAct] = useState<'linear' | 'relu'>('relu');
  const [wStopIdx, setWStopIdx] = useState(2); // d = 64
  const [sigmaW, setSigmaW] = useState(0.125); // Xavier for d=64

  const fanIn = WIDTH_STOPS[wStopIdx];
  const xavierSigma = 1 / Math.sqrt(fanIn);
  const heSigma = Math.sqrt(2 / fanIn);

  const m = useMemo(() => measureLayerVariance(fanIn, sigmaW, act), [fanIn, sigmaW, act]);

  // Dot strip plot of 48 samples
  const W = 520;
  const H = 82;
  const pad = { l: 24, r: 24, t: 12, b: 22 };
  const maxAbs = Math.max(3, ...m.samples.map((v) => Math.abs(v)));
  const sx = (v: number) => pad.l + ((v + maxAbs) / (2 * maxAbs)) * (W - pad.l - pad.r);

  const balanced = m.theoryVar >= 0.85 && m.theoryVar <= 1.15;

  return (
    <Widget
      title="Why 1/√d and √(2/d)? Watching one layer’s variance"
      wide
      caption={
        <>
          400 random dot products of width <code>d</code> with unit-variance inputs. Click <strong>Xavier (1/√d)</strong> on <strong>ReLU</strong> and watch half the dots pile up at exact zero, cutting variance to <strong>0.5</strong> — then click <strong>He (√(2/d))</strong> to restore variance to <strong>1.0</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="activation"
          options={[
            { value: 'linear', label: 'Linear (no zeroing)' },
            { value: 'relu', label: 'ReLU (zeroes negatives)' },
          ]}
          value={act}
          onChange={(v) => setAct(v as 'linear' | 'relu')}
        />
        <Segmented
          label="layer width d"
          options={WIDTH_STOPS.map((d, i) => ({ value: i, label: `d = ${d}` }))}
          value={wStopIdx}
          onChange={(i) => {
            setWStopIdx(i);
            const newD = WIDTH_STOPS[i];
            setSigmaW(Number((1 / Math.sqrt(newD)).toFixed(3)));
          }}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 12 }}>
        <div style={{ flex: '1 1 220px' }}>
          <Slider
            label={`Weight std σ_w = ${sigmaW.toFixed(3)}`}
            min={0.02}
            max={1.0}
            step={0.005}
            value={sigmaW}
            onChange={setSigmaW}
            format={(v) => v.toFixed(3)}
          />
        </div>
        <div className="btn-row">
          <button className="btn" onClick={() => setSigmaW(Number(xavierSigma.toFixed(3)))}>
            Xavier ({xavierSigma.toFixed(3)})
          </button>
          <button className="btn primary" onClick={() => setSigmaW(Number(heSigma.toFixed(3)))}>
            He ({heSigma.toFixed(3)})
          </button>
          <button className="btn" onClick={() => setSigmaW(1.0)}>
            Naive (1.0)
          </button>
        </div>
      </div>

      {/* Output distribution strip */}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'var(--bg-sunken)', borderRadius: 8, border: '1px solid var(--border)', marginBottom: 12 }}>
        <line x1={pad.l} y1={H / 2 - 4} x2={W - pad.r} y2={H / 2 - 4} stroke="var(--border-strong)" />
        <line x1={sx(0)} y1={10} x2={sx(0)} y2={H - pad.b} stroke="var(--text-faint)" strokeDasharray="3 3" />
        {/* ±1 reference ticks */}
        {[-1, 0, 1].map((t) => (
          <text key={t} x={sx(t)} y={H - 6} textAnchor="middle" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
            {t}
          </text>
        ))}
        <text x={W - pad.r} y={H - 6} textAnchor="end" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
          +{maxAbs.toFixed(1)}
        </text>
        <text x={pad.l} y={H - 6} textAnchor="start" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
          −{maxAbs.toFixed(1)}
        </text>

        {m.samples.map((v, idx) => (
          <circle
            key={idx}
            cx={sx(v)}
            cy={H / 2 - 4 + ((idx % 5) - 2) * 5.5}
            r={3.2}
            fill={v === 0 && act === 'relu' ? 'var(--c-b)' : 'var(--c-a)'}
            opacity={0.8}
          />
        ))}
      </svg>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 8 }}>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>EXACT FORMULA</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: balanced ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {m.theoryVar.toFixed(3)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {act === 'relu' ? '½ · d · σ_w²' : 'd · σ_w²'} (goal = 1.000)
          </div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>MEASURED (400 TRIALS)</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-a)' }}>
            {m.empiricalVar.toFixed(3)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            E[y²] · RMS = {m.empiricalStd.toFixed(3)}
          </div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>AFTER 16 SUCH LAYERS</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: balanced ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {fmtSci(Math.pow(m.theoryVar, 8))}× RMS
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            (√{m.theoryVar.toFixed(2)})¹⁶
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Real 8-Layer Training Race

const RACE_COLORS = ['var(--c-bad)', 'var(--c-b)', 'var(--c-a)', 'var(--c-good)'];

export function DeepTrainingRace() {
  const [seed, setSeed] = useState(11);
  const [focusIdx, setFocusIdx] = useState(0);

  const results = useMemo(
    () => DEEP_RACE_PRESETS.map((p) => runDeepTrainingRace(p, 150, seed)),
    [seed]
  );

  const focused = results[focusIdx];

  // Loss curve SVG
  const W = 310;
  const H = 175;
  const pad = { l: 38, r: 12, t: 14, b: 28 };
  const nPts = results[0].losses.length;
  const sx = (i: number) => pad.l + (i / Math.max(1, nPts - 1)) * (W - pad.l - pad.r);
  const sy = (loss: number) => {
    const c = Math.max(0, Math.min(1.0, Number.isFinite(loss) ? loss : 1.0));
    return H - pad.b - (c / 1.0) * (H - pad.t - pad.b);
  };

  // Layer movement bar chart
  const moveVals = focused.layerWeightMovement;
  const maxMove = Math.max(0.15, ...moveVals.filter(Number.isFinite));

  return (
    <Widget
      title="Real 8-hidden-layer training race (concentric rings, 150 SGD steps)"
      wide
      caption={
        <>
          Four real <code>2 → 16×8 → 1</code> networks trained live in your browser (each at its grid-searched best learning rate, except Naive σ=1.0 shown exploding at η=0.05). Click any model to inspect <strong>how far each layer’s weights actually moved</strong> — watch Sigmoid’s layer 1 stay frozen at <strong>0.00%</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="btn-row">
          {DEEP_RACE_PRESETS.map((p, i) => (
            <button
              key={p.id}
              className={focusIdx === i ? 'btn primary' : 'btn'}
              onClick={() => setFocusIdx(i)}
            >
              <span style={{ color: focusIdx === i ? 'inherit' : RACE_COLORS[i], marginRight: 5 }}>●</span>
              {p.label}
            </button>
          ))}
        </div>
        <Segmented
          label="seed"
          options={[
            { value: 11, label: 'Seed 11' },
            { value: 22, label: 'Seed 22' },
            { value: 33, label: 'Seed 33' },
          ]}
          value={seed}
          onChange={setSeed}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 12, marginBottom: 12 }}>
        {/* Left: loss curves */}
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>Training loss (0..150 steps)</div>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {[0, 0.35, 0.693, 1.0].map((v) => (
              <g key={v}>
                <line x1={pad.l} y1={sy(v)} x2={W - pad.r} y2={sy(v)} stroke="var(--border)" strokeDasharray={v === 0.693 ? '4 3' : '2 3'} />
                <text x={pad.l - 5} y={sy(v) + 3} textAnchor="end" fill="var(--text-faint)" fontSize={9} fontFamily="var(--font-mono)">
                  {v === 0.693 ? 'ln 2' : v.toFixed(1)}
                </text>
              </g>
            ))}
            {[0, 75, 150].map((st, idx) => (
              <text key={st} x={sx(idx * ((nPts - 1) / 2))} y={H - 8} textAnchor="middle" fill="var(--text-faint)" fontSize={9} fontFamily="var(--font-mono)">
                {st}
              </text>
            ))}
            {results.map((r, i) => {
              const pts = r.losses
                .map((l, j) => (Number.isFinite(l) ? `${sx(j).toFixed(1)},${sy(l).toFixed(1)}` : null))
                .filter(Boolean)
                .join(' ');
              return (
                <polyline
                  key={r.preset.id}
                  fill="none"
                  stroke={RACE_COLORS[i]}
                  strokeWidth={focusIdx === i ? 2.8 : 1.5}
                  opacity={focusIdx === i ? 1 : 0.45}
                  points={pts}
                />
              );
            })}
          </svg>
        </div>

        {/* Right: per-layer weight movement */}
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
            <strong style={{ fontSize: 12, color: RACE_COLORS[focusIdx] }}>{focused.preset.label}</strong>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Acc {(focused.finalAcc * 100).toFixed(0)}% · Loss {Number.isFinite(focused.finalLoss) ? focused.finalLoss.toFixed(3) : 'NaN'}
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)', marginBottom: 8 }}>
            Relative weight movement ‖ΔW‖ / ‖W‖ per layer (L1 = earliest, L8 = output side):
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {moveVals.map((m, l) => {
              const bad = !Number.isFinite(m) || m > 10;
              const pct = bad ? 100 : Math.min(100, (m / maxMove) * 100);
              return (
                <div key={l} style={{ display: 'grid', gridTemplateColumns: '26px 1fr 62px', alignItems: 'center', gap: 6 }}>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>L{l + 1}</span>
                  <div style={{ height: 12, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${pct}%`,
                        height: '100%',
                        background: bad ? 'var(--c-bad)' : RACE_COLORS[focusIdx],
                        opacity: 0.85,
                      }}
                    />
                  </div>
                  <span className="mono" style={{ fontSize: 11, textAlign: 'right', color: bad ? 'var(--c-bad)' : 'var(--text)' }}>
                    {bad ? 'EXPLODED' : `${(m * 100).toFixed(2)}%`}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="live-note">
        <strong>{focused.preset.label} (η = {focused.preset.lr}):</strong> {focused.preset.note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Rescue the Deep Stack

const LAB_KEY = 'nl-vanish-lab-v1';

interface SavedVanishLab {
  recipes: Record<string, RescueRecipe>;
  solved: Record<string, boolean>;
}

function loadVanishLab(): SavedVanishLab {
  const defaults = Object.fromEntries(RESCUE_LEVELS.map((l) => [l.id, { ...l.defaultRecipe }]));
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return { recipes: defaults, solved: {} };
    const parsed = JSON.parse(raw);
    return {
      recipes: { ...defaults, ...(parsed.recipes ?? {}) },
      solved: parsed.solved ?? {},
    };
  } catch {
    return { recipes: defaults, solved: {} };
  }
}

export function RescueDeepNetLab() {
  const [lvlIdx, setLvlIdx] = useState(0);
  const [state, setState] = useState<SavedVanishLab>(() => loadVanishLab());

  const level = RESCUE_LEVELS[lvlIdx];
  const recipe = state.recipes[level.id] ?? level.defaultRecipe;
  const ev = useMemo(() => evaluateRescue(level, recipe), [level, recipe]);
  const sol = useMemo(() => solveRescueLevel(level), [level]);

  useEffect(() => {
    if (ev.passes && !state.solved[level.id]) {
      setState((prev) => ({ ...prev, solved: { ...prev.solved, [level.id]: true } }));
    }
  }, [ev.passes, level.id, state.solved]);

  useEffect(() => {
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(state));
    } catch {
      /* ignore */
    }
  }, [state]);

  const update = (patch: Partial<RescueRecipe>) => {
    setState((prev) => ({
      ...prev,
      recipes: { ...prev.recipes, [level.id]: { ...recipe, ...patch } },
    }));
  };

  const gradLo = level.id === 'deep-32' ? 0.4 : 0.25;
  const gradHi = level.id === 'deep-32' ? 2.5 : 4.0;
  const solvedCount = RESCUE_LEVELS.filter((l) => state.solved[l.id]).length;
  const hitBest = ev.passes && ev.balanceErr <= sol.bestEval.balanceErr + 0.01;

  return (
    <Widget
      title="Lab: Rescue the deep network"
      wide
      caption={
        <>
          Three deep architectures stuck in vanishing or exploding regimes. Keep every hidden layer’s activation RMS in the green corridor and deliver a healthy gradient all the way back to layer 0. All legal recipes are verified exhaustively.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="btn-row">
          {RESCUE_LEVELS.map((l, i) => (
            <button
              key={l.id}
              className={lvlIdx === i ? 'btn primary' : 'btn'}
              onClick={() => {
                setLvlIdx(i);
              }}
            >
              {state.solved[l.id] ? '✓ ' : ''}
              {i === 0 ? 'L1 · Plain 16L' : i === 1 ? 'L2 · Sigmoid 16L' : 'L3 · Tower 32L'}
            </button>
          ))}
        </div>
        <span className="mono" style={{ fontSize: 12, color: 'var(--text-faint)' }}>
          {solvedCount} / {RESCUE_LEVELS.length} levels solved
        </span>
      </div>

      <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', marginBottom: 12, fontSize: 13 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{level.title}</div>
        <div style={{ color: 'var(--text-muted)' }}>{level.brief}</div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        {level.lockedAct ? (
          <div>
            <div className="field-label">ACTIVATION (LOCKED)</div>
            <div className="mono" style={{ padding: '6px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 12 }}>
              🔒 Sigmoid only
            </div>
          </div>
        ) : (
          <Segmented
            label="activation"
            options={RESCUE_ACTS.map((a) => ({ value: a, label: a === 'relu' ? 'ReLU' : a === 'tanh' ? 'Tanh' : 'Sigmoid' }))}
            value={recipe.act}
            onChange={(v) => update({ act: v as ActName })}
          />
        )}

        <Segmented
          label="initialisation scale"
          options={RESCUE_INITS.map((id) => ({ value: id, label: INIT_BY_ID[id].shortLabel }))}
          value={recipe.init}
          onChange={(v) => update({ init: v as InitName })}
        />

        <div className="btn-row">
          <button
            className={level.allowLayerNorm && recipe.useLayerNorm ? 'btn primary' : 'btn'}
            disabled={!level.allowLayerNorm}
            onClick={() => update({ useLayerNorm: !recipe.useLayerNorm })}
          >
            {!level.allowLayerNorm ? '🔒 LayerNorm' : recipe.useLayerNorm ? '✓ LayerNorm ON' : 'LayerNorm OFF'}
          </button>
          <button
            className={level.allowResidual && recipe.useResidual ? 'btn primary' : 'btn'}
            disabled={!level.allowResidual}
            onClick={() => update({ useResidual: !recipe.useResidual })}
          >
            {!level.allowResidual ? '🔒 Residual' : recipe.useResidual ? '✓ Residual ON' : 'Residual OFF'}
          </button>
        </div>
      </div>

      {/* Live trace plots */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10, marginBottom: 12 }}>
        <LogTraceSvg
          title="Forward activation RMS"
          values={ev.trace.actRms}
          color="var(--c-a)"
          bandLo={0.3}
          bandHi={3.0}
          directionNote="goal: [0.3, 3.0]"
        />
        <LogTraceSvg
          title="Backward gradient RMS"
          values={ev.trace.gradRms}
          color="var(--accent)"
          bandLo={gradLo}
          bandHi={gradHi}
          directionNote={`goal: [${gradLo}, ${gradHi}]`}
        />
      </div>

      {/* Checklist */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 8, marginBottom: 10 }}>
        <div
          className="mono"
          style={{
            padding: '8px 10px',
            borderRadius: 6,
            fontSize: 12,
            border: `1px solid color-mix(in srgb, ${ev.actOk ? 'var(--c-good)' : 'var(--c-bad)'} 45%, transparent)`,
            background: `color-mix(in srgb, ${ev.actOk ? 'var(--c-good)' : 'var(--c-bad)'} 10%, transparent)`,
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span style={{ color: ev.actOk ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
            {ev.actOk ? '✓' : '✗'} Hidden RMS in [0.3, 3.0]
          </span>
          <span>
            {fmtSci(ev.trace.minActRms)}..{fmtSci(ev.trace.maxActRms)}
          </span>
        </div>
        <div
          className="mono"
          style={{
            padding: '8px 10px',
            borderRadius: 6,
            fontSize: 12,
            border: `1px solid color-mix(in srgb, ${ev.gradOk ? 'var(--c-good)' : 'var(--c-bad)'} 45%, transparent)`,
            background: `color-mix(in srgb, ${ev.gradOk ? 'var(--c-good)' : 'var(--c-bad)'} 10%, transparent)`,
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span style={{ color: ev.gradOk ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
            {ev.gradOk ? '✓' : '✗'} Grad ratio in [{gradLo}, {gradHi}]
          </span>
          <span>{fmtSci(ev.trace.gradRatio)}×</span>
        </div>
      </div>

      <div
        className="live-note"
        style={{
          borderColor: ev.passes ? 'var(--c-good)' : undefined,
          background: ev.passes ? 'color-mix(in srgb, var(--c-good) 10%, var(--bg-sunken))' : undefined,
        }}
      >
        {ev.passes ? (
          <>
            <strong style={{ color: 'var(--c-good)' }}>
              {hitBest ? 'Level solved — and you hit the tightest gradient balance!' : 'Level solved!'}
            </strong>{' '}
            ({sol.passingCount} of {sol.totalRecipes} legal recipes pass
            {level.allowResidual ? `; ${sol.plainPassingCount} work without LN/Residual` : ''}.)
          </>
        ) : (
          <>
            <strong>Not yet balanced.</strong>{' '}
            {rescueHint(level, recipe, ev)}
          </>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn" onClick={() => update(level.defaultRecipe)}>
          Reset level
        </button>
      </div>

      <AnswerReveal solved={!!state.solved[level.id] || ev.passes} stuckKey={`vanishing-rescue:${level.id}`} label="the exhaustive summary">
        <div className="mono" style={{ marginTop: 8, padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 12 }}>
          Exhaustive check of all {sol.totalRecipes} legal recipes: <strong>{sol.passingCount}</strong> pass (
          <strong>{sol.plainPassingCount}</strong> plain without LN/Residual). Tightest grad balance:{' '}
          <strong>
            {sol.bestRecipe.act.toUpperCase()} + {INIT_BY_ID[sol.bestRecipe.init].shortLabel}
            {sol.bestRecipe.useLayerNorm ? ' + LN' : ''}
            {sol.bestRecipe.useResidual ? ' + Res' : ''}
          </strong>{' '}
          (grad ratio {sol.bestEval.trace.gradRatio.toFixed(3)}×).
        </div>
      </AnswerReveal>
    </Widget>
  );
}
