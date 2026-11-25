// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { LogChart } from '../../components/LogChart';
import {
  carouselGradientFlow,
  cellParamCount,
  evalLstmBrief1,
  evalLstmBrief2,
  evalLstmBrief3,
  LAB_BIAS_CELLS,
  LAB_BIAS_VALS,
  LAB_BUDGET_HS,
  LAB_BUDGET_KINDS,
  LAB_GATE_LEVELS,
  lstmLabExhaustive,
  runGatedRace,
  simulateLstmCorridor,
  type CellKind,
  type LstmBriefId,
} from './gated';

const fmtSci = (v: number) => {
  if (!Number.isFinite(v)) return '∞';
  const a = Math.abs(v);
  if (a === 0) return '0.000';
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  return v.toFixed(3);
};

// ------------------------------------------------------------------ 1. Interactive LSTM Gate Playground

export function LstmGatePlayground() {
  const [f, setF] = useState<number>(0.95);
  const [i, setI] = useState<number>(0.05);
  const [o, setO] = useState<number>(0.05);

  const sim = useMemo(() => simulateLstmCorridor(f, i, o, 0.95), [f, i, o]);

  const applyPreset = (kind: 'hold' | 'overwrite' | 'wipe' | 'leak') => {
    if (kind === 'hold') { setF(0.95); setI(0.05); setO(0.05); }
    if (kind === 'overwrite') { setF(0.95); setI(0.85); setO(0.05); }
    if (kind === 'wipe') { setF(0.15); setI(0.05); setO(0.05); }
    if (kind === 'leak') { setF(0.95); setI(0.05); setO(0.95); }
  };

  return (
    <Widget
      title="Inside an LSTM cell: carry a +1.0 cue past four hostile distractors"
      wide
      caption={
        <>
          At <code>t = 0</code> the cell stores <code>x₀ = +1.0</code> into cell state <code>c₀ = +0.905</code>. Steps <code>t = 1…4</code> throw four negative distractors (<code>−0.8, −0.7, −0.6, −0.7</code>). Try the presets or slide <strong>Forget f</strong>, <strong>Input i</strong>, and <strong>Output o</strong>: notice how <code>f = 0.95, i = 0.05, o = 0.05</code> shields both the highway and <code>hₜ</code> until the read step (<code>t = 5</code>)!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        <button type="button" className="btn" onClick={() => applyPreset('hold')}>Hold & shield (f=0.95, i=0.05, o=0.05)</button>
        <button type="button" className="btn" onClick={() => applyPreset('overwrite')}>Open input gate (distractors overwrite!)</button>
        <button type="button" className="btn" onClick={() => applyPreset('wipe')}>Shut forget gate (memory wiped!)</button>
        <button type="button" className="btn" onClick={() => applyPreset('leak')}>Open output gate (leaks hₜ early)</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 12, marginBottom: 14 }}>
        <Slider label="Forget gate f (keep old c)" value={f} min={0.05} max={0.98} step={0.05} onChange={setF} format={(v) => v.toFixed(2)} />
        <Slider label="Input gate i (write new x)" value={i} min={0.05} max={0.95} step={0.05} onChange={setI} format={(v) => v.toFixed(2)} />
        <Slider label="Corridor Output gate o" value={o} min={0.05} max={0.95} step={0.05} onChange={setO} format={(v) => v.toFixed(2)} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(105px, 1fr))', gap: 8, marginBottom: 12 }}>
        {sim.steps.map((s) => (
          <div
            key={s.t}
            style={{
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '8px 10px',
              background: s.t === 0 || s.t === 5 ? 'var(--bg-elev)' : 'var(--bg-subtle)',
              fontFamily: 'var(--mono)',
              fontSize: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700 }}>
              <span>t = {s.t}</span>
              <span style={{ color: s.t === 0 ? 'var(--c-good)' : s.t === 5 ? 'var(--accent)' : 'var(--c-bad)', fontSize: 11 }}>
                {s.t === 0 ? 'WRITE' : s.t === 5 ? 'READ' : 'NOISE'}
              </span>
            </div>
            <div style={{ color: 'var(--text-muted)' }}>x = {s.x >= 0 ? '+' : ''}{s.x.toFixed(1)}</div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>cₜ: </span>
              <strong style={{ color: s.c >= 0.4 ? 'var(--c-good)' : s.c <= 0 ? 'var(--c-bad)' : 'var(--c-b)' }}>
                {s.c >= 0 ? '+' : ''}{s.c.toFixed(2)}
              </strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>hₜ: </span>
              <span style={{ color: 'var(--c-a)', fontWeight: 600 }}>
                {s.h >= 0 ? '+' : ''}{s.h.toFixed(2)}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
        <div className="readout">
          <div className="readout-label">Cue kept at t = 5</div>
          <div className="readout-value" style={{ color: sim.retentionRatio >= 0.55 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {(100 * sim.retentionRatio).toFixed(1)}%
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Corridor leak max |hₜ|</div>
          <div className="readout-value" style={{ color: sim.maxCorridorH <= 0.15 ? 'var(--c-good)' : 'var(--c-b)' }}>
            {sim.maxCorridorH.toFixed(3)}
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Final readout h₅</div>
          <div className="readout-value" style={{ color: sim.readH > 0.35 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {sim.readH >= 0 ? '+' : ''}{sim.readH.toFixed(3)}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Constant Error Carousel & Forget-Gate Bias

export function CarouselWidget() {
  const [bf, setBf] = useState<number>(2);
  const T = 25;

  const flow = useMemo(() => carouselGradientFlow(T, bf, 0.8), [bf]);
  const b0Flow = useMemo(() => carouselGradientFlow(T, 0, 0.8), []);

  const series = useMemo(
    () => [
      {
        points: flow.lstm.ratios.map((r, k): [number, number] => [k, Math.max(1e-4, r)]),
        color: 'var(--c-good)',
        label: flow.lstm.label,
      },
      {
        points: flow.rnn.ratios.map((r, k): [number, number] => [k, Math.max(1e-4, r)]),
        color: 'var(--c-bad)',
        dashed: true,
        label: flow.rnn.label,
      },
    ],
    [flow],
  );

  const boostVsZero = flow.lstm.step1Ratio / b0Flow.lstm.step1Ratio;

  return (
    <Widget
      title="The Constant Error Carousel: why forget-gate bias b_f = +2 changes everything"
      wide
      caption={
        <>
          Exact cell-highway gradient <code>∂c_25 / ∂c_(25−k) = fᵏ</code> with <code>f = σ(b_f)</code> vs a plain <code>tanh</code> RNN (<code>ρ = 0.80</code>). Slide <strong>b_f</strong> from <code>0</code> to <code>+3</code>: naive <code>b_f = 0</code> STILL vanishes to <code>6.0×10⁻⁸</code>, whereas <code>b_f = +3</code> preserves <strong>31.2%</strong> across 25 steps — over <strong>5 million×</strong> stronger!
        </>
      }
    >
      <div style={{ marginBottom: 10 }}>
        <Slider
          label="Initial forget-gate bias b_f"
          value={bf}
          min={-1}
          max={3}
          step={0.5}
          onChange={setBf}
          format={(v) => `b_f = ${v >= 0 ? '+' : ''}${v.toFixed(1)}  (f = ${flow.lstm.gateVal.toFixed(3)})`}
        />
      </div>

      <LogChart
        series={series}
        xDomain={[0, T - 1]}
        yDomain={[1e-4, 2]}
        width={760}
        height={215}
        xTicks={[0, 4, 8, 12, 16, 20, 24]}
        xLabel="Steps backwards k = 25 − t"
        ariaLabel="LSTM cell-highway gradient vs plain RNN over 25 steps"
        hlines={[{ y: 0.05, label: 'healthy floor (0.05)', color: 'var(--text-muted)' }]}
      />

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, margin: '6px 0 10px', color: 'var(--text-muted)' }}>
        <span><span style={{ color: 'var(--c-good)', fontWeight: 700 }}>━━</span> {flow.lstm.label}</span>
        <span><span style={{ color: 'var(--c-bad)', fontWeight: 700 }}>---</span> {flow.rnn.label}</span>
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8, margin: 0 }}>
        <div className="readout">
          <div className="readout-label">LSTM grad at t = 1</div>
          <div className="readout-value" style={{ color: flow.lstm.step1Ratio >= 0.04 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {fmtSci(flow.lstm.step1Ratio)}
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Plain RNN at t = 1</div>
          <div className="readout-value" style={{ color: 'var(--c-bad)' }}>{fmtSci(flow.rnn.step1Ratio)}</div>
        </div>
        <div className="readout">
          <div className="readout-label">Boost vs b_f = 0</div>
          <div className="readout-value" style={{ color: 'var(--accent)' }}>
            {boostVsZero >= 1000 ? `${boostVsZero.toExponential(1)}×` : `${boostVsZero.toFixed(1)}×`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Fair Race + Parameter Calculator (RNN vs GRU vs LSTM)

export function RaceAndParamsWidget() {
  const [tab, setTab] = useState<'race' | 'params'>('race');
  const [inputDim, setInputDim] = useState<number>(16);
  const [hiddenDim, setHiddenDim] = useState<number>(16);

  const raceRows = useMemo(() => [3, 8, 14].map((lag) => runGatedRace(lag)), []);

  const rnnP = cellParamCount('rnn', inputDim, hiddenDim);
  const gruP = cellParamCount('gru', inputDim, hiddenDim);
  const lstmP = cellParamCount('lstm', inputDim, hiddenDim);

  return (
    <Widget
      title="Head-to-head: Vanilla RNN vs GRU vs LSTM (accuracy & parameter cost)"
      wide
      caption={
        tab === 'race' ? (
          <>
            Real BPTT training averaged across 3 fixed seeds on noisy delayed recall (<code>±0.9</code> distractors). Plain RNN (even given extra width <code>H = 4</code>) collapses from <strong>100% → 79.6% → 50.9%</strong> at lag 14, while both <strong>GRU (H = 3)</strong> and <strong>LSTM (H = 3)</strong> hold <strong>100.0%</strong> across all lags!
          </>
        ) : (
          <>
            Every gate adds a full <code>H × (D + H + 1)</code> matrix. A <strong>GRU</strong> merges cell & hidden state and uses 3 gates (<strong>25% fewer parameters</strong> than an LSTM’s 4 gates) while keeping an additive highway!
          </>
        )
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="Comparison"
          value={tab}
          onChange={setTab}
          options={[
            { value: 'race', label: 'Long-lag memory race (3 seeds)' },
            { value: 'params', label: 'Parameter count calculator' },
          ]}
        />
      </div>

      {tab === 'race' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {raceRows.map((row) => (
            <div key={row.lag} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'var(--bg-subtle)' }}>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 12.5, fontWeight: 700, marginBottom: 8 }}>
                Delay lag L = {row.lag} distractor steps (sequence length T = {row.lag + 1})
              </div>
              {(
                [
                  ['Vanilla RNN (H=4)', row.rnnAcc, row.rnnAcc >= 0.9 ? 'var(--c-a)' : 'var(--c-bad)'],
                  ['GRU (H=3, 3× gates)', row.gruAcc, 'var(--accent)'],
                  ['LSTM (H=3, 4× gates)', row.lstmAcc, 'var(--c-good)'],
                ] as [string, number, string][]
              ).map(([label, acc, col]) => (
                <div key={label} style={{ display: 'grid', gridTemplateColumns: '155px 1fr 56px', alignItems: 'center', gap: 10, fontSize: 12.5, marginBottom: 4 }}>
                  <span>{label}</span>
                  <div style={{ height: 12, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                    <div style={{ width: `${(100 * acc).toFixed(1)}%`, height: '100%', background: col }} />
                  </div>
                  <span style={{ fontFamily: 'var(--mono)', fontWeight: 700, textAlign: 'right' }}>{(100 * acc).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 12 }}>
            <Slider label="Input dim D" value={inputDim} min={4} max={128} step={4} onChange={setInputDim} format={(v) => `D = ${v}`} />
            <Slider label="Hidden width H" value={hiddenDim} min={4} max={128} step={4} onChange={setHiddenDim} format={(v) => `H = ${v}`} />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Vanilla RNN (1×)</div>
              <div className="readout-value">{rnnP.toLocaleString()}</div>
            </div>
            <div className="readout">
              <div className="readout-label">GRU (3× · −25% vs LSTM)</div>
              <div className="readout-value" style={{ color: 'var(--accent)' }}>{gruP.toLocaleString()}</div>
            </div>
            <div className="readout">
              <div className="readout-label">LSTM (4×)</div>
              <div className="readout-value" style={{ color: 'var(--c-good)' }}>{lstmP.toLocaleString()}</div>
            </div>
          </div>
        </div>
      )}
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Wire the Gated Cell

type LstmLabState = {
  b1F: number;
  b1I: number;
  b1O: number;
  b2Cell: 'rnn' | 'lstm';
  b2Bias: number;
  b3Kind: CellKind;
  b3H: number;
  solved: Record<LstmBriefId, boolean>;
};

const LSTM_LAB_KEY = 'nl-lstm-lab-v1';
const LSTM_LAB_DEFAULT: LstmLabState = {
  b1F: 0.50,
  b1I: 0.50,
  b1O: 0.50,
  b2Cell: 'lstm',
  b2Bias: 0,
  b3Kind: 'lstm',
  b3H: 16,
  solved: { gates: false, bias: false, budget: false },
};

function loadLstmLab(): LstmLabState {
  try {
    const raw = localStorage.getItem(LSTM_LAB_KEY);
    if (!raw) return LSTM_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...LSTM_LAB_DEFAULT, ...p, solved: { ...LSTM_LAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return LSTM_LAB_DEFAULT;
  }
}

export function LstmLab() {
  const [briefId, setBriefId] = useState<LstmBriefId>('gates');
  const [st, setSt] = useState<LstmLabState>(loadLstmLab);
  const [revealed, setRevealed] = useState<Record<LstmBriefId, boolean>>({ gates: false, bias: false, budget: false });

  useEffect(() => {
    try {
      localStorage.setItem(LSTM_LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalLstmBrief1(st.b1F, st.b1I, st.b1O), [st.b1F, st.b1I, st.b1O]);
  const r1Sim = useMemo(() => simulateLstmCorridor(st.b1F, st.b1I, st.b1O, 0.95), [st.b1F, st.b1I, st.b1O]);
  const r2 = useMemo(() => evalLstmBrief2(st.b2Cell, st.b2Bias), [st.b2Cell, st.b2Bias]);
  const r2Flow = useMemo(() => carouselGradientFlow(25, st.b2Bias, 0.8), [st.b2Bias]);
  const r3 = useMemo(() => evalLstmBrief3(st.b3Kind, st.b3H), [st.b3Kind, st.b3H]);
  const exhaust = useMemo(() => lstmLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'gates' ? r1.pass : briefId === 'bias' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const solvedCount = Object.values(st.solved).filter(Boolean).length;
  const canReveal = st.solved[briefId] || revealed[briefId];

  return (
    <Widget
      title="Lab: Wire the Gated Cell"
      wide
      caption={
        <>
          Protect a memory cell from hostile distractors, rescue a 25-step highway with forget-gate bias, and pick the largest gated cell under an edge-device parameter budget. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
        {(
          [
            ['gates', 'Brief 1 · Shield Memory'],
            ['bias', 'Brief 2 · Forget Bias'],
            ['budget', 'Brief 3 · Edge Budget'],
          ] as [LstmBriefId, string][]
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

      {briefId === 'gates' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> Four hostile negative pulses try to flip a stored <code>+1.0</code> cue before read time. Set corridor gate levels <code>(f, i, o)</code> to preserve <strong>≥ 55%</strong> of the cue in cell state <code>c₅</code> while keeping corridor output leak <strong>max |hₜ| ≤ 0.15</strong>.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Forget gate f"
              value={st.b1F}
              onChange={(v) => setSt((p) => ({ ...p, b1F: v }))}
              options={LAB_GATE_LEVELS.map((v) => ({ value: v, label: `${v.toFixed(2)}` }))}
            />
            <Segmented
              label="Input gate i"
              value={st.b1I}
              onChange={(v) => setSt((p) => ({ ...p, b1I: v }))}
              options={LAB_GATE_LEVELS.map((v) => ({ value: v, label: `${v.toFixed(2)}` }))}
            />
            <Segmented
              label="Output gate o"
              value={st.b1O}
              onChange={(v) => setSt((p) => ({ ...p, b1O: v }))}
              options={LAB_GATE_LEVELS.map((v) => ({ value: v, label: `${v.toFixed(2)}` }))}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 6 }}>
            {r1Sim.steps.map((s) => (
              <div
                key={s.t}
                style={{
                  textAlign: 'center',
                  padding: '6px 4px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-subtle)',
                  fontFamily: 'var(--mono)',
                  fontSize: 11.5,
                }}
              >
                <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>t={s.t}</div>
                <div style={{ fontWeight: 700, color: s.c >= 0.4 ? 'var(--c-good)' : s.c <= 0 ? 'var(--c-bad)' : 'var(--c-b)' }}>
                  c={s.c >= 0 ? '+' : ''}{s.c.toFixed(2)}
                </div>
                <div style={{ color: 'var(--c-a)', fontSize: 10.5 }}>h={s.h >= 0 ? '+' : ''}{s.h.toFixed(2)}</div>
              </div>
            ))}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Retention (≥ 55%)</div>
              <div className="readout-value" style={{ color: r1.retentionRatio >= 0.55 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {(100 * r1.retentionRatio).toFixed(1)}%
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Leak max |h| (≤ 0.15)</div>
              <div className="readout-value" style={{ color: r1.maxCorridorH <= 0.15 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r1.maxCorridorH.toFixed(3)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'bias' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> A 25-step sequence model stalls on epoch 1 because step-1 gradients arrive at <code>6×10⁻⁸</code>. Pick the architecture and initial forget-gate bias <code>b_f</code> so step-1 gradient stays <strong>≥ 0.04</strong> across 25 steps!
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Cell type"
              value={st.b2Cell}
              onChange={(v) => setSt((p) => ({ ...p, b2Cell: v }))}
              options={LAB_BIAS_CELLS.map((c) => ({ value: c, label: c === 'rnn' ? 'Vanilla RNN' : 'LSTM' }))}
            />
            <Segmented
              label="Forget bias b_f"
              value={st.b2Bias}
              onChange={(v) => setSt((p) => ({ ...p, b2Bias: v }))}
              options={LAB_BIAS_VALS.map((b) => ({ value: b, label: `${b >= 0 ? '+' : ''}${b}` }))}
            />
          </div>
          <LogChart
            series={[
              {
                points: (st.b2Cell === 'rnn' ? r2Flow.rnn.ratios : r2Flow.lstm.ratios).map((v, k): [number, number] => [k, Math.max(1e-4, v)]),
                color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)',
              },
            ]}
            xDomain={[0, 24]}
            yDomain={[1e-4, 2]}
            width={760}
            height={170}
            xTicks={[0, 6, 12, 18, 24]}
            xLabel="Steps back k"
            ariaLabel="Lab 2 gradient survival over 25 steps"
            hlines={[{ y: 0.04, label: 'target 0.04', color: 'var(--text-muted)' }]}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Step-1 ratio (≥ 0.04)</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{fmtSci(r2.ratio)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'budget' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> You are deploying an audio wake-word detector (<code>D = 16</code> mel bands) on a microcontroller capped at <strong>≤ 1,800 parameters</strong>. It needs a <strong>gated highway</strong> (long-range memory) and hidden width <strong>H ≥ 12</strong>. Pick the cell and width that pack the <strong>highest hidden capacity</strong> inside the cap!
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Cell architecture"
              value={st.b3Kind}
              onChange={(v) => setSt((p) => ({ ...p, b3Kind: v }))}
              options={LAB_BUDGET_KINDS.map((k) => ({ value: k, label: k.toUpperCase() }))}
            />
            <Segmented
              label="Hidden width H"
              value={st.b3H}
              onChange={(v) => setSt((p) => ({ ...p, b3H: v }))}
              options={LAB_BUDGET_HS.map((h) => ({ value: h, label: `H = ${h}` }))}
            />
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 12, padding: '6px 10px', borderRadius: 6, background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
            {`nn.${st.b3Kind.toUpperCase()}(input_size=16, hidden_size=${st.b3H})  # ${r3.params.toLocaleString()} params (budget <= 1,800)`}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Params (≤ 1,800)</div>
              <div className="readout-value" style={{ color: r3.params <= 1800 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.params.toLocaleString()}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Gated highway</div>
              <div className="readout-value" style={{ color: r3.isGated ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.isGated ? 'Yes' : 'No (vanishes)'}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Hidden width (≥ 12)</div>
              <div className="readout-value" style={{ color: r3.H >= 12 ? 'var(--c-good)' : 'var(--c-bad)' }}>H = {r3.H}</div>
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
