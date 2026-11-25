// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { LogChart } from '../../components/LogChart';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
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
  lstmLabHint,
  runGatedRace,
  simulateLstmCorridor,
  type CellKind,
  type LstmBriefId,
} from './gated';
import { lstmLabRealCode } from './realCode';

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

// // ------------------------------------------------------------------ 4. Lab: Wire the Gated Cell

type LstmLabState = {
  b1F: number;
  b1I: number;
  b1O: number;
  b2Cell: 'rnn' | 'lstm';
  b2Bias: number;
  b3Kind: CellKind;
  b3H: number;
};

const LSTM_LAB_KEY = 'nl-lstm-lab-v1';
const LSTM_LAB_DEFAULT: LstmLabState = {
  b1F: 0.5,
  b1I: 0.5,
  b1O: 0.5,
  b2Cell: 'rnn',
  b2Bias: 0,
  b3Kind: 'lstm',
  b3H: 16,
};

function readLegacyLstmSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(LSTM_LAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved && typeof p.solved === 'object' ? p.solved : {};
  } catch {
    return {};
  }
}

const LSTM_BRIEFS: { id: LstmBriefId; tab: string; title: string; text: string }[] = [
  {
    id: 'gates',
    tab: 'Brief 1 · Shield Memory',
    title: 'Guard the Cell Conveyor Through Hostile Distractors',
    text: 'Four hostile negative pulses try to flip a stored +1.0 cue before read time. Set corridor gate levels (f, i, o) to preserve ≥ 55% of the cue in cell state c₅ while keeping corridor output leak max |hₜ| ≤ 0.15.',
  },
  {
    id: 'bias',
    tab: 'Brief 2 · Forget Bias',
    title: 'Rescue a 25-Step Gradient Highway',
    text: 'A 25-step sequence model stalls on epoch 1 because step-1 gradients arrive at 6×10⁻⁸. Pick the cell architecture and initial forget-gate bias b_f so step-1 gradient stays ≥ 0.04 across 25 steps.',
  },
  {
    id: 'budget',
    tab: 'Brief 3 · Edge Budget',
    title: 'Pack Maximum Gated Width Inside 1,800 Parameters',
    text: 'You are deploying an audio wake-word detector (D = 16 mel bands) on a microcontroller capped at ≤ 1,800 parameters. It needs a gated highway and hidden width H ≥ 12. Pick the cell and width that pack the highest hidden capacity inside the cap.',
  },
];

const GATE_BLOCK_NAMES: Record<CellKind, string[]> = {
  rnn: ['candidate h'],
  gru: ['reset r', 'update z', 'candidate h̃'],
  lstm: ['input i', 'forget f', 'cell g', 'output o'],
};

export function LstmLab() {
  const [active, setActive] = useState(0);
  const [st, setSt] = useState<LstmLabState>(LSTM_LAB_DEFAULT);
  const progress = useLabProgress('nl-lstm-lab-v2', () => ({ solved: readLegacyLstmSolved() }));
  const log = useMachineLog(12);

  const briefId = LSTM_BRIEFS[active].id;

  const r1 = useMemo(() => evalLstmBrief1(st.b1F, st.b1I, st.b1O), [st.b1F, st.b1I, st.b1O]);
  const r1Sim = useMemo(() => simulateLstmCorridor(st.b1F, st.b1I, st.b1O, 0.95), [st.b1F, st.b1I, st.b1O]);
  const r2 = useMemo(() => evalLstmBrief2(st.b2Cell, st.b2Bias), [st.b2Cell, st.b2Bias]);
  const r2Flow = useMemo(() => carouselGradientFlow(25, st.b2Bias, 0.8), [st.b2Bias]);
  const r3 = useMemo(() => evalLstmBrief3(st.b3Kind, st.b3H), [st.b3Kind, st.b3H]);
  const exhaust = useMemo(() => lstmLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'gates' ? r1.pass : briefId === 'bias' ? r2.pass : r3.pass;
  const curOptimal =
    briefId === 'gates'
      ? r1.pass
      : briefId === 'bias'
        ? r2.pass && st.b2Bias === 3
        : r3.pass && st.b3Kind === 'gru' && st.b3H === 16;

  const forgetOpen = 1 / (1 + Math.exp(-st.b2Bias));
  const tweened = useTweenedArray([
    Math.max(0, Math.min(100, r1.retentionRatio * 100)),
    Math.max(0, Math.min(100, (r1.maxCorridorH / 0.6) * 100)),
    forgetOpen * 100,
    Math.min(100, (r3.params / 3000) * 100),
  ]);

  const hintText = useMemo(() => lstmLabHint(briefId, st), [briefId, st]);
  const liveCode = useMemo(() => lstmLabRealCode(briefId, st), [briefId, st]);

  const gridB1 = useMemo(
    () => LAB_GATE_LEVELS.map((f) => ({ f, cells: LAB_GATE_LEVELS.map((i) => evalLstmBrief1(f, i, st.b1O)) })),
    [st.b1O],
  );
  const gridB2 = useMemo(
    () => LAB_BIAS_CELLS.map((c) => ({ c, cells: LAB_BIAS_VALS.map((b) => evalLstmBrief2(c, b)) })),
    [],
  );
  const gridB3 = useMemo(
    () => LAB_BUDGET_KINDS.map((k) => ({ k, cells: LAB_BUDGET_HS.map((h) => evalLstmBrief3(k, h)) })),
    [],
  );

  const updateB1 = (f: number, i: number, o: number) => {
    setSt((p) => ({ ...p, b1F: f, b1I: i, b1O: o }));
    const res = evalLstmBrief1(f, i, o);
    log.push(
      `B1: f=${f.toFixed(2)} i=${i.toFixed(2)} o=${o.toFixed(2)} -> keep ${(100 * res.retentionRatio).toFixed(1)}%, leak ${res.maxCorridorH.toFixed(3)} (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const updateB2 = (cell: 'rnn' | 'lstm', bias: number) => {
    setSt((p) => ({ ...p, b2Cell: cell, b2Bias: bias }));
    const res = evalLstmBrief2(cell, bias);
    log.push(
      `B2: ${cell.toUpperCase()} b_f=${bias >= 0 ? '+' : ''}${bias} -> step-1 ratio ${fmtSci(res.ratio)} (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const updateB3 = (kind: CellKind, H: number) => {
    setSt((p) => ({ ...p, b3Kind: kind, b3H: H }));
    const res = evalLstmBrief3(kind, H);
    log.push(
      `B3: ${kind.toUpperCase()} H=${H} -> ${res.params.toLocaleString()} params (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const resetBrief = () => {
    if (briefId === 'gates') setSt((p) => ({ ...p, b1F: LSTM_LAB_DEFAULT.b1F, b1I: LSTM_LAB_DEFAULT.b1I, b1O: LSTM_LAB_DEFAULT.b1O }));
    else if (briefId === 'bias') setSt((p) => ({ ...p, b2Cell: LSTM_LAB_DEFAULT.b2Cell, b2Bias: LSTM_LAB_DEFAULT.b2Bias }));
    else setSt((p) => ({ ...p, b3Kind: LSTM_LAB_DEFAULT.b3Kind, b3H: LSTM_LAB_DEFAULT.b3H }));
    log.push(`Reset ${LSTM_BRIEFS[active].tab}`, 'info');
  };

  const statusText =
    briefId === 'gates'
      ? `Cue retention ${(100 * r1.retentionRatio).toFixed(1)}% (≥ 55%) · corridor leak max |h| ${r1.maxCorridorH.toFixed(3)} (≤ 0.15).`
      : briefId === 'bias'
        ? `${st.b2Cell.toUpperCase()} (b_f = ${st.b2Bias >= 0 ? '+' : ''}${st.b2Bias}) · step-1 ratio ${fmtSci(r2.ratio)} (need ≥ 0.04).`
        : `${st.b3Kind.toUpperCase()} H=${st.b3H} · ${r3.params.toLocaleString()} / 1,800 params · ${r3.isGated ? 'gated highway' : 'no gate highway'}.`;

  const revealContent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
        <strong>Exhaustive search ({exhaust.total} configs):</strong> {exhaust.passingCount}/{exhaust.total} pass · Best: <code>{exhaust.bestLabel}</code>
      </div>
      {briefId === 'gates' && (
        <div style={{ overflowX: 'auto' }}>
          <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
            Slice at output gate <code>o = {st.b1O.toFixed(2)}</code> (retention% / leak):
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>f \ i</th>
                {LAB_GATE_LEVELS.map((iv) => (
                  <th key={iv} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>i={iv.toFixed(2)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB1.map((row) => (
                <tr key={row.f}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>f={row.f.toFixed(2)}</td>
                  {row.cells.map((c) => (
                    <td
                      key={c.i}
                      style={{
                        textAlign: 'center',
                        padding: '4px 6px',
                        borderRadius: 4,
                        color: c.pass ? 'var(--c-good)' : 'var(--text-muted)',
                        fontWeight: c.pass ? 700 : 400,
                        background: c.pass ? 'color-mix(in srgb, var(--c-good) 12%, transparent)' : 'transparent',
                      }}
                    >
                      {(100 * c.retentionRatio).toFixed(0)}% / {c.maxCorridorH.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {briefId === 'bias' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>cell \ b_f</th>
                {LAB_BIAS_VALS.map((b) => (
                  <th key={b} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>
                    {b >= 0 ? '+' : ''}{b}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB2.map((row) => (
                <tr key={row.c}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>{row.c.toUpperCase()}</td>
                  {row.cells.map((c) => (
                    <td
                      key={c.forgetBias}
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
      {briefId === 'budget' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>cell \ H</th>
                {LAB_BUDGET_HS.map((h) => (
                  <th key={h} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>H={h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB3.map((row) => (
                <tr key={row.k}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>{row.k.toUpperCase()}</td>
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
                      {c.params.toLocaleString()}
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

  const perGateParams = st.b3H * (16 + st.b3H + 1);
  const blockNames = GATE_BLOCK_NAMES[st.b3Kind];

  return (
    <LabShell
      title="Lab: Wire the Gated Cell"
      caption="Shield a stored cue along the LSTM cell conveyor, open the 25-step forget-gate highway, and pack maximum gated capacity into a 1,800-parameter edge budget."
      briefs={LSTM_BRIEFS}
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
        title: 'GATED CELL BENCH LOG',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Adjust gate levels, forget-gate bias, or cell width to log runs.',
      }}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {briefId === 'gates' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Forget gate f"
              value={st.b1F}
              onChange={(v) => updateB1(v, st.b1I, st.b1O)}
              options={LAB_GATE_LEVELS.map((v) => ({ value: v, label: `${v.toFixed(2)}` }))}
            />
            <Segmented
              label="Input gate i"
              value={st.b1I}
              onChange={(v) => updateB1(st.b1F, v, st.b1O)}
              options={LAB_GATE_LEVELS.map((v) => ({ value: v, label: `${v.toFixed(2)}` }))}
            />
            <Segmented
              label="Output gate o"
              value={st.b1O}
              onChange={(v) => updateB1(st.b1F, st.b1I, v)}
              options={LAB_GATE_LEVELS.map((v) => ({ value: v, label: `${v.toFixed(2)}` }))}
            />
          </div>

          {/* Live SVG Cell-State Conveyor Highway + Corridor Leak Trace */}
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
              <span>Cell-state conveyor <code>cₜ</code> (top) &amp; corridor hidden leak <code>|hₜ|</code> (bottom) across hostile pulses <code>t=1…4</code></span>
              <span style={{ fontFamily: 'var(--mono)' }}>
                valves: f={Math.round(st.b1F * 100)}% · i={Math.round(st.b1I * 100)}% · o={Math.round(st.b1O * 100)}%
              </span>
            </div>
            <svg viewBox="0 0 640 165" style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="LSTM cell-state conveyor and hidden leak across 6 steps">
              {/* Top track: c_t in [-1, +1] mapped to y=16..84 (zero at y=50) */}
              <line x1={38} x2={624} y1={50} y2={50} stroke="var(--border)" strokeWidth={1} />
              <line x1={38} x2={624} y1={31.3} y2={31.3} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1} opacity={0.7} />
              <text x={32} y={53} textAnchor="end" fontSize={10} fill="var(--text-muted)" fontFamily="var(--mono)">c=0</text>
              <text x={32} y={22} textAnchor="end" fontSize={10} fill="var(--text-muted)" fontFamily="var(--mono)">+1</text>
              <text x={32} y={82} textAnchor="end" fontSize={10} fill="var(--text-muted)" fontFamily="var(--mono)">−1</text>

              {/* Conveyor polyline */}
              <polyline
                points={r1Sim.steps.map((s, idx) => `${76 + idx * 100},${50 - Math.max(-1, Math.min(1, s.c)) * 32}`).join(' ')}
                fill="none"
                stroke={r1.retentionRatio >= 0.55 ? 'var(--c-good)' : 'var(--c-bad)'}
                strokeWidth={2.4}
              />

              {/* Bottom track: |h_t| in [0, 0.8] mapped to y=144..102 */}
              <line x1={38} x2={624} y1={144} y2={144} stroke="var(--border)" strokeWidth={1} />
              <line x1={38} x2={624} y1={144 - (0.15 / 0.8) * 40} y2={144 - (0.15 / 0.8) * 40} stroke="var(--c-b)" strokeDasharray="3 3" strokeWidth={1.1} />
              <text x={32} y={146} textAnchor="end" fontSize={10} fill="var(--text-muted)" fontFamily="var(--mono)">|h|</text>
              <text x={636} y={114} textAnchor="end" fontSize={9.5} fill="var(--c-b)" fontFamily="var(--mono)">cap 0.15</text>

              {r1Sim.steps.map((s, idx) => {
                const cx = 76 + idx * 100;
                const cy = 50 - Math.max(-1, Math.min(1, s.c)) * 32;
                const absH = Math.min(0.8, Math.abs(s.h));
                const bh = (absH / 0.8) * 40;
                const isCorridor = s.t >= 1 && s.t <= 4;
                const leakFail = isCorridor && Math.abs(s.h) > 0.15;
                return (
                  <g key={s.t}>
                    {isCorridor && (
                      <rect x={cx - 32} y={10} width={64} height={148} rx={6} fill="var(--bg-elev)" opacity={0.45} />
                    )}
                    <circle cx={cx} cy={cy} r={4.5} fill={s.c >= 0.4 ? 'var(--c-good)' : s.c <= 0 ? 'var(--c-bad)' : 'var(--c-b)'} />
                    <text x={cx} y={cy + (s.c >= 0 ? -8 : 13)} textAnchor="middle" fontSize={10} fontWeight={700} fontFamily="var(--mono)" fill={s.c >= 0.4 ? 'var(--c-good)' : s.c <= 0 ? 'var(--c-bad)' : 'var(--c-b)'}>
                      {s.c >= 0 ? '+' : ''}{s.c.toFixed(2)}
                    </text>
                    <rect
                      x={cx - 14}
                      y={144 - bh}
                      width={28}
                      height={Math.max(2, bh)}
                      rx={3}
                      fill={leakFail ? 'var(--c-bad)' : isCorridor ? 'var(--c-good)' : 'var(--c-a)'}
                    />
                    <text x={cx} y={159} textAnchor="middle" fontSize={10} fontFamily="var(--mono)" fill="var(--text-muted)">
                      t={s.t}{s.t === 0 ? ' cue' : s.t === 5 ? ' read' : ' −1'}
                    </text>
                  </g>
                );
              })}
            </svg>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 12 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                  <span>Cue retention at c₅ (≥ 55%)</span>
                  <strong style={{ color: r1.retentionRatio >= 0.55 ? 'var(--c-good)' : 'var(--c-bad)' }}>{(100 * r1.retentionRatio).toFixed(1)}%</strong>
                </div>
                <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                  <div style={{ width: `${tweened[0]}%`, height: '100%', borderRadius: 4, background: r1.retentionRatio >= 0.55 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', left: '55%', top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                  <span>Corridor leak max |hₜ| (≤ 0.15)</span>
                  <strong style={{ color: r1.maxCorridorH <= 0.15 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.maxCorridorH.toFixed(3)}</strong>
                </div>
                <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                  <div style={{ width: `${tweened[1]}%`, height: '100%', borderRadius: 4, background: r1.maxCorridorH <= 0.15 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', left: '25%', top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
                </div>
              </div>
            </div>
          </div>
          <LiveCode note="PyTorch LSTM cell update" code={liveCode} />
        </div>
      )}

      {briefId === 'bias' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Cell type"
              value={st.b2Cell}
              onChange={(v) => updateB2(v, st.b2Bias)}
              options={LAB_BIAS_CELLS.map((c) => ({ value: c, label: c === 'rnn' ? 'Vanilla RNN' : 'LSTM' }))}
            />
            <Segmented
              label="Forget bias b_f"
              value={st.b2Bias}
              onChange={(v) => updateB2(st.b2Cell, v)}
              options={LAB_BIAS_VALS.map((b) => ({ value: b, label: `${b >= 0 ? '+' : ''}${b}` }))}
            />
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <LogChart
              series={[
                ...(st.b2Cell === 'lstm'
                  ? [{ points: r2Flow.rnn.ratios.map((v, k): [number, number] => [k, Math.max(1e-4, v)]), color: 'var(--text-muted)', dashed: true }]
                  : []),
                {
                  points: (st.b2Cell === 'rnn' ? r2Flow.rnn.ratios : r2Flow.lstm.ratios).map((v, k): [number, number] => [k, Math.max(1e-4, v)]),
                  color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)',
                },
              ]}
              xDomain={[0, 24]}
              yDomain={[1e-4, 2]}
              width={740}
              height={170}
              xTicks={[0, 6, 12, 18, 24]}
              xLabel="Steps back k"
              ariaLabel="Lab 2 gradient survival over 25 steps"
              hlines={[{ y: 0.04, label: 'target 0.04', color: 'var(--text-muted)' }]}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)' }}>
                <span>Initial forget-gate openness σ(b_f = {st.b2Bias >= 0 ? '+' : ''}{st.b2Bias})</span>
                <strong style={{ color: st.b2Cell === 'lstm' && r2.pass ? 'var(--c-good)' : 'var(--text-muted)' }}>
                  {st.b2Cell === 'rnn' ? 'N/A (no forget gate)' : `${(forgetOpen * 100).toFixed(1)}% per step → ${fmtSci(r2.ratio)} at step 1`}
                </strong>
              </div>
              <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                <div
                  style={{
                    width: `${st.b2Cell === 'rnn' ? 0 : tweened[2]}%`,
                    height: '100%',
                    borderRadius: 4,
                    background: r2.pass ? 'var(--c-good)' : 'var(--c-b)',
                  }}
                />
              </div>
            </div>
          </div>
          <LiveCode note="PyTorch LSTM forget-gate bias init" code={liveCode} />
        </div>
      )}

      {briefId === 'budget' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Cell architecture"
              value={st.b3Kind}
              onChange={(v) => updateB3(v, st.b3H)}
              options={LAB_BUDGET_KINDS.map((k) => ({ value: k, label: k.toUpperCase() }))}
            />
            <Segmented
              label="Hidden width H"
              value={st.b3H}
              onChange={(v) => updateB3(st.b3Kind, v)}
              options={LAB_BUDGET_HS.map((h) => ({ value: h, label: `H = ${h}` }))}
            />
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Gate-matrix parameter anatomy: <strong>{blockNames.length}×</strong> gate block{blockNames.length > 1 ? 's' : ''} × <code>{perGateParams}</code> params each (<code>H·(16 + H + 1)</code>)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${blockNames.length}, 1fr)`, gap: 8 }}>
              {blockNames.map((name) => (
                <div
                  key={name}
                  style={{
                    padding: '8px 6px',
                    borderRadius: 6,
                    textAlign: 'center',
                    fontFamily: 'var(--mono)',
                    fontSize: 11.5,
                    background: 'var(--bg-elev)',
                    border: `1px solid ${r3.pass ? 'var(--c-good)' : r3.isGated ? 'var(--c-b)' : 'var(--c-bad)'}`,
                  }}
                >
                  <div style={{ fontWeight: 700 }}>{name}</div>
                  <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>{perGateParams} params</div>
                </div>
              ))}
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                <span>Microcontroller parameter budget (cap = 1,800)</span>
                <strong style={{ color: r3.params <= 1800 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {r3.params.toLocaleString()} / 1,800
                </strong>
              </div>
              <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                <div
                  style={{
                    width: `${tweened[3]}%`,
                    height: '100%',
                    borderRadius: 4,
                    background: r3.params <= 1800 ? (r3.isGated && r3.H >= 12 ? 'var(--c-good)' : 'var(--c-b)') : 'var(--c-bad)',
                  }}
                />
                <div style={{ position: 'absolute', left: '60%', top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
              </div>
            </div>
          </div>
          <LiveCode note="PyTorch recurrent cell sizing" code={liveCode} />
        </div>
      )}
    </LabShell>
  );
}
