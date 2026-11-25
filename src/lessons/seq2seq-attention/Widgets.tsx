// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
import {
  computeArchMetrics,
  computeCrossAttention,
  evalSeqBrief1,
  evalSeqBrief2,
  evalSeqBrief3,
  evaluateBottleneck,
  GOLD_ALIGNMENT,
  LAB_ALIGN_KINDS,
  LAB_ALIGN_TAUS,
  LAB_BN_ARCHS,
  LAB_BN_DS,
  LAB_LAT_ARCHS,
  seqLabExhaustive,
  seqLabHint,
  SOURCE_TOKENS,
  TARGET_TOKENS,
  type ScoreKind,
  type SeqArchId,
  type SeqBriefId,
} from './seq2seq';
import { seqLabRealCode } from './realCode';

// ------------------------------------------------------------------ 1. Fixed-Vector Bottleneck vs Cross-Attention

export function BottleneckWidget() {
  const [d, setD] = useState<number>(16);
  const rows = useMemo(() => evaluateBottleneck(d), [d]);

  return (
    <Widget
      title="The information bottleneck: one fixed vector vs attending to all L states"
      wide
      caption={
        <>
          Exact token-copying accuracy as sentence length grows from <code>L = 2 → 24</code> tokens. Squeezing the entire sentence into a single <strong>d = {d}</strong> vector collapses from <strong>100% → {(100 * rows[rows.length - 1].fixedAcc).toFixed(1)}%</strong> by <code>L = 24</code> due to superposition cross-talk — while <strong>Cross-Attention</strong> keeps all <code>L</code> encoder states and holds <strong>100.0%</strong> at every length!
        </>
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="Encoder state width d"
          value={d}
          onChange={setD}
          options={[
            { value: 8, label: 'd = 8' },
            { value: 16, label: 'd = 16' },
            { value: 32, label: 'd = 32' },
          ]}
        />
      </div>

      <Plot x={[2, 24]} y={[0, 105]} width={760} height={215} xLabel="Sentence length L (tokens)" yLabel="Token accuracy (%)">
        {({ x: sx, y: sy }) => (
          <>
            <line x1={sx(2)} x2={sx(24)} y1={sy(12.5)} y2={sy(12.5)} stroke="var(--text-muted)" strokeDasharray="3 4" strokeWidth={1.1} />
            <text x={sx(23.5)} y={sy(12.5) - 5} textAnchor="end" fontSize={10.5} fill="var(--text-muted)" fontFamily="var(--mono)">
              random chance (12.5%)
            </text>
            <polyline
              points={rows.map((r) => `${sx(r.length)},${sy(100 * r.attnAcc)}`).join(' ')}
              fill="none"
              stroke="var(--c-good)"
              strokeWidth={2.6}
            />
            <polyline
              points={rows.map((r) => `${sx(r.length)},${sy(100 * r.fixedAcc)}`).join(' ')}
              fill="none"
              stroke="var(--c-bad)"
              strokeWidth={2.6}
            />
            {rows.map((r) => (
              <g key={r.length}>
                <circle cx={sx(r.length)} cy={sy(100 * r.attnAcc)} r={4} fill="var(--c-good)" />
                <circle cx={sx(r.length)} cy={sy(100 * r.fixedAcc)} r={4} fill="var(--c-bad)" />
              </g>
            ))}
          </>
        )}
      </Plot>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, margin: '6px 0 10px', color: 'var(--text-muted)' }}>
        <span><span style={{ color: 'var(--c-good)', fontWeight: 700 }}>━━</span> Cross-Attention (O(L·d) memory)</span>
        <span><span style={{ color: 'var(--c-bad)', fontWeight: 700 }}>━━</span> Fixed-Vector Bottleneck (single vector c ∈ ℝ^{d})</span>
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8, margin: 0 }}>
        <div className="readout">
          <div className="readout-label">Fixed at L = 4</div>
          <div className="readout-value">{(100 * rows[1].fixedAcc).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Fixed at L = 12</div>
          <div className="readout-value" style={{ color: 'var(--c-b)' }}>{(100 * rows[3].fixedAcc).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Fixed at L = 24</div>
          <div className="readout-value" style={{ color: 'var(--c-bad)' }}>{(100 * rows[rows.length - 1].fixedAcc).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Cross-Attn at L = 24</div>
          <div className="readout-value" style={{ color: 'var(--c-good)' }}>100.0%</div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Cross-Attention Alignment Heatmap

export function AlignmentHeatmapWidget() {
  const [kind, setKind] = useState<ScoreKind>('luong');
  const [tau, setTau] = useState<number>(0.15);
  const [selectedT, setSelectedT] = useState<number>(2); // "noir" -> "black" (swap!)

  const res = useMemo(() => computeCrossAttention(kind, tau), [kind, tau]);
  const activeRow = res.weights[selectedT];
  const goldS = GOLD_ALIGNMENT[selectedT];

  return (
    <Widget
      title="Cross-attention alignment: translating & reordering English → French"
      wide
      caption={
        <>
          Click any French target row to inspect its exact softmax attention weights over the English source tokens. Notice the two off-diagonal swaps: <strong>“chat”</strong> skips ahead to <strong>“cat”</strong>, then <strong>“noir”</strong> jumps back to <strong>“black”</strong> (<code>{(100 * res.weights[2][1]).toFixed(1)}%</code> weight) — and the exact same swap happens for <strong>“l’oiseau rouge”</strong>!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Scoring rule"
          value={kind}
          onChange={setKind}
          options={[
            { value: 'luong', label: 'Luong dot (sₜ · hₛ)' },
            { value: 'bahdanau', label: 'Bahdanau additive (vᵀ tanh)' },
          ]}
        />
        <div style={{ flex: '1 1 200px' }}>
          <Slider
            label="Temperature τ"
            value={tau}
            min={0.05}
            max={1.0}
            step={0.05}
            onChange={setTau}
            format={(v) => `τ = ${v.toFixed(2)}`}
          />
        </div>
      </div>

      <div style={{ overflowX: 'auto', marginBottom: 12 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-muted)' }}>Target ↓ \ Source →</th>
              {SOURCE_TOKENS.map((tok, s) => (
                <th key={s} style={{ textAlign: 'center', padding: '6px 6px', color: s === goldS ? 'var(--c-good)' : 'var(--text)' }}>
                  {tok}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {TARGET_TOKENS.map((tok, t) => {
              const isSel = t === selectedT;
              return (
                <tr
                  key={t}
                  onClick={() => setSelectedT(t)}
                  style={{
                    cursor: 'pointer',
                    outline: isSel ? '2px solid var(--accent)' : 'none',
                    borderRadius: 4,
                  }}
                >
                  <td style={{ padding: '6px 8px', fontWeight: isSel ? 700 : 500, color: isSel ? 'var(--accent)' : 'var(--text)' }}>
                    {isSel ? '▸ ' : ''}{tok}
                  </td>
                  {res.weights[t].map((w, s) => {
                    const isGold = s === GOLD_ALIGNMENT[t];
                    const opacity = Math.max(0.04, Math.min(0.92, w));
                    return (
                      <td
                        key={s}
                        style={{
                          textAlign: 'center',
                          padding: '6px 4px',
                          border: isGold ? '1.5px solid var(--c-good)' : '1px solid var(--border)',
                          background: `color-mix(in srgb, var(--accent) ${Math.round(opacity * 100)}%, transparent)`,
                          color: w > 0.45 ? 'var(--bg)' : 'var(--text)',
                          fontWeight: w > 0.3 ? 700 : 400,
                        }}
                      >
                        {(100 * w).toFixed(0)}%
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
        <div className="readout">
          <div className="readout-label">Selected query</div>
          <div className="readout-value" style={{ color: 'var(--accent)' }}>
            {TARGET_TOKENS[selectedT]} → {SOURCE_TOKENS[goldS]}
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Weight on gold word</div>
          <div className="readout-value" style={{ color: activeRow[goldS] >= 0.8 ? 'var(--c-good)' : 'var(--c-b)' }}>
            {(100 * activeRow[goldS]).toFixed(1)}%
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Sentence avg gold</div>
          <div className="readout-value">{(100 * res.meanGoldWeight).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Max |logit|</div>
          <div className="readout-value" style={{ color: res.maxAbsLogit > 15 ? 'var(--c-bad)' : 'var(--text)' }}>
            {res.maxAbsLogit.toFixed(1)}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Why Attention Killed Recurrence (Parallelism)

export function ParallelismWidget() {
  const [L, setL] = useState<number>(512);
  const rows = useMemo(() => LAB_LAT_ARCHS.map((a) => computeArchMetrics(a, L)), [L]);
  const maxLat = Math.max(...rows.map((r) => r.trainLatencyMs));

  return (
    <Widget
      title="Why attention killed recurrence: sequential depth O(L) vs GPU parallelism O(1)"
      wide
      caption={
        <>
          Bahdanau’s <strong>LSTM + Cross-Attention</strong> broke the fixed-vector bottleneck (max path drops from <code>L → 1</code>), yet training still runs <strong>2L serial steps</strong>! At <code>L = 512</code> tokens, dropping recurrence for a parallel <strong>Transformer</strong> cuts training step latency <strong>11×</strong> (<code>85.07 ms → 7.75 ms</code>) — bridging directly into Module 12!
        </>
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="Sequence length L"
          value={L}
          onChange={setL}
          options={[
            { value: 64, label: 'L = 64' },
            { value: 256, label: 'L = 256' },
            { value: 512, label: 'L = 512' },
            { value: 2048, label: 'L = 2,048' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {rows.map((m) => {
          const pct = (100 * m.trainLatencyMs) / Math.max(1, maxLat);
          const col = m.arch === 'transformer' ? 'var(--c-good)' : m.arch === 'lstm-attn' ? 'var(--accent)' : 'var(--c-bad)';
          return (
            <div key={m.arch} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'var(--bg-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 6, fontSize: 13 }}>
                <strong>{m.label}</strong>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-muted)' }}>
                  Serial depth: <strong>{m.seqDepth.toLocaleString()}</strong> · Max path: <strong>{m.maxPathLen}</strong> · Latency: <strong style={{ color: col }}>{m.trainLatencyMs.toFixed(2)} ms</strong>
                </span>
              </div>
              <div style={{ height: 12, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ width: `${Math.max(2, pct).toFixed(1)}%`, height: '100%', background: col }} />
              </div>
            </div>
          );
        })}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Break the Bottleneck

type SeqLabState = {
  b1Arch: 'fixed' | 'cross-attn';
  b1D: number;
  b2Kind: ScoreKind;
  b2Tau: number;
  b3Arch: SeqArchId;
};

const SEQ_LAB_KEY = 'nl-seq2seq-lab-v1';
const SEQ_LAB_DEFAULT: SeqLabState = {
  b1Arch: 'fixed',
  b1D: 16,
  b2Kind: 'bahdanau',
  b2Tau: 1.0,
  b3Arch: 'lstm-attn',
};

function readLegacySeqSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(SEQ_LAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved && typeof p.solved === 'object' ? p.solved : {};
  } catch {
    return {};
  }
}

const SEQ_BRIEFS: { id: SeqBriefId; tab: string; title: string; text: string }[] = [
  {
    id: 'bottleneck',
    tab: 'Brief 1 · 20-Token Copy',
    title: 'Escape the Fixed-Vector Sentence Crash',
    text: 'Your translation model works on 2-word phrases but drops to 33.8% on 20-word sentences. Pick the decoder bridge and state width d to reach ≥ 95% accuracy on L = 20 tokens (optional goal: smallest state width d).',
  },
  {
    id: 'align',
    tab: 'Brief 2 · Sharp Alignment',
    title: 'Sharpen Cross-Attention Without Logit Overflow',
    text: 'Tune scoring rule and temperature on “the black cat saw the red bird” → “le chat noir vit l’oiseau rouge” so mean gold alignment weight reaches ≥ 85% while keeping max |logit| ≤ 15.0 (safe FP16 softmax).',
  },
  {
    id: 'latency',
    tab: 'Brief 3 · 20 ms GPU SLA',
    title: 'Hit a 20 ms Training SLA on 512-Token Documents',
    text: 'You are training on L = 512-token documents. Your cluster requires both a direct O(1) token path (no fixed-vector bottleneck) AND forward-step latency ≤ 20.0 ms.',
  },
];

export function Seq2SeqLab() {
  const [active, setActive] = useState(0);
  const [st, setSt] = useState<SeqLabState>(SEQ_LAB_DEFAULT);
  const progress = useLabProgress('nl-seq2seq-lab-v2', () => ({ solved: readLegacySeqSolved() }));
  const log = useMachineLog(12);

  const briefId = SEQ_BRIEFS[active].id;

  const r1 = useMemo(() => evalSeqBrief1(st.b1Arch, st.b1D), [st.b1Arch, st.b1D]);
  const b1Curve = useMemo(() => evaluateBottleneck(st.b1D), [st.b1D]);
  const r2 = useMemo(() => evalSeqBrief2(st.b2Kind, st.b2Tau), [st.b2Kind, st.b2Tau]);
  const r2Align = useMemo(() => computeCrossAttention(st.b2Kind, st.b2Tau), [st.b2Kind, st.b2Tau]);
  const r3 = useMemo(() => evalSeqBrief3(st.b3Arch), [st.b3Arch]);
  const exhaust = useMemo(() => seqLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'bottleneck' ? r1.pass : briefId === 'align' ? r2.pass : r3.pass;
  const curOptimal =
    briefId === 'bottleneck'
      ? r1.pass && st.b1D === 8
      : briefId === 'align'
        ? r2.pass
        : r3.pass;

  const tweened = useTweenedArray([
    r1.acc * 100,
    r2.meanGoldWeight * 100,
    Math.min(100, (r2.maxAbsLogit / 25) * 100),
    Math.min(100, (r3.trainLatencyMs / 95) * 100),
  ]);

  const hintText = useMemo(() => seqLabHint(briefId, st), [briefId, st]);
  const liveCode = useMemo(() => seqLabRealCode(briefId, st), [briefId, st]);

  const gridB1 = useMemo(
    () => LAB_BN_ARCHS.map((arch) => ({ arch, cells: LAB_BN_DS.map((d) => evalSeqBrief1(arch, d)) })),
    [],
  );
  const gridB2 = useMemo(
    () => LAB_ALIGN_KINDS.map((kind) => ({ kind, cells: LAB_ALIGN_TAUS.map((tau) => evalSeqBrief2(kind, tau)) })),
    [],
  );
  const gridB3 = useMemo(() => LAB_LAT_ARCHS.map((arch) => evalSeqBrief3(arch)), []);

  const updateB1 = (arch: 'fixed' | 'cross-attn', d: number) => {
    setSt((p) => ({ ...p, b1Arch: arch, b1D: d }));
    const res = evalSeqBrief1(arch, d);
    log.push(
      `B1: ${arch} d=${d} -> L=20 acc ${(100 * res.acc).toFixed(1)}% (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const updateB2 = (kind: ScoreKind, tau: number) => {
    setSt((p) => ({ ...p, b2Kind: kind, b2Tau: tau }));
    const res = evalSeqBrief2(kind, tau);
    log.push(
      `B2: ${kind} tau=${tau} -> gold ${(100 * res.meanGoldWeight).toFixed(1)}%, max|e|=${res.maxAbsLogit.toFixed(1)} (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const updateB3 = (arch: SeqArchId) => {
    setSt((p) => ({ ...p, b3Arch: arch }));
    const res = evalSeqBrief3(arch);
    log.push(
      `B3: ${res.label} -> ${res.trainLatencyMs.toFixed(2)} ms, depth=${res.seqDepth}, path=${res.maxPathLen} (${res.pass ? 'PASS' : 'FAIL'})`,
      res.pass ? 'good' : 'bad',
    );
  };

  const resetBrief = () => {
    if (briefId === 'bottleneck') setSt((p) => ({ ...p, b1Arch: SEQ_LAB_DEFAULT.b1Arch, b1D: SEQ_LAB_DEFAULT.b1D }));
    else if (briefId === 'align') setSt((p) => ({ ...p, b2Kind: SEQ_LAB_DEFAULT.b2Kind, b2Tau: SEQ_LAB_DEFAULT.b2Tau }));
    else setSt((p) => ({ ...p, b3Arch: SEQ_LAB_DEFAULT.b3Arch }));
    log.push(`Reset ${SEQ_BRIEFS[active].tab}`, 'info');
  };

  const statusText =
    briefId === 'bottleneck'
      ? `L = 20 copy accuracy ${(100 * r1.acc).toFixed(1)}% (≥ 95%) · bridge footprint ${st.b1Arch === 'fixed' ? `${st.b1D} floats` : `${20 * st.b1D} floats (20×${st.b1D})`}.`
      : briefId === 'align'
        ? `Mean gold alignment ${(100 * r2.meanGoldWeight).toFixed(1)}% (≥ 85%) · max |logit| ${r2.maxAbsLogit.toFixed(2)} (≤ 15.0).`
        : `${r3.label} · ${r3.trainLatencyMs.toFixed(2)} ms (≤ 20 ms) · serial depth ${r3.seqDepth.toLocaleString()} · max path ${r3.maxPathLen}.`;

  const revealContent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
        <strong>Exhaustive search ({exhaust.total} configs):</strong> {exhaust.passingCount}/{exhaust.total} pass · Best: <code>{exhaust.bestLabel}</code>
      </div>
      {briefId === 'bottleneck' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>bridge \ d</th>
                {LAB_BN_DS.map((d) => (
                  <th key={d} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>d = {d}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB1.map((row) => (
                <tr key={row.arch}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>{row.arch}</td>
                  {row.cells.map((c) => (
                    <td
                      key={c.d}
                      style={{
                        textAlign: 'center',
                        padding: '4px 6px',
                        borderRadius: 4,
                        color: c.pass ? 'var(--c-good)' : 'var(--text-muted)',
                        fontWeight: c.pass ? 700 : 400,
                        background: c.pass ? 'color-mix(in srgb, var(--c-good) 12%, transparent)' : 'transparent',
                      }}
                    >
                      {(100 * c.acc).toFixed(1)}%
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {briefId === 'align' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>score \ τ</th>
                {LAB_ALIGN_TAUS.map((t) => (
                  <th key={t} style={{ textAlign: 'center', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>τ = {t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gridB2.map((row) => (
                <tr key={row.kind}>
                  <td style={{ padding: '4px 6px', fontWeight: 600 }}>{row.kind}</td>
                  {row.cells.map((c) => (
                    <td
                      key={c.tau}
                      style={{
                        textAlign: 'center',
                        padding: '4px 6px',
                        borderRadius: 4,
                        color: c.pass ? 'var(--c-good)' : c.maxAbsLogit > 15 ? 'var(--c-bad)' : 'var(--text-muted)',
                        fontWeight: c.pass ? 700 : 400,
                        background: c.pass ? 'color-mix(in srgb, var(--c-good) 12%, transparent)' : 'transparent',
                      }}
                    >
                      {(100 * c.meanGoldWeight).toFixed(0)}% / |e|={c.maxAbsLogit.toFixed(1)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {briefId === 'latency' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>architecture</th>
                <th style={{ textAlign: 'right', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>serial depth</th>
                <th style={{ textAlign: 'right', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>max path</th>
                <th style={{ textAlign: 'right', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>latency</th>
              </tr>
            </thead>
            <tbody>
              {gridB3.map((c) => (
                <tr
                  key={c.arch}
                  style={{
                    color: c.pass ? 'var(--c-good)' : 'var(--text-muted)',
                    fontWeight: c.pass ? 700 : 400,
                    background: c.pass ? 'color-mix(in srgb, var(--c-good) 12%, transparent)' : 'transparent',
                  }}
                >
                  <td style={{ padding: '4px 6px' }}>{c.label}</td>
                  <td style={{ textAlign: 'right', padding: '4px 6px' }}>{c.seqDepth.toLocaleString()}</td>
                  <td style={{ textAlign: 'right', padding: '4px 6px' }}>{c.maxPathLen}</td>
                  <td style={{ textAlign: 'right', padding: '4px 6px' }}>{c.trainLatencyMs.toFixed(2)} ms</td>
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
      title="Lab: Break the Bottleneck"
      caption="Escape the 20-token fixed-vector crash, sharpen cross-attention across an adjective–noun swap without logit overflow, and hit a 20 ms GPU SLA on 512-token documents."
      briefs={SEQ_BRIEFS}
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
        title: 'SEQ2SEQ BENCH LOG',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Switch bridge, score rule, temperature, or architecture to log runs.',
      }}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {briefId === 'bottleneck' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Encoder–Decoder bridge"
              value={st.b1Arch}
              onChange={(v) => updateB1(v, st.b1D)}
              options={LAB_BN_ARCHS.map((a) => ({ value: a, label: a === 'fixed' ? 'Fixed single vector c' : 'Cross-Attention (all hₛ)' }))}
            />
            <Segmented
              label="State width d"
              value={st.b1D}
              onChange={(v) => updateB1(st.b1Arch, v)}
              options={LAB_BN_DS.map((d) => ({ value: d, label: `d = ${d}` }))}
            />
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
              <span style={{ padding: '4px 8px', borderRadius: 5, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                Encoder (20 tokens)
              </span>
              <span>→</span>
              <span
                style={{
                  padding: '4px 8px',
                  borderRadius: 5,
                  background: 'var(--bg-elev)',
                  border: `1.5px solid ${st.b1Arch === 'fixed' ? 'var(--c-bad)' : 'var(--c-good)'}`,
                  color: st.b1Arch === 'fixed' ? 'var(--c-bad)' : 'var(--c-good)',
                  fontWeight: 700,
                }}
              >
                {st.b1Arch === 'fixed' ? `1 vector c ∈ ℝ^${st.b1D} (bottleneck!)` : `All 20 states H ∈ ℝ^(20×${st.b1D})`}
              </span>
              <span>→</span>
              <span style={{ padding: '4px 8px', borderRadius: 5, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                Decoder ({(100 * r1.acc).toFixed(1)}% at L=20)
              </span>
            </div>
            <Plot x={[2, 24]} y={[0, 105]} width={740} height={165} xLabel="Sentence length L" yLabel="Copy acc (%)">
              {({ x: sx, y: sy }) => (
                <>
                  <line x1={sx(2)} x2={sx(24)} y1={sy(95)} y2={sy(95)} stroke="var(--c-good)" strokeDasharray="4 4" strokeWidth={1.1} />
                  <line x1={sx(20)} x2={sx(20)} y1={sy(0)} y2={sy(105)} stroke="var(--text-muted)" strokeDasharray="3 3" strokeWidth={1.1} />
                  <text x={sx(20) - 4} y={sy(12)} textAnchor="end" fontSize={10} fill="var(--text-muted)" fontFamily="var(--mono)">L = 20 target</text>
                  <polyline
                    points={b1Curve.map((pt) => `${sx(pt.length)},${sy((st.b1Arch === 'fixed' ? pt.fixedAcc : pt.attnAcc) * 100)}`).join(' ')}
                    fill="none"
                    stroke={r1.pass ? 'var(--c-good)' : 'var(--c-bad)'}
                    strokeWidth={2.5}
                  />
                  {b1Curve.map((pt) => {
                    const yVal = (st.b1Arch === 'fixed' ? pt.fixedAcc : pt.attnAcc) * 100;
                    const isTarget = pt.length === 20;
                    return (
                      <circle
                        key={pt.length}
                        cx={sx(pt.length)}
                        cy={sy(yVal)}
                        r={isTarget ? 5.5 : 3.2}
                        fill={r1.pass ? 'var(--c-good)' : 'var(--c-bad)'}
                        stroke={isTarget ? 'var(--bg)' : 'none'}
                        strokeWidth={isTarget ? 1.8 : 0}
                      />
                    );
                  })}
                </>
              )}
            </Plot>
          </div>
          <LiveCode note="PyTorch Seq2Seq bridge" code={liveCode} />
        </div>
      )}

      {briefId === 'align' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Score function"
              value={st.b2Kind}
              onChange={(v) => updateB2(v, st.b2Tau)}
              options={LAB_ALIGN_KINDS.map((k) => ({ value: k, label: k === 'luong' ? 'Luong dot' : 'Bahdanau additive' }))}
            />
            <Segmented
              label="Temperature"
              value={st.b2Tau}
              onChange={(v) => updateB2(st.b2Kind, v)}
              options={LAB_ALIGN_TAUS.map((t) => ({ value: t, label: `τ = ${t}` }))}
            />
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Live 6×7 French → English cross-attention matrix (outlined cells = gold translation alignment, incl. <em>chat↔cat / noir↔black</em> swap)
            </div>
            <div style={{ overflowX: 'auto' }}>
              <div style={{ display: 'grid', gridTemplateColumns: `78px repeat(${SOURCE_TOKENS.length}, minmax(44px, 1fr))`, gap: 4, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 11 }}>
                <div />
                {SOURCE_TOKENS.map((src, s) => (
                  <div key={s} style={{ textAlign: 'center', color: 'var(--text-muted)', fontWeight: 600 }}>{src}</div>
                ))}
                {TARGET_TOKENS.map((tgt, t) => (
                  <div key={t} style={{ display: 'contents' }}>
                    <div style={{ textAlign: 'right', paddingRight: 6, fontWeight: 600 }}>{tgt}</div>
                    {r2Align.weights[t].map((w, s) => {
                      const isGold = GOLD_ALIGNMENT[t] === s;
                      const pct = Math.round(w * 100);
                      return (
                        <div
                          key={s}
                          style={{
                            textAlign: 'center',
                            padding: '5px 2px',
                            borderRadius: 4,
                            background: `color-mix(in srgb, var(--accent) ${Math.round(8 + w * 82)}%, var(--bg-elev))`,
                            border: isGold ? `1.8px solid ${w >= 0.85 ? 'var(--c-good)' : 'var(--c-b)'}` : '1px solid var(--border)',
                            color: w > 0.55 ? 'var(--bg)' : 'var(--text)',
                            fontWeight: isGold ? 700 : 400,
                          }}
                        >
                          {pct}%
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 12 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                  <span>Mean gold weight (≥ 85%)</span>
                  <strong style={{ color: r2.meanGoldWeight >= 0.85 ? 'var(--c-good)' : 'var(--c-bad)' }}>{(100 * r2.meanGoldWeight).toFixed(1)}%</strong>
                </div>
                <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                  <div style={{ width: `${tweened[1]}%`, height: '100%', borderRadius: 4, background: r2.meanGoldWeight >= 0.85 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', left: '85%', top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                  <span>Max |logit| (FP16 ceiling ≤ 15.0)</span>
                  <strong style={{ color: r2.maxAbsLogit <= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.maxAbsLogit.toFixed(2)}</strong>
                </div>
                <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                  <div style={{ width: `${tweened[2]}%`, height: '100%', borderRadius: 4, background: r2.maxAbsLogit <= 15 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', left: '60%', top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
                </div>
              </div>
            </div>
          </div>
          <LiveCode note="PyTorch cross-attention scoring" code={liveCode} />
        </div>
      )}

      {briefId === 'latency' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="Sequence architecture (L = 512 tokens)"
            value={st.b3Arch}
            onChange={updateB3}
            options={LAB_LAT_ARCHS.map((a) => ({
              value: a,
              label: a === 'fixed-lstm' ? 'Fixed LSTM' : a === 'lstm-attn' ? 'LSTM + Cross-Attn' : 'Transformer',
            }))}
          />
          <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              GPU execution schedule for <code>L = 512</code> tokens — {r3.seqDepth === 1 ? 'all 512 positions launch in 1 parallel matmul wave' : `${r3.seqDepth.toLocaleString()} serial token-by-token dependencies stall GPU cores`}
            </div>
            <svg viewBox="0 0 640 72" style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="GPU execution waves for selected sequence architecture">
              {r3.seqDepth === 1 ? (
                <g>
                  {Array.from({ length: 16 }, (_, idx) => (
                    <rect key={idx} x={16 + idx * 38} y={14} width={32} height={24} rx={4} fill="var(--c-good)" opacity={0.85} />
                  ))}
                  <text x={320} y={58} textAnchor="middle" fontSize={11} fontFamily="var(--mono)" fill="var(--c-good)" fontWeight={700}>
                    1 parallel wave · O(1) token path · {r3.trainLatencyMs.toFixed(2)} ms (under 20 ms SLA)
                  </text>
                </g>
              ) : (
                <g>
                  {Array.from({ length: 16 }, (_, idx) => (
                    <g key={idx}>
                      <rect x={12 + idx * 38} y={16} width={24} height={22} rx={4} fill="var(--c-bad)" opacity={0.4 + 0.035 * idx} />
                      {idx < 15 && <line x1={36 + idx * 38} x2={50 + idx * 38} y1={27} y2={27} stroke="var(--text-muted)" strokeWidth={1.3} />}
                    </g>
                  ))}
                  <text x={320} y={58} textAnchor="middle" fontSize={11} fontFamily="var(--mono)" fill="var(--c-bad)" fontWeight={700}>
                    {r3.seqDepth.toLocaleString()} sequential steps (t=1 → t=512 → dec) · max path {r3.maxPathLen} · {r3.trainLatencyMs.toFixed(2)} ms
                  </text>
                </g>
              )}
            </svg>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 4 }}>
                <span>Forward-step latency (SLA ≤ 20.0 ms)</span>
                <strong style={{ color: r3.trainLatencyMs <= 20 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {r3.trainLatencyMs.toFixed(2)} ms / 20.0 ms
                </strong>
              </div>
              <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                <div
                  style={{
                    width: `${tweened[3]}%`,
                    height: '100%',
                    borderRadius: 4,
                    background: r3.pass ? 'var(--c-good)' : 'var(--c-bad)',
                  }}
                />
                <div style={{ position: 'absolute', left: `${(20 / 95) * 100}%`, top: -2, bottom: -2, width: 2, background: 'var(--text)' }} />
              </div>
            </div>
          </div>
          <LiveCode note="PyTorch sequence layer" code={liveCode} />
        </div>
      )}
    </LabShell>
  );
}
