// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
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
  SOURCE_TOKENS,
  TARGET_TOKENS,
  type ScoreKind,
  type SeqArchId,
  type SeqBriefId,
} from './seq2seq';

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
  solved: Record<SeqBriefId, boolean>;
};

const SEQ_LAB_KEY = 'nl-seq2seq-lab-v1';
const SEQ_LAB_DEFAULT: SeqLabState = {
  b1Arch: 'fixed',
  b1D: 16,
  b2Kind: 'bahdanau',
  b2Tau: 1.0,
  b3Arch: 'lstm-attn',
  solved: { bottleneck: false, align: false, latency: false },
};

function loadSeqLab(): SeqLabState {
  try {
    const raw = localStorage.getItem(SEQ_LAB_KEY);
    if (!raw) return SEQ_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...SEQ_LAB_DEFAULT, ...p, solved: { ...SEQ_LAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return SEQ_LAB_DEFAULT;
  }
}

export function Seq2SeqLab() {
  const [briefId, setBriefId] = useState<SeqBriefId>('bottleneck');
  const [st, setSt] = useState<SeqLabState>(loadSeqLab);
  const [revealed, setRevealed] = useState<Record<SeqBriefId, boolean>>({ bottleneck: false, align: false, latency: false });

  useEffect(() => {
    try {
      localStorage.setItem(SEQ_LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalSeqBrief1(st.b1Arch, st.b1D), [st.b1Arch, st.b1D]);
  const r2 = useMemo(() => evalSeqBrief2(st.b2Kind, st.b2Tau), [st.b2Kind, st.b2Tau]);
  const r2Align = useMemo(() => computeCrossAttention(st.b2Kind, st.b2Tau), [st.b2Kind, st.b2Tau]);
  const r3 = useMemo(() => evalSeqBrief3(st.b3Arch), [st.b3Arch]);
  const exhaust = useMemo(() => seqLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'bottleneck' ? r1.pass : briefId === 'align' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const solvedCount = Object.values(st.solved).filter(Boolean).length;
  const canReveal = st.solved[briefId] || revealed[briefId];

  return (
    <Widget
      title="Lab: Break the Bottleneck"
      wide
      caption={
        <>
          Three real translation-engineering briefs — escape the 20-token fixed-vector crash, sharpen cross-attention without logit overflow, and hit a 20 ms training SLA on 512-token documents. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
        {(
          [
            ['bottleneck', 'Brief 1 · 20-Token Copy'],
            ['align', 'Brief 2 · Sharp Alignment'],
            ['latency', 'Brief 3 · 20 ms GPU SLA'],
          ] as [SeqBriefId, string][]
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

      {briefId === 'bottleneck' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> Your translation model works on 2-word phrases but drops to 33.8% on 20-word sentences. Pick the decoder interface and state width <code>d</code> to reach <strong>≥ 95%</strong> accuracy on <code>L = 20</code> tokens (optional goal: smallest state width <code>d</code>!).
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Encoder–Decoder bridge"
              value={st.b1Arch}
              onChange={(v) => setSt((p) => ({ ...p, b1Arch: v }))}
              options={LAB_BN_ARCHS.map((a) => ({ value: a, label: a === 'fixed' ? 'Fixed single vector c' : 'Cross-Attention (all hₛ)' }))}
            />
            <Segmented
              label="State width d"
              value={st.b1D}
              onChange={(v) => setSt((p) => ({ ...p, b1D: v }))}
              options={LAB_BN_DS.map((d) => ({ value: d, label: `d = ${d}` }))}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">L=20 accuracy (≥ 95%)</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{(100 * r1.acc).toFixed(1)}%</div>
            </div>
            <div className="readout">
              <div className="readout-label">Memory footprint</div>
              <div className="readout-value">{st.b1Arch === 'fixed' ? `${st.b1D} floats` : `${20 * st.b1D} floats (20×d)`}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'align' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> Tune the cross-attention scoring rule and temperature on <em>“the black cat saw the red bird”</em> → <em>“le chat noir vit l’oiseau rouge”</em> so mean gold alignment weight reaches <strong>≥ 85%</strong> while keeping <strong>max |logit| ≤ 15.0</strong> (safe FP16 softmax).
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Score function"
              value={st.b2Kind}
              onChange={(v) => setSt((p) => ({ ...p, b2Kind: v }))}
              options={LAB_ALIGN_KINDS.map((k) => ({ value: k, label: k === 'luong' ? 'Luong dot' : 'Bahdanau additive' }))}
            />
            <Segmented
              label="Temperature"
              value={st.b2Tau}
              onChange={(v) => setSt((p) => ({ ...p, b2Tau: v }))}
              options={LAB_ALIGN_TAUS.map((t) => ({ value: t, label: `τ = ${t}` }))}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 6 }}>
            {TARGET_TOKENS.map((tok, t) => {
              const g = r2Align.weights[t][GOLD_ALIGNMENT[t]];
              return (
                <div
                  key={t}
                  style={{
                    textAlign: 'center',
                    padding: '6px 4px',
                    borderRadius: 6,
                    border: '1px solid var(--border)',
                    background: 'var(--bg-subtle)',
                    fontFamily: 'var(--mono)',
                    fontSize: 11,
                  }}
                >
                  <div>{tok}</div>
                  <div style={{ fontWeight: 700, color: g >= 0.85 ? 'var(--c-good)' : 'var(--c-b)' }}>{(100 * g).toFixed(0)}%</div>
                </div>
              );
            })}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Avg gold (≥ 85%)</div>
              <div className="readout-value" style={{ color: r2.meanGoldWeight >= 0.85 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {(100 * r2.meanGoldWeight).toFixed(1)}%
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Max |logit| (≤ 15)</div>
              <div className="readout-value" style={{ color: r2.maxAbsLogit <= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.maxAbsLogit.toFixed(2)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'latency' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
            <strong>On-call brief:</strong> You are training on <code>L = 512</code>-token documents. Your cluster requires both a direct <strong>O(1) token path</strong> (no fixed-vector bottleneck) AND forward-step latency <strong>≤ 20.0 ms</strong>!
          </div>
          <Segmented
            label="Sequence architecture"
            value={st.b3Arch}
            onChange={(v) => setSt((p) => ({ ...p, b3Arch: v }))}
            options={LAB_LAT_ARCHS.map((a) => ({
              value: a,
              label: a === 'fixed-lstm' ? 'Fixed LSTM' : a === 'lstm-attn' ? 'LSTM + Cross-Attn' : 'Transformer',
            }))}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Serial depth</div>
              <div className="readout-value">{r3.seqDepth.toLocaleString()}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Max path (= 1)</div>
              <div className="readout-value" style={{ color: r3.maxPathLen === 1 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.maxPathLen}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Step latency (≤ 20 ms)</div>
              <div className="readout-value" style={{ color: r3.trainLatencyMs <= 20 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.trainLatencyMs.toFixed(2)} ms
              </div>
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
