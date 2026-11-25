// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweened, useTweenedArray } from '../../components/motion';
import {
  CAT_ITEMS,
  EMBED_LAB_BRIEFS,
  LAB_DIMS_B2,
  LAB_DTYPES_B2,
  LAB_SKU_MODES,
  MINI_VOCAB,
  MODEL_EMBED_SPECS,
  embedLabExhaustive,
  embedLabHint,
  embeddingCosts,
  evalEmbedBrief1,
  evalEmbedBrief2,
  evalEmbedBrief3,
  oneHot,
  similarityMatrix,
  trainRepClassifier,
  verifyLookupEquivalence,
  type RepKind,
  type SkuMode,
} from './embed';
import { embedRealCode } from './realCode';

function simBg(v: number): { background: string; color: string } {
  const t = Math.min(1, Math.abs(v));
  if (t < 0.03) return { background: 'var(--bg-sunken)', color: 'var(--text-muted)' };
  const pct = Math.round(14 + 68 * t);
  const base = v > 0 ? 'var(--c-a)' : 'var(--c-b)';
  return {
    background: `color-mix(in srgb, ${base} ${pct}%, var(--bg))`,
    color: t > 0.6 ? 'var(--bg)' : 'var(--text)',
  };
}

// ------------------------------------------------------------------ 1. Orthogonality Curse vs Dense Similarity Matrix

export function OrthogonalityWidget() {
  const [mode, setMode] = useState<'onehot' | 'dense'>('onehot');
  const [pair, setPair] = useState<[number, number]>([0, 1]);

  const mat = useMemo(() => similarityMatrix(mode), [mode]);
  const wA = MINI_VOCAB[pair[0]];
  const wB = MINI_VOCAB[pair[1]];
  const sim = mat[pair[0]][pair[1]];

  return (
    <Widget
      title="The orthogonality curse: one-hot similarity matrix vs 3D dense embeddings"
      wide
      caption={
        <>
          Click any cell to compare two words. Under <strong>One-Hot</strong>, every distinct pair has cosine similarity <strong>0.000</strong> — <code>cat</code> is just as foreign to <code>kitten</code> as to <code>car</code>! Flip to <strong>3D Dense</strong> and semantic blocks light up instantly (<code>cat·kitten = +0.996</code>, <code>car·truck = +0.931</code>, <code>cat·car = −0.923</code>).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <Segmented
          label="Representation"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'onehot', label: '8D One-Hot (sparse)' },
            { value: 'dense', label: '3D Dense embedding' },
          ]}
        />
        <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Selected: <strong>{wA.word}</strong> vs <strong>{wB.word}</strong> → cos ={' '}
          <strong style={{ color: sim > 0.5 ? 'var(--c-a)' : sim < -0.5 ? 'var(--c-b)' : 'var(--text)' }}>
            {sim.toFixed(3)}
          </strong>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="num-table" style={{ fontSize: 12, width: '100%' }}>
          <thead>
            <tr>
              <th>Word</th>
              {MINI_VOCAB.map((w) => (
                <th key={w.id} style={{ textAlign: 'center', padding: '4px 5px' }}>{w.word}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MINI_VOCAB.map((rowW, r) => (
              <tr key={rowW.id}>
                <td style={{ fontWeight: 600 }}>{rowW.word}</td>
                {MINI_VOCAB.map((colW, c) => {
                  const v = mat[r][c];
                  const isSel = r === pair[0] && c === pair[1];
                  return (
                    <td
                      key={colW.id}
                      onClick={() => setPair([r, c])}
                      style={{
                        ...simBg(v),
                        textAlign: 'center',
                        cursor: 'pointer',
                        padding: '5px 4px',
                        fontFamily: 'var(--mono)',
                        outline: isSel ? '2px solid var(--accent)' : undefined,
                      }}
                    >
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Embedding Table = Matrix Multiply Lookup

export function LookupEquivalenceWidget() {
  const [tokId, setTokId] = useState<number>(1);
  const [modelId, setModelId] = useState<string>('llama3');

  const eq = useMemo(() => verifyLookupEquivalence(tokId), [tokId]);
  const oh = useMemo(() => oneHot(tokId, MINI_VOCAB.length), [tokId]);
  const spec = MODEL_EMBED_SPECS.find((m) => m.id === modelId) ?? MODEL_EMBED_SPECS[2];
  const cost = useMemo(() => embeddingCosts(spec.vocab, spec.dim, 2), [spec]);

  return (
    <Widget
      title="An embedding layer is literally E[token_id] — skipping 99.999% zero multiplications"
      wide
      caption={
        <>
          Pick any word ID to see <code>e_idᵀ @ E</code> select row <code>id</code> verbatim (<code>max error = 0.0</code>). In Llama-3 8B (<code>V = 128,256</code>), naive matrix multiplication wastes <strong>99.9992%</strong> of its 1.05 billion FLOPs multiplying by zero!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {MINI_VOCAB.map((w) => (
          <button
            key={w.id}
            type="button"
            className="btn"
            style={{
              fontSize: 12.5,
              padding: '4px 9px',
              borderColor: tokId === w.id ? 'var(--accent)' : 'var(--border)',
              background: tokId === w.id ? 'var(--bg-sunken)' : 'transparent',
              fontWeight: tokId === w.id ? 600 : 400,
            }}
            onClick={() => setTokId(w.id)}
          >
            {w.id}: {w.word}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center', marginBottom: 14 }}>
        <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', fontSize: 12.5, fontFamily: 'var(--mono)' }}>
          <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--sans)', marginBottom: 4 }}>1-hot row vector e[{tokId}]</div>
          <div>[{oh.join(', ')}]</div>
          <div style={{ marginTop: 8, color: 'var(--text-muted)', fontFamily: 'var(--sans)' }}>
            Selected row E[{tokId}] ({eq.word}):
          </div>
          <div style={{ color: 'var(--c-a)', fontWeight: 700 }}>
            [{eq.lookupOut.map((v) => v.toFixed(2)).join(', ')}]
          </div>
          <div style={{ fontSize: 11.5, color: 'var(--c-good)', marginTop: 4 }}>
            Max |eᵀE − E[id]| = {eq.maxErr.toFixed(1)} · Skipped {eq.zeroMults}/{eq.zeroMults + eq.nonZeroMults} zero mults
          </div>
        </div>

        <div style={{ flex: '1 1 260px' }}>
          <div style={{ marginBottom: 8 }}>
            <Segmented
              label="Real model scale (FP16)"
              value={modelId}
              onChange={setModelId}
              options={MODEL_EMBED_SPECS.map((m) => ({ value: m.id, label: m.name }))}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Table shape V×d</div>
              <div className="readout-value" style={{ fontSize: 14 }}>{spec.vocab.toLocaleString()} × {spec.dim}</div>
            </div>
            <div className="readout">
              <div className="readout-label">FP16 table RAM</div>
              <div className="readout-value" style={{ color: 'var(--c-b)', fontSize: 15 }}>
                {cost.mib < 0.1 ? `${cost.bytes} B` : `${cost.mib.toFixed(1)} MiB`}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Zero-mult waste</div>
              <div className="readout-value" style={{ color: 'var(--accent)', fontSize: 15 }}>
                {(100 * cost.wastedFraction).toFixed(4)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Zero-Shot Synonym Transfer Widget

export function SynonymTransferWidget() {
  const [rep, setRep] = useState<RepKind>('onehot');
  const res = useMemo(() => trainRepClassifier(rep), [rep]);

  return (
    <Widget
      title="Why dense embeddings generalise to unseen words (and integer IDs fail)"
      wide
      caption={
        <>
          A logistic classifier trains on 8 nouns (<code>cat, car, dog, bus, bird, train, horse, boat</code>) to predict <strong>Living vs Vehicle</strong>, then faces 6 <strong>unseen synonyms</strong> (<code>kitten, truck, puppy, tram, robin, ferry</code>). One-Hot memorises training (<strong>100%</strong>) and guesses blindly on unseen IDs (<strong>50%</strong>); scalar integer IDs fail even on training (<strong>50%</strong>); 2D dense embeddings nail <strong>100%</strong> zero-shot!
        </>
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="Input encoding"
          value={rep}
          onChange={setRep}
          options={[
            { value: 'onehot', label: 'One-Hot (14D)' },
            { value: 'scalar-id', label: 'Scalar ID (1D)' },
            { value: 'dense-2', label: 'Dense (2D)' },
            { value: 'dense-4', label: 'Dense (4D)' },
          ]}
        />
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, marginBottom: 12 }}>
        <div className="readout">
          <div className="readout-label">Parameters</div>
          <div className="readout-value">{res.params}</div>
        </div>
        <div className="readout">
          <div className="readout-label">Train acc (8 seen)</div>
          <div className="readout-value">{(100 * res.trainAcc).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Unseen synonyms (6)</div>
          <div className="readout-value" style={{ color: res.testAcc >= 0.9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {(100 * res.testAcc).toFixed(1)}%
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 8 }}>
        {res.testPreds.map((p) => (
          <div key={p.name} style={{ padding: '7px 9px', borderRadius: 7, border: '1px solid var(--border)', background: 'var(--bg-sunken)', fontSize: 12.5 }}>
            <div style={{ fontWeight: 600 }}>{p.name}</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>Truth: {p.y === 1 ? 'Living' : 'Vehicle'}</div>
            <div style={{ fontFamily: 'var(--mono)', fontWeight: 600, color: p.ok ? 'var(--c-good)' : 'var(--c-bad)' }}>
              P(liv) = {p.prob.toFixed(2)} {p.ok ? '✓' : '✕'}
            </div>
          </div>
        ))}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Choose & Size the Representation

type ELabState = {
  rep: RepKind;
  b2Dim: number;
  b2Bytes: 1 | 2 | 4;
  sku: SkuMode;
};

const ELAB_KEY = 'nl-onehot-lab-v1';
const ELAB_PROG_KEY = 'nl-onehot-lab-progress-v2';
const ELAB_DEFAULT: ELabState = {
  rep: 'onehot',
  b2Dim: 128,
  b2Bytes: 4,
  sku: 'onehot',
};

function loadLegacyELabSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(ELAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved ?? {};
  } catch {
    return {};
  }
}

function loadELab(): ELabState {
  try {
    const raw = localStorage.getItem(ELAB_KEY);
    if (!raw) return ELAB_DEFAULT;
    const p = JSON.parse(raw);
    return {
      rep: ['onehot', 'scalar-id', 'dense-2', 'dense-4'].includes(p.rep) ? p.rep : ELAB_DEFAULT.rep,
      b2Dim: (LAB_DIMS_B2 as readonly number[]).includes(p.b2Dim) ? p.b2Dim : ELAB_DEFAULT.b2Dim,
      b2Bytes: p.b2Bytes === 1 || p.b2Bytes === 2 || p.b2Bytes === 4 ? p.b2Bytes : ELAB_DEFAULT.b2Bytes,
      sku: (LAB_SKU_MODES as readonly string[]).includes(p.sku) ? p.sku : ELAB_DEFAULT.sku,
    };
  } catch {
    return ELAB_DEFAULT;
  }
}

export function RepresentationLab() {
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const briefId = EMBED_LAB_BRIEFS[activeIdx].id;
  const [st, setSt] = useState<ELabState>(loadELab);
  const progress = useLabProgress(ELAB_PROG_KEY, () => ({ solved: loadLegacyELabSolved() }));
  const log = useMachineLog(18);

  useEffect(() => {
    try {
      localStorage.setItem(ELAB_KEY, JSON.stringify({ ...st, solved: progress.solved }));
    } catch {
      /* ignore */
    }
  }, [st, progress.solved]);

  const r1 = useMemo(() => evalEmbedBrief1(st.rep), [st.rep]);
  const r2 = useMemo(() => evalEmbedBrief2(st.b2Dim, st.b2Bytes), [st.b2Dim, st.b2Bytes]);
  const r3 = useMemo(() => evalEmbedBrief3(st.sku), [st.sku]);
  const exhaust = useMemo(() => embedLabExhaustive(briefId), [briefId]);
  const hint = useMemo(() => embedLabHint(briefId, st), [briefId, st]);
  const realCode = useMemo(() => embedRealCode(briefId, st), [briefId, st]);

  const allB1 = useMemo(() => (['onehot', 'scalar-id', 'dense-2', 'dense-4'] as RepKind[]).map((r) => evalEmbedBrief1(r)), []);
  const allB3 = useMemo(() => LAB_SKU_MODES.map((m) => evalEmbedBrief3(m)), []);

  const curPass = briefId === 'transfer' ? r1.pass : briefId === 'edge-ram' ? r2.pass : r3.pass;
  const curOptimal =
    briefId === 'transfer'
      ? r1.pass && r1.params === 3
      : briefId === 'edge-ram'
        ? r2.pass && r2.d === 128
        : r3.pass && r3.d === 16;

  const tweenedAcc1 = useTweenedArray([r1.trainAcc, r1.testAcc], 240);
  const tweenedRam2 = useTweened(r2.mib, 240);
  const tweenedSkuParams3 = useTweened(r3.params, 240);

  const updateRep = (rep: RepKind) => {
    const ev = evalEmbedBrief1(rep);
    setSt((p) => ({ ...p, rep }));
    log.push(
      `B1: ${rep} (${ev.params} params) → seen ${(100 * ev.trainAcc).toFixed(0)}%, unseen ${(100 * ev.testAcc).toFixed(1)}% (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : 'bad',
    );
  };

  const updateB2 = (patch: Partial<Pick<ELabState, 'b2Dim' | 'b2Bytes'>>) => {
    const d = patch.b2Dim ?? st.b2Dim;
    const b = patch.b2Bytes ?? st.b2Bytes;
    const ev = evalEmbedBrief2(d, b);
    const dtName = b === 4 ? 'FP32' : b === 2 ? 'FP16' : 'INT8';
    setSt((p) => ({ ...p, b2Dim: d, b2Bytes: b }));
    log.push(
      `B2: V=50k × ${d}D × ${dtName} → ${ev.mib.toFixed(3)} MiB (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const updateSku = (sku: SkuMode) => {
    const ev = evalEmbedBrief3(sku);
    setSt((p) => ({ ...p, sku }));
    log.push(
      `B3: ${sku === 'onehot' ? 'One-Hot 100k→128' : `Emb ${ev.d}D→128`} → ${(ev.params / 1e6).toFixed(3)}M params (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const resetBrief = () => {
    if (briefId === 'transfer') {
      setSt((p) => ({ ...p, rep: ELAB_DEFAULT.rep }));
      log.push('B1: reset to One-Hot');
    } else if (briefId === 'edge-ram') {
      setSt((p) => ({ ...p, b2Dim: ELAB_DEFAULT.b2Dim, b2Bytes: ELAB_DEFAULT.b2Bytes }));
      log.push('B2: reset to 128D FP32');
    } else {
      setSt((p) => ({ ...p, sku: ELAB_DEFAULT.sku }));
      log.push('B3: reset to One-Hot 100k→128');
    }
  };

  const statusText = useMemo(() => {
    if (briefId === 'transfer') {
      if (r1.pass) {
        return r1.params === 3
          ? `Unseen accuracy ${(100 * r1.testAcc).toFixed(1)}% with just 3 parameters — leanest representation in this suite.`
          : `Unseen accuracy ${(100 * r1.testAcc).toFixed(1)}% with ${r1.params} parameters. Optional goal: can you use fewer coordinates?`;
      }
      return `Seen accuracy ${(100 * r1.trainAcc).toFixed(1)}% · unseen accuracy ${(100 * r1.testAcc).toFixed(1)}% (need ≥ 90.0%) · ${r1.params} params.`;
    }
    if (briefId === 'edge-ram') {
      if (r2.pass) {
        return r2.d === 128
          ? `Table fits in ${r2.mib.toFixed(3)} MiB (≤ 8.0 MiB) at the maximum capacity (128D)!`
          : `Table fits in ${r2.mib.toFixed(3)} MiB with ${r2.d}D vectors. Optional goal: can you fit an even larger dimension inside 8.0 MiB?`;
      }
      return `Table needs ${r2.mib.toFixed(3)} MiB (limit 8.0 MiB) · dimension ${r2.d}D (need ≥ 64D).`;
    }
    if (r3.pass) {
      return r3.d === 16
        ? `Total weights ${(r3.params / 1e6).toFixed(3)}M (≤ 3.50M) with d = 16D — 8.0× smaller than direct One-Hot.`
        : `Total weights ${(r3.params / 1e6).toFixed(3)}M (≤ 3.50M) with d = ${r3.d}D. Optional goal: can a narrower embedding still clear d ≥ 16?`;
    }
    return `Total weights ${(r3.params / 1e6).toFixed(3)}M (limit 3.50M) · embedding dim ${r3.d === 0 ? 'none (sparse 100k)' : `${r3.d}D (need ≥ 16D)`}.`;
  }, [briefId, r1, r2, r3]);

  return (
    <LabShell
      title="Lab: Choose & Size the Representation"
      caption={
        <>
          Inspect learned weights, decision boundaries, byte-level table footprints, and matrix factorisations across three representation-engineering briefs.
        </>
      }
      briefs={EMBED_LAB_BRIEFS.map((b, i) => ({
        id: b.id,
        tab: `Brief ${i + 1}`,
        title: b.title,
        text: b.story,
      }))}
      active={activeIdx}
      onSelect={setActiveIdx}
      progress={progress}
      passing={curPass}
      optimal={curOptimal}
      status={statusText}
      hint={hint}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
      log={{
        title: 'Representation bench log',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Change encoding, dimension, or precision to record an architecture test.',
      }}
      reveal={{
        label: `exhaustive search (${exhaust.total} configs)`,
        content: (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 13 }}>
              <strong>
                {exhaust.passing}/{exhaust.total}
              </strong>{' '}
              pass · {exhaust.summary}
            </div>

            {briefId === 'transfer' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Encoding</th>
                    <th>Params</th>
                    <th>Seen acc</th>
                    <th>Unseen acc</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {allB1.map((row) => (
                    <tr key={row.rep} style={{ fontWeight: row.rep === st.rep ? 600 : 400 }}>
                      <td>{row.rep}</td>
                      <td>{row.params}</td>
                      <td>{(100 * row.trainAcc).toFixed(1)}%</td>
                      <td>{(100 * row.testAcc).toFixed(1)}%</td>
                      <td style={{ color: row.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{row.pass ? 'PASS' : 'FAIL'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {briefId === 'edge-ram' && (
              <div style={{ overflowX: 'auto' }}>
                <table className="num-table" style={{ width: '100%', fontSize: 12, textAlign: 'center' }}>
                  <thead>
                    <tr>
                      <th>Dim</th>
                      {LAB_DTYPES_B2.map((dt) => (
                        <th key={dt.value}>{dt.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {LAB_DIMS_B2.map((d) => (
                      <tr key={d}>
                        <td style={{ fontWeight: 600 }}>{d}D</td>
                        {LAB_DTYPES_B2.map((dt) => {
                          const cell = evalEmbedBrief2(d, dt.value);
                          const isCur = d === st.b2Dim && dt.value === st.b2Bytes;
                          return (
                            <td
                              key={dt.value}
                              style={{
                                background: cell.pass
                                  ? 'color-mix(in srgb, var(--c-good) 20%, var(--bg))'
                                  : 'color-mix(in srgb, var(--c-bad) 9%, var(--bg))',
                                color: cell.pass ? 'var(--c-good)' : 'var(--text-muted)',
                                fontWeight: cell.pass || isCur ? 700 : 400,
                                outline: isCur ? '2px solid var(--accent)' : undefined,
                                outlineOffset: -2,
                              }}
                            >
                              {cell.mib.toFixed(2)} MiB
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {briefId === 'sku-mlp' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>SKU Layer</th>
                    <th>Dim</th>
                    <th>Total params</th>
                    <th>vs One-Hot</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {allB3.map((row) => (
                    <tr key={row.mode} style={{ fontWeight: row.mode === st.sku ? 600 : 400 }}>
                      <td>{row.mode === 'onehot' ? 'One-Hot → 128' : `Embedding(${row.d}) → 128`}</td>
                      <td>{row.d === 0 ? '—' : `${row.d}D`}</td>
                      <td>{(row.params / 1e6).toFixed(3)}M</td>
                      <td>{(12_800_128 / row.params).toFixed(1)}×</td>
                      <td style={{ color: row.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{row.pass ? 'PASS' : 'FAIL'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ),
      }}
    >
      {briefId === 'transfer' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <Segmented
            label="Input encoding"
            value={st.rep}
            onChange={updateRep}
            options={[
              { value: 'onehot', label: 'One-Hot' },
              { value: 'scalar-id', label: 'Scalar ID' },
              { value: 'dense-2', label: 'Dense 2D' },
              { value: 'dense-4', label: 'Dense 4D' },
            ]}
          />

          {/* Animated seen vs unseen bars */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
            {[
              { label: 'Seen 8 nouns accuracy', val: tweenedAcc1[0], raw: r1.trainAcc, ok: r1.trainAcc >= 0.9 },
              { label: 'Unseen 6 synonyms accuracy (≥ 90%)', val: tweenedAcc1[1], raw: r1.testAcc, ok: r1.pass },
            ].map((b) => (
              <div key={b.label} style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>{b.label}</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: b.ok ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {(100 * b.raw).toFixed(1)}%
                  </strong>
                </div>
                <div style={{ height: 9, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.max(0, Math.min(100, b.val * 100))}%`, background: b.ok ? 'var(--c-good)' : 'var(--c-bad)' }} />
                </div>
              </div>
            ))}
          </div>

          {/* Inside the trained linear classifier */}
          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
            {st.rep === 'onehot' && (
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                  Inside One-Hot classifier: learned weight w[i] per word ID (seen 0..7 vs unseen 8..13)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(56px, 1fr))', gap: 4 }}>
                  {CAT_ITEMS.map((item, i) => {
                    const w = r1.weights[i] ?? 0;
                    const maxW = Math.max(...r1.weights.map(Math.abs), 1);
                    const pct = (Math.abs(w) / maxW) * 100;
                    const unseen = item.split === 'test';
                    return (
                      <div
                        key={item.id}
                        style={{
                          padding: '4px 5px',
                          borderRadius: 5,
                          border: `1px solid ${unseen ? 'var(--c-bad)' : 'var(--border)'}`,
                          background: 'var(--bg)',
                          textAlign: 'center',
                          fontSize: 10.5,
                        }}
                      >
                        <div style={{ fontWeight: 600, color: unseen ? 'var(--c-bad)' : 'var(--text)' }}>{item.name}</div>
                        <div style={{ height: 22, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', margin: '3px 0' }}>
                          <div
                            style={{
                              width: 14,
                              height: `${Math.max(2, pct * 0.2)}px`,
                              borderRadius: 2,
                              background: unseen ? 'var(--text-muted)' : w >= 0 ? 'var(--c-a)' : 'var(--c-b)',
                            }}
                          />
                        </div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: unseen ? 'var(--c-bad)' : 'var(--text-muted)' }}>
                          {w.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 6 }}>
                  Every unseen word&apos;s column stayed frozen at <strong>0.00</strong> because its one-hot bit was never 1 during training.
                </div>
              </div>
            )}

            {st.rep === 'scalar-id' && (
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                  Inside Scalar-ID classifier: 1D number line (● Living vs ■ Vehicle) + learned threshold cut
                </div>
                <svg viewBox="0 0 380 86" style={{ width: '100%', maxWidth: 520, height: 'auto', display: 'block' }}>
                  <line x1={18} y1={42} x2={362} y2={42} stroke="var(--border)" strokeWidth={2} />
                  {(() => {
                    const cutId = Math.abs(r1.weights[0]) > 1e-6 ? 6.5 - (4 * r1.bias) / r1.weights[0] : 6.5;
                    const cxCut = 24 + (Math.max(0, Math.min(13, cutId)) / 13) * 332;
                    return (
                      <>
                        <line x1={cxCut} y1={10} x2={cxCut} y2={74} stroke="var(--c-bad)" strokeWidth={2} strokeDasharray="4 2" />
                        <text x={cxCut} y={9} textAnchor="middle" fontSize={9} fill="var(--c-bad)" fontWeight={600}>
                          cut ≈ {cutId.toFixed(1)}
                        </text>
                      </>
                    );
                  })()}
                  {CAT_ITEMS.map((item) => {
                    const cx = 24 + (item.id / 13) * 332;
                    const isLiv = item.y === 1;
                    const fill = isLiv ? 'var(--c-a)' : 'var(--c-b)';
                    return (
                      <g key={item.id}>
                        {isLiv ? (
                          <circle cx={cx} cy={42} r={5.5} fill={fill} fillOpacity={item.split === 'test' ? 0.4 : 0.9} stroke={fill} strokeWidth={1.5} />
                        ) : (
                          <rect x={cx - 5} y={37} width={10} height={10} fill={fill} fillOpacity={item.split === 'test' ? 0.4 : 0.9} stroke={fill} strokeWidth={1.5} />
                        )}
                        <text x={cx} y={item.id % 2 === 0 ? 28 : 66} textAnchor="middle" fontSize={8.5} fill="var(--text-muted)">
                          {item.name}
                        </text>
                      </g>
                    );
                  })}
                </svg>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }}>
                  Living and Vehicle IDs alternate along the line, so a single threshold cannot separate even the 8 training nouns.
                </div>
              </div>
            )}

            {(st.rep === 'dense-2' || st.rep === 'dense-4') && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
                    Inside {st.rep === 'dense-2' ? 'Dense 2D' : 'Dense 4D'} classifier: semantic space + learned boundary
                  </div>
                  <svg viewBox="0 0 280 175" style={{ width: '100%', maxWidth: 320, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', display: 'block' }}>
                    {(() => {
                      const sx = (x: number) => 22 + ((x + 1.1) / 2.2) * 236;
                      const sy = (y: number) => 155 - ((y + 1.0) / 2.1) * 136;
                      // Decision line w0*x + w1*y + b = 0 -> x = -(w1*y + b)/w0
                      const w0 = r1.weights[0];
                      const w1 = r1.weights[1];
                      const b = r1.bias;
                      const xTop = Math.abs(w0) > 1e-6 ? -(w1 * 1.1 + b) / w0 : 0;
                      const xBot = Math.abs(w0) > 1e-6 ? -(w1 * -1.0 + b) / w0 : 0;
                      return (
                        <>
                          <line x1={sx(xBot)} y1={sy(-1.0)} x2={sx(xTop)} y2={sy(1.1)} stroke="var(--c-good)" strokeWidth={2} strokeDasharray="4 3" />
                          {CAT_ITEMS.map((item) => {
                            const cx = sx(item.emb2[0]);
                            const cy = sy(item.emb2[1]);
                            const isTest = item.split === 'test';
                            const col = item.y === 1 ? 'var(--c-a)' : 'var(--c-b)';
                            return (
                              <g key={item.id}>
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r={isTest ? 5.2 : 4}
                                  fill={isTest ? 'var(--bg)' : col}
                                  stroke={col}
                                  strokeWidth={isTest ? 2.2 : 1.2}
                                />
                                <text
                                  x={cx + (item.emb2[0] > 0 ? -6 : 6)}
                                  y={cy + 3}
                                  textAnchor={item.emb2[0] > 0 ? 'end' : 'start'}
                                  fontSize={8.5}
                                  fontWeight={isTest ? 700 : 400}
                                  fill={isTest ? 'var(--text)' : 'var(--text-muted)'}
                                >
                                  {item.name}
                                </text>
                              </g>
                            );
                          })}
                        </>
                      );
                    })()}
                  </svg>
                </div>
                <div style={{ flex: '1 1 180px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(95px, 1fr))', gap: 6 }}>
                  {r1.testPreds.map((p) => (
                    <div key={p.name} style={{ padding: '5px 7px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', fontSize: 11.5 }}>
                      <div style={{ fontWeight: 600 }}>{p.name}</div>
                      <div style={{ fontFamily: 'var(--mono)', color: p.ok ? 'var(--c-good)' : 'var(--c-bad)' }}>
                        P={p.prob.toFixed(2)} {p.ok ? '✓' : '✕'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {briefId === 'edge-ram' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Embedding dim"
              value={st.b2Dim}
              onChange={(v) => updateB2({ b2Dim: v })}
              options={LAB_DIMS_B2.map((d) => ({ value: d, label: `${d}D` }))}
            />
            <Segmented
              label="Precision"
              value={st.b2Bytes}
              onChange={(v) => updateB2({ b2Bytes: v })}
              options={LAB_DTYPES_B2}
            />
          </div>

          {/* Animated RAM budget bar + per-row byte anatomy */}
          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span>Table RAM (V = 50,000 · ceiling = 8.0 MiB)</span>
                <strong style={{ fontFamily: 'var(--mono)', color: r2.mib <= 8.0 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {tweenedRam2.toFixed(3)} / 8.000 MiB
                </strong>
              </div>
              <div style={{ position: 'relative', height: 14, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.min(100, (tweenedRam2 / 25) * 100)}%`,
                    background: r2.mib <= 8.0 ? 'var(--c-good)' : 'var(--c-bad)',
                  }}
                />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(8 / 25) * 100}%`, width: 2, background: 'var(--text)' }} />
              </div>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 12 }}>
              <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                1 token row = {st.b2Dim} × {st.b2Bytes} B = <strong>{st.b2Dim * st.b2Bytes} B</strong>
              </span>
              <span>× 50,000 rows =</span>
              <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: `1px solid ${r2.pass ? 'var(--c-good)' : 'var(--c-bad)'}`, fontWeight: 600 }}>
                {r2.mib.toFixed(3)} MiB ({r2.d}D {r2.d >= 64 ? '≥ 64D ✓' : '< 64D ✕'})
              </span>
            </div>
          </div>
        </div>
      )}

      {briefId === 'sku-mlp' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <Segmented
            label="SKU input layer"
            value={st.sku}
            onChange={updateSku}
            options={LAB_SKU_MODES.map((m) => ({ value: m, label: m === 'onehot' ? 'One-Hot → 128' : `Emb ${m.split('-')[1]}D → 128` }))}
          />

          {/* Proportional matrix-factorisation diagram + parameter meter */}
          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 12 }}>
              {st.sku === 'onehot' ? (
                <>
                  <span style={{ padding: '5px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                    One-Hot (100,000)
                  </span>
                  <span>→</span>
                  <span style={{ padding: '5px 10px', borderRadius: 6, background: 'color-mix(in srgb, var(--c-bad) 18%, var(--bg))', border: '1px solid var(--c-bad)', fontWeight: 600 }}>
                    Dense [100,000 × 128] = 12.800M params
                  </span>
                </>
              ) : (
                <>
                  <span style={{ padding: '5px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                    SKU ID
                  </span>
                  <span>→</span>
                  <span style={{ padding: '5px 9px', borderRadius: 6, background: 'color-mix(in srgb, var(--c-a) 18%, var(--bg))', border: '1px solid var(--c-a)', fontWeight: 600 }}>
                    Emb [100k × {r3.d}] = {((100_000 * r3.d) / 1e6).toFixed(2)}M
                  </span>
                  <span>→</span>
                  <span style={{ padding: '5px 9px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                    Linear [{r3.d} × 128] = {(r3.d * 128 + 128).toLocaleString()}
                  </span>
                </>
              )}
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span>Total input-stage parameters (ceiling = 3.50M)</span>
                <strong style={{ fontFamily: 'var(--mono)', color: r3.params <= 3_500_000 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {(tweenedSkuParams3 / 1e6).toFixed(3)}M / 3.500M
                </strong>
              </div>
              <div style={{ position: 'relative', height: 14, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.min(100, (tweenedSkuParams3 / 13_000_000) * 100)}%`,
                    background: r3.params <= 3_500_000 ? 'var(--c-good)' : 'var(--c-bad)',
                  }}
                />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(3_500_000 / 13_000_000) * 100}%`, width: 2, background: 'var(--text)' }} />
              </div>
            </div>
          </div>
        </div>
      )}

      <LiveCode note="PyTorch nn.Embedding / F.one_hot" code={realCode} />
    </LabShell>
  );
}
