// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import {
  EMBED_LAB_BRIEFS,
  LAB_DIMS_B2,
  LAB_DTYPES_B2,
  LAB_SKU_MODES,
  MINI_VOCAB,
  MODEL_EMBED_SPECS,
  embedLabExhaustive,
  embeddingCosts,
  evalEmbedBrief1,
  evalEmbedBrief2,
  evalEmbedBrief3,
  oneHot,
  similarityMatrix,
  trainRepClassifier,
  verifyLookupEquivalence,
  type EmbedBriefId,
  type RepKind,
  type SkuMode,
} from './embed';

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
  solved: Record<EmbedBriefId, boolean>;
};

const ELAB_KEY = 'nl-onehot-lab-v1';
const ELAB_DEFAULT: ELabState = {
  rep: 'onehot',
  b2Dim: 128,
  b2Bytes: 4,
  sku: 'onehot',
  solved: { transfer: false, 'edge-ram': false, 'sku-mlp': false },
};

function loadELab(): ELabState {
  try {
    const raw = localStorage.getItem(ELAB_KEY);
    if (!raw) return ELAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...ELAB_DEFAULT, ...p, solved: { ...ELAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return ELAB_DEFAULT;
  }
}

export function RepresentationLab() {
  const [briefId, setBriefId] = useState<EmbedBriefId>('transfer');
  const [st, setSt] = useState<ELabState>(loadELab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(ELAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalEmbedBrief1(st.rep), [st.rep]);
  const r2 = useMemo(() => evalEmbedBrief2(st.b2Dim, st.b2Bytes), [st.b2Dim, st.b2Bytes]);
  const r3 = useMemo(() => evalEmbedBrief3(st.sku), [st.sku]);
  const exhaust = useMemo(() => embedLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'transfer' ? r1.pass : briefId === 'edge-ram' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const brief = EMBED_LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Choose & Size the Representation"
      wide
      caption={
        <>
          Three real representation-engineering briefs checked exhaustively over every setting. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {EMBED_LAB_BRIEFS.map((b, i) => (
          <button
            key={b.id}
            type="button"
            className="btn"
            style={{
              borderColor: briefId === b.id ? 'var(--accent)' : 'var(--border)',
              background: briefId === b.id ? 'var(--bg-sunken)' : 'transparent',
              fontWeight: briefId === b.id ? 600 : 400,
            }}
            onClick={() => {
              setBriefId(b.id);
              setShowOpt(false);
            }}
          >
            {st.solved[b.id] ? '✓ ' : ''}Brief {i + 1}
          </button>
        ))}
      </div>

      <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{brief.title}</div>
      <p style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 0, marginBottom: 12, lineHeight: 1.5 }}>{brief.story}</p>

      {briefId === 'transfer' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="Input encoding"
            value={st.rep}
            onChange={(v) => setSt((p) => ({ ...p, rep: v }))}
            options={[
              { value: 'onehot', label: 'One-Hot' },
              { value: 'scalar-id', label: 'Scalar ID' },
              { value: 'dense-2', label: 'Dense 2D' },
              { value: 'dense-4', label: 'Dense 4D' },
            ]}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Unseen acc (≥ 90%)</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{(100 * r1.testAcc).toFixed(1)}%</div>
            </div>
            <div className="readout">
              <div className="readout-label">Parameters</div>
              <div className="readout-value">{r1.params}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r1.pass ? (r1.params === 3 ? 'PASS (Opt 3!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'edge-ram' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Embedding dim"
              value={st.b2Dim}
              onChange={(v) => setSt((p) => ({ ...p, b2Dim: v }))}
              options={LAB_DIMS_B2.map((d) => ({ value: d, label: `${d}D` }))}
            />
            <Segmented
              label="Precision"
              value={st.b2Bytes}
              onChange={(v) => setSt((p) => ({ ...p, b2Bytes: v }))}
              options={LAB_DTYPES_B2}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Table RAM (≤ 8 MiB)</div>
              <div className="readout-value" style={{ color: r2.mib <= 8 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.mib.toFixed(3)} MiB</div>
            </div>
            <div className="readout">
              <div className="readout-label">Capacity (≥ 64D)</div>
              <div className="readout-value" style={{ color: r2.d >= 64 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.d}D</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.pass ? (r2.d === 128 ? 'PASS (Max 128D!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'sku-mlp' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="SKU input layer"
            value={st.sku}
            onChange={(v) => setSt((p) => ({ ...p, sku: v }))}
            options={LAB_SKU_MODES.map((m) => ({ value: m, label: m === 'onehot' ? 'One-Hot → 128' : `Emb ${m.split('-')[1]}D → 128` }))}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Total params (≤ 3.5M)</div>
              <div className="readout-value" style={{ color: r3.params <= 3_500_000 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {(r3.params / 1e6).toFixed(3)}M
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Embedding dim (≥ 16)</div>
              <div className="readout-value">{r3.d === 0 ? 'None (100k sparse)' : `${r3.d}D`}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r3.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.pass ? (r3.d === 16 ? 'PASS (Opt 1.60M!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
        <button type="button" className="btn" onClick={() => setShowOpt((v) => !v)}>
          {showOpt ? 'Hide exhaustive check' : `Reveal exhaustive search (${exhaust.total} configs)`}
        </button>
        {showOpt && (
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            <strong>{exhaust.passing}/{exhaust.total}</strong> pass · {exhaust.summary}
          </div>
        )}
      </div>
    </Widget>
  );
}
