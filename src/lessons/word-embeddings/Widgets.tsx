// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  ANALOGIES,
  LAB_DIMS_B2,
  LAB_KS_B1,
  LAB_SCALE_MODES,
  LAB_WINS_B2,
  PLOT_WORDS,
  W2V_LAB_BRIEFS,
  evalW2vBrief1,
  evalW2vBrief2,
  evalW2vBrief3,
  getVocab,
  projectTo2D,
  solveAnalogy,
  trainSgns,
  w2vLabExhaustive,
  type ScaleMode,
  type W2vBriefId,
} from './word2vec';

const ROYAL_SET = new Set(['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']);

// ------------------------------------------------------------------ 1. Live Trained 2D Embedding Map & Collapse Toggle

export function EmbeddingMapWidget() {
  const [epochs, setEpochs] = useState<number>(140);
  const [negK, setNegK] = useState<number>(5);

  const vocab = useMemo(() => getVocab(), []);
  const res = useMemo(() => trainSgns(6, negK, 2, epochs, 42), [negK, epochs]);
  const coords = useMemo(() => projectTo2D(res.vecs), [res.vecs]);

  const plotted = useMemo(() => {
    const pts = PLOT_WORDS.map((w) => {
      const id = vocab.toId.get(w)!;
      return { word: w, x: coords[id][0], y: coords[id][1], isRoyal: ROYAL_SET.has(w) };
    });
    const maxAbs = Math.max(1.2, ...pts.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y)))) * 1.42;
    return { pts, bound: maxAbs };
  }, [coords, vocab]);

  const byWord = useMemo(() => new Map<string, (typeof plotted.pts)[number]>(plotted.pts.map((p) => [p.word, p])), [plotted]);

  const arrows: [string, string, string][] = [
    ['man', 'woman', 'var(--c-a)'],
    ['king', 'queen', 'var(--c-a)'],
    ['france', 'paris', 'var(--c-b)'],
    ['germany', 'berlin', 'var(--c-b)'],
    ['japan', 'tokyo', 'var(--c-b)'],
  ];

  return (
    <Widget
      title="Meaning emerging from raw text: watch SGNS organise 41 words from scratch"
      wide
      caption={
        <>
          Drag <strong>Training epochs</strong> from <code>0</code> to <code>140</code>: every word starts in random noise, then separates into Royalty/Gender (blue) and Capitals/Countries (orange) with parallel arrows! Now flip <strong>Negative samples</strong> to <code>K = 0</code> (positives only) to watch <strong>embedding collapse</strong> squeeze all words into one direction (mean cosine <strong>+0.84</strong>)!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <div style={{ flex: '1 1 200px' }}>
          <Slider label="Training epochs" value={epochs} min={0} max={140} step={20} onChange={setEpochs} format={(v) => `${v} epochs`} />
        </div>
        <Segmented
          label="Negative samples"
          value={negK}
          onChange={setNegK}
          options={[
            { value: 0, label: 'K = 0 (collapse!)' },
            { value: 1, label: 'K = 1' },
            { value: 5, label: 'K = 5 (SGNS)' },
          ]}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 14, alignItems: 'center' }}>
        <Plot x={[-plotted.bound, plotted.bound]} y={[-plotted.bound, plotted.bound]} height={280} xLabel="PCA 1" yLabel="PCA 2">
          {({ x: sx, y: sy }) => (
            <>
              {arrows.map(([from, to, col]) => {
                const a = byWord.get(from);
                const b = byWord.get(to);
                if (!a || !b) return null;
                return (
                  <line
                    key={`${from}-${to}`}
                    x1={sx(a.x)}
                    y1={sy(a.y)}
                    x2={sx(b.x)}
                    y2={sy(b.y)}
                    stroke={col}
                    strokeWidth={1.6}
                    strokeDasharray="4 2"
                    opacity={0.75}
                  />
                );
              })}
              {plotted.pts.map((p) => (
                <g key={p.word}>
                  <circle cx={sx(p.x)} cy={sy(p.y)} r={4.5} fill={p.isRoyal ? 'var(--c-a)' : 'var(--c-b)'} />
                  <text
                    x={sx(p.x) + 6}
                    y={sy(p.y) + 3}
                    fill="var(--text)"
                    fontSize={11}
                    fontWeight={600}
                  >
                    {p.word}
                  </text>
                </g>
              ))}
            </>
          )}
        </Plot>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Mean pairwise cos</div>
              <div className="readout-value" style={{ color: res.meanCos > 0.5 ? 'var(--c-bad)' : 'var(--c-good)' }}>
                {res.meanCos.toFixed(2)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Analogies solved</div>
              <div className="readout-value" style={{ color: res.analogyHits === 4 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {res.analogyHits} / {res.analogyTotal}
              </div>
            </div>
          </div>

          <div style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', fontSize: 13, lineHeight: 1.5 }}>
            {epochs === 0 ? (
              <span>Epoch 0: random Gaussian initialisation. Words sit in meaningless static.</span>
            ) : negK === 0 ? (
              <span>
                <strong style={{ color: 'var(--c-bad)' }}>Representation collapse!</strong> With only positive pairs and zero negatives, the easiest way to make every dot product large is to push <em>every</em> word vector in the exact same direction (mean cosine <strong>{res.meanCos.toFixed(2)}</strong>).
              </span>
            ) : (
              <span>
                <strong style={{ color: 'var(--c-good)' }}>Geometric structure learned from text alone:</strong> without ever being told what “gender” or “capital” means, SGNS places parallel offsets between <code>man→woman</code> / <code>king→queen</code> and <code>france→paris</code> / <code>germany→berlin</code>.
              </span>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Vector Analogy Calculator (a - b + c)

export function AnalogyWidget() {
  const [idx, setIdx] = useState<number>(0);
  const res = useMemo(() => trainSgns(6, 5, 2, 140, 42), []);
  const spec = ANALOGIES[idx];
  const solved = useMemo(() => solveAnalogy(res.vecs, spec.a, spec.b, spec.c), [res, spec]);

  return (
    <Widget
      title="Vector arithmetic on real SGNS embeddings: a − b + c ≈ ?"
      wide
      caption={
        <>
          Pick any analogy to evaluate <code>norm(a) − norm(b) + norm(c)</code> against our real trained 6D Word2Vec vectors (excluding the three input words <code>a, b, c</code>, just as Mikolov et al. 2013 did). All four analogies hit the exact answer at cosine <strong>0.975–0.998</strong>!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {ANALOGIES.map((a, i) => (
          <button
            key={a.id}
            type="button"
            className="btn"
            style={{
              fontSize: 12.5,
              padding: '5px 10px',
              borderColor: idx === i ? 'var(--accent)' : 'var(--border)',
              background: idx === i ? 'var(--bg-sunken)' : 'transparent',
              fontWeight: idx === i ? 600 : 400,
            }}
            onClick={() => setIdx(i)}
          >
            {a.a} − {a.b} + {a.c}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center' }}>
        <div style={{ padding: '12px 14px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', fontFamily: 'var(--mono)', fontSize: 14 }}>
          <span style={{ color: 'var(--c-a)', fontWeight: 700 }}>{spec.a}</span> −{' '}
          <span style={{ color: 'var(--c-b)', fontWeight: 700 }}>{spec.b}</span> +{' '}
          <span style={{ color: 'var(--c-c)', fontWeight: 700 }}>{spec.c}</span> ≈{' '}
          <strong style={{ color: 'var(--c-good)', fontSize: 16 }}>{solved.top1.word}</strong>
        </div>

        <div style={{ flex: '1 1 240px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 8 }}>
          {solved.top3.map((cand, rank) => (
            <div key={cand.word} style={{ padding: '8px 10px', borderRadius: 7, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>Rank #{rank + 1}</div>
              <div style={{ fontWeight: 700, color: rank === 0 ? 'var(--c-good)' : 'var(--text)' }}>{cand.word}</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-muted)' }}>cos = {cand.sim.toFixed(3)}</div>
            </div>
          ))}
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Lab: Train & Debug Word2Vec

type W2vLabState = {
  b1K: number;
  b2Dim: number;
  b2Win: number;
  b3Mode: ScaleMode;
  solved: Record<W2vBriefId, boolean>;
};

const W2V_LAB_KEY = 'nl-w2v-lab-v1';
const W2V_LAB_DEFAULT: W2vLabState = {
  b1K: 0,
  b2Dim: 2,
  b2Win: 2,
  b3Mode: 'full-softmax',
  solved: { collapse: false, analogies: false, scale: false },
};

function loadW2vLab(): W2vLabState {
  try {
    const raw = localStorage.getItem(W2V_LAB_KEY);
    if (!raw) return W2V_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...W2V_LAB_DEFAULT, ...p, solved: { ...W2V_LAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return W2V_LAB_DEFAULT;
  }
}

export function Word2VecLab() {
  const [briefId, setBriefId] = useState<W2vBriefId>('collapse');
  const [st, setSt] = useState<W2vLabState>(loadW2vLab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(W2V_LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalW2vBrief1(st.b1K), [st.b1K]);
  const r2 = useMemo(() => evalW2vBrief2(st.b2Dim, st.b2Win), [st.b2Dim, st.b2Win]);
  const r3 = useMemo(() => evalW2vBrief3(st.b3Mode), [st.b3Mode]);
  const exhaust = useMemo(() => w2vLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'collapse' ? r1.pass : briefId === 'analogies' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const brief = W2V_LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Train & Debug Word2Vec"
      wide
      caption={
        <>
          Diagnose collapse, squeeze analogies into minimal dimensions, and tame the 100,000-word softmax bottleneck — every brief exhaustively checked. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {W2V_LAB_BRIEFS.map((b, i) => (
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

      {briefId === 'collapse' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="Negative samples"
            value={st.b1K}
            onChange={(v) => setSt((p) => ({ ...p, b1K: v }))}
            options={LAB_KS_B1.map((k) => ({ value: k, label: `K = ${k}` }))}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Mean cos (&lt; 0.35)</div>
              <div className="readout-value" style={{ color: r1.meanCos < 0.35 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.meanCos.toFixed(3)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Analogies (≥ 3/4)</div>
              <div className="readout-value" style={{ color: r1.hits >= 3 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.hits}/{r1.total}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'analogies' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Dimension"
              value={st.b2Dim}
              onChange={(v) => setSt((p) => ({ ...p, b2Dim: v }))}
              options={LAB_DIMS_B2.map((d) => ({ value: d, label: `${d}D` }))}
            />
            <Segmented
              label="Context window"
              value={st.b2Win}
              onChange={(v) => setSt((p) => ({ ...p, b2Win: v }))}
              options={LAB_WINS_B2.map((w) => ({ value: w, label: `±${w}` }))}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Analogies (4/4)</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.hits}/{r2.total}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Dimension</div>
              <div className="readout-value">{r2.dim}D</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.pass ? (r2.dim === 3 ? 'PASS (Min 3D!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'scale' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="Training objective"
            value={st.b3Mode}
            onChange={(v) => setSt((p) => ({ ...p, b3Mode: v }))}
            options={LAB_SCALE_MODES.map((m) => ({
              value: m,
              label: m === 'full-softmax' ? 'Full Softmax' : `SGNS K=${m.split('-')[1]}`,
            }))}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Dot-prods / pair (≤ 15)</div>
              <div className="readout-value" style={{ color: r3.dotsPerPair <= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.dotsPerPair.toLocaleString()}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Speedup vs Softmax</div>
              <div className="readout-value" style={{ color: 'var(--accent)' }}>{Math.round(r3.speedup).toLocaleString()}×</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r3.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.pass ? (r3.k === 5 ? 'PASS (Fastest!)' : 'PASS') : 'FAIL'}
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
