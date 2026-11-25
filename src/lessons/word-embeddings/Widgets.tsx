// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweened } from '../../components/motion';
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
  w2vLabHint,
  type ScaleMode,
} from './word2vec';
import { word2vecRealCode } from './realCode';

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
};

const W2V_LAB_KEY = 'nl-w2v-lab-v1';
const W2V_LAB_PROG_KEY = 'nl-w2v-lab-progress-v2';
const W2V_LAB_DEFAULT: W2vLabState = {
  b1K: 0,
  b2Dim: 2,
  b2Win: 2,
  b3Mode: 'full-softmax',
};

function loadLegacyW2vSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(W2V_LAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved ?? {};
  } catch {
    return {};
  }
}

function loadW2vLab(): W2vLabState {
  try {
    const raw = localStorage.getItem(W2V_LAB_KEY);
    if (!raw) return W2V_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return {
      b1K: (LAB_KS_B1 as readonly number[]).includes(p.b1K) ? p.b1K : W2V_LAB_DEFAULT.b1K,
      b2Dim: (LAB_DIMS_B2 as readonly number[]).includes(p.b2Dim) ? p.b2Dim : W2V_LAB_DEFAULT.b2Dim,
      b2Win: (LAB_WINS_B2 as readonly number[]).includes(p.b2Win) ? p.b2Win : W2V_LAB_DEFAULT.b2Win,
      b3Mode: (LAB_SCALE_MODES as readonly string[]).includes(p.b3Mode) ? p.b3Mode : W2V_LAB_DEFAULT.b3Mode,
    };
  } catch {
    return W2V_LAB_DEFAULT;
  }
}

function LabConstellation({
  vecs,
  analogyDetails,
}: {
  vecs: readonly number[][];
  analogyDetails: { spec: (typeof ANALOGIES)[number]; top1: string; top1Sim: number; ok: boolean }[];
}) {
  const vocab = useMemo(() => getVocab(), []);
  const pts = useMemo(() => projectTo2D(vecs), [vecs]);

  const subset = PLOT_WORDS.map((w) => {
    const id = vocab.toId.get(w)!;
    return { word: w, x: pts[id][0], y: pts[id][1] };
  });
  const xs = subset.map((s) => s.x);
  const ys = subset.map((s) => s.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padX = Math.max(0.25, (maxX - minX) * 0.22);
  const padY = Math.max(0.25, (maxY - minY) * 0.22);
  const sx = (v: number) => 22 + ((v - (minX - padX)) / (maxX - minX + 2 * padX)) * 236;
  const sy = (v: number) => 158 - ((v - (minY - padY)) / (maxY - minY + 2 * padY)) * 136;

  const byWord = new Map<string, { word: string; x: number; y: number }>(subset.map((s) => [s.word, s]));
  const arrows: [string, string, string][] = [
    ['man', 'woman', 'var(--c-a)'],
    ['king', 'queen', 'var(--c-a)'],
    ['france', 'paris', 'var(--c-b)'],
    ['germany', 'berlin', 'var(--c-b)'],
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12, alignItems: 'start' }}>
      <div>
        <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
          2D PCA projection of trained SGNS vectors (arrows = gender &amp; capital shifts)
        </div>
        <svg viewBox="0 0 280 176" style={{ width: '100%', maxWidth: 340, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'block' }}>
          {arrows.map(([a, b, col]) => {
            const pa = byWord.get(a);
            const pb = byWord.get(b);
            if (!pa || !pb) return null;
            return (
              <line
                key={`${a}-${b}`}
                x1={sx(pa.x)}
                y1={sy(pa.y)}
                x2={sx(pb.x)}
                y2={sy(pb.y)}
                stroke={col}
                strokeOpacity={0.55}
                strokeWidth={1.6}
                strokeDasharray="3 2"
              />
            );
          })}
          {subset.map((s) => {
            const isRoyal = ROYAL_SET.has(s.word);
            const col = isRoyal ? 'var(--c-a)' : 'var(--c-b)';
            const cx = sx(s.x);
            const cy = sy(s.y);
            return (
              <g key={s.word}>
                <circle cx={cx} cy={cy} r={4} fill={col} />
                <text x={cx + 5} y={cy + 3} fontSize={8.5} fill="var(--text)" fontWeight={600}>
                  {s.word}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>Live analogy scorecard (4 parallelograms)</div>
        {analogyDetails.map((d) => (
          <div
            key={d.spec.id}
            style={{
              padding: '6px 9px',
              borderRadius: 7,
              border: `1px solid ${d.ok ? 'var(--c-good)' : 'var(--c-bad)'}`,
              background: 'var(--bg-sunken)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 8,
              fontSize: 12,
            }}
          >
            <span>
              <strong>{d.spec.a}</strong> − {d.spec.b} + {d.spec.c}
            </span>
            <span style={{ fontFamily: 'var(--mono)', fontWeight: 600, color: d.ok ? 'var(--c-good)' : 'var(--c-bad)' }}>
              → {d.top1} ({d.top1Sim.toFixed(2)}) {d.ok ? '✓' : `≠ ${d.spec.expected}`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Word2VecLab() {
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const briefId = W2V_LAB_BRIEFS[activeIdx].id;
  const [st, setSt] = useState<W2vLabState>(loadW2vLab);
  const progress = useLabProgress(W2V_LAB_PROG_KEY, () => ({ solved: loadLegacyW2vSolved() }));
  const log = useMachineLog(18);

  useEffect(() => {
    try {
      localStorage.setItem(W2V_LAB_KEY, JSON.stringify({ ...st, solved: progress.solved }));
    } catch {
      /* ignore */
    }
  }, [st, progress.solved]);

  const r1 = useMemo(() => evalW2vBrief1(st.b1K), [st.b1K]);
  const r2 = useMemo(() => evalW2vBrief2(st.b2Dim, st.b2Win), [st.b2Dim, st.b2Win]);
  const r3 = useMemo(() => evalW2vBrief3(st.b3Mode), [st.b3Mode]);
  const exhaust = useMemo(() => w2vLabExhaustive(briefId), [briefId]);
  const hint = useMemo(() => w2vLabHint(briefId, st), [briefId, st]);
  const realCode = useMemo(() => word2vecRealCode(briefId, st), [briefId, st]);

  const allB1 = useMemo(() => LAB_KS_B1.map((k) => evalW2vBrief1(k)), []);
  const allB3 = useMemo(() => LAB_SCALE_MODES.map((m) => evalW2vBrief3(m)), []);

  const curPass = briefId === 'collapse' ? r1.pass : briefId === 'analogies' ? r2.pass : r3.pass;
  const curOptimal =
    briefId === 'collapse'
      ? r1.pass && r1.hits === 4
      : briefId === 'analogies'
        ? r2.pass && r2.dim === 3
        : r3.pass && r3.k === 5;

  const tweenedCos1 = useTweened(Math.max(0, r1.meanCos), 240);
  const tweenedLogDots3 = useTweened(Math.log10(Math.max(1, r3.dotsPerPair)), 240);

  const updateB1 = (b1K: number) => {
    const ev = evalW2vBrief1(b1K);
    setSt((p) => ({ ...p, b1K }));
    log.push(
      `B1: K=${b1K} (d=6, W=2) → meanCos ${ev.meanCos.toFixed(3)}, analogies ${ev.hits}/${ev.total} (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : 'bad',
    );
  };

  const updateB2 = (patch: Partial<Pick<W2vLabState, 'b2Dim' | 'b2Win'>>) => {
    const d = patch.b2Dim ?? st.b2Dim;
    const w = patch.b2Win ?? st.b2Win;
    const ev = evalW2vBrief2(d, w);
    setSt((p) => ({ ...p, b2Dim: d, b2Win: w }));
    log.push(
      `B2: d=${d}D, window=±${w} → analogies ${ev.hits}/${ev.total}, meanCos ${ev.meanCos.toFixed(2)} (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const updateB3 = (b3Mode: ScaleMode) => {
    const ev = evalW2vBrief3(b3Mode);
    setSt((p) => ({ ...p, b3Mode }));
    log.push(
      `B3: ${b3Mode} → ${ev.dotsPerPair.toLocaleString()} dots/pair (${Math.round(ev.speedup).toLocaleString()}× speedup, ${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const resetBrief = () => {
    if (briefId === 'collapse') {
      setSt((p) => ({ ...p, b1K: W2V_LAB_DEFAULT.b1K }));
      log.push('B1: reset to K=0');
    } else if (briefId === 'analogies') {
      setSt((p) => ({ ...p, b2Dim: W2V_LAB_DEFAULT.b2Dim, b2Win: W2V_LAB_DEFAULT.b2Win }));
      log.push('B2: reset to 2D, window ±2');
    } else {
      setSt((p) => ({ ...p, b3Mode: W2V_LAB_DEFAULT.b3Mode }));
      log.push('B3: reset to Full Softmax');
    }
  };

  const statusText = useMemo(() => {
    if (briefId === 'collapse') {
      return r1.pass
        ? `Isotropic space restored: mean cos ${r1.meanCos.toFixed(3)} (< 0.35) · ${r1.hits}/${r1.total} analogies solved.`
        : `Collapsed cone: mean cos ${r1.meanCos.toFixed(3)} (need < 0.35) · ${r1.hits}/${r1.total} analogies (need ≥ 3/4).`;
    }
    if (briefId === 'analogies') {
      if (r2.pass) {
        return r2.dim === 3
          ? `All 4/4 analogies solved in just 3D (window ±${r2.win}) — minimum viable dimension!`
          : `All 4/4 analogies solved in ${r2.dim}D (window ±${r2.win}). Optional goal: can you pack them into fewer dimensions?`;
      }
      return `${r2.hits}/${r2.total} analogies solved in ${r2.dim}D (window ±${r2.win}) — need 4/4.`;
    }
    if (r3.pass) {
      return r3.k === 5
        ? `${r3.dotsPerPair} dot-products/pair (≤ 15) · ${Math.round(r3.speedup).toLocaleString()}× faster than Full Softmax — fastest qualifying setting!`
        : `${r3.dotsPerPair} dot-products/pair (≤ 15) · ${Math.round(r3.speedup).toLocaleString()}× speedup. Optional goal: can you go even faster while keeping K ≥ 5?`;
    }
    return `${r3.dotsPerPair.toLocaleString()} dot-products/pair (limit 15) · ${r3.k < 0 ? 'Full Softmax' : `K = ${r3.k} (need ≥ 5)`}.`;
  }, [briefId, r1, r2, r3]);

  return (
    <LabShell
      title="Lab: Train & Debug Word2Vec"
      caption={
        <>
          Watch SGNS vectors collapse or open into semantic parallelograms live, squeeze 4/4 analogies into minimal dimensions, and scale past the 100,000-word softmax wall.
        </>
      }
      briefs={W2V_LAB_BRIEFS.map((b, i) => ({
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
        title: 'SGNS training log',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Change negative samples K, dimension, window, or objective to train and inspect.',
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

            {briefId === 'collapse' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Neg samples K</th>
                    <th>Mean cos</th>
                    <th>Analogies</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {allB1.map((row) => (
                    <tr key={row.k} style={{ fontWeight: row.k === st.b1K ? 600 : 400 }}>
                      <td>K = {row.k}</td>
                      <td>{row.meanCos.toFixed(3)}</td>
                      <td>{row.hits}/{row.total}</td>
                      <td style={{ color: row.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{row.pass ? 'PASS' : 'FAIL'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {briefId === 'analogies' && (
              <div style={{ overflowX: 'auto' }}>
                <table className="num-table" style={{ width: '100%', fontSize: 12, textAlign: 'center' }}>
                  <thead>
                    <tr>
                      <th>Dim</th>
                      {LAB_WINS_B2.map((w) => (
                        <th key={w}>Window ±{w}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {LAB_DIMS_B2.map((d) => (
                      <tr key={d}>
                        <td style={{ fontWeight: 600 }}>{d}D</td>
                        {LAB_WINS_B2.map((w) => {
                          const cell = evalW2vBrief2(d, w);
                          const isCur = d === st.b2Dim && w === st.b2Win;
                          return (
                            <td
                              key={w}
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
                              {cell.hits}/{cell.total}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {briefId === 'scale' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Objective</th>
                    <th>Dots / pair</th>
                    <th>Speedup</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {allB3.map((row) => (
                    <tr key={row.mode} style={{ fontWeight: row.mode === st.b3Mode ? 600 : 400 }}>
                      <td>{row.mode === 'full-softmax' ? 'Full Softmax' : `SGNS K=${row.k}`}</td>
                      <td>{row.dotsPerPair.toLocaleString()}</td>
                      <td>{Math.round(row.speedup).toLocaleString()}×</td>
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
      {briefId === 'collapse' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <Segmented
            label="Negative samples"
            value={st.b1K}
            onChange={updateB1}
            options={LAB_KS_B1.map((k) => ({ value: k, label: `K = ${k}` }))}
          />
          <div style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
              <span>Mean pairwise cosine (collapse indicator · ceiling = 0.35)</span>
              <strong style={{ fontFamily: 'var(--mono)', color: r1.meanCos < 0.35 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r1.meanCos.toFixed(3)}
              </strong>
            </div>
            <div style={{ position: 'relative', height: 10, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${Math.min(100, tweenedCos1 * 100)}%`, background: r1.meanCos < 0.35 ? 'var(--c-good)' : 'var(--c-bad)' }} />
              <div style={{ position: 'absolute', top: 0, bottom: 0, left: '35%', width: 2, background: 'var(--text)' }} />
            </div>
          </div>
          <LabConstellation vecs={r1.vecs} analogyDetails={r1.analogyDetails} />
        </div>
      )}

      {briefId === 'analogies' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Dimension"
              value={st.b2Dim}
              onChange={(v) => updateB2({ b2Dim: v })}
              options={LAB_DIMS_B2.map((d) => ({ value: d, label: `${d}D` }))}
            />
            <Segmented
              label="Context window"
              value={st.b2Win}
              onChange={(v) => updateB2({ b2Win: v })}
              options={LAB_WINS_B2.map((w) => ({ value: w, label: `±${w}` }))}
            />
          </div>
          <LabConstellation vecs={r2.vecs} analogyDetails={r2.analogyDetails} />
        </div>
      )}

      {briefId === 'scale' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <Segmented
            label="Training objective"
            value={st.b3Mode}
            onChange={updateB3}
            options={LAB_SCALE_MODES.map((m) => ({
              value: m,
              label: m === 'full-softmax' ? 'Full Softmax' : `SGNS K=${m.split('-')[1]}`,
            }))}
          />

          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 12 }}>
              <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                centre v_c
              </span>
              <span>·</span>
              <span
                style={{
                  padding: '4px 9px',
                  borderRadius: 6,
                  background: 'var(--bg)',
                  border: `1px solid ${r3.pass ? 'var(--c-good)' : 'var(--c-bad)'}`,
                  fontWeight: 600,
                }}
              >
                {r3.mode === 'full-softmax'
                  ? 'ALL 100,000 context rows (V = 100k)'
                  : `1 real neighbour + ${r3.k} random noise rows`}
              </span>
              <span>=</span>
              <span style={{ fontWeight: 700, color: r3.dotsPerPair <= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.dotsPerPair.toLocaleString()} dots/pair ({Math.round(r3.speedup).toLocaleString()}× faster)
              </span>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span>Dot-products per pair (log₁₀ scale · marker = 15)</span>
                <strong style={{ fontFamily: 'var(--mono)', color: r3.dotsPerPair <= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {r3.dotsPerPair.toLocaleString()} / 15
                </strong>
              </div>
              <div style={{ position: 'relative', height: 14, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.max(4, Math.min(100, (tweenedLogDots3 / 5) * 100))}%`,
                    background: r3.dotsPerPair <= 15 ? 'var(--c-good)' : 'var(--c-bad)',
                  }}
                />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(Math.log10(15) / 5) * 100}%`, width: 2, background: 'var(--text)' }} />
              </div>
            </div>
          </div>
        </div>
      )}

      <LiveCode note="Gensim Word2Vec (SGNS)" code={realCode} />
    </LabShell>
  );
}
