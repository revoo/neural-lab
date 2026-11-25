// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  ANN_LAB_BRIEFS,
  COMP_SPECS,
  LAB_LSH_BITS,
  LAB_LSH_TABLES,
  LAB_NLISTS,
  LAB_NPROBES,
  annLabExhaustive,
  benchmarkIvf16D,
  compareMetrics,
  evalAnnBrief1,
  evalAnnBrief2,
  evalAnnBrief3,
  exactTopK,
  getIvf2DScene,
  lshCollisionProb,
  queryIvf,
  type AnnBriefId,
  type CompFormat,
} from './ann';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

// ------------------------------------------------------------------ 1. Cosine vs Dot Product vs L2

export function MetricBattleWidget() {
  const [bScale, setBScale] = useState<number>(2.8);
  const [unitNorm, setUnitNorm] = useState<boolean>(false);

  const cands = useMemo(
    () => [
      { id: 'A', label: 'Doc A (15° angle, short)', vec: [0.95, 0.25] as [number, number] },
      { id: 'B', label: 'Doc B (32° angle, long)', vec: [0.85 * bScale, 0.53 * bScale] as [number, number] },
      { id: 'C', label: 'Doc C (33° angle, unit)', vec: [0.84, 0.54] as [number, number] },
    ],
    [bScale],
  );
  const res = useMemo(() => compareMetrics([1, 0], cands, unitNorm), [cands, unitNorm]);

  return (
    <Widget
      title="Cosine vs Dot Product vs Euclidean: when length changes the winner"
      wide
      caption={
        <>
          Stretch <strong>Doc B’s length</strong> while keeping its angle wider than Doc A’s. Without normalisation, <strong>Dot Product</strong> picks long Doc B while <strong>Cosine</strong> and <strong>L2</strong> pick close-angle Doc A. Flip <strong>L2-normalise all vectors</strong> on: suddenly <code>‖a − b‖² = 2(1 − cos)</code> and all three metrics agree 100%!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Vector length"
          value={unitNorm ? 'unit' : 'raw'}
          onChange={(v) => setUnitNorm(v === 'unit')}
          options={[
            { value: 'raw', label: 'Raw lengths' },
            { value: 'unit', label: 'L2-normalised (‖v‖ = 1)' },
          ]}
        />
        <div style={{ flex: '1 1 200px' }}>
          <Slider label="Doc B length" value={bScale} min={0.6} max={3.2} step={0.1} onChange={setBScale} format={(v) => `${v.toFixed(1)}×`} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 14, alignItems: 'center' }}>
        <Plot x={[-0.2, 3.0]} y={[-0.2, 2.2]} height={220} xLabel="Dim 1" yLabel="Dim 2">
          {({ x: sx, y: sy }) => (
            <>
              {/* Unit circle arc */}
              <ellipse cx={sx(0)} cy={sy(0)} rx={Math.abs(sx(1) - sx(0))} ry={Math.abs(sy(0) - sy(1))} fill="none" stroke="var(--border)" strokeDasharray="3 3" />
              {/* Query */}
              <line x1={sx(0)} y1={sy(0)} x2={sx(res.q[0])} y2={sy(res.q[1])} stroke="var(--accent)" strokeWidth={2.5} />
              <circle cx={sx(res.q[0])} cy={sy(res.q[1])} r={4.5} fill="var(--accent)" />
              <text x={sx(res.q[0]) + 6} y={sy(res.q[1]) - 5} fill="var(--accent)" fontSize={11} fontWeight={700}>Query</text>
              {res.rows.map((r, i) => {
                const col = i === 0 ? 'var(--c-a)' : i === 1 ? 'var(--c-b)' : 'var(--c-c)';
                return (
                  <g key={r.id}>
                    <line x1={sx(0)} y1={sy(0)} x2={sx(r.vec[0])} y2={sy(r.vec[1])} stroke={col} strokeWidth={2} />
                    <circle cx={sx(r.vec[0])} cy={sy(r.vec[1])} r={4} fill={col} />
                    <text x={sx(r.vec[0]) + 6} y={sy(r.vec[1]) + 3} fill={col} fontSize={11} fontWeight={600}>
                      Doc {r.id}
                    </text>
                  </g>
                );
              })}
            </>
          )}
        </Plot>

        <div style={{ overflowX: 'auto' }}>
          <table className="num-table" style={{ width: '100%', fontSize: 12.5 }}>
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Cosine (↑)</th>
                <th>Dot (↑)</th>
                <th>L2 dist (↓)</th>
              </tr>
            </thead>
            <tbody>
              {res.rows.map((r) => (
                <tr key={r.id}>
                  <td style={{ fontWeight: 600 }}>Doc {r.id}</td>
                  <td style={{ color: res.bestCos === r.id ? 'var(--c-good)' : undefined, fontWeight: res.bestCos === r.id ? 700 : 400 }}>
                    {r.cos.toFixed(3)} {res.bestCos === r.id ? '★' : ''}
                  </td>
                  <td style={{ color: res.bestDot === r.id ? 'var(--c-good)' : undefined, fontWeight: res.bestDot === r.id ? 700 : 400 }}>
                    {r.dotVal.toFixed(3)} {res.bestDot === r.id ? '★' : ''}
                  </td>
                  <td style={{ color: res.bestL2 === r.id ? 'var(--c-good)' : undefined, fontWeight: res.bestL2 === r.id ? 700 : 400 }}>
                    {r.l2.toFixed(3)} {res.bestL2 === r.id ? '★' : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 12.5, color: res.allAgree ? 'var(--c-good)' : 'var(--c-b)', marginTop: 8 }}>
            {res.allAgree
              ? 'All three metrics pick the exact same winner (Doc ' + res.bestCos + ')!'
              : `Metrics disagree: Cosine & L2 pick Doc ${res.bestCos}, while Dot Product picks long Doc ${res.bestDot}.`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. IVF Voronoi Buckets & Multi-Probe Rescue

export function IvfExplorerWidget() {
  const scene = useMemo(() => getIvf2DScene(), []);
  const [nprobe, setNprobe] = useState<number>(1);

  const truth = useMemo(() => new Set(exactTopK(scene.data, scene.boundaryQuery, 5)), [scene]);
  const qRes = useMemo(() => queryIvf(scene.data, scene.ivf, scene.boundaryQuery, nprobe, 5), [scene, nprobe]);
  const probedSet = useMemo(() => new Set(qRes.probedBuckets), [qRes]);
  const foundSet = useMemo(() => new Set(qRes.topK), [qRes]);
  const hits = qRes.topK.filter((i) => truth.has(i)).length;

  const sweep16D = useMemo(() => [1, 2, 3, 4, 6].map((np) => ({ np, ...benchmarkIvf16D(16, np) })), []);

  return (
    <Widget
      title="IVF (Inverted File Index): skip 90% of vectors — and rescue Voronoi fence misses"
      wide
      caption={
        <>
          240 vectors partitioned into 12 k-Means Voronoi buckets. The query (purple star) sits right near a cell boundary: at <code>nprobe = 1</code>, we scan only <strong>20/240</strong> points (32 total distance checks) but catch just <strong>1/5</strong> true neighbours! Slide <strong>Probed buckets</strong> to <code>2</code>: scanning 40 points immediately rescues <strong>5/5 (100%)</strong>!
        </>
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Slider label="Probed buckets (nprobe)" value={nprobe} min={1} max={6} step={1} onChange={setNprobe} format={(v) => `${v} / 12 buckets`} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 14, alignItems: 'center' }}>
        <Plot x={[-4.8, 4.8]} y={[-4.5, 4.5]} height={270} xLabel="Dim 1" yLabel="Dim 2">
          {({ x: sx, y: sy }) => (
            <>
              {scene.data.map((p, i) => {
                const b = scene.ivf.assign[i];
                const isProbed = probedSet.has(b);
                const isTruth = truth.has(i);
                const isFound = foundSet.has(i) && isTruth;
                return (
                  <circle
                    key={i}
                    cx={sx(p[0])}
                    cy={sy(p[1])}
                    r={isTruth ? 5.5 : 3.2}
                    fill={isFound ? 'var(--c-good)' : isTruth ? 'var(--c-bad)' : isProbed ? 'var(--c-a)' : 'var(--text-muted)'}
                    opacity={isTruth || isProbed ? 0.92 : 0.22}
                  />
                );
              })}
              {/* Centroids */}
              {scene.ivf.centroids.map((c, idx) => (
                <circle
                  key={`c-${idx}`}
                  cx={sx(c[0])}
                  cy={sy(c[1])}
                  r={5}
                  fill="var(--bg)"
                  stroke={probedSet.has(idx) ? 'var(--c-a)' : 'var(--border)'}
                  strokeWidth={2}
                />
              ))}
              {/* Query */}
              <circle cx={sx(scene.boundaryQuery[0])} cy={sy(scene.boundaryQuery[1])} r={7} fill="var(--accent)" stroke="var(--bg)" strokeWidth={2} />
            </>
          )}
        </Plot>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">True Top-5 found</div>
              <div className="readout-value" style={{ color: hits === 5 ? 'var(--c-good)' : 'var(--c-bad)' }}>{hits} / 5</div>
            </div>
            <div className="readout">
              <div className="readout-label">Distance checks</div>
              <div className="readout-value">{qRes.distEvals} / 240</div>
            </div>
            <div className="readout">
              <div className="readout-label">Speedup</div>
              <div className="readout-value" style={{ color: 'var(--c-a)' }}>{(240 / qRes.distEvals).toFixed(1)}×</div>
            </div>
          </div>
          <div style={{ fontSize: 12.5, padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', lineHeight: 1.5 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>16D benchmark (600 vectors, nlist = 16, 60 queries):</div>
            {sweep16D.map((r) => (
              <div key={r.np} style={{ fontFamily: 'var(--mono)', fontSize: 11.5 }}>
                nprobe={r.np}: Recall@5 <strong>{pct1(r.recall5)}</strong> · scan {pct1(r.scanFrac)} ({r.speedup.toFixed(1)}×)
              </div>
            ))}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Random-Hyperplane LSH S-Curve & Compression RAM

export function LshCurveWidget() {
  const [bits, setBits] = useState<number>(8);
  const [tables, setTables] = useState<number>(8);

  const pts = useMemo(
    () => Array.from({ length: 37 }, (_, i) => {
      const deg = i * 5;
      return { deg, p: lshCollisionProb(deg, bits, tables) };
    }),
    [bits, tables],
  );
  const p20 = lshCollisionProb(20, bits, tables);
  const p90 = lshCollisionProb(90, bits, tables);

  return (
    <Widget
      title="Locality-Sensitive Hashing (SimHash): shaping the collision S-curve"
      wide
      caption={
        <>
          Each random hyperplane separates two vectors with probability <code>θ / 180°</code>. More <strong>hash bits b</strong> narrows buckets (blocking distant 90° false positives); more <strong>hash tables L</strong> boosts recall on close 20° neighbours!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 12 }}>
        <div style={{ flex: '1 1 180px' }}>
          <Slider label="Hash bits per table (b)" value={bits} min={2} max={12} step={1} onChange={setBits} format={(v) => `${v} bits (${1 << v} buckets)`} />
        </div>
        <div style={{ flex: '1 1 180px' }}>
          <Slider label="Independent tables (L)" value={tables} min={1} max={16} step={1} onChange={setTables} format={(v) => `${v} tables`} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 14, alignItems: 'center' }}>
        <Plot x={[0, 180]} y={[0, 1.05]} height={200} xLabel="Angle θ between vectors (degrees)" yLabel="P(match in ≥1 table)">
          {({ x: sx, y: sy }) => {
            const path = pts.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${sx(pt.deg).toFixed(1)} ${sy(pt.p).toFixed(1)}`).join(' ');
            return (
              <>
                <line x1={sx(20)} y1={sy(0)} x2={sx(20)} y2={sy(1.05)} stroke="var(--c-good)" strokeDasharray="3 3" />
                <line x1={sx(90)} y1={sy(0)} x2={sx(90)} y2={sy(1.05)} stroke="var(--c-bad)" strokeDasharray="3 3" />
                <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2.4} />
                <circle cx={sx(20)} cy={sy(p20)} r={4.5} fill="var(--c-good)" />
                <circle cx={sx(90)} cy={sy(p90)} r={4.5} fill="var(--c-bad)" />
              </>
            );
          }}
        </Plot>

        <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
          <div className="readout">
            <div className="readout-label">Close pair (20°)</div>
            <div className="readout-value" style={{ color: p20 >= 0.8 ? 'var(--c-good)' : 'var(--c-b)' }}>{pct1(p20)}</div>
          </div>
          <div className="readout">
            <div className="readout-label">Unrelated pair (90°)</div>
            <div className="readout-value" style={{ color: p90 <= 0.15 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(p90)}</div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tune the Vector Index

type AnnLabState = {
  nlist: number;
  nprobe: number;
  bits: number;
  tables: number;
  fmt: CompFormat;
  solved: Record<AnnBriefId, boolean>;
};

const ANN_LAB_KEY = 'nl-ann-lab-v1';
const ANN_LAB_DEFAULT: AnnLabState = {
  nlist: 8,
  nprobe: 1,
  bits: 6,
  tables: 1,
  fmt: 'fp32',
  solved: { ivf: false, lsh: false, ram: false },
};

function loadAnnLab(): AnnLabState {
  try {
    const raw = localStorage.getItem(ANN_LAB_KEY);
    if (!raw) return ANN_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...ANN_LAB_DEFAULT, ...p, solved: { ...ANN_LAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return ANN_LAB_DEFAULT;
  }
}

export function VectorIndexLab() {
  const [briefId, setBriefId] = useState<AnnBriefId>('ivf');
  const [st, setSt] = useState<AnnLabState>(loadAnnLab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(ANN_LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalAnnBrief1(st.nlist, st.nprobe), [st.nlist, st.nprobe]);
  const r2 = useMemo(() => evalAnnBrief2(st.bits, st.tables), [st.bits, st.tables]);
  const r3 = useMemo(() => evalAnnBrief3(st.fmt), [st.fmt]);
  const exhaust = useMemo(() => annLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'ivf' ? r1.pass : briefId === 'lsh' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const brief = ANN_LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Tune the Vector Index"
      wide
      caption={
        <>
          Balance Recall@5, query speedup, and vector-database RAM across three exhaustive briefs. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {ANN_LAB_BRIEFS.map((b, i) => (
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

      {briefId === 'ivf' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Voronoi buckets (nlist)"
              value={st.nlist}
              onChange={(v) => setSt((p) => ({ ...p, nlist: v }))}
              options={LAB_NLISTS.map((n) => ({ value: n, label: String(n) }))}
            />
            <Segmented
              label="Probed buckets (nprobe)"
              value={st.nprobe}
              onChange={(v) => setSt((p) => ({ ...p, nprobe: v }))}
              options={LAB_NPROBES.map((n) => ({ value: n, label: String(n) }))}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Recall@5 (≥ 90%)</div>
              <div className="readout-value" style={{ color: r1.recall5 >= 0.9 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r1.recall5)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Scan work (≤ 25%)</div>
              <div className="readout-value" style={{ color: r1.scanFrac <= 0.25 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r1.scanFrac)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Speedup</div>
              <div className="readout-value">{r1.speedup.toFixed(1)}×</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.pass ? 'PASS (Unique!)' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'lsh' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Hash bits"
              value={st.bits}
              onChange={(v) => setSt((p) => ({ ...p, bits: v }))}
              options={LAB_LSH_BITS.map((b) => ({ value: b, label: `${b} bits` }))}
            />
            <Segmented
              label="Hash tables"
              value={st.tables}
              onChange={(v) => setSt((p) => ({ ...p, tables: v }))}
              options={LAB_LSH_TABLES.map((t) => ({ value: t, label: `${t} tables` }))}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Recall@5 (≥ 85%)</div>
              <div className="readout-value" style={{ color: r2.recall5 >= 0.85 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r2.recall5)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Scan work (≤ 35%)</div>
              <div className="readout-value" style={{ color: r2.scanFrac <= 0.35 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r2.scanFrac)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Speedup</div>
              <div className="readout-value">{r2.speedup.toFixed(1)}×</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.pass ? 'PASS (Unique!)' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'ram' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {COMP_SPECS.map((s) => (
              <button
                key={s.id}
                type="button"
                className="btn"
                style={{
                  fontSize: 12.5,
                  padding: '5px 10px',
                  borderColor: st.fmt === s.id ? 'var(--accent)' : 'var(--border)',
                  background: st.fmt === s.id ? 'var(--bg-sunken)' : 'transparent',
                  fontWeight: st.fmt === s.id ? 600 : 400,
                }}
                onClick={() => setSt((p) => ({ ...p, fmt: s.id }))}
              >
                {s.label}
              </button>
            ))}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">RAM (≤ 16.0 GiB)</div>
              <div className="readout-value" style={{ color: r3.gib <= 16 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.gib.toFixed(2)} GiB</div>
            </div>
            <div className="readout">
              <div className="readout-label">Compression</div>
              <div className="readout-value">{r3.ratio.toFixed(0)}×</div>
            </div>
            <div className="readout">
              <div className="readout-label">Recall kept (≥ 95%)</div>
              <div className="readout-value" style={{ color: r3.typicalRecall >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r3.typicalRecall)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r3.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.pass ? 'PASS (Unique!)' : 'FAIL'}</div>
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
