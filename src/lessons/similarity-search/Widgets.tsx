// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweened, useTweenedArray } from '../../components/motion';
import {
  ANN_LAB_BRIEFS,
  COMP_SPECS,
  LAB_LSH_BITS,
  LAB_LSH_TABLES,
  LAB_NLISTS,
  LAB_NPROBES,
  annLabExhaustive,
  annLabHint,
  benchmarkIvf16D,
  compareMetrics,
  evalAnnBrief1,
  evalAnnBrief2,
  evalAnnBrief3,
  exactTopK,
  getIvf2DScene,
  lshCollisionProb,
  queryIvf,
  type CompFormat,
} from './ann';
import { faissRealCode } from './realCode';

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
};

const ANN_LAB_KEY = 'nl-ann-lab-v1';
const ANN_LAB_PROG_KEY = 'nl-ann-lab-progress-v2';
const ANN_LAB_DEFAULT: AnnLabState = {
  nlist: 8,
  nprobe: 1,
  bits: 6,
  tables: 1,
  fmt: 'fp32',
};

function loadLegacyAnnSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(ANN_LAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved ?? {};
  } catch {
    return {};
  }
}

function loadAnnLab(): AnnLabState {
  try {
    const raw = localStorage.getItem(ANN_LAB_KEY);
    if (!raw) return ANN_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return {
      nlist: (LAB_NLISTS as readonly number[]).includes(p.nlist) ? p.nlist : ANN_LAB_DEFAULT.nlist,
      nprobe: (LAB_NPROBES as readonly number[]).includes(p.nprobe) ? p.nprobe : ANN_LAB_DEFAULT.nprobe,
      bits: (LAB_LSH_BITS as readonly number[]).includes(p.bits) ? p.bits : ANN_LAB_DEFAULT.bits,
      tables: (LAB_LSH_TABLES as readonly number[]).includes(p.tables) ? p.tables : ANN_LAB_DEFAULT.tables,
      fmt: COMP_SPECS.some((s) => s.id === p.fmt) ? p.fmt : ANN_LAB_DEFAULT.fmt,
    };
  } catch {
    return ANN_LAB_DEFAULT;
  }
}

function QueryRecallStrip({ hits }: { hits: readonly number[] }) {
  const perfect = hits.filter((h) => h === 5).length;
  return (
    <div>
      <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
        60 held-out queries (each tile = Recall@5: <strong>{perfect}/60</strong> hit 5/5)
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(15, 1fr)', gap: 3 }}>
        {hits.map((h, i) => {
          const bg =
            h === 5
              ? 'color-mix(in srgb, var(--c-good) 58%, var(--bg))'
              : h >= 3
                ? 'color-mix(in srgb, var(--c-a) 45%, var(--bg))'
                : 'color-mix(in srgb, var(--c-bad) 52%, var(--bg))';
          return (
            <div
              key={i}
              title={`Query ${i + 1}: ${h}/5 exact top-5 found`}
              style={{
                height: 16,
                borderRadius: 3,
                background: bg,
                border: '1px solid var(--border)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: 'var(--mono)',
                fontSize: 9.5,
                fontWeight: 600,
              }}
            >
              {h}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function VectorIndexLab() {
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const briefId = ANN_LAB_BRIEFS[activeIdx].id;
  const [st, setSt] = useState<AnnLabState>(loadAnnLab);
  const progress = useLabProgress(ANN_LAB_PROG_KEY, () => ({ solved: loadLegacyAnnSolved() }));
  const log = useMachineLog(18);

  useEffect(() => {
    try {
      localStorage.setItem(ANN_LAB_KEY, JSON.stringify({ ...st, solved: progress.solved }));
    } catch {
      /* ignore */
    }
  }, [st, progress.solved]);

  const r1 = useMemo(() => evalAnnBrief1(st.nlist, st.nprobe), [st.nlist, st.nprobe]);
  const r2 = useMemo(() => evalAnnBrief2(st.bits, st.tables), [st.bits, st.tables]);
  const r3 = useMemo(() => evalAnnBrief3(st.fmt), [st.fmt]);
  const exhaust = useMemo(() => annLabExhaustive(briefId), [briefId]);
  const hint = useMemo(() => annLabHint(briefId, st), [briefId, st]);
  const realCode = useMemo(() => faissRealCode(briefId, st), [briefId, st]);

  const b1All = useMemo(() => {
    const out: ReturnType<typeof evalAnnBrief1>[] = [];
    for (const nl of LAB_NLISTS) for (const np of LAB_NPROBES) out.push(evalAnnBrief1(nl, np));
    return out;
  }, []);
  const b3All = useMemo(() => COMP_SPECS.map((s) => ({ spec: s, r: evalAnnBrief3(s.id) })), []);

  const curPass = briefId === 'ivf' ? r1.pass : briefId === 'lsh' ? r2.pass : r3.pass;

  const tweenedIvf = useTweenedArray([r1.recall5, r1.scanFrac], 240);
  const tweenedLsh = useTweenedArray([r2.recall5, r2.scanFrac], 240);
  const tweenedRam = useTweened(r3.gib, 240);
  const tweenedRec3 = useTweened(r3.typicalRecall, 240);

  const updateIvf = (patch: Partial<Pick<AnnLabState, 'nlist' | 'nprobe'>>) => {
    const nl = patch.nlist ?? st.nlist;
    const np = patch.nprobe ?? st.nprobe;
    const ev = evalAnnBrief1(nl, np);
    setSt((p) => ({ ...p, nlist: nl, nprobe: np }));
    log.push(
      `B1 IVF: nlist=${nl}, nprobe=${np} → Recall@5 ${pct1(ev.recall5)}, scan ${pct1(ev.scanFrac)} (${ev.speedup.toFixed(1)}×, ${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const updateLsh = (patch: Partial<Pick<AnnLabState, 'bits' | 'tables'>>) => {
    const b = patch.bits ?? st.bits;
    const t = patch.tables ?? st.tables;
    const ev = evalAnnBrief2(b, t);
    setSt((p) => ({ ...p, bits: b, tables: t }));
    log.push(
      `B2 LSH: b=${b}, L=${t} → Recall@5 ${pct1(ev.recall5)}, scan ${pct1(ev.scanFrac)} (${ev.speedup.toFixed(1)}×, ${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const updateFmt = (fmt: CompFormat) => {
    const ev = evalAnnBrief3(fmt);
    setSt((p) => ({ ...p, fmt }));
    log.push(
      `B3 RAM: ${fmt} → ${ev.gib.toFixed(2)} GiB (${ev.ratio.toFixed(0)}×), recall ${pct1(ev.typicalRecall)} (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const resetBrief = () => {
    if (briefId === 'ivf') {
      setSt((p) => ({ ...p, nlist: ANN_LAB_DEFAULT.nlist, nprobe: ANN_LAB_DEFAULT.nprobe }));
      log.push('B1: reset to nlist=8, nprobe=1');
    } else if (briefId === 'lsh') {
      setSt((p) => ({ ...p, bits: ANN_LAB_DEFAULT.bits, tables: ANN_LAB_DEFAULT.tables }));
      log.push('B2: reset to b=6 bits, L=1 table');
    } else {
      setSt((p) => ({ ...p, fmt: ANN_LAB_DEFAULT.fmt }));
      log.push('B3: reset to FP32');
    }
  };

  const statusText = useMemo(() => {
    if (briefId === 'ivf') {
      return r1.pass
        ? `Unique IVF sweet spot: Recall@5 ${pct1(r1.recall5)} (≥ 90%) at ${pct1(r1.scanFrac)} scan (${r1.speedup.toFixed(1)}× faster than brute force).`
        : `Recall@5 ${pct1(r1.recall5)} (need ≥ 90%) · scan work ${pct1(r1.scanFrac)} (limit 25%) · ${r1.speedup.toFixed(1)}× speedup.`;
    }
    if (briefId === 'lsh') {
      return r2.pass
        ? `Unique LSH sweet spot: Recall@5 ${pct1(r2.recall5)} (≥ 85%) at ${pct1(r2.scanFrac)} scan (${r2.speedup.toFixed(1)}× speedup).`
        : `Recall@5 ${pct1(r2.recall5)} (need ≥ 85%) · candidate scan ${pct1(r2.scanFrac)} (limit 35%) · ${r2.speedup.toFixed(1)}× speedup.`;
    }
    return r3.pass
      ? `Unique passing format: ${r3.gib.toFixed(2)} GiB (≤ 16.0 GiB, ${r3.ratio.toFixed(0)}× smaller than FP32) at ${pct1(r3.typicalRecall)} recall.`
      : `${r3.gib.toFixed(2)} GiB (limit 16.0 GiB) · typical recall ${pct1(r3.typicalRecall)} (need ≥ 95.0%).`;
  }, [briefId, r1, r2, r3]);

  const lshCurvePath = useMemo(() => {
    const pts: string[] = [];
    for (let deg = 0; deg <= 120; deg += 4) {
      const p = lshCollisionProb(deg, st.bits, st.tables);
      const cx = 34 + (deg / 120) * 232;
      const cy = 112 - p * 96;
      pts.push(`${deg === 0 ? 'M' : 'L'} ${cx.toFixed(1)} ${cy.toFixed(1)}`);
    }
    return pts.join(' ');
  }, [st.bits, st.tables]);

  return (
    <LabShell
      title="Lab: Tune the Vector Index"
      caption={
        <>
          Inspect Voronoi bucket occupancy, per-query Recall@5 tiles, LSH collision S-curves, and 10M-vector RAM compression live across three exhaustive briefs.
        </>
      }
      briefs={ANN_LAB_BRIEFS.map((b, i) => ({
        id: b.id,
        tab: `Brief ${i + 1}`,
        title: b.title,
        text: b.story,
      }))}
      active={activeIdx}
      onSelect={setActiveIdx}
      progress={progress}
      passing={curPass}
      optimal={curPass}
      status={statusText}
      hint={hint}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
      log={{
        title: 'Vector index bench log',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Adjust nlist/nprobe, LSH bits/tables, or quantisation format to run a benchmark.',
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

            {briefId === 'ivf' && (
              <div>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
                  All 15 (nlist, nprobe) configs: scan work vs Recall@5 (shaded quadrant = scan ≤ 25% &amp; recall ≥ 90%)
                </div>
                <svg viewBox="0 0 360 144" style={{ width: '100%', maxWidth: 420, height: 'auto', display: 'block' }}>
                  <rect x={42} y={10} width={304} height={102} fill="var(--bg)" stroke="var(--border)" rx={4} />
                  {(() => {
                    const x25 = 42 + (0.25 / 0.85) * 304;
                    const y90 = 112 - ((0.9 - 0.35) / 0.67) * 102;
                    return (
                      <>
                        <rect x={42} y={10} width={x25 - 42} height={y90 - 10} fill="color-mix(in srgb, var(--c-good) 15%, transparent)" />
                        <line x1={x25} y1={10} x2={x25} y2={112} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1} />
                        <line x1={42} y1={y90} x2={346} y2={y90} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1} />
                      </>
                    );
                  })()}
                  {b1All.map((pt) => {
                    const cx = 42 + (Math.min(0.85, pt.scanFrac) / 0.85) * 304;
                    const cy = 112 - ((Math.max(0.35, pt.recall5) - 0.35) / 0.67) * 102;
                    const isCur = pt.nlist === st.nlist && pt.nprobe === st.nprobe;
                    const col = pt.pass ? 'var(--c-good)' : pt.nlist === 32 ? 'var(--c-a)' : pt.nlist === 16 ? 'var(--c-b)' : 'var(--text-muted)';
                    return (
                      <circle
                        key={`${pt.nlist}-${pt.nprobe}`}
                        cx={cx}
                        cy={cy}
                        r={isCur ? 5.5 : 3.8}
                        fill={col}
                        stroke={isCur ? 'var(--text)' : 'none'}
                        strokeWidth={2}
                      />
                    );
                  })}
                  <text x={42} y={128} fontSize={9} fill="var(--text-muted)">0% scan</text>
                  <text x={118} y={128} fontSize={9} fill="var(--c-good)">25%</text>
                  <text x={306} y={128} fontSize={9} fill="var(--text-muted)">85% scan</text>
                  <text x={4} y={31} fontSize={9} fill="var(--c-good)">90%</text>
                  <text x={4} y={110} fontSize={9} fill="var(--text-muted)">35%</text>
                </svg>
              </div>
            )}

            {briefId === 'lsh' && (
              <div style={{ overflowX: 'auto' }}>
                <table className="num-table" style={{ width: '100%', fontSize: 11.5, textAlign: 'center' }}>
                  <thead>
                    <tr>
                      <th>Bits b</th>
                      {LAB_LSH_TABLES.map((t) => (
                        <th key={t}>L = {t}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {LAB_LSH_BITS.map((b) => (
                      <tr key={b}>
                        <td style={{ fontWeight: 600 }}>{b} bits</td>
                        {LAB_LSH_TABLES.map((t) => {
                          const cell = evalAnnBrief2(b, t);
                          const isCur = b === st.bits && t === st.tables;
                          return (
                            <td
                              key={t}
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
                              {(100 * cell.recall5).toFixed(0)}% / {(100 * cell.scanFrac).toFixed(0)}%
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {briefId === 'ram' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Format</th>
                    <th>B / vec</th>
                    <th>10M RAM</th>
                    <th>Recall kept</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {b3All.map(({ spec, r }) => (
                    <tr key={spec.id} style={{ fontWeight: spec.id === st.fmt ? 600 : 400 }}>
                      <td>{spec.label}</td>
                      <td>{spec.bytesPerVec.toLocaleString()}</td>
                      <td>{r.gib.toFixed(2)} GiB</td>
                      <td>{pct1(r.typicalRecall)}</td>
                      <td style={{ color: r.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r.pass ? 'PASS' : 'FAIL'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ),
      }}
    >
      {briefId === 'ivf' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Voronoi buckets (nlist)"
              value={st.nlist}
              onChange={(v) => updateIvf({ nlist: v })}
              options={LAB_NLISTS.map((n) => ({ value: n, label: String(n) }))}
            />
            <Segmented
              label="Probed buckets (nprobe)"
              value={st.nprobe}
              onChange={(v) => updateIvf({ nprobe: v })}
              options={LAB_NPROBES.map((n) => ({ value: n, label: String(n) }))}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
            <div style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span>Recall@5 (target ≥ 90%)</span>
                <strong style={{ fontFamily: 'var(--mono)', color: r1.recall5 >= 0.9 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r1.recall5)}</strong>
              </div>
              <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, tweenedIvf[0] * 100)}%`, background: r1.recall5 >= 0.9 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: '90%', width: 2, background: 'var(--text)' }} />
              </div>
            </div>
            <div style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span>Scan work (ceiling ≤ 25% · {r1.speedup.toFixed(1)}×)</span>
                <strong style={{ fontFamily: 'var(--mono)', color: r1.scanFrac <= 0.25 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r1.scanFrac)}</strong>
              </div>
              <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, tweenedIvf[1] * 100)}%`, background: r1.scanFrac <= 0.25 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: '25%', width: 2, background: 'var(--text)' }} />
              </div>
            </div>
          </div>

          {/* Voronoi bucket occupancy strip + 60-query Recall@5 heatmap */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12, padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
            <div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 6 }}>
                Inside IVF ({st.nlist} Voronoi buckets · highlighted = {st.nprobe} probed for query #1)
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 68 }}>
                {(() => {
                  const maxB = Math.max(...r1.bucketSizes, 1);
                  const probedSet = new Set(r1.sampleProbed);
                  return r1.bucketSizes.map((sz, i) => {
                    const isProbed = probedSet.has(i);
                    return (
                      <div
                        key={i}
                        title={`Bucket ${i}: ${sz} vectors${isProbed ? ' (probed)' : ''}`}
                        style={{
                          flex: 1,
                          height: `${Math.max(8, (sz / maxB) * 100)}%`,
                          borderRadius: '3px 3px 0 0',
                          background: isProbed ? 'var(--accent)' : 'color-mix(in srgb, var(--c-a) 30%, var(--bg))',
                          border: isProbed ? '1px solid var(--text)' : '1px solid var(--border)',
                        }}
                      />
                    );
                  });
                })()}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                Avg bucket size ≈ {(600 / st.nlist).toFixed(1)} vecs · {st.nlist} centroid checks + {st.nprobe} buckets
              </div>
            </div>
            <QueryRecallStrip hits={r1.perQueryHits} />
          </div>
        </div>
      )}

      {briefId === 'lsh' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Hash bits"
              value={st.bits}
              onChange={(v) => updateLsh({ bits: v })}
              options={LAB_LSH_BITS.map((b) => ({ value: b, label: `${b} bits` }))}
            />
            <Segmented
              label="Hash tables"
              value={st.tables}
              onChange={(v) => updateLsh({ tables: v })}
              options={LAB_LSH_TABLES.map((t) => ({ value: t, label: `${t} tables` }))}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
            <div style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span>Recall@5 (target ≥ 85%)</span>
                <strong style={{ fontFamily: 'var(--mono)', color: r2.recall5 >= 0.85 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r2.recall5)}</strong>
              </div>
              <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, tweenedLsh[0] * 100)}%`, background: r2.recall5 >= 0.85 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: '85%', width: 2, background: 'var(--text)' }} />
              </div>
            </div>
            <div style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span>Candidate scan (ceiling ≤ 35% · {r2.speedup.toFixed(1)}×)</span>
                <strong style={{ fontFamily: 'var(--mono)', color: r2.scanFrac <= 0.35 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r2.scanFrac)}</strong>
              </div>
              <div style={{ position: 'relative', height: 9, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, tweenedLsh[1] * 100)}%`, background: r2.scanFrac <= 0.35 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: '35%', width: 2, background: 'var(--text)' }} />
              </div>
            </div>
          </div>

          {/* Live LSH collision S-curve + 60-query Recall@5 strip */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12, padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
                SimHash collision probability P_hit(θ) for b={st.bits}, L={st.tables}
              </div>
              <svg viewBox="0 0 280 132" style={{ width: '100%', maxWidth: 320, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', display: 'block' }}>
                <line x1={34} y1={112} x2={266} y2={112} stroke="var(--border)" strokeWidth={1} />
                <line x1={34} y1={16} x2={34} y2={112} stroke="var(--border)" strokeWidth={1} />
                {/* 25° near-neighbour marker & 75° far-noise marker */}
                <line x1={34 + (25 / 120) * 232} y1={16} x2={34 + (25 / 120) * 232} y2={112} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1} />
                <line x1={34 + (75 / 120) * 232} y1={16} x2={34 + (75 / 120) * 232} y2={112} stroke="var(--c-bad)" strokeDasharray="3 3" strokeWidth={1} />
                <path d={lshCurvePath} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
                <text x={34 + (25 / 120) * 232} y={125} textAnchor="middle" fontSize={8.5} fill="var(--c-good)">
                  25° ({(100 * lshCollisionProb(25, st.bits, st.tables)).toFixed(0)}%)
                </text>
                <text x={34 + (75 / 120) * 232} y={125} textAnchor="middle" fontSize={8.5} fill="var(--c-bad)">
                  75° ({(100 * lshCollisionProb(75, st.bits, st.tables)).toFixed(0)}%)
                </text>
                <text x={6} y={22} fontSize={8.5} fill="var(--text-muted)">1.0</text>
                <text x={6} y={112} fontSize={8.5} fill="var(--text-muted)">0.0</text>
              </svg>
            </div>
            <QueryRecallStrip hits={r2.perQueryHits} />
          </div>
        </div>
      )}

      {briefId === 'ram' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
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
                onClick={() => updateFmt(s.id)}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 12 }}>
              <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                1 vector (1,536D) = <strong>{r3.bytesPerVec.toLocaleString()} B</strong> ({r3.ratio.toFixed(0)}× smaller than FP32)
              </span>
              <span>× 10,000,000 =</span>
              <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: `1px solid ${r3.gib <= 16 ? 'var(--c-good)' : 'var(--c-bad)'}`, fontWeight: 600 }}>
                {r3.gib.toFixed(2)} GiB
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>Node RAM (ceiling = 16.0 GiB)</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: r3.gib <= 16 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {tweenedRam.toFixed(2)} / 16.0 GiB
                  </strong>
                </div>
                <div style={{ position: 'relative', height: 12, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.min(100, (tweenedRam / 60) * 100)}%`, background: r3.gib <= 16 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(16 / 60) * 100}%`, width: 2, background: 'var(--text)' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>Typical recall retention (floor ≥ 95%)</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: r3.typicalRecall >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {pct1(r3.typicalRecall)}
                  </strong>
                </div>
                <div style={{ position: 'relative', height: 12, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.min(100, tweenedRec3 * 100)}%`, background: r3.typicalRecall >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: '95%', width: 2, background: 'var(--text)' }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <LiveCode note="FAISS / NumPy ANN" code={realCode} />
    </LabShell>
  );
}
