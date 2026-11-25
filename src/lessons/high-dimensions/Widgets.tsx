// authored-by: Claude Opus 5.5
import { useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { Segmented, Slider } from '../../components/controls';
import { Plot, usePlot } from '../../components/plot';
import {
  CONTRAST_DIMS,
  PACK_BRIEFS,
  ballInCube,
  checkPacking,
  cornerDistance,
  cosineDensity,
  distancesFromQuery,
  dotInt,
  log10BallInCube,
  maxPacking,
  meanContrast,
  probAbsCosAtLeast,
  sampleCosines,
  shellFraction,
} from './hd';

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const superscript = (n: number) => String(n).replace(/-/g, '⁻').replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)]);
/** Format 10^l without ever computing 10^l (which underflows to 0 past about 10⁻³⁰⁸). */
const sciLog10 = (l: number) => {
  const e = Math.floor(l);
  return `${(10 ** (l - e)).toFixed(1)} × 10${superscript(e)}`;
};
const sci = (v: number) => {
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(v));
  if (e >= -3) return v.toPrecision(3);
  const sup = String(e).replace(/-/g, '⁻').replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)]);
  return `${(v / 10 ** e).toFixed(1)} × 10${sup}`;
};

function Hist({ values, lo, hi, bins, color = 'var(--c-a)', density = true }: { values: number[]; lo: number; hi: number; bins: number; color?: string; density?: boolean }) {
  const { x, y } = usePlot();
  const w = (hi - lo) / bins;
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) {
    const k = Math.floor((v - lo) / w);
    if (k >= 0 && k < bins) counts[k]++;
  }
  return (
    <g>
      {counts.map((c, k) => {
        const h = density ? c / values.length / w : c;
        if (!h) return null;
        return <rect key={k} x={x(lo + k * w) + 0.5} y={y(h)} width={Math.max(0.5, x(lo + (k + 1) * w) - x(lo + k * w) - 1)} height={y(0) - y(h)} fill={color} opacity={0.7} />;
      })}
    </g>
  );
}

function Curve({ f, lo, hi, color = 'var(--c-b)' }: { f: (t: number) => number; lo: number; hi: number; color?: string }) {
  const { x, y } = usePlot();
  const ymax = Math.max(...y.domain);
  let d = '';
  for (let k = 0; k <= 300; k++) {
    const t = lo + ((hi - lo) * k) / 300;
    const v = Math.min(ymax * 1.5, f(t));
    d += `${k ? 'L' : 'M'}${x(t).toFixed(1)},${y(v).toFixed(1)}`;
  }
  return <path d={d} fill="none" stroke={color} strokeWidth={2.4} />;
}

// ------------------------------------------------------------------ 1. Peel the orange

const LOG_DIMS = { min: 0, max: 3 }; // 10^0 … 10^3

export function OrangePeel() {
  const [ld, setLd] = useState(Math.log10(3));
  const [eps, setEps] = useState(0.01);
  const d = Math.max(1, Math.round(10 ** ld));
  const frac = shellFraction(d, eps);
  const curve = useMemo(() => Array.from({ length: 121 }, (_, k) => {
    const dd = 10 ** ((k / 120) * 3);
    return [k / 40, shellFraction(dd, eps)] as [number, number];
  }), [eps]);

  return (
    <Widget
      title="Peel the orange: where a ball’s volume lives"
      wide
      caption={
        <>
          A ball in <em>d</em> dimensions. What fraction of its volume lies in the outer peel, the shell within ε of the surface? Shrinking the radius by a factor (1 − ε) shrinks the volume by (1 − ε)<sup>d</sup>, so the peel holds 1 − (1 − ε)<sup>d</sup>. Start with a 1% peel in 3D, then drag the dimension up.
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(240px, 100%), 1fr))', gap: 12 }}>
        <Slider label="dimensions d" value={ld} min={LOG_DIMS.min} max={LOG_DIMS.max} step={0.01} onChange={setLd} format={() => String(d)} color="accent" />
        <Slider label="peel thickness ε" value={eps} min={0.005} max={0.2} step={0.005} onChange={setEps} format={(v) => pct(v, 1)} color="c-b" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 8 }}>
        <Plot width={440} height={240} x={[-0.12, 3.12]} y={[0, 1.04]} pad={34} xLabel="dimensions (log scale)" tickLabels={false} ariaLabel="Fraction of volume in the outer shell against dimension">
          {({ x, y }) => (
            <g>
              {[0, 1, 2, 3].map((k) => (
                <text key={k} x={x(k)} y={y(0) + 14} textAnchor="middle" fontSize={11} fill="var(--text-faint)">
                  {10 ** k}
                </text>
              ))}
              {[0.5, 1].map((v) => (
                <text key={v} x={x(-0.12) + 30} y={y(v) + 4} textAnchor="end" fontSize={11} fill="var(--text-faint)">
                  {pct(v, 0)}
                </text>
              ))}
              <polyline points={curve.map(([a, b]) => `${x(a)},${y(b)}`).join(' ')} fill="none" stroke="var(--c-b)" strokeWidth={2.5} />
              <circle cx={x(Math.log10(d))} cy={y(frac)} r={6} fill="var(--accent)" stroke="var(--bg)" strokeWidth={1.5} />
            </g>
          )}
        </Plot>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">volume in the outer {pct(eps, 1)} peel </span>
            {pct(frac, frac > 0.999 ? 3 : 1)}
            {'\n'}
            <span className="k">ball’s share of the cube around it </span>
            {d <= 6 ? pct(ballInCube(d), 1) : log10BallInCube(d) > -3 ? pct(ballInCube(d), 2) : sciLog10(log10BallInCube(d))}
            {'\n'}
            <span className="k">centre-to-corner of that cube </span>
            √{d} = {cornerDistance(d).toFixed(2)} radii
          </div>
          <div className="live-note">
            {d <= 3 ? (
              <>In {d}D the peel is a thin rind: {pct(frac)} of the volume. This is the geometry your intuition was trained on.</>
            ) : frac < 0.9 ? (
              <>In {d} dimensions the {pct(eps, 1)} peel already holds {pct(frac)} of the volume. Every extra dimension multiplies the inner part by another (1 − ε).</>
            ) : (
              <>
                In {d} dimensions, {pct(frac, 2)} of the ball is within {pct(eps, 1)} of its surface: the “inside” is almost empty. Meanwhile the ball fills only {log10BallInCube(d) > -3 ? pct(ballInCube(d), 2) : sciLog10(log10BallInCube(d))} of the cube around it: nearly all of the cube is in its corners, √{d} ≈ {cornerDistance(d).toFixed(1)} radii out.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Random arrows are perpendicular

const COS_DIMS = [3, 10, 100, 768] as const;

export function RandomCosines() {
  const [d, setD] = useState<(typeof COS_DIMS)[number]>(3);
  const [seed, setSeed] = useState(1);
  const cs = useMemo(() => sampleCosines(d, 2000, seed * 1000 + d), [d, seed]);
  const m = cs.reduce((a, b) => a + b, 0) / cs.length;
  const sdSample = Math.sqrt(cs.reduce((a, b) => a + (b - m) ** 2, 0) / cs.length);
  const peak = cosineDensity(d, 0);
  const yMax = d === 3 ? 0.9 : peak * 1.2;

  return (
    <Widget
      title="Two random arrows: almost always nearly perpendicular"
      wide
      caption={
        <>
          2,000 pairs of random arrows (each entry drawn from a bell curve) in <em>d</em> dimensions. The bars are the cosines of the angles between them; the orange curve is the exact distribution. In 3D every cosine is equally likely. Step up to 768 dimensions, the size of BERT’s and GPT-2’s embeddings.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'end', marginBottom: 8 }}>
        <Segmented value={d} onChange={setD} options={COS_DIMS.map((v) => ({ value: v, label: `d = ${v}` }))} />
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>Draw 2,000 new pairs</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <Plot width={440} height={240} x={[-1, 1]} y={[0, yMax]} xLabel="cosine" ariaLabel="Histogram of cosines between random vectors">
          <Hist values={cs} lo={-1} hi={1} bins={80} />
          <Curve f={(t) => cosineDensity(d, t)} lo={-0.999} hi={0.999} />
        </Plot>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">spread of the cosines (these 2,000) </span>
            {sdSample.toFixed(3)}
            {'\n'}
            <span className="k">exact: 1/√d </span>
            {(1 / Math.sqrt(d)).toFixed(3)}
            {'\n'}
            <span className="k">chance |cos| ≥ 0.1 </span>
            {pct(probAbsCosAtLeast(d, 0.1), 1)}
            {'\n'}
            <span className="k">chance |cos| ≥ 0.3 </span>
            {probAbsCosAtLeast(d, 0.3) < 1e-3 ? sci(probAbsCosAtLeast(d, 0.3)) : pct(probAbsCosAtLeast(d, 0.3), 1)}
          </div>
          <div className="live-note">
            {d === 3 ? (
              <>In 3D the cosine is spread evenly from −1 to 1 (a fact Archimedes would have recognised): random arrows are as likely to be nearly parallel as nearly perpendicular.</>
            ) : d < 768 ? (
              <>In {d} dimensions the cosines bunch around 0 with spread 1/√{d} ≈ {(1 / Math.sqrt(d)).toFixed(2)}. The dot product is a sum of {d} independent products, so it grows like √{d}, while the lengths grow like √{d} each: their ratio shrinks like 1/√{d}.</>
            ) : (
              <>
                In 768 dimensions, two random arrows have |cos| ≥ 0.3 with probability {sci(probAbsCosAtLeast(768, 0.3))}. So when two embeddings have cosine 0.3, that is not a coincidence: it is a strong signal. “Unrelated” in high dimensions means “nearly perpendicular”, and there’s room for an enormous number of nearly perpendicular directions.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Nearest and farthest

export function DistanceConcentration() {
  const [d, setD] = useState<(typeof CONTRAST_DIMS)[number]>(2);
  const [seed, setSeed] = useState(1);
  const [table, setTable] = useState<null | { d: number; mean: number; min: number; max: number }[]>(null);
  const run = useMemo(() => distancesFromQuery(d, 1000, seed * 101 + d), [d, seed]);
  const lo = 0;
  const hi = run.farthest * 1.05;

  const computeTable = () => {
    setTable(CONTRAST_DIMS.map((dd) => ({ d: dd, ...meanContrast(dd) })));
  };

  return (
    <Widget
      title="Nearest and farthest neighbours look alike"
      wide
      caption={
        <>
          1,000 random points in a unit cube and one random query point. The bars show the distance from the query to every point; green marks the nearest, red the farthest. In 2D the nearest neighbour is far closer than the rest. Step up the dimension and watch the whole crowd move to the same distance.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'end', marginBottom: 8 }}>
        <Segmented value={d} onChange={setD} options={CONTRAST_DIMS.map((v) => ({ value: v, label: `d = ${v}` }))} />
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>New points</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <Plot width={440} height={220} x={[lo, hi]} y={[0, 1]} xLabel="distance from the query" tickLabels ariaLabel="Histogram of distances from a query point" grid={false}>
          {({ x, y }) => {
            const bins = 60;
            const w = (hi - lo) / bins;
            const counts = new Array<number>(bins).fill(0);
            for (const v of run.dists) counts[Math.min(bins - 1, Math.floor((v - lo) / w))]++;
            const mx = Math.max(...counts);
            return (
              <g>
                {counts.map((c, k) =>
                  c ? <rect key={k} x={x(lo + k * w) + 0.5} y={y((c / mx) * 0.9)} width={Math.max(0.5, x(lo + (k + 1) * w) - x(lo + k * w) - 1)} height={y(0) - y((c / mx) * 0.9)} fill="var(--c-a)" opacity={0.7} /> : null,
                )}
                <line x1={x(run.nearest)} x2={x(run.nearest)} y1={y(0)} y2={y(1)} stroke="var(--c-good)" strokeWidth={2.5} />
                <line x1={x(run.farthest)} x2={x(run.farthest)} y1={y(0)} y2={y(1)} stroke="var(--c-bad)" strokeWidth={2.5} />
              </g>
            );
          }}
        </Plot>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">nearest </span>
            {run.nearest.toFixed(3)}
            <span className="k"> · farthest </span>
            {run.farthest.toFixed(3)}
            {'\n'}
            <span className="k">farthest is </span>
            {run.contrast > 10 ? `${(run.farthest / run.nearest).toFixed(0)}× as far` : `${pct(run.contrast, 0)} farther`}
          </div>
          <div className="live-note">
            {d <= 3 ? (
              <>In {d}D, the nearest of 1,000 points is very close and the farthest is {(run.farthest / run.nearest).toFixed(0)} times as far away. “Nearest neighbour” means something sharp.</>
            ) : d < 1000 ? (
              <>In {d} dimensions the distances bunch up: the farthest point is only {pct(run.contrast, 0)} farther than the nearest. A squared distance is a sum of {d} independent pieces, and sums concentrate: the spread grows like √{d} while the total grows like {d}.</>
            ) : (
              <>In 1,000 dimensions the farthest of 1,000 random points is only about {pct(run.contrast, 0)} farther than the nearest. With uniformly random data, “nearest neighbour” has almost stopped meaning anything.</>
            )}
          </div>
          <div className="btn-row">
            <button type="button" className="btn" onClick={computeTable} disabled={!!table}>Average over 20 seeds</button>
          </div>
          {table && (
            <div style={{ overflowX: 'auto' }}>
              <table className="num-table">
                <thead>
                  <tr>
                    <th>dimensions</th>
                    <th>farthest ÷ nearest (mean)</th>
                    <th>range over seeds</th>
                  </tr>
                </thead>
                <tbody>
                  {table.map((r) => (
                    <tr key={r.d}>
                      <td>{r.d}</td>
                      <td>{(1 + r.mean).toFixed(r.mean > 10 ? 0 : 2)}</td>
                      <td>
                        {(1 + r.min).toFixed(r.mean > 10 ? 0 : 2)} – {(1 + r.max).toFixed(r.mean > 10 ? 0 : 2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: superposition

const PACK_KEY = 'nl-hd-pack-lab-v1';
type PackSave = Record<string, { rows: number[][]; solved: boolean; declared: boolean; optimal: boolean }>;

function loadPack(): PackSave {
  const d: PackSave = Object.fromEntries(PACK_BRIEFS.map((b) => [b.id, { rows: [new Array(b.d).fill(1)], solved: false, declared: false, optimal: false }]));
  try {
    const raw = localStorage.getItem(PACK_KEY);
    if (!raw) return d;
    const parsed = JSON.parse(raw) as PackSave;
    for (const b of PACK_BRIEFS) {
      const v = parsed[b.id];
      if (!v || !Array.isArray(v.rows)) continue;
      const ok = v.rows.length >= 1 && v.rows.length <= 16 && v.rows.every((r) => Array.isArray(r) && r.length === b.d && r.every((x) => x === 1 || x === -1));
      if (ok) d[b.id] = { rows: v.rows, solved: !!v.solved, declared: !!v.declared, optimal: !!v.optimal };
    }
  } catch {
    // ignore storage errors
  }
  return d;
}


/**
 * Brief 1's arrows drawn inside a cube you can turn by dragging: ±1 vectors in
 * 3D are the cube's corners, so the arrows are diagonals. Clashing arrows are red.
 */
function Cube3D({ rows, bad }: { rows: number[][]; bad: Set<string> }) {
  const [ang, setAng] = useState({ yaw: 0.6, pitch: -0.45 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const S = 220;
  const sc = 58;
  const proj = ([x, y, z]: number[]): [number, number, number] => {
    const cy = Math.cos(ang.yaw), sy = Math.sin(ang.yaw), cp = Math.cos(ang.pitch), sp = Math.sin(ang.pitch);
    const x1 = cy * x + sy * z;
    const z1 = -sy * x + cy * z;
    const y1 = cp * y - sp * z1;
    const z2 = sp * y + cp * z1;
    return [S / 2 + sc * x1, S / 2 - sc * y1, z2];
  };
  const corners = [-1, 1].flatMap((a) => [-1, 1].flatMap((b) => [-1, 1].map((c) => [a, b, c])));
  const edges: [number[], number[]][] = [];
  for (const a of corners) for (const b of corners) if (a.reduce((t, v, i) => t + (v !== b[i] ? 1 : 0), 0) === 1 && a.join() < b.join()) edges.push([a, b]);
  const clash = (i: number) => [...bad].some((k) => k.split(',').map(Number).includes(i));
  const o = proj([0, 0, 0]);
  return (
    <div>
      <svg
        viewBox={`0 0 ${S} ${S}`}
        width={S}
        height={S}
        style={{ touchAction: 'none', cursor: 'grab', display: 'block' }}
        role="img"
        aria-label="The arrows inside a cube; drag to turn it"
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY };
          (e.target as Element).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          const dx = e.clientX - drag.current.x;
          const dy = e.clientY - drag.current.y;
          drag.current = { x: e.clientX, y: e.clientY };
          setAng((a) => ({ yaw: a.yaw + dx * 0.012, pitch: Math.max(-1.4, Math.min(1.4, a.pitch + dy * 0.012)) }));
        }}
        onPointerUp={() => (drag.current = null)}
      >
        {edges.map(([a, b], k) => {
          const p = proj(a), q = proj(b);
          return <line key={k} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} stroke="var(--border-strong)" strokeWidth={1} />;
        })}
        {rows.map((r, i) => {
          const p = proj(r);
          const col = clash(i) ? 'var(--c-bad)' : 'var(--accent)';
          return (
            <g key={i}>
              <line x1={o[0]} y1={o[1]} x2={p[0]} y2={p[1]} stroke={col} strokeWidth={2.5} />
              <circle cx={p[0]} cy={p[1]} r={4} fill={col} />
              <text x={p[0] + 6} y={p[1] - 6} fontSize={12} fontWeight={700} fill={col}>{i + 1}</text>
            </g>
          );
        })}
      </svg>
      <div className="muted" style={{ fontSize: 12, maxWidth: 260 }}>Drag to turn the cube. Every ±1 arrow in 3D ends at a corner, so it runs along a diagonal of the cube.</div>
    </div>
  );
}

export function PackLab() {
  const [save, setSave] = useState<PackSave>(() => loadPack());
  const [bi, setBi] = useState(0);
  const [showOpt, setShowOpt] = useState(false);
  const brief = PACK_BRIEFS[bi];
  const entry = save[brief.id];
  const rows = entry.rows;
  const check = checkPacking(brief, rows);
  const badSet = new Set(check.bad.map(([i, j]) => `${i},${j}`));
  const max = useMemo(() => (entry.solved || entry.declared ? maxPacking(brief.d, brief.maxDot).length : null), [brief, entry.solved, entry.declared]);

  const persist = (next: PackSave) => {
    setSave(next);
    try {
      localStorage.setItem(PACK_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };
  const setRows = (nr: number[][]) => {
    const c = checkPacking(brief, nr);
    let optimal = entry.optimal;
    if (c.pass && c.bad.length === 0) optimal = optimal || nr.length >= maxPacking(brief.d, brief.maxDot).length;
    persist({ ...save, [brief.id]: { ...entry, rows: nr, solved: entry.solved || c.pass, optimal } });
  };
  const log = useMachineLog();
  const flip = (i: number, j: number) => {
    const nr = rows.map((r, k) => (k === i ? r.map((x, m) => (m === j ? -x : x)) : r));
    setRows(nr);
    const after = checkPacking(brief, nr);
    const withI = after.bad.filter(([a, b]) => a === i || b === i).map(([a, b]) => (a === i ? b : a) + 1);
    log.push(
      `row ${i + 1}, entry ${j + 1} flipped: ${withI.length ? `now clashes with row${withI.length > 1 ? 's' : ''} ${withI.join(', ')}` : `clashes with nothing`} · ${after.bad.length} clashing pair${after.bad.length === 1 ? '' : 's'} in all`,
      withI.length ? 'bad' : 'good',
    );
  };
  const addRow = () => rows.length < 16 && setRows([...rows, new Array(brief.d).fill(1)]);
  const removeRow = (i: number) => rows.length > 1 && setRows(rows.filter((_, k) => k !== i));
  const declare = () => persist({ ...save, [brief.id]: { ...entry, declared: true } });

  const cell = 30;

  return (
    <Widget
      title="Lab: Superposition — more concepts than dimensions"
      wide
      caption={
        <>
          Each row is a “concept vector” whose entries are +1 or −1 (click a cell to flip it). Every pair of rows must be nearly perpendicular: its dot product may be at most the brief’s limit (the cosine is the dot product divided by d). Pairs that break the rule light up red in the grid of dot products. Your arrows are saved in this browser.
        </>
      }
    >
      <Segmented
        value={bi}
        onChange={(i) => {
          setBi(i);
          setShowOpt(false);
        }}
        options={PACK_BRIEFS.map((b, i) => ({ value: i, label: `${save[b.id].solved || (b.impossible && save[b.id].declared) ? '✓ ' : ''}${b.title.split(' · ')[0]}` }))}
      />
      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
        <div style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 6 }}>
          d = {brief.d} · every |dot product| ≤ {brief.maxDot} (|cos| ≤ {(brief.maxDot / brief.d).toFixed(3).replace(/\.?0+$/, '') || '0'}) · goal: {brief.goal} rows
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 4 }}>
          {rows.map((r, i) => (
            <div key={i} style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
              <span className="mono" style={{ width: 26, fontSize: 12, color: 'var(--text-faint)' }}>{i + 1}</span>
              {r.map((x, j) => (
                <button
                  key={j}
                  type="button"
                  onClick={() => flip(i, j)}
                  aria-label={`row ${i + 1}, entry ${j + 1}: ${x > 0 ? 'plus' : 'minus'} one`}
                  style={{
                    width: cell,
                    height: cell,
                    borderRadius: 6,
                    border: '1px solid var(--border-strong)',
                    background: x > 0 ? 'color-mix(in srgb, var(--c-a) 22%, var(--bg-elev))' : 'color-mix(in srgb, var(--c-b) 22%, var(--bg-elev))',
                    color: x > 0 ? 'var(--c-a)' : 'var(--c-b)',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    cursor: 'pointer',
                    padding: 0,
                  }}
                >
                  {x > 0 ? '+' : '−'}
                </button>
              ))}
              <button type="button" className="btn small" onClick={() => removeRow(i)} disabled={rows.length <= 1} aria-label={`remove row ${i + 1}`} style={{ marginLeft: 6 }}>
                ✕
              </button>
            </div>
          ))}
          <div className="btn-row" style={{ marginTop: 6 }}>
            <button type="button" className="btn" onClick={addRow} disabled={rows.length >= 16}>Add a row</button>
            {brief.impossible && (
              <button type="button" className="btn" onClick={declare} disabled={entry.declared}>Declare it impossible</button>
            )}
          </div>
        </div>
        <div style={{ display: 'grid', gap: 10 }}>
          {brief.d === 3 && <Cube3D rows={rows} bad={badSet} />}
          <div style={{ overflowX: 'auto' }}>
            <table className="num-table">
              <tbody>
                {rows.map((a, i) => (
                  <tr key={i}>
                    {rows.map((b, j) => {
                      const dot = dotInt(a, b);
                      const bad = i !== j && badSet.has(`${Math.min(i, j)},${Math.max(i, j)}`);
                      return (
                        <td
                          key={j}
                          style={{
                            background: i === j ? 'var(--bg-sunken)' : bad ? 'color-mix(in srgb, var(--c-bad) 22%, transparent)' : 'color-mix(in srgb, var(--c-good) 12%, transparent)',
                            color: i === j ? 'var(--text-faint)' : bad ? 'var(--c-bad)' : 'var(--text)',
                            padding: '2px 5px',
                            fontSize: 12,
                          }}
                        >
                          {i === j ? '·' : dot}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            {check.pass ? (
              <span className="status good">Brief met: {rows.length} vectors in {brief.d} dimensions</span>
            ) : brief.impossible && entry.declared ? (
              <span className="status good">Correct: it’s impossible</span>
            ) : (
              <span className="status neutral">
                {rows.length} row{rows.length === 1 ? '' : 's'}, {check.bad.length} clashing pair{check.bad.length === 1 ? '' : 's'}
              </span>
            )}
          </div>
          <div className="live-note">
            {brief.impossible && entry.declared ? (
              <>
                Right. Take any three ±1 vectors and flip the sign of the same coordinates in all of them (that changes no dot product) until the first is all +1s. For the second to be perpendicular to it, it needs exactly 3 minus signs; the third needs 3 too, overlapping the second’s in exactly 1.5 places to be perpendicular to both — and you can’t have half a place. Three mutually perpendicular ±1 vectors need d to be a multiple of 4. The exhaustive search agrees: the maximum in 6D is {max ?? 2}.
              </>
            ) : check.pass ? (
              entry.optimal ? (
                <>That’s the most possible: an exhaustive search over every set of ±1 vectors finds no larger arrangement than {max}.</>
              ) : (
                <>Done. Optional: how many can you fit? An exhaustive search knows the maximum{max ? ` (reveal below)` : ''}.</>
              )
            ) : check.bad.length ? (
              <>Some pairs overlap too much: their dot product counts (agreements − disagreements), and it must stay within ±{brief.maxDot}. Flip entries in one of the red pair’s rows to balance agreements and disagreements.</>
            ) : (
              <>No clashes so far. Add a row and find a pattern that agrees and disagrees with every existing row in roughly equal measure.</>
            )}
          </div>
          {(entry.solved || entry.declared) && max !== null && !brief.impossible && (
            <div>
              <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>{showOpt ? 'Hide' : 'Show'} the exhaustive maximum</button>
              {showOpt && (
                <div className="live-note" style={{ marginTop: 8 }}>
                  Searching every set of ±1 vectors in {brief.d} dimensions (one per line through the origin, {2 ** (brief.d - 1)} candidates), the largest arrangement that keeps every |dot| ≤ {brief.maxDot} has <strong>{max}</strong> vectors{max > brief.d ? `: ${(max / brief.d).toFixed(max % brief.d ? 2 : 0)}× as many concepts as dimensions` : ''}.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <MachineLog lines={log.lines} onClear={log.clear} empty="Flip an entry: which rows it now clashes with is logged here." />
    </Widget>
  );
}
