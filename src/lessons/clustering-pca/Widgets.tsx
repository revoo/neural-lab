// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  B3_MS,
  BAD_INIT_BLOBS4,
  LAB_BRIEFS,
  elbowSweep,
  evalBrief1,
  evalBrief2,
  evalBrief3,
  getBlobs4,
  getInitRace,
  getRingsData,
  labExhaustive,
  pca2DAtAngle,
  runKMeans,
  runKMeansFrom,
  type InitMode,
  type UnsupBrief,
} from './unsupervised';

const CLUSTER_COLORS = [
  'var(--c-a)',
  'var(--c-b)',
  'var(--c-c)',
  'var(--accent)',
  'var(--c-good)',
  'var(--c-bad)',
  'var(--text-muted)',
];

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

// ------------------------------------------------------------------ 1. k-Means Step-by-Step & Init Trap

export function KMeansStepWidget() {
  const { pts } = useMemo(() => getBlobs4(), []);
  const race = useMemo(() => getInitRace(), []);
  const [preset, setPreset] = useState<'bad' | 'pp'>('bad');
  const [seed, setSeed] = useState<number>(17);
  const [step, setStep] = useState<number>(0);

  const trace = useMemo(() => {
    if (preset === 'bad') return runKMeansFrom(pts, BAD_INIT_BLOBS4);
    return runKMeans(pts, 4, 'kmeans++', seed).trace;
  }, [pts, preset, seed]);

  const maxStep = trace.length - 1;
  const clampedStep = Math.min(step, maxStep);
  const snap = trace[clampedStep];
  const isConverged = clampedStep === maxStep;
  const isOptimal = snap.inertia <= race.globalInertia * 1.05;

  const switchPreset = (p: 'bad' | 'pp') => {
    setPreset(p);
    setStep(0);
  };

  return (
    <Widget
      title="Lloyd’s algorithm step by step: why starting positions make or break k-Means"
      wide
      caption={
        <>
          Step through <strong>Assign → Update</strong> on four clean blobs. With the unlucky start, two centroids begin inside the top-left blob and one sits between the two bottom blobs — it converges in 3 steps to a terrible local minimum (inertia <strong>415.6</strong>). Switch to <strong>k-Means++</strong> to spread the seeds out and hit the global minimum (<strong>73.2</strong>).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 10 }}>
        <Segmented
          label="Initialisation"
          value={preset}
          onChange={switchPreset}
          options={[
            { value: 'bad', label: 'Unlucky random start' },
            { value: 'pp', label: 'k-Means++ seeding' },
          ]}
        />
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            type="button"
            className="btn"
            disabled={clampedStep >= maxStep}
            onClick={() => setStep((s) => Math.min(maxStep, s + 1))}
          >
            Step ({clampedStep}/{maxStep})
          </button>
          <button
            type="button"
            className="btn"
            disabled={clampedStep >= maxStep}
            onClick={() => setStep(maxStep)}
          >
            Run to convergence
          </button>
          <button type="button" className="btn" onClick={() => setStep(0)}>
            Reset
          </button>
          {preset === 'pp' && (
            <button
              type="button"
              className="btn"
              onClick={() => {
                setSeed((s) => s + 13);
                setStep(0);
              }}
            >
              New seed
            </button>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(290px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        <Plot x={[-4.5, 4.5]} y={[-4.2, 4.2]} height={290} xLabel="Feature 1" yLabel="Feature 2">
          {({ x: sx, y: sy }) => (
            <>
              {/* Spokes from points to their assigned centroid */}
              {pts.map((p, i) => {
                const c = snap.assign[i];
                const mu = snap.centroids[c];
                return (
                  <line
                    key={`sp-${i}`}
                    x1={sx(p[0])}
                    y1={sy(p[1])}
                    x2={sx(mu[0])}
                    y2={sy(mu[1])}
                    stroke={CLUSTER_COLORS[c]}
                    strokeWidth={0.8}
                    opacity={0.25}
                  />
                );
              })}
              {/* Points */}
              {pts.map((p, i) => (
                <circle
                  key={`pt-${i}`}
                  cx={sx(p[0])}
                  cy={sy(p[1])}
                  r={4}
                  fill={CLUSTER_COLORS[snap.assign[i]]}
                  opacity={0.88}
                />
              ))}
              {/* Centroid movement trails up to current step */}
              {clampedStep > 0 &&
                snap.centroids.map((_, c) => {
                  const dPath = trace
                    .slice(0, clampedStep + 1)
                    .map((t, idx) => `${idx === 0 ? 'M' : 'L'} ${sx(t.centroids[c][0]).toFixed(1)} ${sy(t.centroids[c][1]).toFixed(1)}`)
                    .join(' ');
                  return (
                    <path
                      key={`tr-${c}`}
                      d={dPath}
                      fill="none"
                      stroke={CLUSTER_COLORS[c]}
                      strokeWidth={2}
                      strokeDasharray="4 3"
                    />
                  );
                })}
              {/* Current centroids (crosshairs) */}
              {snap.centroids.map((mu, c) => (
                <g key={`mu-${c}`}>
                  <circle
                    cx={sx(mu[0])}
                    cy={sy(mu[1])}
                    r={9}
                    fill="var(--bg)"
                    stroke={CLUSTER_COLORS[c]}
                    strokeWidth={2.5}
                  />
                  <line x1={sx(mu[0]) - 5} y1={sy(mu[1])} x2={sx(mu[0]) + 5} y2={sy(mu[1])} stroke={CLUSTER_COLORS[c]} strokeWidth={2} />
                  <line x1={sx(mu[0])} y1={sy(mu[1]) - 5} x2={sx(mu[0])} y2={sy(mu[1]) + 5} stroke={CLUSTER_COLORS[c]} strokeWidth={2} />
                </g>
              ))}
            </>
          )}
        </Plot>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="readouts" style={{ margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Iteration</div>
              <div className="readout-value">{clampedStep} / {maxStep}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Inertia (SSE)</div>
              <div className="readout-value" style={{ color: isOptimal ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {snap.inertia.toFixed(1)}
              </div>
            </div>
          </div>

          <div
            style={{
              padding: '10px 12px',
              borderRadius: 8,
              border: '1px solid var(--border)',
              background: 'var(--bg-sunken)',
              fontSize: 13.5,
              lineHeight: 1.5,
            }}
          >
            {isConverged ? (
              isOptimal ? (
                <span>
                  <strong style={{ color: 'var(--c-good)' }}>Global optimum reached!</strong> Each centroid owns one true blob, and inertia sits at the floor (<strong>{snap.inertia.toFixed(1)}</strong>).
                </span>
              ) : (
                <span>
                  <strong style={{ color: 'var(--c-bad)' }}>Trapped in a local minimum.</strong> No single point wants to switch centroids, so Lloyd’s algorithm halts forever with the bottom two blobs merged and the top-left blob split in half (<strong>5.7×</strong> worse inertia).
                </span>
              )
            ) : (
              <span>
                Step {clampedStep}: points are colour-coded by their nearest centroid. Click <strong>Step</strong> to move every crosshair to the mean of its coloured points and re-assign.
              </span>
            )}
          </div>

          <div
            style={{
              padding: '10px 12px',
              borderRadius: 8,
              border: '1px solid var(--border)',
              background: 'var(--bg-sunken)',
              fontSize: 13,
              lineHeight: 1.5,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Fair 40-seed race on these 4 blobs</div>
            <div>
              Random start: <strong>{race.randomOptimal}/{race.seeds}</strong> hit the global minimum (mean inertia <strong>{race.randomMeanInertia.toFixed(1)}</strong>)
            </div>
            <div>
              k-Means++: <strong style={{ color: 'var(--c-good)' }}>{race.ppOptimal}/{race.seeds}</strong> hit the global minimum (mean inertia <strong>{race.ppMeanInertia.toFixed(1)}</strong>)
            </div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Elbow, Silhouette & Concentric Rings

export function ElbowRingsWidget() {
  const [shape, setShape] = useState<'blobs' | 'rings'>('blobs');
  const [k, setK] = useState<number>(4);

  const blobsSweep = useMemo(() => elbowSweep(getBlobs4().pts), []);
  const ringsSweep = useMemo(() => elbowSweep(getRingsData().pts), []);
  const pts = shape === 'blobs' ? getBlobs4().pts : getRingsData().pts;
  const sweep = shape === 'blobs' ? blobsSweep : ringsSweep;
  const cur = sweep.find((r) => r.k === k) ?? sweep[0];

  const maxInertia = sweep[0].inertia;

  return (
    <Widget
      title="Choosing k — and watching k-Means break on non-round shapes"
      wide
      caption={
        <>
          Slide <strong>Clusters</strong> from 2 to 7. On <strong>Four blobs</strong>, inertia drops steeply until <code>k = 4</code> and then flattens (the elbow), while the <strong>Silhouette score</strong> peaks sharply at <strong>0.763</strong>. Now toggle to <strong>Concentric rings</strong>: k-Means can only draw straight Voronoi fences, so at <code>k = 2</code> it slices both rings straight down the middle (Silhouette <strong>0.326</strong>) and stays stuck in <strong>0.31–0.41</strong> across every <code>k</code>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 10 }}>
        <Segmented
          label="Dataset"
          value={shape}
          onChange={setShape}
          options={[
            { value: 'blobs', label: 'Four round blobs' },
            { value: 'rings', label: 'Concentric rings' },
          ]}
        />
        <div style={{ minWidth: 200, flex: '1 1 200px' }}>
          <Slider label="Clusters" value={k} min={2} max={7} step={1} onChange={setK} format={(v) => `k = ${v}`} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        <Plot
          x={shape === 'blobs' ? [-4.5, 4.5] : [-3.5, 3.5]}
          y={shape === 'blobs' ? [-4.2, 4.2] : [-3.5, 3.5]}
          height={270}
          xLabel="Feature 1"
          yLabel="Feature 2"
        >
          {({ x: sx, y: sy }) => (
            <>
              {pts.map((p, i) => (
                <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={3.8} fill={CLUSTER_COLORS[cur.assign[i] % CLUSTER_COLORS.length]} opacity={0.85} />
              ))}
              {cur.centroids.map((mu, c) => (
                <circle
                  key={`c-${c}`}
                  cx={sx(mu[0])}
                  cy={sy(mu[1])}
                  r={7.5}
                  fill="var(--bg)"
                  stroke={CLUSTER_COLORS[c % CLUSTER_COLORS.length]}
                  strokeWidth={2.5}
                />
              ))}
            </>
          )}
        </Plot>

        <div>
          <Plot x={[1.7, 7.3]} y={[0, 1.05]} height={210} xLabel="Number of clusters k" yLabel="Normalised inertia / Silhouette">
            {({ x: sx, y: sy }) => {
              const inertiaPath = sweep
                .map((r, i) => `${i === 0 ? 'M' : 'L'} ${sx(r.k).toFixed(1)} ${sy(r.inertia / maxInertia).toFixed(1)}`)
                .join(' ');
              const silPath = sweep
                .map((r, i) => `${i === 0 ? 'M' : 'L'} ${sx(r.k).toFixed(1)} ${sy(Math.max(0, r.silhouette)).toFixed(1)}`)
                .join(' ');
              return (
                <>
                  <line x1={sx(k)} y1={sy(0)} x2={sx(k)} y2={sy(1.05)} stroke="var(--text-muted)" strokeDasharray="3 3" strokeWidth={1.2} />
                  <path d={inertiaPath} fill="none" stroke="var(--c-b)" strokeWidth={2.2} />
                  <path d={silPath} fill="none" stroke="var(--c-a)" strokeWidth={2.2} />
                  {sweep.map((r) => (
                    <g key={r.k}>
                      <circle cx={sx(r.k)} cy={sy(r.inertia / maxInertia)} r={r.k === k ? 5.5 : 3.5} fill="var(--c-b)" />
                      <circle cx={sx(r.k)} cy={sy(Math.max(0, r.silhouette))} r={r.k === k ? 5.5 : 3.5} fill="var(--c-a)" />
                    </g>
                  ))}
                </>
              );
            }}
          </Plot>
          <div style={{ display: 'flex', gap: 14, fontSize: 12.5, marginTop: 4, flexWrap: 'wrap' }}>
            <span style={{ color: 'var(--c-b)' }}>● Inertia (normalised): <strong>{cur.inertia.toFixed(1)}</strong></span>
            <span style={{ color: 'var(--c-a)' }}>● Silhouette: <strong>{cur.silhouette.toFixed(3)}</strong></span>
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 6, lineHeight: 1.45 }}>
            {shape === 'blobs'
              ? cur.k === 4
                ? 'Sharp elbow + highest Silhouette (0.763): four tight, well-separated clusters.'
                : cur.k < 4
                  ? 'Too few centroids: distinct blobs are forced to share a cluster, inflating inertia.'
                  : 'Too many centroids: inertia always edges down, but splitting clean blobs cuts Silhouette.'
              : 'Every cluster boundary in k-Means is a perpendicular bisector. A centroid cannot wrap around an inner ring without grabbing the outer ring too.'}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. 2D PCA Rotation Explorer (Pythagoras Live)

export function PcaRotateWidget() {
  const [deg, setDeg] = useState<number>(80);
  const res = useMemo(() => pca2DAtAngle(deg), [deg]);

  const len = 6.2;
  const x1 = -res.ux * len;
  const y1 = -res.uy * len;
  const x2 = res.ux * len;
  const y2 = res.uy * len;
  const nearOpt = Math.abs(deg - res.optDeg) < 1.5;

  return (
    <Widget
      title="PCA is Pythagoras: maximising spread and minimising error are the same line"
      wide
      caption={
        <>
          Rotate the projection axis from <code>0°</code> to <code>180°</code>. Orange dashed drop-lines measure reconstruction error; blue dots along the axis measure projected spread. Watch the two numbers add up to the exact same total variance (<strong>{res.totalVar.toFixed(3)}</strong>) at every angle — and click <strong>Snap to PC1</strong> to lock onto the exact eigenvector (<strong>{res.optDeg.toFixed(1)}°</strong>, keeping <strong>{pct1(res.optShare)}</strong> of the variance).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 10 }}>
        <div style={{ flex: '1 1 240px' }}>
          <Slider
            label="Axis angle"
            value={deg}
            min={0}
            max={180}
            step={0.5}
            onChange={setDeg}
            format={(v) => `${v.toFixed(1)}°`}
          />
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button type="button" className="btn" onClick={() => setDeg(Number(res.optDeg.toFixed(1)))}>
            Snap to PC1 ({res.optDeg.toFixed(1)}°)
          </button>
          <button type="button" className="btn" onClick={() => setDeg(Number(((res.optDeg + 90) % 180).toFixed(1)))}>
            Snap to PC2 ({((res.optDeg + 90) % 180).toFixed(1)}°)
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 14, alignItems: 'center' }}>
        <Plot x={[-5.2, 5.2]} y={[-4.5, 4.5]} height={290} xLabel="Centred x₁" yLabel="Centred x₂">
          {({ x: sx, y: sy }) => (
            <>
              {/* Projection line */}
              <line x1={sx(x1)} y1={sy(y1)} x2={sx(x2)} y2={sy(y2)} stroke="var(--accent)" strokeWidth={2.2} />
              {/* Perpendicular drop lines (reconstruction error) */}
              {res.projections.map((p, i) => (
                <line
                  key={`drop-${i}`}
                  x1={sx(p.x)}
                  y1={sy(p.y)}
                  x2={sx(p.px)}
                  y2={sy(p.py)}
                  stroke="var(--c-b)"
                  strokeWidth={1.1}
                  strokeDasharray="2 2"
                  opacity={0.7}
                />
              ))}
              {/* Projected shadows along axis */}
              {res.projections.map((p, i) => (
                <circle key={`sh-${i}`} cx={sx(p.px)} cy={sy(p.py)} r={2.8} fill="var(--c-a)" opacity={0.85} />
              ))}
              {/* Original points */}
              {res.projections.map((p, i) => (
                <circle key={`pt-${i}`} cx={sx(p.x)} cy={sy(p.y)} r={3.6} fill="var(--text)" opacity={0.78} />
              ))}
            </>
          )}
        </Plot>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="readouts" style={{ margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Projected variance</div>
              <div className="readout-value" style={{ color: 'var(--c-a)' }}>
                {res.projVar.toFixed(3)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Reconstruction MSE</div>
              <div className="readout-value" style={{ color: 'var(--c-b)' }}>
                {res.reconMse.toFixed(3)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Sum (Total var)</div>
              <div className="readout-value">{res.totalVar.toFixed(3)}</div>
            </div>
          </div>

          {/* Stacked Pythagoras bar */}
          <div style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
            <div style={{ fontSize: 12.5, marginBottom: 6, display: 'flex', justifyContent: 'space-between' }}>
              <span>Variance kept: <strong style={{ color: 'var(--c-a)' }}>{pct1(res.share)}</strong></span>
              <span>Lost: <strong style={{ color: 'var(--c-b)' }}>{pct1(1 - res.share)}</strong></span>
            </div>
            <div style={{ height: 14, borderRadius: 7, overflow: 'hidden', display: 'flex', background: 'var(--border)' }}>
              <div style={{ width: `${(100 * res.share).toFixed(1)}%`, background: 'var(--c-a)', transition: 'width 60ms linear' }} />
              <div style={{ width: `${(100 * (1 - res.share)).toFixed(1)}%`, background: 'var(--c-b)', transition: 'width 60ms linear' }} />
            </div>
            <div style={{ fontSize: 13, marginTop: 8, lineHeight: 1.45, color: 'var(--text-muted)' }}>
              {nearOpt
                ? 'Right on Principal Component 1! Every extra unit of spread along the line is a unit subtracted from the orange error drops.'
                : 'Notice the total bar length never changes: for every point, distance² from origin = shadow² + error².'}
            </div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Compress & Segment

type LabState = {
  b1Scaled: boolean;
  b1M: number;
  b2Scaled: boolean;
  b2K: number;
  b2Init: InitMode;
  b3M: number;
  b3K: number;
  solved: Record<UnsupBrief['id'], boolean>;
};

const LAB_KEY = 'nl-unsup-lab-v1';

const DEFAULT_STATE: LabState = {
  b1Scaled: false,
  b1M: 1,
  b2Scaled: false,
  b2K: 4,
  b2Init: 'random',
  b3M: 8,
  b3K: 4,
  solved: { 'pca-scale': false, segment: false, denoise: false },
};

function loadLab(): LabState {
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return DEFAULT_STATE;
    const p = JSON.parse(raw);
    return { ...DEFAULT_STATE, ...p, solved: { ...DEFAULT_STATE.solved, ...(p.solved ?? {}) } };
  } catch {
    return DEFAULT_STATE;
  }
}

export function UnsupervisedLab() {
  const [briefId, setBriefId] = useState<UnsupBrief['id']>('pca-scale');
  const [st, setSt] = useState<LabState>(loadLab);
  const [showOpt, setShowOpt] = useState<boolean>(false);

  useEffect(() => {
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore storage errors */
    }
  }, [st]);

  const b1Res = useMemo(() => evalBrief1(st.b1Scaled, st.b1M), [st.b1Scaled, st.b1M]);
  const b2Res = useMemo(() => evalBrief2(st.b2Scaled, st.b2K, st.b2Init), [st.b2Scaled, st.b2K, st.b2Init]);
  const b3Res = useMemo(() => evalBrief3(st.b3M, st.b3K), [st.b3M, st.b3K]);
  const exhaust = useMemo(() => labExhaustive(briefId), [briefId]);

  const curPass = briefId === 'pca-scale' ? b1Res.pass : briefId === 'segment' ? b2Res.pass : b3Res.pass;

  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((prev) => ({ ...prev, solved: { ...prev.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const brief = LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Compress & Segment"
      wide
      caption={
        <>
          Three real unsupervised pipelines with exhaustive search checks. First find any passing pipeline for each brief; then try to hit the optional minimal-dimension or optimal-score goal. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {LAB_BRIEFS.map((b, idx) => (
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
            {st.solved[b.id] ? '✓ ' : ''}Brief {idx + 1}
          </button>
        ))}
      </div>

      <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{brief.title}</div>
      <p style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 0, marginBottom: 12, lineHeight: 1.5 }}>{brief.story}</p>

      {briefId === 'pca-scale' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end' }}>
            <Segmented
              label="Feature scaling"
              value={st.b1Scaled ? 'std' : 'raw'}
              onChange={(v) => setSt((p) => ({ ...p, b1Scaled: v === 'std' }))}
              options={[
                { value: 'raw', label: 'Raw units (mm + m)' },
                { value: 'std', label: 'Standardised (z-score)' },
              ]}
            />
            <div style={{ flex: '1 1 200px' }}>
              <Slider
                label="PCA components kept"
                value={st.b1M}
                min={1}
                max={6}
                step={1}
                onChange={(v) => setSt((p) => ({ ...p, b1M: v }))}
                format={(v) => `m = ${v}`}
              />
            </div>
          </div>

          <div className="readouts" style={{ margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Apparent variance</div>
              <div className="readout-value">{(100 * b1Res.apparentVar).toFixed(2)}%</div>
            </div>
            <div className="readout">
              <div className="readout-label">True std variance (≥ 90%)</div>
              <div className="readout-value" style={{ color: b1Res.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {pct1(b1Res.trueVar)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: b1Res.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {b1Res.pass ? (st.b1M === 2 ? 'PASS (Optimal m=2!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {!st.b1Scaled && st.b1M === 1
              ? 'Trap sprung! Because Sensor 1 is 100× larger in raw millimetres, PC1 points almost entirely along Sensor 1 and boasts 99.97% apparent variance while throwing away 5 of the 6 sensors (only 17.3% of true standardised variance).'
              : !st.b1Scaled && st.b1M === 2
                ? 'Still unscaled: PC1 was wasted on Sensor 1 alone, so keeping 2 raw components captures only 66.4% of true standardised variance.'
                : b1Res.pass && st.b1M === 2
                  ? 'Spot on: standardising puts all 6 sensors on equal footing, so just 2 components capture 98.3% of the true physical signal.'
                  : b1Res.pass
                    ? `This passes (${pct1(b1Res.trueVar)}), but can you hit ≥ 90% using fewer than ${st.b1M} components?`
                    : 'Not enough variance preserved yet. Try standardising or keeping more components.'}
          </div>
        </div>
      )}

      {briefId === 'segment' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end' }}>
            <Segmented
              label="Feature scaling"
              value={st.b2Scaled ? 'std' : 'raw'}
              onChange={(v) => setSt((p) => ({ ...p, b2Scaled: v === 'std' }))}
              options={[
                { value: 'raw', label: 'Raw (€ vs visits)' },
                { value: 'std', label: 'Standardised' },
              ]}
            />
            <Segmented
              label="Seeding"
              value={st.b2Init}
              onChange={(v) => setSt((p) => ({ ...p, b2Init: v }))}
              options={[
                { value: 'random', label: 'Random' },
                { value: 'kmeans++', label: 'k-Means++' },
              ]}
            />
            <div style={{ flex: '1 1 180px' }}>
              <Slider
                label="Clusters"
                value={st.b2K}
                min={2}
                max={6}
                step={1}
                onChange={(v) => setSt((p) => ({ ...p, b2K: v }))}
                format={(v) => `k = ${v}`}
              />
            </div>
          </div>

          <div className="readouts" style={{ margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Mean Silhouette (≥ 0.72)</div>
              <div className="readout-value" style={{ color: b2Res.meanSil >= 0.72 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {b2Res.meanSil.toFixed(3)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Seeds optimal (≥ 36/40)</div>
              <div className="readout-value" style={{ color: b2Res.stableSeeds >= 36 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {b2Res.stableSeeds}/{b2Res.seeds}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: b2Res.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {b2Res.pass ? 'PASS (Unique!)' : 'FAIL'}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {!st.b2Scaled
              ? 'Raw euro spend (€500–€4,500) completely swamps visit count (2–18) inside Euclidean distance, so k-Means slices vertically along spend alone.'
              : st.b2K === 4 && st.b2Init === 'random'
                ? 'So close! Standardised random init clears Silhouette (0.721), but 11 of 40 seeds still land in bad local minima (29/40 = 72.5%). Switch seeding to k-Means++.'
                : b2Res.pass
                  ? 'Nailed it — out of all 20 configurations, standardised k = 4 with k-Means++ is the unique solution that clears both separation and stability.'
                  : 'Check both cluster count and seeding stability across the 40 seeds.'}
          </div>
        </div>
      )}

      {briefId === 'denoise' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end' }}>
            <Segmented
              label="PCA dimensions before k-Means"
              value={st.b3M}
              onChange={(v) => setSt((p) => ({ ...p, b3M: v }))}
              options={B3_MS.map((m) => ({ value: m, label: m === 8 ? '8 (no PCA)' : `${m}D` }))}
            />
            <div style={{ flex: '1 1 180px' }}>
              <Slider
                label="Clusters"
                value={st.b3K}
                min={2}
                max={5}
                step={1}
                onChange={(v) => setSt((p) => ({ ...p, b3K: v }))}
                format={(v) => `k = ${v}`}
              />
            </div>
          </div>

          <div className="readouts" style={{ margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Silhouette (≥ 0.65)</div>
              <div className="readout-value" style={{ color: b3Res.sil >= 0.65 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {b3Res.sil.toFixed(3)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Hidden purity (≥ 95%)</div>
              <div className="readout-value" style={{ color: b3Res.purity >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {pct1(b3Res.purity)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: b3Res.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {b3Res.pass ? (st.b3M === 2 ? 'PASS (Optimal 2D!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {st.b3M === 8
              ? 'With all 8 columns intact, the 6 random noise columns pad every pairwise distance, crushing Silhouette even when cluster purity is decent.'
              : st.b3M === 1
                ? 'Compressing all the way to 1D collapses two of the four square corners on top of each other, hurting purity.'
                : b3Res.pass && st.b3M === 2
                  ? 'Optimal pipeline! Projecting onto the top 2 principal components strips away all 6 noise columns, lifting Silhouette to 0.805 with 100.0% cluster purity.'
                  : b3Res.pass
                    ? `This passes, though keeping ${st.b3M} dimensions still carries noise columns. Can you compress further?`
                    : 'Try trimming the noisy dimensions with PCA and matching k to the four underlying segments.'}
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
