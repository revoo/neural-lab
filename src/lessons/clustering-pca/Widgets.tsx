// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
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
  unsupLabHint,
  type InitMode,
  type UnsupBrief,
} from './unsupervised';
import { sklearnUnsupLines } from './realCode';

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

function seedShellProgress() {
  const old = loadLab();
  const solved: Record<string, boolean> = {};
  for (const b of LAB_BRIEFS) {
    if (old.solved[b.id]) solved[b.id] = true;
  }
  return { solved, optimal: {} };
}

export function UnsupervisedLab() {
  const [briefIdx, setBriefIdx] = useState(0);
  const [st, setSt] = useState<LabState>(loadLab);
  const progress = useLabProgress('nl-unsup-lab-v1-shell', seedShellProgress);
  const log = useMachineLog();

  useEffect(() => {
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore storage errors */
    }
  }, [st]);

  const brief = LAB_BRIEFS[briefIdx];
  const briefId = brief.id;

  const b1Res = useMemo(() => evalBrief1(st.b1Scaled, st.b1M), [st.b1Scaled, st.b1M]);
  const b2Res = useMemo(() => evalBrief2(st.b2Scaled, st.b2K, st.b2Init), [st.b2Scaled, st.b2K, st.b2Init]);
  const b3Res = useMemo(() => evalBrief3(st.b3M, st.b3K), [st.b3M, st.b3K]);
  const exhaust = useMemo(() => labExhaustive(briefId), [briefId]);

  const curPass = briefId === 'pca-scale' ? b1Res.pass : briefId === 'segment' ? b2Res.pass : b3Res.pass;
  const curOpt =
    curPass &&
    (briefId === 'pca-scale' ? st.b1M === 2 : briefId === 'segment' ? true : st.b3M === 2);

  // Animated bars for Brief 1 per-sensor variance retention & PC spectrum
  const shownSensors = useTweenedArray(
    b1Res.sensorRetained.map((v) => v * 100),
    240,
  );
  const shownPcs = useTweenedArray(
    b1Res.pcShares.map((v) => v * 100),
    240,
  );

  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((prev) => ({ ...prev, solved: { ...prev.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const updateState = (patch: Partial<LabState>) => {
    const next = { ...st, ...patch };
    setSt(next);
    if (briefId === 'pca-scale') {
      const r = evalBrief1(next.b1Scaled, next.b1M);
      log.push(
        `PCA (${next.b1Scaled ? 'z-score' : 'raw'}, m=${next.b1M}) → apparent ${(100 * r.apparentVar).toFixed(2)}% · true ${pct1(r.trueVar)} ${r.pass ? '✓' : '✗'}`,
        r.pass ? 'good' : 'info',
      );
    } else if (briefId === 'segment') {
      const r = evalBrief2(next.b2Scaled, next.b2K, next.b2Init);
      log.push(
        `k-Means (${next.b2Scaled ? 'z-score' : 'raw'}, k=${next.b2K}, ${next.b2Init}) → Sil ${r.meanSil.toFixed(3)} · ${r.stableSeeds}/40 seeds opt ${r.pass ? '✓' : '✗'}`,
        r.pass ? 'good' : 'info',
      );
    } else {
      const r = evalBrief3(next.b3M, next.b3K);
      log.push(
        `PCA(${next.b3M === 8 ? 'none' : `m=${next.b3M}`}) → k-Means(k=${next.b3K}) → Sil ${r.sil.toFixed(3)} · purity ${pct1(r.purity)} ${r.pass ? '✓' : '✗'}`,
        r.pass ? 'good' : 'info',
      );
    }
  };

  const resetBrief = () => {
    if (briefId === 'pca-scale') updateState({ b1Scaled: false, b1M: 1 });
    else if (briefId === 'segment') updateState({ b2Scaled: false, b2K: 4, b2Init: 'random' });
    else updateState({ b3M: 8, b3K: 4 });
  };

  const liveCode = useMemo(() => sklearnUnsupLines(briefId, st), [briefId, st]);
  const shellBriefs = useMemo(
    () => LAB_BRIEFS.map((b, i) => ({ id: b.id, tab: `Brief ${i + 1}`, title: b.title, text: b.story })),
    [],
  );

  const statusNode =
    briefId === 'pca-scale' ? (
      b1Res.pass ? (
        <>
          <strong>{curOpt ? 'Optimal — minimum 2 components!' : 'Passes!'}</strong> True standardised variance kept: <strong>{pct1(b1Res.trueVar)}</strong> (apparent {(100 * b1Res.apparentVar).toFixed(2)}%).{' '}
          {!curOpt && `Optional: can you keep ≥ 90% with fewer than ${st.b1M} components?`}
        </>
      ) : (
        <>
          <strong>Not yet.</strong> Apparent variance claims {(100 * b1Res.apparentVar).toFixed(2)}%, yet true standardised variance kept is only <strong>{pct1(b1Res.trueVar)}</strong> (need ≥ 90%). Compare the 6 sensor bars above.
        </>
      )
    ) : briefId === 'segment' ? (
      b2Res.pass ? (
        <>
          <strong>Unique optimal pipeline found!</strong> Mean Silhouette <strong>{b2Res.meanSil.toFixed(3)}</strong> · <strong>{b2Res.stableSeeds}/40</strong> seeds reach the global optimum.
        </>
      ) : (
        <>
          <strong>Not yet.</strong> Mean Silhouette {b2Res.meanSil.toFixed(3)} (need ≥ 0.72) · {b2Res.stableSeeds}/40 seeds optimal (need ≥ 36/40). Watch both the cluster boundaries and the 40-seed strip.
        </>
      )
    ) : b3Res.pass ? (
      <>
        <strong>{curOpt ? 'Optimal — all 6 noise dimensions stripped!' : 'Passes!'}</strong> Silhouette <strong>{b3Res.sil.toFixed(3)}</strong> · hidden purity <strong>{pct1(b3Res.purity)}</strong>.{' '}
        {!curOpt && `Optional: can you compress below ${st.b3M} dimensions and still pass?`}
      </>
    ) : (
      <>
        <strong>Not yet.</strong> Silhouette {b3Res.sil.toFixed(3)} (need ≥ 0.65) · hidden purity {pct1(b3Res.purity)} (need ≥ 95%). Inspect the 2D projection and the 8-component scree chart.
      </>
    );

  const maxEigB3 = Math.max(1, ...b3Res.eigenvalues);

  return (
    <LabShell
      title="Lab: Compress & Segment"
      caption={
        <>
          Three real unsupervised pipelines with exhaustive checks. Watch the live sensor retention bars (Brief 1), customer cluster boundaries & 40-seed stability strip (Brief 2), and 2D PCA projection & eigenvalue scree cliff (Brief 3).
        </>
      }
      briefs={shellBriefs}
      active={briefIdx}
      onSelect={(i) => setBriefIdx(i)}
      progress={progress}
      passing={curPass}
      optimal={curOpt}
      status={statusNode}
      statusTone={curPass ? 'good' : 'neutral'}
      hint={unsupLabHint(briefId, st)}
      reveal={{
        label: `the exhaustive search (${exhaust.total} configs)`,
        content: (
          <div style={{ fontSize: 13.5, lineHeight: 1.55 }}>
            All <strong>{exhaust.total}</strong> configurations were checked exhaustively: <strong>{exhaust.passing}</strong> pass. {exhaust.summary}.
          </div>
        ),
      }}
      log={{
        lines: log.lines,
        onClear: log.clear,
        title: 'Unsupervised pipeline log',
        empty: 'Adjust scaling, components, k or seeding to record a pipeline run.',
      }}
      actions={
        <button type="button" className="btn small" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {/* Brief 1: PCA Scaling Trap + Per-Sensor Retention X-Ray */}
      {briefId === 'pca-scale' && (
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end' }}>
            <Segmented
              label="Feature scaling"
              value={st.b1Scaled ? 'std' : 'raw'}
              onChange={(v) => updateState({ b1Scaled: v === 'std' })}
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
                onChange={(v) => updateState({ b1M: v })}
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
              <div className="readout-label">Components kept</div>
              <div className="readout-value">{st.b1M} / 6</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12 }}>
            {/* Apparent PC eigenvalue spectrum */}
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 6 }}>
              <div className="field-label" style={{ margin: 0 }}>PCA spectrum (what PCA thinks each PC is worth)</div>
              {shownPcs.map((pct, i) => {
                const kept = i < st.b1M;
                return (
                  <div key={i} style={{ display: 'grid', gridTemplateColumns: '44px 1fr 50px', gap: 8, alignItems: 'center', fontSize: 12 }} className="mono">
                    <span style={{ color: kept ? 'var(--text)' : 'var(--text-muted)', fontWeight: kept ? 700 : 400 }}>PC{i + 1}</span>
                    <div style={{ height: 10, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                      <div style={{ width: `${Math.min(100, pct)}%`, height: '100%', background: kept ? 'var(--accent)' : 'var(--text-muted)', opacity: kept ? 0.9 : 0.35 }} />
                    </div>
                    <span style={{ textAlign: 'right', color: kept ? 'var(--text)' : 'var(--text-muted)' }}>{pct.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>

            {/* Actual per-sensor reconstruction R^2 */}
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 6 }}>
              <div className="field-label" style={{ margin: 0 }}>True signal kept per sensor (S1 is logged in mm!)</div>
              {shownSensors.map((pct, i) => (
                <div key={i} style={{ display: 'grid', gridTemplateColumns: '64px 1fr 48px', gap: 8, alignItems: 'center', fontSize: 12 }} className="mono">
                  <span>{i === 0 ? 'S1 (mm)' : `S${i + 1} (m)`}</span>
                  <div style={{ height: 10, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, pct)}%`, height: '100%', background: pct >= 85 ? 'var(--c-good)' : pct >= 40 ? 'var(--c-b)' : 'var(--c-bad)' }} />
                  </div>
                  <span style={{ textAlign: 'right', fontWeight: 600 }}>{pct.toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Brief 2: Live Customer Clusters + 40-Seed Stability Strip */}
      {briefId === 'segment' && (
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end' }}>
            <Segmented
              label="Feature scaling"
              value={st.b2Scaled ? 'std' : 'raw'}
              onChange={(v) => updateState({ b2Scaled: v === 'std' })}
              options={[
                { value: 'raw', label: 'Raw (€ vs visits)' },
                { value: 'std', label: 'Standardised' },
              ]}
            />
            <Segmented
              label="Seeding"
              value={st.b2Init}
              onChange={(v) => updateState({ b2Init: v })}
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
                onChange={(v) => updateState({ b2K: v })}
                format={(v) => `k = ${v}`}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(265px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
            <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-muted)', textAlign: 'center' }}>
                Sample run clusters (Spend € vs Visits)
              </div>
              <Plot width={300} height={250} x={[200, 4800]} y={[0, 20]} pad={34} xLabel="Annual spend (€)" yLabel="Visits / yr" ariaLabel="Customer k-Means clustering scatter plot">
                {({ x, y }) => (
                  <>
                    {b2Res.rawPts.map((pt, i) => (
                      <circle
                        key={i}
                        cx={x(pt[0])}
                        cy={y(pt[1])}
                        r={3}
                        fill={CLUSTER_COLORS[b2Res.sampleAssign[i] % CLUSTER_COLORS.length]}
                        opacity={0.85}
                      />
                    ))}
                    {b2Res.sampleCentroidsRaw.map((c, ci) => (
                      <g key={`c-${ci}`}>
                        <circle cx={x(c[0])} cy={y(c[1])} r={6.5} fill="var(--bg)" stroke={CLUSTER_COLORS[ci % CLUSTER_COLORS.length]} strokeWidth={2.2} />
                        <text x={x(c[0])} y={y(c[1]) + 3.5} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--text)">×</text>
                      </g>
                    ))}
                  </>
                )}
              </Plot>
            </div>

            <div style={{ display: 'grid', gap: 10 }}>
              <div className="readout" style={{ margin: 0 }}>
                <span style={{ color: b2Res.meanSil >= 0.72 - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {b2Res.meanSil >= 0.72 - 1e-9 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">mean Silhouette </span>
                <strong>{b2Res.meanSil.toFixed(3)}</strong> (need ≥ 0.720)
                {'\n'}
                <span style={{ color: b2Res.stableSeeds >= 36 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {b2Res.stableSeeds >= 36 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">seeds at global optimum </span>
                <strong>{b2Res.stableSeeds} / {b2Res.seeds}</strong> (need ≥ 36/40)
              </div>

              <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 6 }}>
                <div className="field-label" style={{ margin: 0 }}>40-seed stability grid (each tile = 1 random start)</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: 4 }}>
                  {b2Res.seedOptimal.map((opt, idx) => (
                    <div
                      key={idx}
                      title={`Seed ${idx + 1}: Silhouette ${b2Res.seedSils[idx].toFixed(3)} (${opt ? 'global optimum' : 'local minimum trap'})`}
                      style={{
                        height: 18,
                        borderRadius: 4,
                        background: opt ? 'var(--c-good)' : 'var(--c-bad)',
                        opacity: opt ? 0.85 : 0.75,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 10,
                        color: 'var(--bg)',
                        fontWeight: 700,
                      }}
                    >
                      {opt ? '✓' : '×'}
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                  ✓ landed on the global optimum · × got trapped in a suboptimal local minimum.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Brief 3: PCA -> k-Means Denoising (2D projection + 8D Scree Cliff) */}
      {briefId === 'denoise' && (
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end' }}>
            <Segmented
              label="PCA dimensions before k-Means"
              value={st.b3M}
              onChange={(v) => updateState({ b3M: v })}
              options={B3_MS.map((m) => ({ value: m, label: m === 8 ? '8 (no PCA)' : `${m}D` }))}
            />
            <div style={{ flex: '1 1 180px' }}>
              <Slider
                label="Clusters"
                value={st.b3K}
                min={2}
                max={5}
                step={1}
                onChange={(v) => updateState({ b3K: v })}
                format={(v) => `k = ${v}`}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(265px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
            <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-muted)', textAlign: 'center' }}>
                2D (PC1, PC2) view · coloured by clusters found in {st.b3M === 8 ? 'all 8D' : `${st.b3M}D PCA`}
              </div>
              <Plot width={300} height={250} x={[-5.2, 5.2]} y={[-5.2, 5.2]} pad={32} xLabel="PC1" yLabel="PC2" ariaLabel="2D PCA projection of 8D logs coloured by cluster">
                {({ x, y }) => (
                  <>
                    {st.b3M === 1 && (
                      <line x1={x(-5)} x2={x(5)} y1={y(0)} y2={y(0)} stroke="var(--accent)" strokeDasharray="4 3" strokeWidth={1.6} />
                    )}
                    {b3Res.pc2Pts.map((pt, i) => (
                      <circle
                        key={i}
                        cx={x(pt[0])}
                        cy={y(st.b3M === 1 ? pt[1] * 0.15 : pt[1])}
                        r={3.1}
                        fill={CLUSTER_COLORS[b3Res.assign[i] % CLUSTER_COLORS.length]}
                        opacity={0.88}
                      />
                    ))}
                  </>
                )}
              </Plot>
            </div>

            <div style={{ display: 'grid', gap: 10 }}>
              <div className="readout" style={{ margin: 0 }}>
                <span style={{ color: b3Res.sil >= 0.65 - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {b3Res.sil >= 0.65 - 1e-9 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">Silhouette in {st.b3M}D </span>
                <strong>{b3Res.sil.toFixed(3)}</strong> (need ≥ 0.650)
                {'\n'}
                <span style={{ color: b3Res.purity >= 0.95 - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {b3Res.purity >= 0.95 - 1e-9 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">hidden segment purity </span>
                <strong>{pct1(b3Res.purity)}</strong> (need ≥ 95.0%)
              </div>

              <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 5 }}>
                <div className="field-label" style={{ margin: 0 }}>8D eigenvalue scree (kept {st.b3M} of 8 dims)</div>
                {b3Res.eigenvalues.map((eig, idx) => {
                  const kept = idx < st.b3M;
                  const pct = (eig / maxEigB3) * 100;
                  return (
                    <div key={idx} style={{ display: 'grid', gridTemplateColumns: '38px 1fr 44px', gap: 8, alignItems: 'center', fontSize: 11.5 }} className="mono">
                      <span style={{ color: kept ? 'var(--text)' : 'var(--text-muted)', fontWeight: kept ? 700 : 400 }}>PC{idx + 1}</span>
                      <div style={{ height: 9, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                        <div style={{ width: `${pct}%`, height: '100%', background: kept ? 'var(--c-a)' : 'var(--text-muted)', opacity: kept ? 0.9 : 0.3 }} />
                      </div>
                      <span style={{ textAlign: 'right', color: kept ? 'var(--text)' : 'var(--text-muted)' }}>{eig.toFixed(2)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      <LiveCode code={liveCode} note="scikit-learn PCA / KMeans pipeline" />
    </LabShell>
  );
}
