// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Clip, Handle, Plot, usePlot } from '../../components/plot';
import { Segmented, Slider, Toggle } from '../../components/controls';
import {
  OVERLAP_POINTS,
  RING_POINTS,
  RIVAL_LINES,
  SEPARABLE_POINTS,
  evaluateLine2D,
  generateSeparableTestCloud,
  solveHardMargin2D,
  solveSoftMarginSMO,
  type Pt2,
} from './svm';

/** Clip line w[0]*x + w[1]*y + c = 0 to the box [x0, x1] x [y0, y1]. */
export function lineEndpoints(
  wx: number,
  wy: number,
  c: number,
  domain: [number, number] = [-3.2, 3.2],
): [[number, number], [number, number]] | null {
  const [lo, hi] = domain;
  const pts: [number, number][] = [];
  if (Math.abs(wy) > 1e-9) {
    for (const x of [lo, hi]) {
      const y = -(wx * x + c) / wy;
      if (y >= lo - 1e-6 && y <= hi + 1e-6) pts.push([x, y]);
    }
  }
  if (Math.abs(wx) > 1e-9) {
    for (const y of [lo, hi]) {
      const x = -(wy * y + c) / wx;
      if (x >= lo - 1e-6 && x <= hi + 1e-6) pts.push([x, y]);
    }
  }
  // Deduplicate corners
  const uniq: [number, number][] = [];
  for (const p of pts) {
    if (!uniq.some((q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-4)) uniq.push(p);
  }
  if (uniq.length < 2) return null;
  return [uniq[0], uniq[1]];
}

/** Draw the shaded street band between w·x + b = +1 and w·x + b = -1, plus gutters and centre line. */
export function StreetOverlay({
  wx,
  wy,
  b,
  color = 'var(--accent)',
  dashedCentre = false,
  opacity = 0.14,
}: {
  wx: number;
  wy: number;
  b: number;
  color?: string;
  dashedCentre?: boolean;
  opacity?: number;
}) {
  const { x, y } = usePlot();
  const n = Math.hypot(wx, wy);
  if (n < 1e-9) return null;
  const centre = lineEndpoints(wx, wy, b, [-3.5, 3.5]);
  const plus = lineEndpoints(wx, wy, b - 1, [-3.5, 3.5]);
  const minus = lineEndpoints(wx, wy, b + 1, [-3.5, 3.5]);

  // Build street quad from a large segment along centre ± halfMargin along unit normal
  const ux = wx / n;
  const uy = wy / n;
  const half = 1 / n;
  let polyPoints = '';
  if (centre) {
    const [a, c] = centre;
    // Extend slightly beyond box
    const dx = c[0] - a[0];
    const dy = c[1] - a[1];
    const p1: [number, number] = [a[0] - dx * 0.4 + ux * half, a[1] - dy * 0.4 + uy * half];
    const p2: [number, number] = [c[0] + dx * 0.4 + ux * half, c[1] + dy * 0.4 + uy * half];
    const p3: [number, number] = [c[0] + dx * 0.4 - ux * half, c[1] + dy * 0.4 - uy * half];
    const p4: [number, number] = [a[0] - dx * 0.4 - ux * half, a[1] - dy * 0.4 - uy * half];
    polyPoints = [p1, p2, p3, p4].map(([px, py]) => `${x(px).toFixed(1)},${y(py).toFixed(1)}`).join(' ');
  }

  return (
    <Clip>
      <g pointerEvents="none">
        {polyPoints && <polygon points={polyPoints} fill={color} opacity={opacity} />}
        {plus && (
          <line
            x1={x(plus[0][0])}
            y1={y(plus[0][1])}
            x2={x(plus[1][0])}
            y2={y(plus[1][1])}
            stroke={color}
            strokeWidth={1.5}
            strokeDasharray="5 4"
            opacity={0.75}
          />
        )}
        {minus && (
          <line
            x1={x(minus[0][0])}
            y1={y(minus[0][1])}
            x2={x(minus[1][0])}
            y2={y(minus[1][1])}
            stroke={color}
            strokeWidth={1.5}
            strokeDasharray="5 4"
            opacity={0.75}
          />
        )}
        {centre && (
          <line
            x1={x(centre[0][0])}
            y1={y(centre[0][1])}
            x2={x(centre[1][0])}
            y2={y(centre[1][1])}
            stroke={color}
            strokeWidth={2.4}
            strokeDasharray={dashedCentre ? '6 4' : undefined}
          />
        )}
      </g>
    </Clip>
  );
}

/** Point marker: circle for +1, diamond for -1, with optional support-vector halo. */
export function ClassMarker({
  pt,
  isSV = false,
  r = 5.5,
  opacity = 1,
}: {
  pt: Pt2;
  isSV?: boolean;
  r?: number;
  opacity?: number;
}) {
  const { x, y } = usePlot();
  const cx = x(pt.x);
  const cy = y(pt.y);
  const col = pt.label === 1 ? 'var(--c-a)' : 'var(--c-b)';
  return (
    <g opacity={opacity} pointerEvents="none">
      {isSV && (
        <circle
          cx={cx}
          cy={cy}
          r={r + 4.5}
          fill="none"
          stroke="var(--text)"
          strokeWidth={1.8}
          strokeDasharray="3 2"
        />
      )}
      {pt.label === 1 ? (
        <circle cx={cx} cy={cy} r={r} fill={col} stroke="var(--bg-elev)" strokeWidth={1.4} />
      ) : (
        <rect
          x={cx - r * 0.85}
          y={cy - r * 0.85}
          width={r * 1.7}
          height={r * 1.7}
          transform={`rotate(45 ${cx} ${cy})`}
          fill={col}
          stroke="var(--bg-elev)"
          strokeWidth={1.4}
        />
      )}
    </g>
  );
}

// ------------------------------------------------------------------ 1. MarginExplorer

export function MarginExplorer() {
  const [mode, setMode] = useState<'rivals' | 'drag'>('drag');
  const [rivalIdx, setRivalIdx] = useState(2);
  const [showTest, setShowTest] = useState(false);
  const [pts, setPts] = useState<Pt2[]>(() => SEPARABLE_POINTS.map((p) => ({ ...p })));
  const [lastMoved, setLastMoved] = useState<{ id: string; wasSV: boolean } | null>(null);

  const testCloud = useMemo(() => generateSeparableTestCloud(), []);
  const hard = useMemo(() => solveHardMargin2D(pts), [pts]);
  const baseHard = useMemo(() => solveHardMargin2D(SEPARABLE_POINTS), []);

  const rivalStats = useMemo(() => {
    return RIVAL_LINES.map((rl) => ({
      ...rl,
      train: evaluateLine2D(SEPARABLE_POINTS, rl.a, rl.b),
      test: evaluateLine2D(testCloud, rl.a, rl.b),
    }));
  }, [testCloud]);

  const activeRival = rivalStats[rivalIdx];
  const svSet = new Set(hard.separable ? hard.supportIndices : []);

  const onDragPoint = (idx: number, nx: number, ny: number) => {
    const wasSV = svSet.has(idx);
    setLastMoved({ id: pts[idx].id, wasSV });
    setPts((prev) => prev.map((p, i) => (i === idx ? { ...p, x: +nx.toFixed(2), y: +ny.toFixed(2) } : p)));
  };

  return (
    <Widget
      title="The widest street — and the few points that pin it"
      wide
      caption="In “Drag points” mode, grab any ringed support vector and watch the whole street pivot. Now drag an unringed interior point (like P4 or N3): as long as it stays behind its gutter, the street does not budge by a single pixel. Switch to “Compare 3 lines” to test rival boundaries against 200 fresh points."
    >
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 320px', maxWidth: 400 }}>
          <Plot
            width={380}
            height={380}
            x={[-3, 3]}
            y={[-3, 3]}
            xLabel="x₁"
            yLabel="x₂"
            ariaLabel="2D Support Vector Machine street and draggable points"
            style={{ width: '100%', height: 'auto' }}
          >
            {({ x, y }) => (
              <g>
                {mode === 'rivals' && showTest &&
                  testCloud.map((tp) => <ClassMarker key={tp.id} pt={tp} r={2.6} opacity={0.38} />)}

                {mode === 'rivals' && activeRival.train.separates && (
                  <StreetOverlay
                    wx={activeRival.train.normal[0] / activeRival.train.minSignedDist}
                    wy={activeRival.train.normal[1] / activeRival.train.minSignedDist}
                    b={activeRival.train.offset / activeRival.train.minSignedDist}
                    color={rivalIdx === 2 ? 'var(--c-c)' : 'var(--accent)'}
                  />
                )}

                {mode === 'drag' && hard.separable && (
                  <StreetOverlay wx={hard.w[0]} wy={hard.w[1]} b={hard.b} color="var(--accent)" />
                )}

                {(mode === 'drag' ? pts : SEPARABLE_POINTS).map((pt, i) => (
                  <g key={pt.id}>
                    <ClassMarker pt={pt} isSV={mode === 'drag' && svSet.has(i)} r={6} />
                    <text
                      x={x(pt.x) + 8}
                      y={y(pt.y) - 6}
                      fontSize={10.5}
                      fill="var(--text-muted)"
                      style={{ fontFamily: 'var(--font-mono)', pointerEvents: 'none' }}
                    >
                      {pt.id}
                    </text>
                  </g>
                ))}

                {mode === 'drag' &&
                  pts.map((pt, i) => (
                    <Handle
                      key={`h-${pt.id}`}
                      x={pt.x}
                      y={pt.y}
                      r={7}
                      color={pt.label === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                      label={`Move point ${pt.id}`}
                      onDrag={(nx, ny) => onDragPoint(i, nx, ny)}
                    />
                  ))}
              </g>
            )}
          </Plot>
        </div>

        <div style={{ flex: '1 1 250px', display: 'grid', gap: 12 }}>
          <Segmented
            label="Mode"
            value={mode}
            onChange={(m) => setMode(m)}
            options={[
              { value: 'drag', label: 'Drag points (live SVM)' },
              { value: 'rivals', label: 'Compare 3 lines' },
            ]}
          />

          {mode === 'drag' ? (
            <>
              <div className="readout">
                {hard.separable ? (
                  <>
                    <div>
                      <strong>Street width (2/‖w‖):</strong>{' '}
                      <span className="mono">{hard.streetWidth.toFixed(3)}</span>
                    </div>
                    <div>
                      <strong>Weight length ‖w‖:</strong>{' '}
                      <span className="mono">{Math.hypot(hard.w[0], hard.w[1]).toFixed(3)}</span>
                    </div>
                    <div>
                      <strong>Support vectors ({hard.supportIndices.length}/10):</strong>{' '}
                      <span className="mono">{hard.supportIndices.map((i) => pts[i].id).join(', ')}</span>
                    </div>
                  </>
                ) : (
                  <div style={{ color: 'var(--c-bad)' }}>
                    <strong>Not linearly separable!</strong> Two points crossed past the opposite class — no hard-margin street exists. That is why we need the soft margin below.
                  </div>
                )}
              </div>

              <div className="live-note">
                {!hard.separable
                  ? 'Classes overlap — press Reset or drag the crossed point back to restore a clean street.'
                  : lastMoved
                    ? !lastMoved.wasSV &&
                      baseHard.separable &&
                      Math.abs(hard.streetWidth - baseHard.streetWidth) < 1e-4
                      ? `Moved non-support point ${lastMoved.id} — the street width stayed ${hard.streetWidth.toFixed(3)} (untouched).`
                      : `Moved ${lastMoved.id}: street width is now ${hard.streetWidth.toFixed(3)}, pinned by ${hard.supportIndices.map((i) => pts[i].id).join(', ')}.`
                    : 'Only 3 of the 10 points (P1, P2, N1) sit on the gutters. The other 7 have zero influence on the line.'}
              </div>

              <div className="btn-row">
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setPts(SEPARABLE_POINTS.map((p) => ({ ...p })));
                    setLastMoved(null);
                  }}
                >
                  Reset points
                </button>
              </div>
            </>
          ) : (
            <>
              <Segmented
                label="Candidate boundary (all 10/10 on training points)"
                value={rivalIdx}
                onChange={(v) => setRivalIdx(v)}
                options={RIVAL_LINES.map((rl, i) => ({ value: i, label: rl.label.split(' ')[0] + ' ' + rl.label.split(' ')[1] }))}
              />
              <Toggle label="Show 200 fresh test points" checked={showTest} onChange={setShowTest} />

              <table className="num-table">
                <thead>
                  <tr>
                    <th>Line</th>
                    <th>Train</th>
                    <th>Street</th>
                    <th>Test (200)</th>
                  </tr>
                </thead>
                <tbody>
                  {rivalStats.map((r, i) => (
                    <tr key={r.id} style={i === rivalIdx ? { fontWeight: 700, background: 'var(--accent-soft)' } : undefined}>
                      <td>{r.label}</td>
                      <td className="mono">100%</td>
                      <td className="mono">{r.train.streetWidth.toFixed(2)}</td>
                      <td className="mono">{showTest ? `${(r.test.accuracy * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="live-note">
                All three lines classify all 10 training points cleanly, yet their street widths differ more than 6× ({rivalStats[0].train.streetWidth.toFixed(2)} vs {rivalStats[2].train.streetWidth.toFixed(2)}). Toggle the 200 fresh test points to see how extra breathing room translates directly into fewer errors on new data.
              </div>
            </>
          )}
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. SoftMarginWidget

export function SoftMarginWidget() {
  // Log slider: log10(C) in [-1.7, 1.7] -> C in [0.02, 50]
  const [logC, setLogC] = useState(-0.3); // C ≈ 0.5
  const C = +(10 ** logC).toFixed(logC < 0 ? 3 : 1);

  const sol = useMemo(() => {
    const X = OVERLAP_POINTS.map((p) => [p.x, p.y]);
    const y = OVERLAP_POINTS.map((p) => p.label);
    return solveSoftMarginSMO(X, y, C, 120);
  }, [C]);

  const svSet = new Set(sol.supportIndices);
  const sumSlack = sol.slacks.reduce((s, v) => s + v, 0);
  const wNorm = Math.hypot(sol.w[0], sol.w[1]);

  return (
    <Widget
      title="Soft margin: trading street width against slack (C)"
      wide
      caption="Two outliers (O1, O2) cross into the opposite cluster, so a hard margin is impossible. Slide C from small (relaxed, wide street) to large (strict, narrow street). The dotted stems show each violating point’s slack ξ — how far it intrudes past its own gutter."
    >
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 320px', maxWidth: 400 }}>
          <Plot
            width={380}
            height={380}
            x={[-2.8, 2.8]}
            y={[-2.8, 2.8]}
            xLabel="x₁"
            yLabel="x₂"
            ariaLabel="Soft-margin SVM street and slack stems"
            style={{ width: '100%', height: 'auto' }}
          >
            {({ x, y }) => (
              <g>
                <StreetOverlay wx={sol.w[0]} wy={sol.w[1]} b={sol.b} color="var(--accent)" />

                {/* Draw slack violation segments from each violating point perpendicularly to its own gutter */}
                {wNorm > 1e-6 &&
                  OVERLAP_POINTS.map((pt, i) => {
                    const xi = sol.slacks[i];
                    if (xi <= 0.02) return null;
                    // Move point along label * w / ||w||^2 by xi to land on its gutter w·x + b = label
                    const gx = pt.x + (pt.label * xi * sol.w[0]) / (wNorm * wNorm);
                    const gy = pt.y + (pt.label * xi * sol.w[1]) / (wNorm * wNorm);
                    return (
                      <line
                        key={`sl-${pt.id}`}
                        x1={x(pt.x)}
                        y1={y(pt.y)}
                        x2={x(gx)}
                        y2={y(gy)}
                        stroke="var(--c-bad)"
                        strokeWidth={2}
                        strokeDasharray="3 2"
                      />
                    );
                  })}

                {OVERLAP_POINTS.map((pt, i) => (
                  <g key={pt.id}>
                    <ClassMarker pt={pt} isSV={svSet.has(i)} r={5.5} />
                    {(pt.id.startsWith('O') || sol.slacks[i] > 0.3) && (
                      <text
                        x={x(pt.x) + 7}
                        y={y(pt.y) - 5}
                        fontSize={10}
                        fill="var(--text-muted)"
                        style={{ fontFamily: 'var(--font-mono)', pointerEvents: 'none' }}
                      >
                        {pt.id}
                      </text>
                    )}
                  </g>
                ))}
              </g>
            )}
          </Plot>
        </div>

        <div style={{ flex: '1 1 250px', display: 'grid', gap: 12 }}>
          <Slider
            label="Penalty C (log scale)"
            value={logC}
            min={-1.5}
            max={1.7}
            step={0.05}
            onChange={setLogC}
            format={() => C < 1 ? C.toFixed(2) : C.toFixed(1)}
            color="accent"
          />

          <div className="btn-row">
            <button type="button" className="btn" onClick={() => setLogC(Math.log10(0.05))}>
              Relaxed (C = 0.05)
            </button>
            <button type="button" className="btn" onClick={() => setLogC(Math.log10(0.5))}>
              Balanced (C = 0.5)
            </button>
            <button type="button" className="btn" onClick={() => setLogC(Math.log10(50))}>
              Strict (C = 50)
            </button>
          </div>

          <div className="readout">
            <div>
              <strong>Street width (2/‖w‖):</strong> <span className="mono">{sol.streetWidth.toFixed(3)}</span>
            </div>
            <div>
              <strong>Support vectors:</strong> <span className="mono">{sol.supportIndices.length} / {OVERLAP_POINTS.length}</span>
            </div>
            <div>
              <strong>Total slack (Σξ):</strong> <span className="mono">{sumSlack.toFixed(2)}</span>
            </div>
            <div>
              <strong>Training accuracy:</strong> <span className="mono">{(sol.trainAcc * 100).toFixed(1)}%</span>
            </div>
          </div>

          <div className="live-note">
            {C < 0.12
              ? 'Small C: violations are cheap, so the SVM keeps a very wide street and lets many points lean into the gutter.'
              : C > 10
                ? 'Large C: every slack stem is expensive, so the SVM squeezes the street narrower trying to shrink the violations around O1 and O2.'
                : 'Moderate C: the street stays wide between the two main clusters while paying a bounded slack budget for O1 and O2.'}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. KernelLiftWidget

export function KernelLiftWidget() {
  const [lift, setLift] = useState(1); // 0 = flat 2D, 1 = lifted to z = x1^2 + x2^2
  const [yawDeg, setYawDeg] = useState(28);
  const [pitchDeg, setPitchDeg] = useState(22);

  // Optimal 1D threshold in z = x1^2 + x2^2 separating RING_POINTS
  const innerMaxZ = Math.max(...RING_POINTS.filter((p) => p.label === 1).map((p) => p.x * p.x + p.y * p.y));
  const outerMinZ = Math.min(...RING_POINTS.filter((p) => p.label === -1).map((p) => p.x * p.x + p.y * p.y));
  const cutZ = (innerMaxZ + outerMinZ) / 2;
  const cutR = Math.sqrt(cutZ);

  // Simple orthographic 3D projection into SVG [0..380] x [0..320]
  const project = (px: number, py: number, pz: number): [number, number, number] => {
    const yaw = (yawDeg * Math.PI) / 180;
    const pitch = (pitchDeg * Math.PI) / 180;
    // Rotate around z axis by yaw
    const x1 = px * Math.cos(yaw) - py * Math.sin(yaw);
    const y1 = px * Math.sin(yaw) + py * Math.cos(yaw);
    const z1 = (pz - 2.2) * 0.58;
    // Tilt by pitch
    const y2 = y1 * Math.cos(pitch) - z1 * Math.sin(pitch);
    const z2 = y1 * Math.sin(pitch) + z1 * Math.cos(pitch);
    const sx = 190 + x1 * 56;
    const sy = 175 - z2 * 64;
    return [sx, sy, y2]; // y2 = depth for painter's sort
  };

  // Cutting plane square corners at z = cutZ * lift
  const planeCorners = ([-2.3, 2.3] as const).flatMap((gx) =>
    ([-2.3, 2.3] as const).map((gy) => [gx, gy] as [number, number]),
  );
  const orderedCorners: [number, number][] = [planeCorners[0], planeCorners[1], planeCorners[3], planeCorners[2]];
  const planeSvg = orderedCorners
    .map(([gx, gy]) => {
      const [sx, sy] = project(gx, gy, cutZ * lift);
      return `${sx.toFixed(1)},${sy.toFixed(1)}`;
    })
    .join(' ');

  // Base floor corners at z = 0
  const floorSvg = orderedCorners
    .map(([gx, gy]) => {
      const [sx, sy] = project(gx, gy, 0);
      return `${sx.toFixed(1)},${sy.toFixed(1)}`;
    })
    .join(' ');

  const projectedPts = RING_POINTS.map((p) => {
    const zVal = (p.x * p.x + p.y * p.y) * lift;
    const [sx, sy, depth] = project(p.x, p.y, zVal);
    const [bx, by] = project(p.x, p.y, 0);
    return { p, sx, sy, bx, by, depth, zVal };
  }).sort((a, b) => a.depth - b.depth);

  return (
    <Widget
      title="The kernel trick: lift the rings, cut with a flat plane"
      wide
      caption="Slide “Lift to z = x₁² + x₂²” from 0 (flat 2D) to 1 (3D paraboloid). In 2D no straight line separates the core from the ring; lifted into a third dimension, one flat horizontal sheet slices cleanly between them — and its shadow back in 2D is a circle."
    >
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 260px', maxWidth: 320 }}>
          <Plot
            width={300}
            height={300}
            x={[-2.6, 2.6]}
            y={[-2.6, 2.6]}
            xLabel="x₁"
            yLabel="x₂"
            ariaLabel="2D concentric rings with circular shadow of cutting plane"
            style={{ width: '100%', height: 'auto' }}
          >
            {({ x, y }) => (
              <g>
                {lift > 0.15 && (
                  <circle
                    cx={x(0)}
                    cy={y(0)}
                    r={Math.abs(x(cutR) - x(0))}
                    fill="color-mix(in srgb, var(--c-a) 10%, transparent)"
                    stroke="var(--accent)"
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    opacity={lift}
                  />
                )}
                {RING_POINTS.map((pt) => (
                  <ClassMarker key={pt.id} pt={pt} r={5} />
                ))}
              </g>
            )}
          </Plot>
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', textAlign: 'center', marginTop: 4 }}>
            Original 2D space: shadow of the cut is r = {cutR.toFixed(2)}
          </div>
        </div>

        <div style={{ flex: '1 1 320px', maxWidth: 400 }}>
          <svg
            viewBox="0 0 380 320"
            width={380}
            height={320}
            role="img"
            aria-label="3D lifted view of concentric rings separated by a flat horizontal plane"
            style={{ width: '100%', height: 'auto', display: 'block', background: 'var(--bg-sunken)', borderRadius: 8, border: '1px solid var(--border)' }}
          >
            {/* Floor grid plane */}
            <polygon points={floorSvg} fill="var(--bg-elev)" stroke="var(--border)" strokeWidth={1} opacity={0.7} />

            {/* Stems from floor to lifted points */}
            {lift > 0.05 &&
              projectedPts.map(({ p, sx, sy, bx, by }) => (
                <line
                  key={`st-${p.id}`}
                  x1={bx}
                  y1={by}
                  x2={sx}
                  y2={sy}
                  stroke="var(--border-strong)"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  opacity={0.7}
                />
              ))}

            {/* Points below the plane */}
            {projectedPts
              .filter((d) => d.zVal <= cutZ * lift + 1e-6)
              .map(({ p, sx, sy }) => (
                <circle
                  key={`b-${p.id}`}
                  cx={sx}
                  cy={sy}
                  r={5.5}
                  fill={p.label === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                  stroke="var(--bg-elev)"
                  strokeWidth={1.3}
                />
              ))}

            {/* Flat separating hyperplane z = cutZ */}
            {lift > 0.15 && (
              <polygon
                points={planeSvg}
                fill="color-mix(in srgb, var(--accent) 22%, transparent)"
                stroke="var(--accent)"
                strokeWidth={1.8}
                opacity={lift}
              />
            )}

            {/* Points above the plane */}
            {projectedPts
              .filter((d) => d.zVal > cutZ * lift + 1e-6)
              .map(({ p, sx, sy }) => (
                <rect
                  key={`t-${p.id}`}
                  x={sx - 4.7}
                  y={sy - 4.7}
                  width={9.4}
                  height={9.4}
                  transform={`rotate(45 ${sx} ${sy})`}
                  fill={p.label === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                  stroke="var(--bg-elev)"
                  strokeWidth={1.3}
                />
              ))}

            <text x={12} y={20} fontSize={11.5} fill="var(--text-muted)" fontFamily="var(--font-mono)">
              z = {lift === 0 ? '0 (flat 2D)' : `${lift.toFixed(2)}·(x₁² + x₂²)`}
            </text>
            {lift > 0.5 && (
              <text x={12} y={36} fontSize={11} fill="var(--accent-text)" fontFamily="var(--font-mono)">
                flat cut: z = {(cutZ * lift).toFixed(2)}
              </text>
            )}
          </svg>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 12, flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 180px' }}>
          <Slider
            label="Lift into 3D (z = x₁² + x₂²)"
            value={lift}
            min={0}
            max={1}
            step={0.02}
            onChange={setLift}
            format={(v) => `${Math.round(v * 100)}%`}
            color="accent"
          />
        </div>
        <div style={{ flex: '1 1 140px' }}>
          <Slider
            label="Spin view"
            value={yawDeg}
            min={-90}
            max={90}
            step={2}
            onChange={setYawDeg}
            format={(v) => `${v}°`}
          />
        </div>
        <div style={{ flex: '1 1 140px' }}>
          <Slider
            label="Tilt view"
            value={pitchDeg}
            min={5}
            max={65}
            step={2}
            onChange={setPitchDeg}
            format={(v) => `${v}°`}
          />
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        Inner cluster sits at x₁² + x₂² ≤ {innerMaxZ.toFixed(2)}; outer ring sits at x₁² + x₂² ≥ {outerMinZ.toFixed(2)}. A single flat plane at z = {cutZ.toFixed(2)} separates all 24 points cleanly, with a margin of {(outerMinZ - innerMaxZ).toFixed(2)} along the z-axis.
      </div>
    </Widget>
  );
}
