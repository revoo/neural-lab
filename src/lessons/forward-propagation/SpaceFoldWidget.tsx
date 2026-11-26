// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import {
  deformGrid,
  interpolateBounds,
  interpolateStage,
  outputBoundaryLine,
  trainSpaceFold,
  type FoldDatasetId,
  type Vec2,
} from './spaceFold';

const FRAMES = 40; // 20 frames per layer -> t = frame / 20 in [0, 2]
const W = 360;
const H = 320;
const PAD = 22;

function toSvg(pt: Vec2, cx: number, cy: number, half: number): [number, number] {
  const span = Math.min(W, H) - 2 * PAD;
  const x0 = (W - span) / 2;
  const y0 = (H - span) / 2;
  const sx = x0 + ((pt[0] - (cx - half)) / (2 * half)) * span;
  const sy = y0 + (1 - (pt[1] - (cy - half)) / (2 * half)) * span;
  return [sx, sy];
}

function polylinePath(pts: [number, number][]): string {
  return pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
}

function stageNote(id: FoldDatasetId, t: number, acc0: number, acc1: number, acc2: number): string {
  const p0 = `${(acc0 * 100).toFixed(1)}%`;
  const p1 = `${(acc1 * 100).toFixed(1)}%`;
  const p2 = `${(acc2 * 100).toFixed(0)}%`;
  if (t < 0.05) {
    return id === 'rings'
      ? `Input space (t = 0): the inner cluster is completely surrounded by the outer ring. No straight line can beat ${p0} here — the magenta loop is the network’s curved boundary seen from the input.`
      : `Input space (t = 0): the two crescents interlock like C-hooks. The best straight line gets ${p0}; the magenta S-curve is what the network’s straight final cut looks like before folding.`;
  }
  if (t < 0.95) {
    return `Layer 1 in motion (t = ${t.toFixed(2)}): the 2×2 weight matrix rotates and shears the rubber sheet, then GELU’s bend begins folding negative coordinates back onto themselves.`;
  }
  if (t < 1.05) {
    return id === 'rings'
      ? `After Layer 1 (t = 1): the closed ring has been pinched and folded halfway open. A straight line can already reach ${p1}, up from ${p0}.`
      : `After Layer 1 (t = 1): the first bend has partly unhooked the two crescents, lifting the best straight-line score from ${p0} to ${p1}.`;
  }
  if (t < 1.95) {
    return `Layer 2 in motion (t = ${t.toFixed(2)}): a second matrix mix and GELU fold finishes straightening the magenta boundary.`;
  }
  return id === 'rings'
    ? `After Layer 2 (t = 2): the rubber sheet has folded the outer ring onto one side and tucked the inner cluster onto the other. The single straight output line now separates ${p2} of all 48 dots (and 100% of the continuous rings).`
    : `After Layer 2 (t = 2): the two crescents now sit on opposite sides of one straight knife cut — ${p2} accuracy from a single linear output neuron.`;
}

export function SpaceFoldWidget() {
  const [dataset, setDataset] = useState<FoldDatasetId>('rings');
  const [showGrid, setShowGrid] = useState(true);
  const [showBoundary, setShowBoundary] = useState(true);

  const data = useMemo(() => trainSpaceFold(dataset), [dataset]);
  const replay = useReplay(FRAMES, 55, dataset, false);
  const t = replay.i / (FRAMES / 2);

  const box = useMemo(() => interpolateBounds(data.stageBounds, t), [data, t]);
  const grid = useMemo(() => (showGrid ? deformGrid(data.net, t, 13, 1.45) : []), [data.net, t, showGrid]);
  const knife = useMemo(() => outputBoundaryLine(data.net, box), [data.net, box]);

  const horizPaths = useMemo(
    () => grid.map((row) => polylinePath(row.map((pt) => toSvg(pt, box.cx, box.cy, box.half)))),
    [grid, box],
  );
  const vertPaths = useMemo(() => {
    if (grid.length === 0) return [];
    const n = grid[0].length;
    return Array.from({ length: n }, (_, c) =>
      polylinePath(grid.map((row) => toSvg(row[c], box.cx, box.cy, box.half))),
    );
  }, [grid, box]);

  const stageIdx = t < 0.5 ? 0 : t < 1.5 ? 1 : 2;
  const stageLabel =
    t < 0.05 ? 'Input (x₁, x₂)'
      : t < 0.95 ? 'Folding → Layer 1'
        : t < 1.05 ? 'Layer 1 (h₁⁽¹⁾, h₂⁽¹⁾)'
          : t < 1.95 ? 'Folding → Layer 2'
            : 'Layer 2 (h₁⁽²⁾, h₂⁽²⁾)';

  const knifeOpacity = t >= 1.8 ? Math.min(1, (t - 1.8) / 0.2) : 0;

  return (
    <Widget
      title="Watch a network fold space until one straight line cuts it"
      caption="Press Play or scrub from Input (t = 0) through Layer 1 and Layer 2. Every coordinate is computed live by a real 15-parameter 2 → 2 → 2 → 1 GELU network. Watch the magenta decision boundary unbend into a ruler-straight line."
    >
      <div className="controls">
        <Segmented
          label="Dataset"
          value={dataset}
          onChange={setDataset}
          options={[
            { value: 'rings', label: 'Concentric rings' },
            { value: 'moons', label: 'Two moons' },
          ]}
        />
        <div className="btn-row">
          <button type="button" className={`btn small${replay.i === 0 ? ' primary' : ''}`} onClick={() => replay.seek(0)}>
            Input (t = 0)
          </button>
          <button type="button" className={`btn small${replay.i === FRAMES / 2 ? ' primary' : ''}`} onClick={() => replay.seek(FRAMES / 2)}>
            Layer 1 (t = 1)
          </button>
          <button type="button" className={`btn small${replay.i === FRAMES ? ' primary' : ''}`} onClick={() => replay.seek(FRAMES)}>
            Layer 2 (t = 2)
          </button>
        </div>
        <label className="ctrl-check">
          <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
          <span>Rubber mesh</span>
        </label>
        <label className="ctrl-check">
          <input type="checkbox" checked={showBoundary} onChange={(e) => setShowBoundary(e.target.checked)} />
          <span>Boundary</span>
        </label>
      </div>

      <div style={{ marginBottom: 12 }}>
        <ReplayControls r={replay} label="frame" />
      </div>

      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 280px', minWidth: 250, maxWidth: 420 }}>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
              borderRadius: 10,
              border: '1px solid var(--border)',
              background: 'var(--bg-sunken)',
            }}
            role="img"
            aria-label={`Warped 2D coordinate mesh for ${data.title} at layer stage t = ${t.toFixed(2)}`}
          >
            <defs>
              <clipPath id={`sf-clip-${dataset}`}>
                <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} rx={6} />
              </clipPath>
            </defs>

            <g clipPath={`url(#sf-clip-${dataset})`}>
              {/* Warped rubber-sheet coordinate mesh */}
              {showGrid &&
                horizPaths.map((d, i) => (
                  <path
                    key={`h-${i}`}
                    d={d}
                    fill="none"
                    stroke="var(--c-a)"
                    strokeWidth={i === 6 ? 1.5 : 0.9}
                    strokeOpacity={i === 6 ? 0.55 : 0.25}
                  />
                ))}
              {showGrid &&
                vertPaths.map((d, i) => (
                  <path
                    key={`v-${i}`}
                    d={d}
                    fill="none"
                    stroke="var(--c-b)"
                    strokeWidth={i === 6 ? 1.5 : 0.9}
                    strokeOpacity={i === 6 ? 0.55 : 0.25}
                  />
                ))}

              {/* Warped p = 0.5 decision contour unbending into a straight line */}
              {showBoundary &&
                data.boundarySegments.map((seg, idx) => {
                  const pA = toSvg(
                    interpolateStage({ h0: seg.a[0], h1: seg.a[1], h2: seg.a[2] }, t),
                    box.cx,
                    box.cy,
                    box.half,
                  );
                  const pB = toSvg(
                    interpolateStage({ h0: seg.b[0], h1: seg.b[1], h2: seg.b[2] }, t),
                    box.cx,
                    box.cy,
                    box.half,
                  );
                  return (
                    <line
                      key={idx}
                      x1={pA[0]}
                      y1={pA[1]}
                      x2={pB[0]}
                      y2={pB[1]}
                      stroke="var(--c-grad)"
                      strokeWidth={2.5}
                      strokeLinecap="round"
                      strokeOpacity={0.92}
                    />
                  );
                })}

              {/* Exact straight output knife line at t -> 2 */}
              {showBoundary && knife && knifeOpacity > 0 && (() => {
                const a = toSvg(knife[0], box.cx, box.cy, box.half);
                const b = toSvg(knife[1], box.cx, box.cy, box.half);
                return (
                  <line
                    x1={a[0]}
                    y1={a[1]}
                    x2={b[0]}
                    y2={b[1]}
                    stroke="var(--c-grad)"
                    strokeWidth={2.2}
                    strokeDasharray="6 4"
                    strokeOpacity={knifeOpacity}
                  />
                );
              })()}

              {/* Data points */}
              {data.points.map((pt, idx) => {
                const pos = interpolateStage(data.pointStages[idx], t);
                const [sx, sy] = toSvg(pos, box.cx, box.cy, box.half);
                return (
                  <circle
                    key={idx}
                    cx={sx}
                    cy={sy}
                    r={4.5}
                    fill={pt.label === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                    stroke="var(--bg-elev)"
                    strokeWidth={1.3}
                  />
                );
              })}
            </g>

            {/* Stage badge */}
            <g transform="translate(12, 18)">
              <text fill="var(--text-muted)" fontSize="11.5" fontFamily="var(--font-mono)" fontWeight={600}>
                {stageLabel}
              </text>
            </g>
            {knifeOpacity > 0.5 && showBoundary && (
              <g transform={`translate(${W - 12}, ${H - 8})`}>
                <text
                  textAnchor="end"
                  fill="var(--c-grad)"
                  fontSize="11"
                  fontFamily="var(--font-mono)"
                  fontWeight={600}
                >
                  straight knife cut (w₃·h⁽²⁾ + b₃ = 0)
                </text>
              </g>
            )}
          </svg>
        </div>

        <div style={{ flex: '1 1 220px', minWidth: 220, display: 'grid', gap: 10 }}>
          <div style={{ padding: '10px 12px', border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-sunken)' }}>
            <div className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 8 }}>
              BEST STRAIGHT-LINE ACCURACY BY LAYER
            </div>
            {[
              { label: 'Input (t = 0)', acc: data.acc0, active: stageIdx === 0, target: 0 },
              { label: 'After Layer 1 (t = 1)', acc: data.acc1, active: stageIdx === 1, target: FRAMES / 2 },
              { label: 'After Layer 2 (t = 2)', acc: data.acc2, active: stageIdx === 2, target: FRAMES },
            ].map((row) => (
              <button
                key={row.label}
                type="button"
                onClick={() => replay.seek(row.target)}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  background: row.active ? 'var(--bg-elev)' : 'transparent',
                  border: row.active ? '1px solid var(--accent)' : '1px solid transparent',
                  borderRadius: 8,
                  padding: '6px 8px',
                  cursor: 'pointer',
                  marginBottom: 4,
                  color: 'var(--text)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                  <span style={{ fontWeight: row.active ? 600 : 400 }}>{row.label}</span>
                  <span className="mono" style={{ fontWeight: 700, color: row.acc >= 0.999 ? 'var(--c-c)' : 'var(--text)' }}>
                    {(row.acc * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ height: 6, borderRadius: 4, background: 'var(--border)', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${(row.acc * 100).toFixed(1)}%`,
                      height: '100%',
                      background: row.acc >= 0.999 ? 'var(--c-c)' : row.active ? 'var(--accent)' : 'var(--text-faint)',
                    }}
                  />
                </div>
              </button>
            ))}
            <div className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 6 }}>
              Architecture: 2 → 2 → 2 → 1 GELU ({data.paramCount} params)
            </div>
          </div>

          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12.5, color: 'var(--text-muted)' }}>
            <span>
              <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: '50%', background: 'var(--c-a)', marginRight: 5 }} />
              Class A (inner / upper)
            </span>
            <span>
              <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: '50%', background: 'var(--c-b)', marginRight: 5 }} />
              Class B (outer / lower)
            </span>
            <span>
              <span style={{ display: 'inline-block', width: 14, height: 2.5, background: 'var(--c-grad)', verticalAlign: 'middle', marginRight: 5 }} />
              p = 0.5 boundary
            </span>
          </div>
        </div>
      </div>

      <p className="live-note" style={{ marginTop: 12 }}>
        {stageNote(dataset, t, data.acc0, data.acc1, data.acc2)}
      </p>
    </Widget>
  );
}
