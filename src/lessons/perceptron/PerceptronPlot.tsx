// authored-by: Claude Opus 5.5
import { useMemo } from 'react';
import { Plot, usePlot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import { out, type Pt, type W } from './perceptron';

export const DOMAIN: [number, number] = [-0.5, 1.6];
const fireColour = (t: number) => diverging(1 - t);

/** The segment of w1·x + w2·y + b = 0 inside the plot's domain, or null if there is no line. */
export function lineSegment(w: W, lo = DOMAIN[0], hi = DOMAIN[1]): [[number, number], [number, number]] | null {
  const [a, b, c] = w;
  const pts: [number, number][] = [];
  if (Math.abs(b) > 1e-12) {
    for (const x of [lo, hi]) {
      const y = -(a * x + c) / b;
      if (y >= lo - 1e-9 && y <= hi + 1e-9) pts.push([x, y]);
    }
  }
  if (Math.abs(a) > 1e-12) {
    for (const y of [lo, hi]) {
      const x = -(b * y + c) / a;
      if (x >= lo - 1e-9 && x <= hi + 1e-9) pts.push([x, y]);
    }
  }
  const uniq = pts.filter((p, i) => pts.findIndex((q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9) === i);
  return uniq.length >= 2 ? [uniq[0], uniq[1]] : null;
}

function Line({ w }: { w: W }) {
  const { x, y } = usePlot();
  const s = lineSegment(w);
  if (!s) return null;
  return <line x1={x(s[0][0])} y1={y(s[0][1])} x2={x(s[1][0])} y2={y(s[1][1])} stroke="var(--c-c)" strokeWidth={2.5} />;
}

function Points({ pts, w, focus, faded }: { pts: Pt[]; w: W; focus?: number; faded?: Set<number> }) {
  const { x, y } = usePlot();
  return (
    <g>
      {pts.map((p, i) => {
        const wrong = out(w, p) !== p.y;
        const gone = faded?.has(i);
        return (
          <g key={i} opacity={gone ? 0.25 : 1}>
            {i === focus && <circle cx={x(p.x[0])} cy={y(p.x[1])} r={13} fill="none" stroke="var(--accent)" strokeWidth={2.5} />}
            <circle cx={x(p.x[0])} cy={y(p.x[1])} r={6.5} fill={p.y ? 'var(--c-b)' : 'var(--c-a)'} stroke="var(--bg-elev)" strokeWidth={1.5}>
              <title>{`${p.name}: should say ${p.y}`}</title>
            </circle>
            {wrong && !gone && <circle cx={x(p.x[0])} cy={y(p.x[1])} r={9.5} fill="none" stroke="var(--c-bad)" strokeWidth={1.8} />}
            <text x={x(p.x[0]) + 9} y={y(p.x[1]) - 8} fontSize={11} fill="var(--text-muted)">
              {p.name.startsWith('(') ? '' : p.name}
            </text>
          </g>
        );
      })}
    </g>
  );
}

/** Points, the current line, and the region where the unit fires (shaded). Ringed points are on the wrong side. */
export function PerceptronPlot({ pts, w, focus, faded, size = 300, label }: { pts: Pt[]; w: W; focus?: number; faded?: Set<number>; size?: number; label: string }) {
  const f = useMemo(() => (a: number, b: number) => (w[0] * a + w[1] * b + w[2] >= 0 ? 1 : 0), [w[0], w[1], w[2]]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Plot x={DOMAIN} y={DOMAIN} width={size} height={size} grid tickLabels xLabel="x₁" yLabel="x₂" ariaLabel={label}>
      <Heatmap f={f} range={[0, 1]} colormap={fireColour} resolution={80} opacity={0.32} />
      <Line w={w} />
      <Points pts={pts} w={w} focus={focus} faded={faded} />
    </Plot>
  );
}

export const fmtW = (w: W) => `(${w.map((v) => fmtN(v)).join(', ')})`;
export function fmtN(v: number) {
  const r = Math.round(v * 100) / 100;
  const s = Number.isInteger(r) ? String(r) : r.toFixed(2).replace(/0$/, '');
  return s === '-0' ? '0' : s.replace('-', '−');
}
