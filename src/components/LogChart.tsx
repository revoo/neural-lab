import { Plot } from './plot';

export type Series = { points: [number, number][]; color: string; dashed?: boolean; dots?: boolean; label?: string };

/**
 * A small line chart with a logarithmic vertical axis, for errors that span orders of magnitude
 * (training/test error against degree, steps, strength…). Values are clamped to the y-domain.
 */
export function LogChart({
  series,
  xDomain,
  yDomain,
  marker,
  xLabel,
  width = 420,
  height = 220,
  hlines = [],
  xTicks,
  ariaLabel,
  logX = false,
  xTickFormat = (t: number) => String(t),
}: {
  series: Series[];
  xDomain: [number, number];
  yDomain: [number, number];
  marker?: number;
  xLabel: string;
  width?: number;
  height?: number;
  hlines?: { y: number; label: string; color: string }[];
  xTicks: number[];
  ariaLabel: string;
  /** Logarithmic horizontal axis too (domain and points must then be positive). */
  logX?: boolean;
  xTickFormat?: (t: number) => string;
}) {
  const tx = logX ? (v: number) => Math.log10(v) : (v: number) => v;
  const ly = (v: number) => Math.log10(Math.max(yDomain[0], Math.min(yDomain[1], v)));
  const lo = Math.log10(yDomain[0]);
  const hi = Math.log10(yDomain[1]);
  const decades: number[] = [];
  for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) decades.push(k);
  return (
    <Plot x={[tx(xDomain[0]), tx(xDomain[1])]} y={[lo, hi]} width={width} height={height} grid={false} axes={false} pad={44} ariaLabel={ariaLabel}>
      {({ x: xs, y }) => {
        const x = (v: number) => xs(tx(v));
        return (
        <g>
          {decades.map((k) => (
            <g key={k}>
              <line className="plot-grid" x1={x(xDomain[0])} x2={x(xDomain[1])} y1={y(k)} y2={y(k)} />
              <text className="plot-tick" x={x(xDomain[0]) - 5} y={y(k) + 3.5} textAnchor="end">
                {k >= 0 ? String(10 ** k) : (10 ** k).toFixed(-k)}
              </text>
            </g>
          ))}
          {xTicks.map((t, i) => (
            <text key={t} className="plot-tick" x={x(t)} y={y(lo) + 14} textAnchor={i === xTicks.length - 1 && t >= xDomain[1] ? 'end' : 'middle'}>
              {xTickFormat(t)}
            </text>
          ))}
          <line className="plot-axis" x1={x(xDomain[0])} x2={x(xDomain[1])} y1={y(lo)} y2={y(lo)} />
          <text className="plot-label" x={x(xDomain[1])} y={y(lo) - 6} textAnchor="end">
            {xLabel}
          </text>
          {hlines.map((h) => (
            <g key={h.label}>
              <line x1={x(xDomain[0])} x2={x(xDomain[1])} y1={y(ly(h.y))} y2={y(ly(h.y))} stroke={h.color} strokeDasharray="3 4" strokeWidth={1.2} />
              <text x={x(xDomain[0]) + 4} y={y(ly(h.y)) + 12} fill={h.color} fontSize={10.5} fontFamily="var(--font-ui)">
                {h.label}
              </text>
            </g>
          ))}
          {marker !== undefined && <line x1={x(marker)} x2={x(marker)} y1={y(lo)} y2={y(hi)} stroke="var(--accent)" strokeWidth={1.5} opacity={0.6} />}
          {series.map((s, i) => (
            <g key={i}>
              <polyline
                points={s.points.map(([a, b]) => `${x(a)},${y(ly(b))}`).join(' ')}
                fill="none"
                stroke={s.color}
                strokeWidth={2.2}
                strokeDasharray={s.dashed ? '6 4' : undefined}
                strokeLinejoin="round"
              />
              {s.dots && s.points.map(([a, b], j) => <circle key={j} cx={x(a)} cy={y(ly(b))} r={3} fill={s.color} />)}
            </g>
          ))}
        </g>
        );
      }}
    </Plot>
  );
}

