import { createContext, Fragment, useId, useContext, useRef, type ReactNode, type PointerEvent as RPointerEvent } from 'react';

export type Scale = ((v: number) => number) & { inv: (px: number) => number; domain: [number, number] };

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = (r1 - r0) / (d1 - d0);
  const s = ((v: number) => r0 + (v - d0) * k) as Scale;
  s.inv = (px: number) => d0 + (px - r0) / k;
  s.domain = domain;
  return s;
}

export function ticks(lo: number, hi: number, approx = 8): number[] {
  const span = hi - lo;
  const raw = span / approx;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

type PlotChild = ReactNode | ((s: PlotCtx) => ReactNode);
export type PlotCtx = { x: Scale; y: Scale; svg: React.RefObject<SVGSVGElement | null> };
const Ctx = createContext<PlotCtx | null>(null);
export const usePlot = () => {
  const c = useContext(Ctx);
  if (!c) throw new Error('usePlot outside <Plot>');
  return c;
};

/**
 * A 2D Cartesian plot in SVG. Children are rendered inside and can read the scales via usePlot()
 * or via the render-prop form.
 */
export function Plot({
  width = 420,
  height = 420,
  x: xDomain,
  y: yDomain,
  grid = true,
  axes = true,
  tickLabels = true,
  xLabel,
  yLabel,
  pad = 28,
  children,
  onPointerDown,
  ariaLabel,
  style,
}: {
  width?: number;
  height?: number;
  x: [number, number];
  y: [number, number];
  grid?: boolean;
  axes?: boolean;
  tickLabels?: boolean;
  xLabel?: string;
  yLabel?: string;
  pad?: number;
  children?: PlotChild | PlotChild[];
  onPointerDown?: (p: { x: number; y: number }, e: RPointerEvent<SVGSVGElement>) => void;
  ariaLabel?: string;
  style?: React.CSSProperties;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const x = linear(xDomain, [pad, width - 8]);
  const y = linear(yDomain, [height - pad, 8]);
  const ctx: PlotCtx = { x, y, svg };
  const xt = ticks(xDomain[0], xDomain[1], width / 55);
  const yt = ticks(yDomain[0], yDomain[1], height / 45);
  const x0 = Math.min(Math.max(0, xDomain[0]), xDomain[1]);
  const y0 = Math.min(Math.max(0, yDomain[0]), yDomain[1]);
  return (
    <Ctx.Provider value={ctx}>
      <svg
        ref={svg}
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label={ariaLabel}
        style={style}
        onPointerDown={
          onPointerDown
            ? (e) => {
                const p = svgPoint(svg.current!, e.clientX, e.clientY);
                onPointerDown({ x: x.inv(p.x), y: y.inv(p.y) }, e);
              }
            : undefined
        }
      >
        {grid && (
          <g>
            {xt.map((t) => <line key={'gx' + t} className="plot-grid" x1={x(t)} x2={x(t)} y1={y(yDomain[0])} y2={y(yDomain[1])} />)}
            {yt.map((t) => <line key={'gy' + t} className="plot-grid" y1={y(t)} y2={y(t)} x1={x(xDomain[0])} x2={x(xDomain[1])} />)}
          </g>
        )}
        {axes && (
          <g>
            <line className="plot-axis" x1={x(xDomain[0])} x2={x(xDomain[1])} y1={y(y0)} y2={y(y0)} />
            <line className="plot-axis" x1={x(x0)} x2={x(x0)} y1={y(yDomain[0])} y2={y(yDomain[1])} />
            {tickLabels &&
              xt.filter((t) => t !== x0).map((t) => (
                <text key={'tx' + t} className="plot-tick" x={x(t)} y={Math.min(y(y0) + 13, height - pad + 13)} textAnchor="middle">{fmtTick(t)}</text>
              ))}
            {tickLabels &&
              yt.filter((t) => t !== y0).map((t) => (
                <text key={'ty' + t} className="plot-tick" x={Math.max(x(x0) - 5, pad - 5)} y={y(t) + 3.5} textAnchor="end">{fmtTick(t)}</text>
              ))}
            {xLabel && <text className="plot-label" x={width - 10} y={y(y0) - 6} textAnchor="end">{xLabel}</text>}
            {yLabel && <text className="plot-label" x={x(x0) + 6} y={16}>{yLabel}</text>}
          </g>
        )}
        {(Array.isArray(children) ? children : [children]).map((c, i) => (
          <Fragment key={i}>{typeof c === 'function' ? c(ctx) : c}</Fragment>
        ))}
      </svg>
    </Ctx.Provider>
  );
}

const fmtTick = (t: number) => (Math.abs(t) < 1e-9 ? '0' : Number.isInteger(t) ? String(t) : t.toFixed(Math.abs(t) < 1 ? 2 : 1).replace(/0$/, ''));

export function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number) {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const m = svg.getScreenCTM();
  return m ? pt.matrixTransform(m.inverse()) : pt;
}

/** A draggable point inside a <Plot>, reporting positions in data coordinates. */
export function Handle({
  x,
  y,
  onDrag,
  color = 'var(--accent)',
  r = 7,
  label,
  snap,
}: {
  x: number;
  y: number;
  onDrag: (x: number, y: number) => void;
  color?: string;
  r?: number;
  label?: string;
  snap?: number;
}) {
  const { x: sx, y: sy, svg } = usePlot();
  const drag = (e: RPointerEvent) => {
    const p = svgPoint(svg.current!, e.clientX, e.clientY);
    let nx = sx.inv(p.x);
    let ny = sy.inv(p.y);
    const [x0, x1] = sx.domain;
    const [y0, y1] = sy.domain;
    nx = Math.min(Math.max(nx, Math.min(x0, x1)), Math.max(x0, x1));
    ny = Math.min(Math.max(ny, Math.min(y0, y1)), Math.max(y0, y1));
    if (snap) {
      nx = Math.round(nx / snap) * snap;
      ny = Math.round(ny / snap) * snap;
    }
    onDrag(nx, ny);
  };
  return (
    <g
      className="handle"
      tabIndex={0}
      role="slider"
      aria-label={label ?? 'draggable point'}
      aria-valuetext={`(${x.toFixed(2)}, ${y.toFixed(2)})`}
      onPointerDown={(e) => {
        e.stopPropagation();
        (e.currentTarget as Element).setPointerCapture(e.pointerId);
        drag(e);
      }}
      onPointerMove={(e) => {
        if ((e.currentTarget as Element).hasPointerCapture(e.pointerId)) drag(e);
      }}
      onKeyDown={(e) => {
        const st = snap ?? (sx.domain[1] - sx.domain[0]) / 40;
        const d: Record<string, [number, number]> = { ArrowLeft: [-st, 0], ArrowRight: [st, 0], ArrowUp: [0, st], ArrowDown: [0, -st] };
        if (d[e.key]) {
          e.preventDefault();
          onDrag(x + d[e.key][0], y + d[e.key][1]);
        }
      }}
    >
      <circle cx={sx(x)} cy={sy(y)} r={r + 10} fill="transparent" />
      <circle className="handle-ring" cx={sx(x)} cy={sy(y)} r={r} stroke={color} />
      <circle cx={sx(x)} cy={sy(y)} r={r - 4} fill={color} />
    </g>
  );
}

/** Arrow from (x1,y1) to (x2,y2) in data coordinates. */
export function Arrow({
  from,
  to,
  color,
  width = 2.5,
  dashed = false,
  head = 10,
  opacity = 1,
}: {
  from: [number, number];
  to: [number, number];
  color: string;
  width?: number;
  dashed?: boolean;
  head?: number;
  opacity?: number;
}) {
  const { x, y } = usePlot();
  const x1 = x(from[0]), y1 = y(from[1]), x2 = x(to[0]), y2 = y(to[1]);
  const len = Math.hypot(x2 - x1, y2 - y1);
  if (len < 0.5) return null;
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
  const h = Math.min(head, len * 0.6);
  const bx = x2 - ux * h, by = y2 - uy * h;
  const px = -uy * h * 0.45, py = ux * h * 0.45;
  return (
    <g opacity={opacity}>
      <line x1={x1} y1={y1} x2={bx} y2={by} stroke={color} strokeWidth={width} strokeDasharray={dashed ? '5 4' : undefined} strokeLinecap="round" />
      <polygon points={`${x2},${y2} ${bx + px},${by + py} ${bx - px},${by - py}`} fill={color} />
    </g>
  );
}

/** Polyline of a function y = f(x) across the plot's x-domain. */
export function FnPath({ f, color, width = 2.5, samples = 300, dashed = false, domain }: { f: (x: number) => number; color: string; width?: number; samples?: number; dashed?: boolean; domain?: [number, number] }) {
  const { x, y } = usePlot();
  const [a, b] = domain ?? x.domain;
  const [ylo, yhi] = [Math.min(...y.domain), Math.max(...y.domain)];
  const span = yhi - ylo;
  let d = '';
  let pen = false;
  for (let i = 0; i <= samples; i++) {
    const xv = a + ((b - a) * i) / samples;
    const yv = f(xv);
    if (!Number.isFinite(yv) || yv < ylo - span * 2 || yv > yhi + span * 2) {
      pen = false;
      continue;
    }
    d += `${pen ? 'L' : 'M'}${x(xv).toFixed(2)},${y(yv).toFixed(2)}`;
    pen = true;
  }
  return <path d={d} fill="none" stroke={color} strokeWidth={width} strokeDasharray={dashed ? '6 5' : undefined} strokeLinejoin="round" />;
}

export function Label({ at, children, color = 'var(--text)', dx = 0, dy = 0, anchor = 'start', size = 13, weight = 600 }: { at: [number, number]; children: ReactNode; color?: string; dx?: number; dy?: number; anchor?: 'start' | 'middle' | 'end'; size?: number; weight?: number }) {
  const { x, y } = usePlot();
  return (
    <text x={x(at[0]) + dx} y={y(at[1]) + dy} fill={color} fontSize={size} fontWeight={weight} textAnchor={anchor} style={{ fontFamily: 'var(--font-ui)', paintOrder: 'stroke', stroke: 'var(--bg-elev)', strokeWidth: 3 }}>
      {children}
    </text>
  );
}

/** Line segment between two points in data coordinates. */
export function Seg({ from, to, color = 'var(--text-faint)', width = 1.5, dash, opacity = 1, cap = 'round' }: { from: [number, number]; to: [number, number]; color?: string; width?: number; dash?: string; opacity?: number; cap?: 'round' | 'butt' }) {
  const { x, y } = usePlot();
  return <line x1={x(from[0])} y1={y(from[1])} x2={x(to[0])} y2={y(to[1])} stroke={color} strokeWidth={width} strokeDasharray={dash} opacity={opacity} strokeLinecap={cap} />;
}

/** A small filled point in data coordinates. */
export function Dot({ at, r = 4, color = 'var(--text)', stroke, opacity = 1 }: { at: [number, number]; r?: number; color?: string; stroke?: string; opacity?: number }) {
  const { x, y } = usePlot();
  return <circle cx={x(at[0])} cy={y(at[1])} r={r} fill={color} stroke={stroke} strokeWidth={stroke ? 1.5 : 0} opacity={opacity} />;
}

/** Clips children to the plot's data area. */
export function Clip({ children }: { children: ReactNode }) {
  const { x, y } = usePlot();
  const id = useId();
  const [x0, x1] = x.domain;
  const [y0, y1] = y.domain;
  return (
    <g>
      <defs>
        <clipPath id={id}>
          <rect x={x(x0)} y={y(y1)} width={x(x1) - x(x0)} height={y(y0) - y(y1)} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${id})`}>{children}</g>
    </g>
  );
}
