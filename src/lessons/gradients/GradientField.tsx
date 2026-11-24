import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Toggle } from '../../components/controls';
import { Arrow, Dot, FnPath, Handle, Label, Plot, Seg } from '../../components/plot';
import { Heatmap } from '../../components/heatmap';
import { Tex, col, fmt } from '../../components/Tex';

type F2 = {
  label: string;
  tex: string;
  f: (x: number, y: number) => number;
  gx: (x: number, y: number) => number;
  gy: (x: number, y: number) => number;
  range: [number, number];
  scale: number;
  flat: string;
};

const g1 = (x: number, y: number) => Math.exp(-((x - 1) ** 2 + (y - 0.8) ** 2));
const g2 = (x: number, y: number) => Math.exp(-((x + 1.4) ** 2 + (y + 1) ** 2) / 1.5);

export const SURFACES: Record<string, F2> = {
  bowl: {
    label: 'Round bowl',
    tex: 'f(x, y) = x^2 + y^2',
    f: (x, y) => x * x + y * y,
    gx: (x) => 2 * x,
    gy: (_, y) => 2 * y,
    range: [0, 18],
    scale: 0.22,
    flat: 'This is the bottom of the bowl — a minimum. Every direction leads uphill.',
  },
  valley: {
    label: 'Narrow valley',
    tex: 'f(x, y) = 0.2x^2 + 2y^2',
    f: (x, y) => 0.2 * x * x + 2 * y * y,
    gx: (x) => 0.4 * x,
    gy: (_, y) => 4 * y,
    range: [0, 20],
    scale: 0.15,
    flat: 'The floor of the valley — a minimum. Notice how gentle the slope was along x compared with y.',
  },
  hills: {
    label: 'Two hills',
    tex: 'f(x,y) = 3e^{-((x-1)^2+(y-0.8)^2)} + 2e^{-((x+1.4)^2+(y+1)^2)/1.5}',
    f: (x, y) => 3 * g1(x, y) + 2 * g2(x, y),
    gx: (x, y) => 3 * g1(x, y) * -2 * (x - 1) + 2 * g2(x, y) * ((-2 * (x + 1.4)) / 1.5),
    gy: (x, y) => 3 * g1(x, y) * -2 * (y - 0.8) + 2 * g2(x, y) * ((-2 * (y + 1)) / 1.5),
    range: [0, 3.1],
    scale: 0.5,
    flat: 'A flat spot. If it’s a summit, every direction leads down; between the hills you can find a saddle-like pass, too.',
  },
  saddle: {
    label: 'Saddle',
    tex: 'f(x, y) = x^2 - y^2',
    f: (x, y) => x * x - y * y,
    gx: (x) => 2 * x,
    gy: (_, y) => -2 * y,
    range: [-9, 9],
    scale: 0.22,
    flat: 'A saddle point: the gradient is zero, yet it’s a minimum along x and a maximum along y. Like a mountain pass.',
  },
};
type Key = keyof typeof SURFACES;

export function GradientField() {
  const [key, setKey] = useState<Key>('bowl');
  const s = SURFACES[key];
  const [p, setP] = useState<[number, number]>([1.5, 1]);
  const [field, setField] = useState(false);
  const [down, setDown] = useState(false);
  const [px, py] = p;
  const gx = s.gx(px, py);
  const gy = s.gy(px, py);
  const gn = Math.hypot(gx, gy);
  const k = s.scale;
  const flat = gn < 0.08;

  const fieldArrows = [];
  if (field) {
    for (let i = -2.5; i <= 2.51; i += 0.625)
      for (let j = -2.5; j <= 2.51; j += 0.625) {
        const ax = s.gx(i, j);
        const ay = s.gy(i, j);
        const n = Math.hypot(ax, ay);
        if (n < 1e-6) continue;
        const L = 0.25 * Math.min(1, 0.3 + n / 4);
        fieldArrows.push(<Arrow key={`${i},${j}`} from={[i, j]} to={[i + (L * ax) / n, j + (L * ay) / n]} color="#fff" width={1.5} head={5} opacity={0.75} />);
      }
  }

  const sliceRange: [number, number] = [s.range[0] - (s.range[1] - s.range[0]) * 0.05, s.range[1] * 1.05];

  return (
    <Widget
      title="Partial derivatives and the gradient"
      wide
      caption={
        <>
          Colour shows height: dark is low, bright is high, and the thin lines are contours (paths of constant height). Drag the point. The{' '}
          <span className="c-c">green arrow</span> is ∂f/∂x — the slope if you only move east–west. The <span style={{ color: 'var(--c-bad)' }}>red arrow</span>{' '}
          is ∂f/∂y — the slope if you only move north–south. Together they make the <span className="c-grad">gradient</span>, which always points straight
          uphill, perpendicular to the contour lines. <strong>Challenge:</strong> find a spot where the gradient vanishes.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="Landscape" value={key} onChange={setKey} options={(Object.keys(SURFACES) as Key[]).map((k2) => ({ value: k2, label: SURFACES[k2].label }))} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={[-3, 3]} y={[-3, 3]} width={440} height={440} grid={false} ariaLabel="Heatmap of a two-input function with its gradient">
            <Heatmap f={s.f} range={s.range} contours={14} />
            {fieldArrows}
            <Arrow from={p} to={[px + k * gx, py]} color="var(--c-c)" width={3} />
            <Arrow from={[px + k * gx, py]} to={[px + k * gx, py + k * gy]} color="var(--c-bad)" width={3} />
            <Arrow from={p} to={[px + k * gx, py + k * gy]} color="var(--c-grad)" width={4} />
            {down && <Arrow from={p} to={[px - k * gx, py - k * gy]} color="#fff" width={3} dashed />}
            {down && <Label at={[px - k * gx, py - k * gy]} dx={6} dy={14} color="#fff" size={12}>−∇f (downhill)</Label>}
            <Label at={[px + k * gx, py + k * gy]} dx={6} dy={-6} color="var(--c-grad)">∇f</Label>
            <Handle x={px} y={py} onDrag={(x, y) => setP([Math.round(x * 20) / 20, Math.round(y * 20) / 20])} color="var(--c-grad)" label="point on the landscape" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <div style={{ fontSize: 13.5 }}>
            <Tex block>{s.tex}</Tex>
          </div>
          <div style={{ fontSize: 15 }}>
            <Tex block>{`f(${fmt(px)}, ${fmt(py)}) = ${fmt(s.f(px, py))}`}</Tex>
            <Tex block>{`${col('c-c', `\\frac{\\partial f}{\\partial x} = ${fmt(gx)}`)} \\qquad ${col('c-bad', `\\frac{\\partial f}{\\partial y} = ${fmt(gy)}`)}`}</Tex>
            <Tex block>{`${col('c-grad', '\\nabla f')} = \\left[${col('c-c', fmt(gx))},\\; ${col('c-bad', fmt(gy))}\\right] \\qquad \\|\\nabla f\\| = ${fmt(gn)}`}</Tex>
          </div>
          <div className="field-label">Slice along x (y held at {py.toFixed(2)})</div>
          <Plot x={[-3, 3]} y={sliceRange} width={300} height={110} tickLabels={false} ariaLabel="slice along x">
            <FnPath f={(x) => s.f(x, py)} color="var(--text-muted)" width={2} />
            <Seg from={[px - 2, s.f(px, py) - 2 * gx]} to={[px + 2, s.f(px, py) + 2 * gx]} color="var(--c-c)" width={2} />
            <Dot at={[px, s.f(px, py)]} r={5} color="var(--c-c)" />
          </Plot>
          <div className="field-label">Slice along y (x held at {px.toFixed(2)})</div>
          <Plot x={[-3, 3]} y={sliceRange} width={300} height={110} tickLabels={false} ariaLabel="slice along y">
            <FnPath f={(y) => s.f(px, y)} color="var(--text-muted)" width={2} />
            <Seg from={[py - 2, s.f(px, py) - 2 * gy]} to={[py + 2, s.f(px, py) + 2 * gy]} color="var(--c-bad)" width={2} />
            <Dot at={[py, s.f(px, py)]} r={5} color="var(--c-bad)" />
          </Plot>
          <div className="live-note">
            {flat ? (
              <>
                <strong>Gradient ≈ 0.</strong> {s.flat}
              </>
            ) : (
              <>
                Steepest uphill direction: <strong>[{(gx / gn).toFixed(2)}, {(gy / gn).toFixed(2)}]</strong>, with slope {gn.toFixed(2)}. Each partial derivative is
                the slope of one slice — the green and red lines in the small plots.
              </>
            )}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <Toggle label="Show the gradient everywhere (field)" checked={field} onChange={setField} />
            <Toggle label="Show the downhill direction, −∇f" checked={down} onChange={setDown} />
          </div>
        </div>
      </div>
    </Widget>
  );
}
