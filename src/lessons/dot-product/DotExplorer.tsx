import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Toggle } from '../../components/controls';
import { Arrow, Handle, Label, Plot, Seg, usePlot } from '../../components/plot';
import { Tex, col, fmt, pn } from '../../components/Tex';

type V = [number, number];

function HalfPlane({ a, show }: { a: V; show: boolean }) {
  const { x, y } = usePlot();
  const n = Math.hypot(a[0], a[1]);
  if (!show || n < 1e-6) return null;
  const u: V = [a[0] / n, a[1] / n];
  const p: V = [-u[1], u[0]];
  const L = 30;
  const poly = (sign: number) =>
    [
      [L * p[0], L * p[1]],
      [L * p[0] + sign * L * u[0], L * p[1] + sign * L * u[1]],
      [-L * p[0] + sign * L * u[0], -L * p[1] + sign * L * u[1]],
      [-L * p[0], -L * p[1]],
    ]
      .map(([px, py]) => `${x(px)},${y(py)}`)
      .join(' ');
  const [x0, x1] = x.domain;
  const [y0, y1] = y.domain;
  return (
    <g>
      <defs>
        <clipPath id="dot-clip">
          <rect x={x(x0)} y={y(y1)} width={x(x1) - x(x0)} height={y(y0) - y(y1)} />
        </clipPath>
      </defs>
      <g clipPath="url(#dot-clip)">
        <polygon points={poly(1)} fill="var(--c-a)" opacity={0.08} />
        <polygon points={poly(-1)} fill="var(--c-b)" opacity={0.08} />
        <line x1={x(L * p[0])} y1={y(L * p[1])} x2={x(-L * p[0])} y2={y(-L * p[1])} stroke="var(--text-faint)" strokeDasharray="4 4" />
      </g>
    </g>
  );
}

function AngleArc({ a, b }: { a: V; b: V }) {
  const { x, y } = usePlot();
  const r = 0.9;
  const ta = Math.atan2(a[1], a[0]);
  let tb = Math.atan2(b[1], b[0]);
  let d = tb - ta;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  tb = ta + d;
  const pts: string[] = [];
  for (let i = 0; i <= 30; i++) {
    const t = ta + (d * i) / 30;
    pts.push(`${x(r * Math.cos(t))},${y(r * Math.sin(t))}`);
  }
  const mid = ta + d / 2;
  return (
    <g>
      <polyline points={pts.join(' ')} fill="none" stroke="var(--text-muted)" strokeWidth={1.5} />
      <text x={x(1.35 * Math.cos(mid))} y={y(1.35 * Math.sin(mid)) + 4} fontSize={12} fill="var(--text-muted)" textAnchor="middle" style={{ fontFamily: 'var(--font-ui)' }}>
        θ
      </text>
    </g>
  );
}

export function DotExplorer() {
  const [a, setA] = useState<V>([3, 1]);
  const [b, setB] = useState<V>([1.5, 3]);
  const [showHalf, setShowHalf] = useState(true);
  const [showProj, setShowProj] = useState(true);
  const [unit, setUnit] = useState(false);

  const la = Math.hypot(...a);
  const lb = Math.hypot(...b);
  const d = a[0] * b[0] + a[1] * b[1];
  const cos = la && lb ? d / (la * lb) : 0;
  const theta = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
  const proj = la ? d / la : 0;
  const P: V = la ? [(a[0] / la) * proj, (a[1] / la) * proj] : [0, 0];
  const ua: V = la ? [a[0] / la, a[1] / la] : [0, 0];
  const ub: V = lb ? [b[0] / lb, b[1] / lb] : [0, 0];

  let verdict: React.ReactNode;
  if (Math.abs(d) < 0.05)
    verdict = (
      <>
        <strong>Zero.</strong> The vectors are perpendicular (θ = 90°). Moving along <span className="c-b">b</span> gets you no closer to, or further
        from, the direction of <span className="c-a">a</span>. They’re completely uncorrelated.
      </>
    );
  else if (d > 0)
    verdict = (
      <>
        <strong>Positive.</strong> The vectors broadly agree (θ &lt; 90°). <span className="c-b">b</span>’s shadow on <span className="c-a">a</span>’s
        line points the same way as <span className="c-a">a</span>.
        {cos > 0.98 && ' They point almost exactly the same way, so the dot product is as large as it can be for these lengths.'}
      </>
    );
  else
    verdict = (
      <>
        <strong>Negative.</strong> The vectors disagree (θ &gt; 90°). <span className="c-b">b</span>’s shadow falls <em>behind</em> the origin,
        pointing against <span className="c-a">a</span>.
      </>
    );

  return (
    <Widget
      title="Dot product explorer"
      wide
      caption={
        <>
          Drag the tips of <span className="c-a">a</span> and <span className="c-b">b</span>. The shaded regions show every place b could be: in the{' '}
          <span className="c-a">blue half</span>, a·b is positive; in the <span className="c-b">orange half</span>, it’s negative; on the dashed line,
          exactly zero. <strong>Challenges:</strong> make a·b exactly 0 without either vector being zero. Then make it negative. Then find the
          largest value you can without making b any longer.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <Plot x={[-5, 5]} y={[-5, 5]} width={440} height={440} ariaLabel="Two draggable vectors with their dot product">
            <HalfPlane a={a} show={showHalf} />
            {unit && (
              <>
                <UnitCircle />
                <Arrow from={[0, 0]} to={ua} color="var(--c-a)" width={4} />
                <Arrow from={[0, 0]} to={ub} color="var(--c-b)" width={4} />
              </>
            )}
            {showProj && la > 0 && (
              <>
                <Seg from={b} to={P} dash="3 3" />
                <Seg from={[0, 0]} to={P} color="var(--c-c)" width={7} opacity={0.55} />
              </>
            )}
            <AngleArc a={a} b={b} />
            <Arrow from={[0, 0]} to={a} color="var(--c-a)" width={3} />
            <Arrow from={[0, 0]} to={b} color="var(--c-b)" width={3} />
            <Label at={a} dx={10} dy={4} color="var(--c-a)">a</Label>
            <Label at={b} dx={10} dy={4} color="var(--c-b)">b</Label>
            <Handle x={a[0]} y={a[1]} onDrag={(x, y) => setA([x, y])} color="var(--c-a)" snap={0.25} label="tip of a" />
            <Handle x={b[0]} y={b[1]} onDrag={(x, y) => setB([x, y])} color="var(--c-b)" snap={0.25} label="tip of b" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <div style={{ fontSize: 15 }}>
            <Tex block>{`\\mathbf{a}\\cdot\\mathbf{b} = ${col('c-a', pn(a[0]))}\\times${col('c-b', pn(b[0]))} + ${col('c-a', pn(a[1]))}\\times${col('c-b', pn(b[1]))} = ${col('c-c', fmt(d))}`}</Tex>
            <Tex block>{`\\|\\mathbf{a}\\|\\,\\|\\mathbf{b}\\|\\cos\\theta = ${fmt(la)} \\times ${fmt(lb)} \\times ${fmt(cos, 3)} = ${col('c-c', fmt(la * lb * cos))}`}</Tex>
          </div>
          <div className="readout">
            <span className="k">angle θ          </span> {theta.toFixed(1)}°<br />
            <span className="k">cos θ (cosine similarity)</span> {cos.toFixed(3)}<br />
            <span className="k">shadow of b on a </span> <span className="c-c">{proj.toFixed(2)}</span> = a·b / |a|
          </div>
          <div className="live-note">{verdict}</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <Toggle label="Shade positive / negative regions" checked={showHalf} onChange={setShowHalf} />
            <Toggle label="Show b’s shadow (projection) on a" checked={showProj} onChange={setShowProj} />
            <Toggle label="Show unit vectors (cosine similarity view)" checked={unit} onChange={setUnit} />
          </div>
        </div>
      </div>
    </Widget>
  );
}

function UnitCircle() {
  const { x, y } = usePlot();
  return <circle cx={x(0)} cy={y(0)} r={x(1) - x(0)} fill="none" stroke="var(--text-faint)" strokeDasharray="3 3" />;
}
