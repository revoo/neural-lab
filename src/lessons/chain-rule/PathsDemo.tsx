import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider } from '../../components/controls';
import { Tex, col, fmt, pn } from '../../components/Tex';

// x feeds two paths that recombine: a = x + 2, b = x², y = a · b
export function PathsDemo() {
  const [x, setX] = useState(1);
  const [hl, setHl] = useState<0 | 1 | 2>(0);
  const a = x + 2;
  const b = x * x;
  const y = a * b;
  // local derivatives
  const dyda = b;
  const dydb = a;
  const dadx = 1;
  const dbdx = 2 * x;
  const p1 = dyda * dadx;
  const p2 = dydb * dbdx;
  const total = p1 + p2;
  const h = 1e-5;
  const yAt = (t: number) => (t + 2) * t * t;
  const numeric = (yAt(x + h) - yAt(x - h)) / (2 * h);

  const W = 620;
  const H = 230;
  const node = (cx: number, cy: number, label: string, val: number, on: boolean) => (
    <g>
      <circle cx={cx} cy={cy} r={34} fill="var(--bg)" stroke={on ? 'var(--c-grad)' : 'var(--border-strong)'} strokeWidth={on ? 3 : 1.5} />
      <text x={cx} y={cy - 3} textAnchor="middle" fontSize={14} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-ui)' }}>
        {label}
      </text>
      <text x={cx} y={cy + 14} textAnchor="middle" fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
        {val.toFixed(2)}
      </text>
    </g>
  );
  const edge = (x1: number, y1: number, x2: number, y2: number, label: string, on: boolean, below = false) => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={on ? 'var(--c-grad)' : 'var(--border-strong)'} strokeWidth={on ? 3.5 : 2} />
      <text
        x={(x1 + x2) / 2}
        y={(y1 + y2) / 2 + (below ? 17 : -8)}
        transform={`rotate(${(Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI}, ${(x1 + x2) / 2}, ${(y1 + y2) / 2})`}
        textAnchor="middle"
        fontSize={12}
        fill={on ? 'var(--c-grad)' : 'var(--text-muted)'}
        fontWeight={600}
        style={{ fontFamily: 'var(--font-mono)' }}
      >
        {label}
      </text>
    </g>
  );
  const on1 = hl === 1;
  const on2 = hl === 2;
  return (
    <Widget
      title="When paths branch, their contributions add"
      wide
      caption={
        <>
          Here x influences y along <strong>two routes</strong>: through a = x + 2 and through b = x². Hover or tap each path to see its contribution — the
          product of the local derivatives along it. The total sensitivity is the <strong>sum over paths</strong>, and it matches a direct numerical measurement.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 480px' }}>
          <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Computation graph with two paths">
            <g onMouseEnter={() => setHl(1)} onMouseLeave={() => setHl(0)} onClick={() => setHl(1)} style={{ cursor: 'pointer' }}>
              {edge(60, 115, 310, 45, `∂a/∂x = ${fmt(dadx)}`, on1)}
              {edge(310, 45, 560, 115, `∂y/∂a = b = ${fmt(dyda)}`, on1)}
            </g>
            <g onMouseEnter={() => setHl(2)} onMouseLeave={() => setHl(0)} onClick={() => setHl(2)} style={{ cursor: 'pointer' }}>
              {edge(60, 115, 310, 185, `∂b/∂x = 2x = ${fmt(dbdx)}`, on2, true)}
              {edge(310, 185, 560, 115, `∂y/∂b = a = ${fmt(dydb)}`, on2, true)}
            </g>
            {node(60, 115, 'x', x, hl !== 0)}
            {node(310, 45, 'a = x+2', a, on1)}
            {node(310, 185, 'b = x²', b, on2)}
            {node(560, 115, 'y = a·b', y, hl !== 0)}
          </svg>
          <div className="btn-row">
            <button className={`btn small${on1 ? ' primary' : ''}`} onClick={() => setHl(on1 ? 0 : 1)}>Path through a</button>
            <button className={`btn small${on2 ? ' primary' : ''}`} onClick={() => setHl(on2 ? 0 : 2)}>Path through b</button>
          </div>
        </div>
        <div className="widget-col controls">
          <Slider label="x" value={x} min={-3} max={2} step={0.05} onChange={setX} color="c-a" />
          <div style={{ fontSize: 14.5 }}>
            <Tex block>{`\\frac{dy}{dx} = \\underbrace{${col('c-grad', `${pn(dyda)}\\cdot${pn(dadx)}`)}}_{\\text{via } a} + \\underbrace{${col('c-grad', `${pn(dydb)}\\cdot${pn(dbdx)}`)}}_{\\text{via } b} = ${fmt(p1)} + ${pn(p2)} = ${fmt(total, 3)}`}</Tex>
          </div>
          <div className="readout">
            <span className="k">measured directly:</span> {numeric.toFixed(3)}
            <br />
            <span className="k">exact (expand y = x³ + 2x², slope 3x² + 4x):</span> {(3 * x * x + 4 * x).toFixed(3)}
          </div>
        </div>
      </div>
    </Widget>
  );
}
