import { useEffect, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider } from '../../components/controls';
import { Arrow, Handle, Label, Plot, Seg, usePlot } from '../../components/plot';
import { Tex, col, fmt } from '../../components/Tex';

type M2 = [[number, number], [number, number]];
type V = [number, number];

const I: M2 = [
  [1, 0],
  [0, 1],
];
const apply = (m: M2, v: V): V => [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]];
const lerp = (a: M2, b: M2, t: number): M2 => [
  [a[0][0] + (b[0][0] - a[0][0]) * t, a[0][1] + (b[0][1] - a[0][1]) * t],
  [a[1][0] + (b[1][0] - a[1][0]) * t, a[1][1] + (b[1][1] - a[1][1]) * t],
];
const det = (m: M2) => m[0][0] * m[1][1] - m[0][1] * m[1][0];
const r2 = (x: number) => Math.round(x * 1000) / 1000;
const rot = (deg: number): M2 => {
  const t = (deg * Math.PI) / 180;
  return [
    [r2(Math.cos(t)), r2(-Math.sin(t))],
    [r2(Math.sin(t)), r2(Math.cos(t))],
  ];
};

const PRESETS: { name: string; m: M2 }[] = [
  { name: 'Identity', m: I },
  { name: 'Stretch', m: [[2, 0], [0, 0.5]] },
  { name: 'Rotate 30°', m: rot(30) },
  { name: 'Shear', m: [[1, 1], [0, 1]] },
  { name: 'Reflect', m: [[-1, 0], [0, 1]] },
  { name: 'Squash flat', m: [[1, 2], [0.5, 1]] },
];

const CHALLENGES: { name: string; goal: string; m: M2; done: string }[] = [
  { name: 'Rotate 90°', goal: 'Rotate the whole plane a quarter-turn anticlockwise.', m: [[0, -1], [1, 0]], done: 'î now points up and ĵ points left. Read the matrix column by column: first column [0, 1] is where î landed, second column [−1, 0] is where ĵ landed.' },
  { name: 'Mirror', goal: 'Flip the plane left-to-right, like a mirror along the vertical axis.', m: [[-1, 0], [0, 1]], done: 'Notice the determinant is −1: area is unchanged, but the square has been flipped over (its orientation reversed), which is why it turned orange.' },
  { name: 'Double', goal: 'Make everything twice as big in every direction.', m: [[2, 0], [0, 2]], done: 'Every length doubles, so every area quadruples — and the determinant is 4. That’s no coincidence: the determinant measures area scaling.' },
  { name: 'Flatten', goal: 'Squash the entire plane onto the horizontal axis.', m: [[1, 0], [0, 0]], done: 'Determinant 0: the whole plane now lies on a line. Every point above and below the axis got squashed onto it, so there is no way to undo this — the vertical information is gone.' },
];

function Grid({ m, color, width = 1, opacity = 1 }: { m: M2; color: string; width?: number; opacity?: number }) {
  const lines = [];
  for (let k = -10; k <= 10; k++) {
    lines.push(<Seg key={'v' + k} from={apply(m, [k, -10])} to={apply(m, [k, 10])} color={color} width={k === 0 ? width * 2 : width} opacity={opacity} cap="butt" />);
    lines.push(<Seg key={'h' + k} from={apply(m, [-10, k])} to={apply(m, [10, k])} color={color} width={k === 0 ? width * 2 : width} opacity={opacity} cap="butt" />);
  }
  return <g>{lines}</g>;
}

function UnitSquare({ m, dashed = false }: { m: M2; dashed?: boolean }) {
  const { x, y } = usePlot();
  const pts = [[0, 0], apply(m, [1, 0]), apply(m, [1, 1]), apply(m, [0, 1])] as V[];
  const flipped = det(m) < 0;
  return (
    <polygon
      points={pts.map((p) => `${x(p[0])},${y(p[1])}`).join(' ')}
      fill={dashed ? 'none' : flipped ? 'var(--c-b)' : 'var(--accent)'}
      fillOpacity={dashed ? 0 : 0.22}
      stroke={dashed ? 'var(--text)' : 'none'}
      strokeDasharray={dashed ? '5 4' : undefined}
      strokeWidth={dashed ? 2 : 0}
    />
  );
}

function Clip({ children }: { children: React.ReactNode }) {
  const { x, y } = usePlot();
  const [x0, x1] = x.domain;
  const [y0, y1] = y.domain;
  return (
    <g>
      <defs>
        <clipPath id="grid-clip">
          <rect x={x(x0)} y={y(y1)} width={x(x1) - x(x0)} height={y(y0) - y(y1)} />
        </clipPath>
      </defs>
      <g clipPath="url(#grid-clip)">{children}</g>
    </g>
  );
}

export function GridTransformer() {
  const [m, setM] = useState<M2>([[1, 1], [0, 1]]);
  const [t, setT] = useState(1);
  const [v, setV] = useState<V>([1.5, 1]);
  const [challenge, setChallenge] = useState<number | null>(null);
  const raf = useRef(0);

  const animate = () => {
    cancelAnimationFrame(raf.current);
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / 1200);
      setT(0.5 - 0.5 * Math.cos(Math.PI * p));
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  };
  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const mt = lerp(I, m, t);
  const i1 = apply(mt, [1, 0]);
  const j1 = apply(mt, [0, 1]);
  const mv = apply(mt, v);
  const xi: V = [v[0] * i1[0], v[0] * i1[1]];
  const d = det(m);
  const ch = challenge !== null ? CHALLENGES[challenge] : null;
  const solved = ch && Math.abs(m[0][0] - ch.m[0][0]) + Math.abs(m[0][1] - ch.m[0][1]) + Math.abs(m[1][0] - ch.m[1][0]) + Math.abs(m[1][1] - ch.m[1][1]) < 0.05;

  const setCol = (c: 0 | 1, p: V) => {
    setT(1);
    const next: M2 = [[...m[0]], [...m[1]]] as M2;
    next[0][c] = p[0];
    next[1][c] = p[1];
    setM(next);
  };
  const setEntry = (r: 0 | 1, c: 0 | 1, val: number) => {
    setT(1);
    const next: M2 = [[...m[0]], [...m[1]]] as M2;
    next[r][c] = val;
    setM(next);
  };

  return (
    <Widget
      title="Matrix transformer"
      wide
      caption={
        <>
          The faint grid is the original space; the bold grid is where every point ends up after the transformation. Drag the tips of{' '}
          <span className="c-c">î</span> and <span style={{ color: 'var(--c-bad)' }}>ĵ</span> directly — the matrix’s columns follow them. The shaded
          square is where the original 1×1 square lands, and its area is the determinant. <strong>Try the challenges</strong>: each shows a dashed
          target outline to match.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={[-4, 4]} y={[-4, 4]} width={460} height={460} grid={false} axes={false} ariaLabel="Grid transformed by a 2 by 2 matrix">
            <Clip>
              <Grid m={I} color="var(--c-grid)" />
              <Grid m={mt} color="var(--accent)" width={1} opacity={0.45} />
              <UnitSquare m={mt} />
              {ch && <UnitSquare m={ch.m} dashed />}
            </Clip>
            <Arrow from={[0, 0]} to={xi} color="var(--c-c)" dashed width={2} opacity={0.6} />
            <Arrow from={xi} to={mv} color="var(--c-bad)" dashed width={2} opacity={0.6} />
            <Arrow from={[0, 0]} to={mv} color="var(--c-a)" width={3} />
            <Label at={mv} dx={8} dy={-6} color="var(--c-a)">Mv</Label>
            <Arrow from={[0, 0]} to={i1} color="var(--c-c)" width={3.5} />
            <Arrow from={[0, 0]} to={j1} color="var(--c-bad)" width={3.5} />
            <Label at={i1} dx={10} dy={14} color="var(--c-c)">î</Label>
            <Label at={j1} dx={10} dy={-4} color="var(--c-bad)">ĵ</Label>
            <Handle x={i1[0]} y={i1[1]} onDrag={(x, y) => setCol(0, [x, y])} color="var(--c-c)" snap={0.25} label="where î lands (first column)" />
            <Handle x={j1[0]} y={j1[1]} onDrag={(x, y) => setCol(1, [x, y])} color="var(--c-bad)" snap={0.25} label="where ĵ lands (second column)" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <div>
            <div className="field-label">The matrix M</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 44, fontWeight: 200, color: 'var(--text-faint)' }}>[</span>
              <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: 6 }}>
                {([0, 1] as const).map((r) =>
                  ([0, 1] as const).map((c) => (
                    <input
                      key={`${r}${c}`}
                      className="num-input"
                      type="number"
                      step={0.25}
                      value={m[r][c]}
                      onChange={(e) => !Number.isNaN(Number(e.target.value)) && setEntry(r, c, Number(e.target.value))}
                      style={{ color: c === 0 ? 'var(--c-c)' : 'var(--c-bad)', fontWeight: 600 }}
                      aria-label={`row ${r + 1}, column ${c + 1}`}
                    />
                  )),
                )}
              </div>
              <span style={{ fontSize: 44, fontWeight: 200, color: 'var(--text-faint)' }}>]</span>
              <div className="muted" style={{ fontSize: 12.5, lineHeight: 1.4 }}>
                <span className="c-c">column 1 = where î lands</span>
                <br />
                <span style={{ color: 'var(--c-bad)' }}>column 2 = where ĵ lands</span>
              </div>
            </div>
          </div>
          <div className="btn-row">
            {PRESETS.map((p) => (
              <button key={p.name} className="btn small" onClick={() => { setM(p.m); animate(); }}>
                {p.name}
              </button>
            ))}
          </div>
          <div className="btn-row">
            <button className="btn small primary" onClick={animate}>▶ Animate from identity</button>
          </div>
          <Slider label="Transformation progress" value={t} min={0} max={1} step={0.01} onChange={setT} format={(x) => `${Math.round(x * 100)}%`} />
          <div className="controls inline">
            <Slider label={<>input v, x-part</>} value={v[0]} min={-2} max={2} step={0.25} onChange={(x) => setV([x, v[1]])} color="c-c" format={(x) => x.toFixed(2)} />
            <Slider label={<>input v, y-part</>} value={v[1]} min={-2} max={2} step={0.25} onChange={(y) => setV([v[0], y])} color="c-bad" format={(x) => x.toFixed(2)} />
          </div>
          <div style={{ fontSize: 14 }}>
            <Tex block>{`M\\mathbf{v} = ${col('c-c', fmt(v[0]))}\\begin{bmatrix}${col('c-c', fmt(mt[0][0]))}\\\\${col('c-c', fmt(mt[1][0]))}\\end{bmatrix} + ${col('c-bad', fmt(v[1]))}\\begin{bmatrix}${col('c-bad', fmt(mt[0][1]))}\\\\${col('c-bad', fmt(mt[1][1]))}\\end{bmatrix} = ${col('c-a', `\\begin{bmatrix}${fmt(mv[0])}\\\\${fmt(mv[1])}\\end{bmatrix}`)}`}</Tex>
          </div>
          <div className="readout">
            <span className="k">determinant</span> = {fmt(m[0][0])}×{fmt(m[1][1])} − {fmt(m[0][1])}×{fmt(m[1][0])} = <strong>{fmt(d)}</strong>
            <br />
            <span className="faint">
              {Math.abs(d) < 1e-9
                ? 'Space is squashed flat: this cannot be undone.'
                : d < 0
                  ? `Areas scale by ${fmt(Math.abs(d))}× and space is flipped over.`
                  : `Areas scale by ${fmt(d)}×.`}
            </span>
          </div>
          <div>
            <div className="field-label">Challenges</div>
            <div className="btn-row">
              {CHALLENGES.map((c, k) => (
                <button key={c.name} className={`btn small${challenge === k ? ' primary' : ''}`} onClick={() => setChallenge(challenge === k ? null : k)}>
                  {c.name}
                </button>
              ))}
            </div>
            {ch && (
              <div className="live-note" style={{ marginTop: 8 }}>
                {solved ? (
                  <>
                    <strong>Solved!</strong> {ch.done}
                  </>
                ) : (
                  <>
                    <strong>Goal:</strong> {ch.goal} Match the dashed outline.
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}
