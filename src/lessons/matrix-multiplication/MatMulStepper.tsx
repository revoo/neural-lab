import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Tex, col, pn } from '../../components/Tex';
import { createRng } from '../../lib/rng';
import { matMul } from '../../lib/linalg';

const randMat = (r: number, c: number, seed: number) => {
  const rng = createRng(seed);
  return Array.from({ length: r }, () => Array.from({ length: c }, () => rng.int(7) - 3));
};

function Grid({
  m,
  name,
  color,
  hlRow,
  hlCol,
  hlK,
  axis,
  onCell,
  sel,
}: {
  m: (number | null)[][];
  name: string;
  color: string;
  hlRow?: number;
  hlCol?: number;
  hlK?: number;
  axis?: 'row' | 'col';
  onCell?: (i: number, j: number) => void;
  sel?: [number, number];
}) {
  return (
    <div style={{ display: 'inline-block' }}>
      <div className="mono" style={{ fontSize: 12, color, fontWeight: 700, marginBottom: 3, textAlign: 'center' }}>
        {name} <span className="faint" style={{ fontWeight: 400 }}>({m.length}×{m[0].length})</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${m[0].length}, 40px)`, gap: 3 }}>
        {m.map((row, i) =>
          row.map((v, j) => {
            const inLine = (axis === 'row' && i === hlRow) || (axis === 'col' && j === hlCol) || (sel && sel[0] === i && sel[1] === j);
            const isK = (axis === 'row' && i === hlRow && j === hlK) || (axis === 'col' && j === hlCol && i === hlK);
            return (
              <button
                key={`${i}-${j}`}
                onClick={onCell ? () => onCell(i, j) : undefined}
                className="mono"
                style={{
                  height: 34,
                  border: `1.5px solid ${inLine ? color : 'var(--border)'}`,
                  background: isK ? color : inLine ? `color-mix(in srgb, ${color} 14%, var(--bg-elev))` : 'var(--bg)',
                  color: isK ? '#fff' : v === null ? 'var(--text-faint)' : 'var(--text)',
                  borderRadius: 6,
                  fontSize: 13.5,
                  fontWeight: inLine ? 700 : 500,
                  cursor: onCell ? 'pointer' : 'default',
                  padding: 0,
                }}
                tabIndex={onCell ? 0 : -1}
              >
                {v === null ? '·' : v}
              </button>
            );
          }),
        )}
      </div>
    </div>
  );
}

export function MatMulStepper() {
  const [dims, setDims] = useState({ n: 2, k: 3, m: 2 });
  const [seed, setSeed] = useState(4);
  const A = useMemo(() => randMat(dims.n, dims.k, seed), [dims, seed]);
  const B = useMemo(() => randMat(dims.k, dims.m, seed + 99), [dims, seed]);
  const C = useMemo(() => matMul(A, B), [A, B]);
  const [sel, setSel] = useState<[number, number]>([0, 0]);
  const [k, setK] = useState(dims.k); // number of terms accumulated for the selected cell
  const [filled, setFilled] = useState<Set<string>>(new Set(['0-0']));
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    setSel([0, 0]);
    setK(dims.k);
    setFilled(new Set(['0-0']));
  }, [dims, seed]);
  useEffect(() => () => window.clearInterval(timer.current), []);

  const [i, j] = sel;
  const terms = Array.from({ length: dims.k }, (_, t) => A[i][t] * B[t][j]);
  const partial = terms.slice(0, k).reduce((s, v) => s + v, 0);
  const Cshown = C.map((row, r) => row.map((v, c) => (r === i && c === j ? (k === dims.k ? v : partial) : filled.has(`${r}-${c}`) ? v : null)));

  const pick = (r: number, c: number) => {
    window.clearInterval(timer.current);
    setSel([r, c]);
    setK(dims.k);
    setFilled((f) => new Set(f).add(`${r}-${c}`));
  };

  const playAll = () => {
    window.clearInterval(timer.current);
    setFilled(new Set());
    let r = 0, c = 0, t = 0;
    setSel([0, 0]);
    setK(0);
    timer.current = window.setInterval(() => {
      t++;
      if (t > dims.k) {
        setFilled((f) => new Set(f).add(`${r}-${c}`));
        c++;
        if (c >= dims.m) { c = 0; r++; }
        if (r >= dims.n) { window.clearInterval(timer.current); setSel([dims.n - 1, dims.m - 1]); setK(dims.k); return; }
        t = 0;
        setSel([r, c]);
      }
      setK(t);
    }, 380);
  };

  const texTerms = terms.map((_, t) => {
    const s = `${col('c-a', pn(A[i][t], 0))}\\cdot${col('c-b', pn(B[t][j], 0))}`;
    return t < k ? s : `\\htmlStyle{opacity:0.25}{${s}}`;
  });

  return (
    <Widget
      title="Matrix multiplication, one dot product at a time"
      wide
      caption={
        <>
          Click any cell of <span className="c-c">C</span> to see where it comes from: <strong>row i of A</strong> meets <strong>column j of B</strong>,
          and their dot product lands at position (i, j). The layout lines everything up — A’s rows sit beside C’s rows, B’s columns sit above C’s columns.
          Press <strong>Play all</strong> to watch the triple loop run. Change the shapes and notice which dimension has to agree.
        </>
      }
    >
      <div className="widget-row" style={{ alignItems: 'flex-start' }}>
        <div className="widget-col" style={{ flex: '0 1 auto', overflowX: 'auto' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: 16, alignItems: 'end' }}>
            <div />
            <Grid m={B} name="B" color="var(--c-b)" hlCol={j} hlK={k > 0 && k <= dims.k ? k - 1 : undefined} axis="col" />
            <Grid m={A} name="A" color="var(--c-a)" hlRow={i} hlK={k > 0 && k <= dims.k ? k - 1 : undefined} axis="row" />
            <Grid m={Cshown} name="C = A @ B" color="var(--c-c)" onCell={pick} sel={sel} />
          </div>
        </div>
        <div className="widget-col controls">
          <div className="controls inline">
            {(['n', 'k', 'm'] as const).map((d) => (
              <label key={d} style={{ fontSize: 13 }}>
                <div className="field-label">{d === 'n' ? 'rows of A' : d === 'k' ? 'shared (inner) dim' : 'cols of B'}</div>
                <select
                  value={dims[d]}
                  onChange={(e) => setDims({ ...dims, [d]: Number(e.target.value) })}
                  style={{ padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border-strong)', background: 'var(--bg)', color: 'var(--text)' }}
                >
                  {[1, 2, 3, 4].map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <div className="btn-row">
            <button className="btn small primary" onClick={playAll}>▶ Play all</button>
            <button className="btn small" onClick={() => setK(Math.max(0, k - 1))} disabled={k === 0}>← term</button>
            <button className="btn small" onClick={() => setK(Math.min(dims.k, k + 1))} disabled={k === dims.k}>term →</button>
            <button className="btn small" onClick={() => setSeed(seed + 1)}>New numbers</button>
          </div>
          <div style={{ fontSize: 15 }}>
            <Tex block>{`C_{${i + 1}${j + 1}} = ${texTerms.join(' + ')} = ${col('c-c', String(partial))}`}</Tex>
          </div>
          <div className="mono" style={{ fontSize: 12.5, background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', lineHeight: 1.7, whiteSpace: 'pre', overflowX: 'auto' }}>
            <div>for i in range({dims.n}):        <span className="faint"># row of A    → i = {i}</span></div>
            <div>  for j in range({dims.m}):      <span className="faint"># column of B → j = {j}</span></div>
            <div>    C[i][j] = 0</div>
            <div>    for k in range({dims.k}):    <span className="faint"># walk along both → k = {Math.max(0, k - 1)}</span></div>
            <div style={{ background: 'var(--accent-soft)' }}>      C[i][j] += A[i][k] * B[k][j]</div>
          </div>
          <div className="muted" style={{ fontSize: 13 }}>
            Shapes: ({dims.n}×<strong className="c-grad">{dims.k}</strong>) @ (<strong className="c-grad">{dims.k}</strong>×{dims.m}) → ({dims.n}×{dims.m}). That’s{' '}
            {dims.n * dims.m} dot products of length {dims.k}: <strong>{dims.n * dims.m * dims.k} multiply-adds</strong>.
          </div>
        </div>
      </div>
    </Widget>
  );
}
