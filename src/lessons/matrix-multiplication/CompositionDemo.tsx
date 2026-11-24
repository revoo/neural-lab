import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { Plot, usePlot } from '../../components/plot';
import { Tex } from '../../components/Tex';
import { matMul, type Mat } from '../../lib/linalg';

const T: Record<string, { label: string; m: Mat }> = {
  rot: { label: 'Rotate 90°', m: [[0, -1], [1, 0]] },
  shear: { label: 'Shear →', m: [[1, 1], [0, 1]] },
  stretch: { label: 'Stretch x ×2', m: [[2, 0], [0, 1]] },
  flip: { label: 'Mirror', m: [[-1, 0], [0, 1]] },
};
type Key = keyof typeof T;

// An asymmetric "F" so that rotations and reflections are visible.
const F: [number, number][] = [
  [0, 0], [0.35, 0], [0.35, 0.8], [0.85, 0.8], [0.85, 1.1], [0.35, 1.1], [0.35, 1.4], [1.1, 1.4], [1.1, 1.75], [0, 1.75],
];

function Shape({ m, color, dashed }: { m: Mat; color: string; dashed?: boolean }) {
  const { x, y } = usePlot();
  const pts = F.map(([a, b]) => [m[0][0] * a + m[0][1] * b, m[1][0] * a + m[1][1] * b]);
  return (
    <polygon
      points={pts.map((p) => `${x(p[0])},${y(p[1])}`).join(' ')}
      fill={dashed ? 'none' : color}
      fillOpacity={0.35}
      stroke={color}
      strokeWidth={2}
      strokeDasharray={dashed ? '4 3' : undefined}
    />
  );
}

function Panel({ first, second, title }: { first: Key; second: Key; title: string }) {
  const prod = matMul(T[second].m, T[first].m); // apply `first`, then `second`
  return (
    <div style={{ flex: '1 1 240px', minWidth: 0 }}>
      <div style={{ fontWeight: 650, fontSize: 13.5, marginBottom: 4 }}>{title}</div>
      <Plot x={[-3, 3]} y={[-3, 3]} width={260} height={260} tickLabels={false} ariaLabel={title}>
        <Shape m={[[1, 0], [0, 1]]} color="var(--text-faint)" dashed />
        <Shape m={T[first].m} color="var(--c-b)" dashed />
        <Shape m={prod} color="var(--c-c)" />
      </Plot>
      <div style={{ fontSize: 14 }}>
        <Tex block>{`\\underbrace{${tex(T[second].m)}}_{\\text{${T[second].label}}}\\underbrace{${tex(T[first].m)}}_{\\text{${T[first].label}}} = ${tex(prod)}`}</Tex>
      </div>
    </div>
  );
}

const tex = (m: Mat) => `\\begin{bmatrix}${m.map((r) => r.join('&')).join('\\\\')}\\end{bmatrix}`;

export function CompositionDemo() {
  const [a, setA] = useState<Key>('rot');
  const [b, setB] = useState<Key>('shear');
  const opts = (Object.keys(T) as Key[]).map((k) => ({ value: k, label: T[k].label }));
  const same = JSON.stringify(matMul(T[a].m, T[b].m)) === JSON.stringify(matMul(T[b].m, T[a].m));
  return (
    <Widget
      title="Order matters: A then B is not B then A"
      wide
      caption={
        <>
          The grey outline is the original “F”. The dashed <span className="c-b">orange</span> F shows the result after the first transformation; the
          solid <span className="c-c">green</span> F is the final result. Matrices are written right-to-left: in <strong>BA</strong>, A acts first. Try
          to find a pair where the order <em>doesn’t</em> matter — they exist, but they’re the exception.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 12 }}>
        <Segmented label="Transformation A" options={opts} value={a} onChange={setA} />
        <Segmented label="Transformation B" options={opts} value={b} onChange={setB} />
      </div>
      <div className="widget-row">
        <Panel first={a} second={b} title="First A, then B   (the matrix BA)" />
        <Panel first={b} second={a} title="First B, then A   (the matrix AB)" />
      </div>
      <div className="live-note" style={{ marginTop: 8 }}>
        {a === b ? (
          <>Same transformation twice — of course the order can’t matter. Pick two different ones.</>
        ) : same ? (
          <>
            <strong>These two commute:</strong> AB = BA. Both results are identical. That happens for special pairs (for example, two stretches along
            the axes), but it’s not something you can assume.
          </>
        ) : (
          <>
            <strong>Different results.</strong> AB ≠ BA. Matrix multiplication composes transformations, and — just like “put on socks, then shoes” — the
            order changes the outcome.
          </>
        )}
      </div>
    </Widget>
  );
}
