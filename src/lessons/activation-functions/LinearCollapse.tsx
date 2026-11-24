import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Toggle } from '../../components/controls';
import { Clip, Plot, usePlot } from '../../components/plot';
import { Tex } from '../../components/Tex';
import { matMul, matVec, type Mat } from '../../lib/linalg';

const W1: Mat = [[1.2, -0.6], [0.5, 1.0]];
const B1 = [0.3, -0.2];
const W2: Mat = [[0.9, 0.7], [-0.8, 0.6]];
const B2 = [-0.2, 0.4];
const W3: Mat = [[1.0, 0.3], [0.2, 0.9]];
const B3 = [0.1, 0];

type Pt = [number, number];
const relu = (v: number[]) => v.map((t) => Math.max(0, t));

export function network(p: Pt, useRelu: boolean): Pt {
  let h = matVec(W1, p).map((v, i) => v + B1[i]);
  if (useRelu) h = relu(h);
  h = matVec(W2, h).map((v, i) => v + B2[i]);
  if (useRelu) h = relu(h);
  h = matVec(W3, h).map((v, i) => v + B3[i]);
  return h as Pt;
}

// The single affine map equivalent to the three linear layers: M x + c
const M = matMul(W3, matMul(W2, W1));
const C = (() => {
  const c2 = matVec(W2, B1).map((v, i) => v + B2[i]);
  return matVec(W3, c2).map((v, i) => v + B3[i]);
})();
const single = (p: Pt): Pt => matVec(M, p).map((v, i) => v + C[i]) as Pt;

const LINES: Pt[][] = (() => {
  const out: Pt[][] = [];
  for (let k = -2; k <= 2.001; k += 0.5) {
    const h: Pt[] = [];
    const v: Pt[] = [];
    for (let t = -2; t <= 2.001; t += 0.05) {
      h.push([t, k]);
      v.push([k, t]);
    }
    out.push(h, v);
  }
  return out;
})();

function Lines({ map }: { map: (p: Pt) => Pt }) {
  const { x, y } = usePlot();
  return (
    <g>
      {LINES.map((line, i) => (
        <polyline
          key={i}
          points={line.map((p) => map(p)).map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ')}
          fill="none"
          stroke={i % 2 === 0 ? 'var(--c-a)' : 'var(--c-b)'}
          strokeWidth={1.6}
          opacity={0.8}
        />
      ))}
    </g>
  );
}

export function LinearCollapse() {
  const [useRelu, setRelu] = useState(false);
  const netMap = useMemo(() => (p: Pt) => network(p, useRelu), [useRelu]);
  const tex = (m: Mat) => `\\begin{bmatrix}${m.map((r) => r.map((v) => v.toFixed(2)).join('&')).join('\\\\')}\\end{bmatrix}`;
  return (
    <Widget
      title="Three layers, or one?"
      wide
      caption={
        <>
          Both panels start from the same square grid (blue horizontal lines, orange vertical lines). The left panel pushes it through a three-layer network. The right
          panel uses a <strong>single</strong> matrix, computed by multiplying the three layers’ matrices together. With activations off, they’re identical — the whole
          network is just one matrix in disguise. Now switch on ReLU.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Toggle label={<strong>ReLU between the layers</strong>} checked={useRelu} onChange={setRelu} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="field-label">Three-layer network: W₃ · f(W₂ · f(W₁x + b₁) + b₂) + b₃</div>
          <Plot x={[-3.5, 3.5]} y={[-3.5, 3.5]} width={340} height={340} tickLabels={false} ariaLabel="Grid transformed by a three-layer network">
            <Clip>
              <Lines map={netMap} />
            </Clip>
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="field-label">One matrix: M x + c, with M = W₃W₂W₁</div>
          <Plot x={[-3.5, 3.5]} y={[-3.5, 3.5]} width={340} height={340} tickLabels={false} ariaLabel="Grid transformed by a single matrix">
            <Clip>
              <Lines map={single} />
            </Clip>
          </Plot>
          <div style={{ fontSize: 13.5 }}>
            <Tex block>{`M = ${tex(M)}`}</Tex>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 8 }}>
        {useRelu ? (
          <>
            <strong>Now they differ.</strong> ReLU zeroes out negative values, which <em>folds</em> space along creases: parts of the grid get squashed flat against an axis,
            and straight lines acquire kinks. No single matrix can ever do this — matrices keep lines straight. The layers are finally doing something together that no
            one layer could do alone.
          </>
        ) : (
          <>
            <strong>Identical.</strong> Every grid line is still straight and evenly spaced, exactly as the matrices lesson promised. Three layers, twelve weights and
            six biases, and all of it collapses into one matrix and one shift.
          </>
        )}
      </div>
    </Widget>
  );
}
