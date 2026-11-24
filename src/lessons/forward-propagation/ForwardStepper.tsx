import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Clip, Handle, Plot, Seg, usePlot } from '../../components/plot';
import { Heatmap, sequential } from '../../components/heatmap';
import { NetDiagram, type NetSelection } from '../../components/NetDiagram';
import { Tex } from '../../components/Tex';
import { LAYERS, forward, INPUT_NAMES } from './islands';

/** Trim a number for display inside a matrix: 4 → "4", 3.2 → "3.2", 0.9608 → "0.96". */
export const nf = (v: number, d = 2) => {
  const s = v.toFixed(d).replace(/\.?0+$/, '');
  return s === '-0' ? '0' : s;
};

export const landF = (x: number, y: number) => forward([x, y]).a[3][0];

export function IslandOutlines() {
  const { x, y } = usePlot();
  const tri = (pts: number[][]) => pts.map(([a, b]) => `${x(a)},${y(b)}`).join(' ');
  return (
    <g fill="none" stroke="var(--text)" strokeWidth={1.4} strokeDasharray="5 4" opacity={0.7}>
      <polygon points={tri([[-1.9, -0.8], [0.1, -0.8], [-0.9, 1.2]])} />
      <polygon points={tri([[0.2, 1], [1.8, 1], [1, -0.6]])} />
    </g>
  );
}

/** The straight line where a first-layer neuron's tally is zero. */
export function Coastline({ j, color = 'var(--c-grad)' }: { j: number; color?: string }) {
  const [w1, w2] = LAYERS[0].W[j];
  const b = LAYERS[0].b[j];
  const n2 = w1 * w1 + w2 * w2;
  const px = (-b * w1) / n2;
  const py = (-b * w2) / n2;
  const d = [-w2 / Math.sqrt(n2), w1 / Math.sqrt(n2)];
  return (
    <Clip>
      <Seg from={[px - 6 * d[0], py - 6 * d[1]]} to={[px + 6 * d[0], py + 6 * d[1]]} color={color} width={2.5} />
    </Clip>
  );
}

const STAGES = [
  'Input',
  'Layer 1: weigh and add bias',
  'Layer 1: respond',
  'Layer 2: weigh and add bias',
  'Layer 2: respond',
  'Layer 3: weigh and add bias',
  'Layer 3: respond',
];

const vecTex = (v: number[], hi = -1, color = '') =>
  `\\begin{bmatrix}${v.map((x, i) => (i === hi ? `\\cg{${nf(x)}}` : color ? `${color}{${nf(x)}}` : nf(x))).join('\\\\')}\\end{bmatrix}`;
const matTex = (m: number[][], hi = -1) =>
  `\\begin{bmatrix}${m.map((r, i) => r.map((x) => (i === hi ? `\\cg{${nf(x)}}` : nf(x))).join(' & ')).join('\\\\')}\\end{bmatrix}`;

export function ForwardStepper() {
  const [probe, setProbe] = useState<[number, number]>([-0.9, 0]);
  const [stage, setStage] = useState(0);
  const [sel, setSel] = useState<NetSelection>(null);
  const t = useMemo(() => forward(probe), [probe]);
  const layer = stage === 0 ? -1 : Math.floor((stage - 1) / 2); // 0-based weight layer being computed
  const isZ = stage > 0 && stage % 2 === 1;

  // What the diagram knows so far: inputs, then completed layers, then the current layer's z or a.
  const values: (number[] | undefined)[] = [t.a[0]];
  for (let l = 0; l < LAYERS.length; l++) {
    if (l < layer) values.push(t.a[l + 1]);
    else if (l === layer) values.push(isZ ? t.z[l] : t.a[l + 1]);
    else values.push(undefined);
  }
  const fill = (v: number, col: number) =>
    col === 0 || (col === layer + 1 && isZ)
      ? 'var(--bg-elev)'
      : `color-mix(in srgb, var(--c-c) ${Math.round(Math.max(0, Math.min(1, v)) * 80)}%, var(--bg-elev))`;

  const hiRow = sel?.kind === 'neuron' && sel.col === layer + 1 ? sel.j : -1;
  const L = layer >= 0 ? LAYERS[layer] : null;
  const prev = layer >= 0 ? t.a[layer] : [];
  const nOut = L?.W.length ?? 0;
  const nIn = L?.W[0].length ?? 0;

  let eq = '';
  let shapes = '';
  if (stage === 0) {
    eq = `\\mathbf{x} = \\begin{bmatrix} \\text{east} \\\\ \\text{north} \\end{bmatrix} = ${vecTex(probe)}`;
    shapes = 'a vector of 2 numbers';
  } else if (L && isZ) {
    eq = `\\underbrace{${matTex(L.W, hiRow)}}_{\\ca{W^{(${layer + 1})}}}\\,\\underbrace{${vecTex(prev)}}_{\\mathbf{a}^{(${layer})}} + \\underbrace{${vecTex(L.b, hiRow)}}_{\\cc{\\mathbf{b}^{(${layer + 1})}}} = \\underbrace{${vecTex(t.z[layer], hiRow)}}_{\\mathbf{z}^{(${layer + 1})}}`;
    shapes = `(${nOut} × ${nIn} matrix) · (${nIn}-vector) + (${nOut}-vector) → ${nOut}-vector · ${nOut * nIn} multiply-adds`;
  } else if (L) {
    eq = `\\mathbf{a}^{(${layer + 1})} = \\sigma\\!\\left(${vecTex(t.z[layer], hiRow)}\\right) = ${vecTex(t.a[layer + 1], hiRow)}`;
    shapes = nOut === 1 ? 'σ is applied to the single number' : `σ is applied to each of the ${nOut} numbers separately — no mixing`;
  }

  // Worked row for a clicked neuron in the current layer.
  let rowTex = '';
  if (L && hiRow >= 0) {
    const w = L.W[hiRow];
    const paren = (v: number, first = false) => (v < 0 && !first ? `(${nf(v)})` : nf(v));
    const terms = w.map((wi, i) => `${paren(wi, i === 0)} \\times ${paren(prev[i])}`).join(' + ');
    rowTex = isZ
      ? `z_{${hiRow + 1}} = ${terms} + ${L.b[hiRow] < 0 ? `(${nf(L.b[hiRow])})` : nf(L.b[hiRow])} = ${nf(t.z[layer][hiRow])}`
      : `a_{${hiRow + 1}} = \\sigma(${nf(t.z[layer][hiRow])}) = \\frac{1}{1 + e^{${nf(-t.z[layer][hiRow])}}} = ${nf(t.a[layer + 1][hiRow], 3)}`;
  }

  const notes: React.ReactNode[] = [
    <>
      The point you dragged is the input: two numbers, its position east and north. Nothing has been computed yet. Press <strong>Next</strong>.
    </>,
    <>
      Six neurons, six rows. Each row is one neuron’s weights: a <strong>coastline detector</strong> for one straight stretch of coast. All six dot products happen in one
      matrix multiplication. Click a neuron in the first hidden column to see its row worked out.
    </>,
    <>
      Each tally is squashed by the sigmoid into a number between 0 and 1: close to 1 means “this point is on the land side of my stretch of coast”.{' '}
      {t.a[1].filter((v) => v > 0.5).length} of the 6 detectors say “land side” for this point.
    </>,
    <>
      Layer 2 never sees the map. Its input is the six answers from layer 1. The first row listens only to the west island’s three coastlines (weights 5, 5, 5); the
      second only to the east island’s. The bias of −12.5 means it takes all three to push the tally above zero.
    </>,
    <>
      West island neuron: <strong>{nf(t.a[2][0], 3)}</strong>, east island neuron: <strong>{nf(t.a[2][1], 3)}</strong>.{' '}
      {t.a[2][0] > 0.5 ? 'This point is inside the west island.' : t.a[2][1] > 0.5 ? 'This point is inside the east island.' : 'Neither island claims this point.'}
    </>,
    <>
      The last layer is a single neuron listening to the two island detectors, with a bias of −3: either island alone is enough to push the tally positive. It’s an OR
      gate, like the one you built in the logic lab.
    </>,
    <>
      Final answer: <strong>{(t.a[3][0] * 100).toFixed(1)}% sure this is land</strong>. That’s the whole forward pass: three matrix multiplications, three biases, three
      rounds of squashing. Drag the point to the sea between the islands and step through again.
    </>,
  ];

  return (
    <Widget
      title="Follow one point through the network"
      wide
      caption={
        <>
          A three-layer network, every weight set by hand, that decides whether a point on this map is <strong>land</strong> (the dashed islands). Drag the white point,
          then step through the forward pass one stage at a time. Only values that have actually been computed appear in the diagram. The background shows the network’s
          final answer everywhere: bright means land.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '0 1 300px' }}>
          <Plot x={[-2.2, 2.2]} y={[-2.2, 2.2]} width={300} height={300} grid={false} tickLabels={false} ariaLabel="Map with two islands and a probe point">
            <Heatmap f={landF} range={[0, 1]} resolution={120} opacity={0.75} />
            <IslandOutlines />
            {sel?.kind === 'neuron' && sel.col === 1 && <Coastline j={sel.j} />}
            <Handle x={probe[0]} y={probe[1]} onDrag={(a, b) => setProbe([Math.round(a * 20) / 20, Math.round(b * 20) / 20])} color="#fff" label="probe point" />
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <NetDiagram
            sizes={[2, 6, 2, 1]}
            W={LAYERS.map((l) => l.W)}
            values={values}
            fill={fill}
            selected={sel}
            onSelect={setSel}
            activeCols={layer >= 0 ? [layer + 1] : []}
            inputLabels={['east', 'north']}
            width={460}
            height={290}
            node={26}
            ariaLabel="The land-detecting network"
          />
          <div className="btn-row" style={{ marginTop: 6 }}>
            <button className="btn" disabled={stage === 0} onClick={() => setStage(stage - 1)}>
              ◀ Back
            </button>
            <button className="btn primary" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>
              Next ▶
            </button>
            <button className="btn small" onClick={() => setStage(0)}>
              Restart
            </button>
            <span className="muted" style={{ fontSize: 13 }}>
              Stage {stage + 1} of {STAGES.length}: <strong>{STAGES[stage]}</strong>
            </span>
          </div>
        </div>
      </div>
      <div style={{ marginTop: 12, overflowX: 'auto' }}>
        <Tex block>{eq}</Tex>
        <div className="muted mono" style={{ fontSize: 12.5, textAlign: 'center' }}>
          {shapes}
        </div>
        {rowTex && (
          <div style={{ marginTop: 6 }}>
            <Tex block>{rowTex}</Tex>
          </div>
        )}
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        {notes[stage]}
        {sel?.kind === 'neuron' && sel.col >= 1 && sel.col !== layer + 1 && (
          <span className="muted"> (The neuron you clicked belongs to layer {sel.col}; step to that layer to see its row.)</span>
        )}
        {sel?.kind === 'neuron' && sel.col === 1 && <span className="muted"> Its coastline is drawn on the map in magenta: “{LAYERS[0].names[sel.j]}”.</span>}
      </div>
      <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
        Inputs: {INPUT_NAMES.join(', ')}. Background colour: <span style={{ color: `rgb(${sequential(0).join(',')})` }}>■</span> sea →{' '}
        <span style={{ color: `rgb(${sequential(1).join(',')})` }}>■</span> land.
      </div>
    </Widget>
  );
}
