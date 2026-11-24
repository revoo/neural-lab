import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Dot, FnPath, Plot } from '../../components/plot';
import { evaluate, evaluateLayers, foldNet, linearPieces, mlpMultiplyAdds, mlpParams, neuronCount, paramCount, perturb, randomNet } from './pieces';

const big = (n: number) => n.toLocaleString('en-GB');
function bytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

// ---------------------------------------------------------------- parameter calculator

const PRESETS: { label: string; sizes: number[]; note: string }[] = [
  { label: 'Playground net', sizes: [2, 3, 1], note: 'The 3-neuron circle classifier from the playground.' },
  { label: 'Digit classifier', sizes: [784, 128, 64, 10], note: 'A classic network for 28 × 28 handwritten digits: 784 pixels in, 10 digit scores out.' },
  { label: 'GPT-2 feed-forward block', sizes: [768, 3072, 768], note: 'The two-layer network inside each of GPT-2 small’s 12 transformer blocks.' },
  { label: 'Wide and shallow', sizes: [100, 2000, 10], note: 'One very wide hidden layer.' },
  { label: 'Deep and narrow', sizes: [100, 200, 200, 200, 200, 200, 200, 10], note: 'Six hidden layers of 200 — almost exactly the same parameter budget as the wide one, spent differently.' },
  { label: 'Four layers of 512', sizes: [512, 512, 512, 512, 512], note: 'Every layer the same width, like the inside of a big model. Try both ×2 buttons on this one.' },
];

export function ParamCalculator() {
  const [text, setText] = useState('784, 128, 64, 10');
  const sizes = text
    .split(/[^0-9]+/)
    .map((s) => parseInt(s, 10))
    .filter((n) => Number.isFinite(n) && n > 0)
    .slice(0, 16);
  const ok = sizes.length >= 2;
  const preset = PRESETS.find((p) => p.sizes.join(',') === sizes.join(','));
  const set = (s: number[]) => setText(s.join(', '));
  const hidden = sizes.slice(1, -1);
  const total = ok ? mlpParams(sizes) : 0;
  return (
    <Widget
      title="Parameter calculator"
      caption={
        <>
          Type the size of every layer, input first, output last (or pick a preset). Then pick <strong>Four layers of 512</strong> and try the two buttons: doubling every hidden layer’s
          <em> width</em> versus doubling the <em>number</em> of hidden layers. Predict which costs more before you press them.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        {PRESETS.map((p) => (
          <button key={p.label} className={`btn small${preset === p ? ' primary' : ''}`} onClick={() => set(p.sizes)}>
            {p.label}
          </button>
        ))}
      </div>
      <label style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', fontSize: 13.5 }}>
        <span className="muted">Layer sizes:</span>
        <input className="num-input" style={{ width: 260, textAlign: 'left' }} value={text} onChange={(e) => setText(e.target.value)} aria-label="layer sizes" />
      </label>
      {preset && <div className="muted" style={{ fontSize: 13, marginTop: 6 }}>{preset.note}</div>}
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn small" disabled={!hidden.length} onClick={() => set([sizes[0], ...hidden.map((h) => h * 2), sizes[sizes.length - 1]])}>
          ×2 width (every hidden layer)
        </button>
        <button className="btn small" disabled={!hidden.length} onClick={() => set([sizes[0], ...hidden, ...hidden, sizes[sizes.length - 1]])}>
          ×2 depth (repeat the hidden layers)
        </button>
      </div>
      {ok ? (
        <div style={{ overflowX: 'auto', marginTop: 12 }}>
          <table className="num-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>layer</th>
                <th>weights</th>
                <th>biases</th>
                <th>parameters</th>
                <th>share</th>
              </tr>
            </thead>
            <tbody>
              {sizes.slice(1).map((n, l) => {
                const p = sizes[l] * n + n;
                return (
                  <tr key={l}>
                    <td style={{ textAlign: 'left' }}>
                      {l + 1}: {sizes[l]} → {n}
                    </td>
                    <td>{big(sizes[l] * n)}</td>
                    <td>{big(n)}</td>
                    <td>{big(p)}</td>
                    <td>{((100 * p) / total).toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="readout" style={{ marginTop: 10 }}>
            <span className="k">total parameters      </span> <strong>{big(total)}</strong>
            {'\n'}
            <span className="k">memory (32-bit floats) </span> {bytes(total * 4)}
            {'   '}
            <span className="k">(16-bit) </span> {bytes(total * 2)}
            {'\n'}
            <span className="k">multiply-adds / input  </span> {big(mlpMultiplyAdds(sizes))}
            {'\n'}
            <span className="k">neurons                </span> {big(sizes.slice(1).reduce((s, v) => s + v, 0))} in {sizes.length - 1} layers
          </div>
        </div>
      ) : (
        <div className="muted" style={{ marginTop: 10 }}>Enter at least two sizes.</div>
      )}
    </Widget>
  );
}

// ---------------------------------------------------------------- folding (depth)

function Stage({ f, title }: { f: (x: number) => number; title: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <Plot x={[0, 1]} y={[-0.1, 1.1]} width={120} height={84} grid={false} tickLabels={false} pad={6} ariaLabel={title}>
        <FnPath f={f} color="var(--c-a)" width={1.6} samples={600} />
      </Plot>
      <div className="muted" style={{ fontSize: 11.5 }}>{title}</div>
    </div>
  );
}

export function FoldingDemo() {
  const [L, setL] = useState(3);
  const [sigma, setSigma] = useState(0);
  const [seed, setSeed] = useState(1);
  const net = useMemo(() => (sigma > 0 ? perturb(foldNet(L), sigma, seed) : foldNet(L)), [L, sigma, seed]);
  const { pieces, breaks } = useMemo(() => linearPieces(net, 0, 1), [net]);
  const f = useMemo(() => (x: number) => evaluate(net, x), [net]);
  const samples = Math.max(600, 2 ** L * 12);
  const stagesShown = Math.min(L, 4);
  const wideNeurons = 2 ** L - 1;
  const wideParams = 3 * wideNeurons + 1;
  const ideal = 2 ** L;

  let note: React.ReactNode;
  if (sigma === 0)
    note = (
      <>
        <strong>{L} fold{L > 1 ? 's' : ''}</strong>: {neuronCount(net)} neurons, {paramCount(net)} parameters, and <strong>{pieces} straight pieces</strong>. Each extra fold
        layer costs just 6 more parameters and <em>doubles</em> the count. One wide layer would need at least {big(wideNeurons)} neurons ({big(wideParams)} parameters) to
        make {big(pieces)} pieces.
      </>
    );
  else
    note = (
      <>
        With every weight nudged by random noise of size about {sigma.toFixed(2)}, the network now has <strong>{pieces}</strong> pieces (the perfect construction has {ideal}).{' '}
        {pieces < ideal * 0.6
          ? 'The folds no longer line up, and many of them land outside the interval or flatten out. The exponential count depends on weights being exactly right.'
          : 'Tiny nudges can even add a few slivers of extra pieces — but the neat doubling pattern is already breaking up.'}
      </>
    );

  return (
    <Widget
      title="Folding with depth"
      wide
      caption={
        <>
          Each hidden layer here has just <strong>2 ReLU neurons</strong>, wired to fold the interval from 0 to 1 in half, like folding a strip of paper. Add fold layers and
          watch the number of straight pieces (counted exactly, by tracking where every neuron switches on or off). Then jiggle the weights to see how delicate this is.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 460px' }}>
          <Plot x={[0, 1]} y={[-0.15, 1.15]} width={480} height={240} grid={false} pad={40} ariaLabel="The network's output">
            <FnPath f={f} color="var(--accent)" width={2} samples={samples} />
            {breaks.length <= 64 && breaks.map((b, i) => <Dot key={i} at={[b, f(b)]} r={2.6} color="var(--c-grad)" />)}
          </Plot>
        </div>
        <div className="widget-col controls" style={{ flex: '0 1 250px' }}>
          <Slider label="fold layers (depth)" value={L} min={1} max={10} step={1} onChange={setL} format={(v) => String(v)} color="c-b" />
          <Slider label="jiggle every weight by" value={sigma} min={0} max={0.2} step={0.01} onChange={setSigma} format={(v) => (v ? `±${v.toFixed(2)}` : 'off')} />
          {sigma > 0 && (
            <button className="btn small" onClick={() => setSeed(seed + 1)}>
              Different random jiggle
            </button>
          )}
          <div className="readout">
            <span className="k">neurons    </span> {neuronCount(net)}
            {'\n'}
            <span className="k">parameters </span> {paramCount(net)}
            {'\n'}
            <span className="k">pieces     </span> <strong>{big(pieces)}</strong>
          </div>
        </div>
      </div>
      <div className="field-label" style={{ marginTop: 10 }}>
        After each layer
      </div>
      <div className="btn-row" style={{ gap: 14 }}>
        {Array.from({ length: stagesShown }, (_, k) => (
          <Stage key={k} title={`after fold ${k + 1}`} f={(x) => {
            const out = evaluateLayers(net, x);
            // the stage's value is the next layer's input: 2·h₁ − 4·h₂ of this layer
            const h = out[k];
            return 2 * h[0] - 4 * h[1];
          }} />
        ))}
        {L > 4 && <div className="muted" style={{ alignSelf: 'center', fontSize: 13 }}>… {L - 4} more</div>}
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        {note}
      </div>
    </Widget>
  );
}

// ---------------------------------------------------------------- random deep vs wide

export function RandomDeepWide() {
  const [shape, setShape] = useState<'4x4' | '4x8' | '8x4'>('4x4');
  const [seed, setSeed] = useState(1);
  const [width, depth] = shape === '4x4' ? [4, 4] : shape === '4x8' ? [4, 8] : [8, 4];
  const neurons = width * depth;
  const deep = useMemo(() => randomNet(Array(depth).fill(width), seed), [width, depth, seed]);
  const wide = useMemo(() => randomNet([neurons], seed + 1000), [neurons, seed]);
  const pd = linearPieces(deep, -3, 3).pieces;
  const pw = linearPieces(wide, -3, 3).pieces;
  const stats = useMemo(() => {
    const n = 200;
    let sd = 0;
    let sw = 0;
    let maxD = 0;
    for (let s = 1; s <= n; s++) {
      const a = linearPieces(randomNet(Array(depth).fill(width), s), -3, 3).pieces;
      const b = linearPieces(randomNet([neurons], s + 1000), -3, 3).pieces;
      sd += a;
      sw += b;
      maxD = Math.max(maxD, a);
    }
    return { deep: sd / n, wide: sw / n, maxD, n };
  }, [width, depth, neurons]);
  const fd = useMemo(() => (x: number) => evaluate(deep, x), [deep]);
  const fw = useMemo(() => (x: number) => evaluate(wide, x), [wide]);
  const range = (f: (x: number) => number): [number, number] => {
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k <= 300; k++) {
      const v = f(-3 + (6 * k) / 300);
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const pad = (hi - lo) * 0.1 + 1e-3;
    return [lo - pad, hi + pad];
  };
  return (
    <Widget
      title="What random networks actually do"
      wide
      caption={
        <>
          Two networks with the <strong>same number of ReLU neurons</strong> and ordinary random starting weights — the kind every training run begins with. Left: stacked
          into {depth} layers of {width}. Right: all in one wide layer. Pieces are counted exactly over the input range shown. Press <strong>New random weights</strong> a few
          times, and read the averages over {stats.n} random networks.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 8 }}>
        <Segmented
          label="Deep network"
          value={shape}
          onChange={setShape}
          options={[
            { value: '4x4', label: '4 layers × 4' },
            { value: '4x8', label: '8 layers × 4' },
            { value: '8x4', label: '4 layers × 8' },
          ]}
        />
        <button className="btn small" onClick={() => setSeed(seed + 1)}>
          New random weights
        </button>
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="field-label">
            Deep: {depth} × {width} = {neurons} neurons · {big(paramCount(deep))} params
          </div>
          <Plot x={[-3, 3]} y={range(fd)} width={320} height={170} grid={false} tickLabels={false} ariaLabel="Random deep network">
            <FnPath f={fd} color="var(--c-b)" width={2} samples={1500} />
          </Plot>
          <div className="mono" style={{ fontSize: 13 }}>
            pieces: <strong>{pd}</strong>
          </div>
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="field-label">
            Wide: 1 × {neurons} = {neurons} neurons · {big(paramCount(wide))} params
          </div>
          <Plot x={[-3, 3]} y={range(fw)} width={320} height={170} grid={false} tickLabels={false} ariaLabel="Random wide network">
            <FnPath f={fw} color="var(--c-a)" width={2} samples={1500} />
          </Plot>
          <div className="mono" style={{ fontSize: 13 }}>
            pieces: <strong>{pw}</strong>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        Averaged over {stats.n} random networks: deep <strong>{stats.deep.toFixed(1)}</strong> pieces (the most in any one: {stats.maxD}), wide{' '}
        <strong>{stats.wide.toFixed(1)}</strong>. Its shape allows far more: a network with one input and {depth} layers of {width} can have at most ({width} + 1)
        <sup>{depth}</sup> = {big((width + 1) ** depth)} pieces, and a hand-designed version that folds into {width} pieces per layer reaches {width}<sup>{depth}</sup> = {big(width ** depth)}. Random weights come nowhere near:
        a typical network, deep or wide, has roughly one bend per neuron or fewer.
      </div>
    </Widget>
  );
}
