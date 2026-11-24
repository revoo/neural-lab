import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { Clip, Plot, Seg, usePlot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import { MLP } from '../../lib/mlp';
import { createOptimizer, type Optimizer } from '../../lib/optimizers';
import { createRng } from '../../lib/rng';
import type { ActivationName } from '../../lib/activations';

type Data = { X: number[][]; Y: number[][] };

function xorData(): Data {
  const r = createRng(5);
  const X: number[][] = [];
  const Y: number[][] = [];
  for (const [cx, cy, l] of [[-0.6, -0.6, 0], [0.6, 0.6, 0], [-0.6, 0.6, 1], [0.6, -0.6, 1]]) {
    for (let i = 0; i < 12; i++) {
      X.push([r.normal(cx, 0.18), r.normal(cy, 0.18)]);
      Y.push([l]);
    }
  }
  return { X, Y };
}
const DATA = xorData();

function Points() {
  const { x, y } = usePlot();
  return (
    <g>
      {DATA.X.map((p, i) => (
        <circle key={i} cx={x(p[0])} cy={y(p[1])} r={5} fill={DATA.Y[i][0] ? 'var(--c-b)' : 'var(--c-a)'} stroke="#fff" strokeWidth={1.3} />
      ))}
    </g>
  );
}

const ACTS: { value: ActivationName; label: string }[] = [
  { value: 'linear', label: 'None (linear)' },
  { value: 'relu', label: 'ReLU' },
  { value: 'tanh', label: 'Tanh' },
  { value: 'sigmoid', label: 'Sigmoid' },
];

export function XORTrainer() {
  const [act, setAct] = useState<ActivationName>('linear');
  const [layers, setLayers] = useState(2);
  const [width, setWidth] = useState(4);
  const [seed, setSeed] = useState(1);
  const [running, setRunning] = useState(false);
  const [version, setVersion] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const build = () => {
    const sizes = [2, ...Array(layers).fill(width), 1];
    const acts = [...Array(layers).fill(act), 'sigmoid'] as ActivationName[];
    return new MLP(sizes, acts, createRng(seed), act === 'relu' ? 'he' : 'xavier');
  };
  const net = useRef<MLP>(null as unknown as MLP);
  if (!net.current) net.current = build();
  const opt = useRef<Optimizer>(createOptimizer({ name: 'adam', lr: 0.03 }));
  const first = useRef(true);

  // Rebuild the network whenever the architecture or seed changes.
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    net.current = build();
    opt.current = createOptimizer({ name: 'adam', lr: 0.03 });
    setHistory([]);
    setVersion((v) => v + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [act, layers, width, seed]);

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const tick = () => {
      const n = net.current;
      let loss = 0;
      for (let s = 0; s < 5; s++) {
        loss = n.backprop(DATA.X, DATA.Y, 'bce');
        opt.current.step(n.params(), n.grads());
      }
      setHistory((h) => (h.length > 400 ? [...h.slice(1), loss] : [...h, loss]));
      setVersion((v) => v + 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  const n = net.current;
  const f = useMemo(() => (px: number, py: number) => n.predict([px, py])[0], [n, version]);
  const cmap = useMemo(() => (t: number) => diverging(1 - t), []);
  const correct = DATA.X.filter((p, i) => (n.predict(p)[0] > 0.5 ? 1 : 0) === DATA.Y[i][0]).length;
  const loss = history.length ? history[history.length - 1] : n.loss(DATA.X, DATA.Y, 'bce');
  const epochs = history.length * 5;
  const solved = correct === DATA.X.length;

  // Each first-layer neuron draws its own line; show them so the combination is visible.
  const lines = n.W[0].map((w, j) => ({ w, b: n.b[0][j] }));

  let note: React.ReactNode;
  if (layers === 0) note = <>No hidden layer: this is the single neuron from the last lesson. It can only draw one straight line, so the best it can do is about 75%.</>;
  else if (act === 'linear' && epochs > 200)
    note = (
      <>
        <strong>Stuck.</strong> {layers} hidden layer{layers > 1 ? 's' : ''} of {width} neurons, and the boundary is still a single straight line. Without activations, all
        those layers collapse into one — exactly as the three-layer demo showed. More neurons can’t fix this; a different kind of neuron can.
      </>
    );
  else if (solved)
    note = (
      <>
        <strong>Solved: all {DATA.X.length} points correct.</strong> Look at the faint dashed lines: each is one first-layer neuron’s line. The next layer combines the
        regions those lines carve out into a shape no single line could make.
      </>
    );
  else if (epochs === 0) note = <>Press <strong>Train</strong>. Before you do: will this network be able to separate the two colours?</>;
  else note = <>Training… {epochs} steps so far. Watch the background — the network’s current opinion — reshape itself.</>;

  return (
    <Widget
      title="XOR: the problem a straight line can’t solve"
      wide
      caption={
        <>
          A real neural network, trained live in your browser. Orange points sit in two opposite corners, blue in the other two. The background shows what the network
          currently predicts everywhere. Start with <strong>no activation</strong>, press Train, and see how far it gets. Then switch to ReLU and try again.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="Activation between layers" value={act} onChange={(v) => { setAct(v); setRunning(false); }} options={ACTS} />
        <Segmented label="Hidden layers" value={layers} onChange={(v) => { setLayers(v); setRunning(false); }} options={[0, 1, 2, 3]} />
        <Segmented label="Neurons per layer" value={width} onChange={(v) => { setWidth(v); setRunning(false); }} options={[2, 3, 4, 8]} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <Plot x={[-1.3, 1.3]} y={[-1.3, 1.3]} width={400} height={400} grid={false} tickLabels={false} ariaLabel="XOR data with the network's decision regions">
            <Heatmap f={f} range={[0, 1]} colormap={cmap} resolution={70} opacity={0.65} />
            {layers > 0 && act !== 'linear' && (
              <Clip>
                {lines.map((l, j) => {
                  const nn = Math.hypot(l.w[0], l.w[1]) || 1;
                  const px = (-l.b * l.w[0]) / (nn * nn);
                  const py = (-l.b * l.w[1]) / (nn * nn);
                  const dx = -l.w[1] / nn;
                  const dy = l.w[0] / nn;
                  return <Seg key={j} from={[px - 4 * dx, py - 4 * dy]} to={[px + 4 * dx, py + 4 * dy]} color="var(--text)" width={1.2} dash="4 4" opacity={0.5} />;
                })}
              </Clip>
            )}
            <Points />
          </Plot>
        </div>
        <div className="widget-col controls">
          <div className="btn-row">
            <button className={`btn${running ? '' : ' primary'}`} onClick={() => setRunning(!running)}>
              {running ? '❚❚ Pause' : '▶ Train'}
            </button>
            <button className="btn" onClick={() => { setRunning(false); setSeed(seed + 1); }}>New random start</button>
          </div>
          <div className="readout">
            <span className="k">accuracy</span> <strong className={solved ? 'c-good' : ''}>{correct} / {DATA.X.length}</strong>
            <br />
            <span className="k">loss    </span> {loss.toFixed(3)}
            <br />
            <span className="k">steps   </span> {epochs}
            <br />
            <span className="k">weights </span> {n.paramCount} adjustable numbers
          </div>
          <LossChart history={history} />
          <div className="live-note">{note}</div>
        </div>
      </div>
    </Widget>
  );
}

function LossChart({ history }: { history: number[] }) {
  const W = 300;
  const H = 80;
  if (history.length < 2)
    return (
      <div className="muted" style={{ fontSize: 12.5, height: H, display: 'flex', alignItems: 'center' }}>
        The loss curve will appear here as it trains.
      </div>
    );
  const max = Math.max(0.8, ...history);
  const pts = history.map((v, i) => `${(i / (history.length - 1)) * W},${H - (v / max) * (H - 6) - 3}`).join(' ');
  return (
    <div>
      <div className="field-label">Loss over training</div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ background: 'var(--bg-sunken)', borderRadius: 6 }}>
        <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={2} />
      </svg>
    </div>
  );
}
