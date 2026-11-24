import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Handle, Plot, usePlot } from '../../components/plot';
import { Heatmap, diverging, fieldImage, sequential } from '../../components/heatmap';
import { NetDiagram, type NetSelection } from '../../components/NetDiagram';
import { MLP } from '../../lib/mlp';
import { createOptimizer, type Optimizer } from '../../lib/optimizers';
import { createRng } from '../../lib/rng';
import type { ActivationName } from '../../lib/activations';
import {
  DATASET_LABELS,
  accuracy,
  forwardMasked,
  fullMask,
  importance,
  makeDataset,
  minimalNeurons,
  paramBreakdown,
  paramTotal,
  predict,
  type DatasetName,
  type Mask,
} from './playground';

type Act = 'tanh' | 'relu' | 'sigmoid';
const DOMAIN: [number, number, number, number] = [-1.6, 1.6, -1.6, 1.6];
const MAX_LAYERS = 4;
const MAX_WIDTH = 8;
const STEP_CAP = 3000;

const fmt = (v: number, d = 2) => {
  const s = v.toFixed(d);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
};
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
/** Accuracy lost by removing a neuron: "−12%", "0%", or "+3%" if removing it actually helps. */
const impLabel = (v: number) => {
  const p = Math.round(v * 100);
  return p === 0 ? '0%' : p > 0 ? `−${p}%` : `+${-p}%`;
};

function build(sizes: number[], act: Act, seed: number) {
  const acts = [...Array(sizes.length - 2).fill(act), 'sigmoid'] as ActivationName[];
  return new MLP(sizes, acts, createRng(seed), act === 'relu' ? 'he' : 'xavier');
}

function Points({ X, y, net, mask }: { X: number[][]; y: number[]; net: MLP; mask: Mask }) {
  const { x: sx, y: sy } = usePlot();
  return (
    <g>
      {X.map((p, k) => {
        const wrong = (predict(net, p, mask) > 0.5 ? 1 : 0) !== y[k];
        return (
          <g key={k}>
            <circle cx={sx(p[0])} cy={sy(p[1])} r={4.2} fill={y[k] ? 'var(--c-b)' : 'var(--c-a)'} stroke="#fff" strokeWidth={1} />
            {wrong && <circle cx={sx(p[0])} cy={sy(p[1])} r={7.5} fill="none" stroke="var(--c-bad)" strokeWidth={1.8} />}
          </g>
        );
      })}
    </g>
  );
}

const nodeName = (col: number, j: number, sizes: number[]) =>
  col === 0 ? (j === 0 ? 'input x' : 'input y') : col === sizes.length - 1 ? 'output' : `layer ${col}, neuron ${j + 1}`;
const shortName = (col: number, j: number, sizes: number[]) => (col === 0 ? (j === 0 ? 'x' : 'y') : col === sizes.length - 1 ? 'out' : `L${col}·n${j + 1}`);

export function Playground() {
  const [dataset, setDataset] = useState<DatasetName>('circles');
  const [act, setAct] = useState<Act>('tanh');
  const [hidden, setHidden] = useState<number[]>([3]);
  const [seed, setSeed] = useState(1);
  const sizes = useMemo(() => [2, ...hidden, 1], [hidden]);
  const data = useMemo(() => makeDataset(dataset), [dataset]);
  const Y = useMemo(() => data.y.map((v) => [v]), [data]);

  const net = useRef<MLP>(null as unknown as MLP);
  const opt = useRef<Optimizer>(createOptimizer({ name: 'adam', lr: 0.03 }));
  const builtFor = useRef('');
  const key = `${sizes.join('-')}|${act}|${seed}`;
  if (builtFor.current !== key) {
    net.current = build(sizes, act, seed);
    opt.current = createOptimizer({ name: 'adam', lr: 0.03 });
    builtFor.current = key;
  }
  const [version, setVersion] = useState(0);
  const [steps, setSteps] = useState(0);
  const [running, setRunning] = useState(false);
  const [mask, setMask] = useState<Mask>(() => fullMask(sizes));
  const [sel, setSel] = useState<NetSelection>(null);
  const [probe, setProbe] = useState<[number, number]>([0.3, 0.2]);
  const [showImp, setShowImp] = useState(false);
  const [bgNeuron, setBgNeuron] = useState(true);
  const [bestKept, setBestKept] = useState<number | null>(null);
  const [optimum, setOptimum] = useState<ReturnType<typeof minimalNeurons> | 'pending' | undefined>(undefined);
  const revealed = useRef(false); // once the optimum is shown, the learner's "best" stops updating

  // Any change of architecture, activation, seed or data starts over.
  useEffect(() => {
    setSteps(0);
    setRunning(false);
    setMask(fullMask(sizes));
    setSel(null);
    setBestKept(null);
    setOptimum(undefined);
    revealed.current = false;
    setVersion((v) => v + 1);
  }, [key, dataset]); // eslint-disable-line react-hooks/exhaustive-deps

  const n = net.current;
  const m = mask.length === sizes.length && mask.every((c, i) => c.length === sizes[i]) ? mask : fullMask(sizes);

  // Training loop.
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let s = steps;
    const tick = () => {
      let loss = 0;
      for (let k = 0; k < 6; k++) {
        loss = n.backprop(data.X, Y, 'bce');
        opt.current.step(n.params(), n.grads());
        s++;
      }
      setSteps(s);
      setVersion((v) => v + 1);
      if (s >= STEP_CAP || loss < 0.003) {
        setRunning(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]); // eslint-disable-line react-hooks/exhaustive-deps

  const startTraining = () => {
    setMask(fullMask(sizes));
    setBestKept(null);
    setOptimum(undefined);
    revealed.current = false;
    setRunning(true);
  };

  const acc = useMemo(() => accuracy(n, data, m), [n, data, m, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const accFull = useMemo(() => accuracy(n, data), [n, data, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const loss = useMemo(() => n.loss(data.X, Y, 'bce'), [n, data, Y, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const imp = useMemo(() => (showImp ? importance(n, data, m) : null), [showImp, n, data, m, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const trace = forwardMasked(n, probe, m);

  const totalHidden = hidden.reduce((s, v) => s + v, 0);
  const keptHidden = m.slice(1, -1).reduce((s, c) => s + c.filter(Boolean).length, 0);
  useEffect(() => {
    if (!running && !revealed.current && acc >= 0.95 && keptHidden < totalHidden) setBestKept((b) => (b === null ? keptHidden : Math.min(b, keptHidden)));
  }, [acc, keptHidden, totalHidden, running]);

  // Thumbnails of every node over the input square.
  const thumbs = useMemo(() => {
    const res = 30;
    const vals = sizes.map((s) => Array.from({ length: s }, () => new Float64Array(res * res)));
    for (let r = 0; r < res; r++)
      for (let c = 0; c < res; c++) {
        const x = DOMAIN[0] + ((c + 0.5) / res) * (DOMAIN[1] - DOMAIN[0]);
        const yv = DOMAIN[3] - ((r + 0.5) / res) * (DOMAIN[3] - DOMAIN[2]);
        const t = forwardMasked(n, [x, yv], m);
        t.a.forEach((col, ci) => col.forEach((v, j) => (vals[ci][j][r * res + c] = v)));
      }
    const last = sizes.length - 1;
    return vals.map((col, ci) =>
      col.map((arr) => {
        let max = 0;
        for (const v of arr) max = Math.max(max, Math.abs(v));
        const norm = (v: number) =>
          ci === 0 ? (v - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) : ci === last ? v : act === 'tanh' ? (v + 1) / 2 : act === 'relu' ? (max > 0 ? v / max : 0) : v;
        let k = 0;
        return fieldImage(() => norm(arr[k++]), DOMAIN, res, ci === last ? (v) => diverging(1 - v) : (v) => sequential(v));
      }),
    );
  }, [n, version, m, sizes, act]); // eslint-disable-line react-hooks/exhaustive-deps

  const selNeuron = sel?.kind === 'neuron' ? sel : null;
  const selEdge = sel?.kind === 'edge' ? sel : null;
  const bgIsNeuron = bgNeuron && selNeuron && selNeuron.col > 0 && selNeuron.col < sizes.length - 1;
  const bgF = useMemo(() => {
    if (bgIsNeuron) {
      const { col, j } = selNeuron!;
      return (x: number, y: number) => forwardMasked(n, [x, y], m).a[col][j];
    }
    return (x: number, y: number) => predict(n, [x, y], m);
  }, [n, version, m, bgIsNeuron, selNeuron?.col, selNeuron?.j]); // eslint-disable-line react-hooks/exhaustive-deps
  const bgRange: [number, number] = bgIsNeuron ? (act === 'tanh' ? [-1, 1] : act === 'relu' ? [0, 2] : [0, 1]) : [0, 1];
  const bgMap = useMemo(() => (bgIsNeuron ? sequential : (t: number) => diverging(1 - t)), [bgIsNeuron]);

  const badges = imp?.map((col, c) => col.map((v, j) => (c === 0 || c === sizes.length - 1 ? undefined : !m[c][j] ? 'off' : impLabel(v))));
  const dead = m.map((col) => col.map((on) => !on));

  const toggleNeuron = (col: number, j: number) => {
    setRunning(false);
    const next = m.map((r) => r.slice());
    next[col][j] = !next[col][j];
    setMask(next);
  };
  const setWeight = (l: number, j: number, i: number, w: number) => {
    setRunning(false);
    n.W[l][j][i] = w;
    setOptimum(undefined);
    setVersion((v) => v + 1);
  };
  const changeLayer = (li: number, d: number) => setHidden(hidden.map((v, k) => (k === li ? Math.max(1, Math.min(MAX_WIDTH, v + d)) : v)));

  // ---- Inspector
  let inspector: React.ReactNode;
  if (selNeuron && selNeuron.col === 0) {
    inspector = (
      <>
        <div className="field-label">{nodeName(0, selNeuron.j, sizes)}</div>
        <p style={{ margin: 0 }}>
          An input neuron doesn’t compute anything: it just holds the probe point’s {selNeuron.j ? 'vertical' : 'horizontal'} position,{' '}
          <strong className="mono">{fmt(probe[selNeuron.j])}</strong>. Its little map is a plain ramp across the square.
        </p>
      </>
    );
  } else if (selNeuron) {
    const { col, j } = selNeuron;
    const l = col - 1;
    const W = n.W[l][j];
    const prev = trace.a[l];
    const terms = W.map((w, i) => w * prev[i]);
    const z = trace.z[l][j];
    const maxT = Math.max(1e-9, ...terms.map(Math.abs), Math.abs(n.b[l][j]));
    const off = !m[col][j];
    const isOut = col === sizes.length - 1;
    const without = !isOut ? (() => {
      const mm = m.map((r) => r.slice());
      mm[col][j] = !mm[col][j];
      return accuracy(n, data, mm);
    })() : acc;
    inspector = (
      <>
        <div className="field-label">{nodeName(col, j, sizes)}</div>
        <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginBottom: 4 }}>Its tally at the probe point, term by term (weight × incoming value):</div>
        <div style={{ display: 'grid', gridTemplateColumns: '52px 1fr 52px', gap: '3px 6px', alignItems: 'center', fontSize: 12 }} className="mono">
          {terms.map((t, i) => (
            <Bar key={i} label={shortName(col - 1, i, sizes)} v={t} max={maxT} title={`${fmt(W[i])} × ${fmt(prev[i])}`} />
          ))}
          <Bar label="bias" v={n.b[l][j]} max={maxT} />
          <span style={{ fontWeight: 700 }}>sum z</span>
          <span />
          <strong style={{ textAlign: 'right' }}>{fmt(z)}</strong>
        </div>
        <div style={{ marginTop: 6, fontSize: 13 }}>
          Output: {isOut ? 'σ' : act}({fmt(z)}) = <strong className="mono">{off ? '0 (switched off)' : fmt(trace.a[col][j], 3)}</strong>
        </div>
        {!isOut && (
          <>
            <button className={`btn small${off ? ' primary' : ''}`} style={{ marginTop: 8 }} onClick={() => toggleNeuron(col, j)}>
              {off ? 'Switch back on' : 'Switch this neuron off'}
            </button>
            <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
              Accuracy {off ? 'with it back on' : 'without it'}: <strong>{pct(without)}</strong> (now {pct(acc)})
            </div>
          </>
        )}
      </>
    );
  } else if (selEdge) {
    const { l, j, i } = selEdge;
    const w = n.W[l][j][i];
    const src = trace.a[l][i];
    const zt = trace.z[l][j];
    const zeroAcc = (() => {
      const old = n.W[l][j][i];
      n.W[l][j][i] = 0;
      const a = accuracy(n, data, m);
      n.W[l][j][i] = old;
      return a;
    })();
    inspector = (
      <>
        <div className="field-label">
          connection {shortName(l, i, sizes)} → {shortName(l + 1, j, sizes)}
        </div>
        <Slider label="weight" value={w} min={-Math.max(6, Math.ceil(Math.abs(w)))} max={Math.max(6, Math.ceil(Math.abs(w)))} step={0.05} onChange={(v) => setWeight(l, j, i, v)} color={w >= 0 ? 'c-a' : 'c-b'} />
        <div className="readout" style={{ marginTop: 8, fontSize: 12.5 }}>
          <span className="k">at the probe: </span>
          {'\n'}
          {fmt(w)} × {src < 0 ? `(${fmt(src)})` : fmt(src)} = <strong>{fmt(w * src)}</strong>
          {'\n'}
          <span className="k">of {shortName(l + 1, j, sizes)}’s tally z = {fmt(zt)}</span>
        </div>
        <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
          Accuracy if this weight were 0: <strong>{pct(zeroAcc)}</strong> (now {pct(acc)})
        </div>
        <button className="btn small" style={{ marginTop: 6 }} onClick={() => setWeight(l, j, i, 0)} disabled={w === 0}>
          Set to 0
        </button>
      </>
    );
  } else {
    inspector = (
      <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>
        Click any <strong>neuron</strong> to see how its tally is built and what happens without it, or any <strong>connection</strong> to see (and change) what it
        contributes.
      </p>
    );
  }

  let note: React.ReactNode;
  if (steps === 0 && !running) note = <>This network has random weights, so its background is a meaningless smear. Press <strong>Train</strong> to let it find weights that fit the points.</>;
  else if (running) note = <>Training… {steps} steps. Watch the neuron pictures in the diagram settle into shapes.</>;
  else if (keptHidden < totalHidden)
    note = (
      <>
        {totalHidden - keptHidden} of {totalHidden} hidden neurons switched off; accuracy {pct(acc)} (it was {pct(accFull)} with everything on).
        {acc >= 0.95 ? ' Still above 95% — those neurons weren’t pulling their weight.' : ' Below 95%: something you switched off was doing real work.'}
      </>
    );
  else if (accFull >= 0.95)
    note = (
      <>
        Trained: {pct(accFull)} of points correct with {paramTotal(sizes)} parameters. Now take it apart. Click hidden neurons to see what each one learned, and switch some
        off.
      </>
    );
  else note = <>Paused at {pct(accFull)}. This network may be too small for this data, or it needs more training.</>;

  return (
    <Widget
      title="The network playground"
      wide
      caption={
        <>
          A real network, trained in your browser. Pick a dataset and an architecture, press <strong>Train</strong>, then take the trained network apart: every node shows a
          tiny map of its own output; click one to inspect it. The white handle is a <strong>probe point</strong> — every number in the inspector and the tensor list is for
          that point. Circled points are currently misclassified.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="Data" value={dataset} onChange={(v) => setDataset(v)} options={(Object.keys(DATASET_LABELS) as DatasetName[]).map((k) => ({ value: k, label: DATASET_LABELS[k] }))} />
        <Segmented label="Activation" value={act} onChange={setAct} options={[{ value: 'tanh', label: 'tanh' }, { value: 'relu', label: 'ReLU' }, { value: 'sigmoid', label: 'sigmoid' }]} />
        <div>
          <div className="field-label">Hidden layers (neurons each)</div>
          <div className="btn-row">
            {hidden.map((v, li) => (
              <span key={li} className="seg" style={{ alignItems: 'center' }}>
                <button onClick={() => changeLayer(li, -1)} aria-label={`fewer neurons in layer ${li + 1}`}>−</button>
                <span className="mono" style={{ padding: '0 6px', fontWeight: 700 }}>{v}</span>
                <button onClick={() => changeLayer(li, 1)} aria-label={`more neurons in layer ${li + 1}`}>+</button>
              </span>
            ))}
            <button className="btn small" disabled={hidden.length >= MAX_LAYERS} onClick={() => setHidden([...hidden, hidden[hidden.length - 1] ?? 3])}>+ layer</button>
            <button className="btn small" disabled={hidden.length <= 1} onClick={() => setHidden(hidden.slice(0, -1))}>− layer</button>
          </div>
        </div>
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '0 1 290px' }}>
          <Plot x={[DOMAIN[0], DOMAIN[1]]} y={[DOMAIN[2], DOMAIN[3]]} width={290} height={290} grid={false} tickLabels={false} ariaLabel="Data and the network's predictions">
            <Heatmap f={bgF} range={bgRange} colormap={bgMap} resolution={64} opacity={0.7} />
            <Points X={data.X} y={data.y} net={n} mask={m} />
            <Handle x={probe[0]} y={probe[1]} onDrag={(a, b) => setProbe([Math.round(a * 20) / 20, Math.round(b * 20) / 20])} color="#fff" label="probe point" />
          </Plot>
          <div style={{ fontSize: 12.5, marginTop: 4 }} className="muted">
            Background: {bgIsNeuron ? <>what <strong>{nodeName(selNeuron!.col, selNeuron!.j, sizes)}</strong> outputs (bright = high)</> : <>the network’s answer (blue / orange)</>}
          </div>
          <Toggle label="Background follows the selected neuron" checked={bgNeuron} onChange={setBgNeuron} />
        </div>
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <NetDiagram
            sizes={sizes}
            W={n.W}
            values={trace.a}
            thumbs={thumbs}
            selected={sel}
            onSelect={setSel}
            dead={dead}
            badges={badges}
            inputLabels={['x', 'y']}
            width={400}
            height={Math.max(220, Math.max(...sizes) * 50 + 40)}
            showValues={!showImp}
            node={30}
            maxAbsW={4}
            ariaLabel="The network: click a neuron or connection to inspect it"
          />
          <div className="btn-row" style={{ marginTop: 6 }}>
            <button className={`btn${running ? '' : ' primary'}`} onClick={() => (running ? setRunning(false) : startTraining())}>
              {running ? '❚❚ Pause' : steps ? '▶ Keep training' : '▶ Train'}
            </button>
            <span className="seg" style={{ alignItems: 'center' }} aria-label="random start">
              <button onClick={() => setSeed(Math.max(1, seed - 1))} aria-label="previous random start">‹</button>
              <span className="mono" style={{ padding: '0 6px', fontSize: 12.5 }}>start #{seed}</span>
              <button onClick={() => setSeed(seed + 1)} aria-label="next random start">›</button>
            </span>
            <Toggle label="Show importance" checked={showImp} onChange={setShowImp} />
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
        {showImp && <span className="muted"> Magenta badges: accuracy lost if that neuron alone is switched off.</span>}
      </div>
      <div className="widget-row" style={{ marginTop: 12 }}>
        <div className="widget-col" style={{ flex: '1 1 330px', fontSize: 13.5, border: '1px solid var(--border)', borderRadius: 10, padding: '10px 12px' }}>
          <div className="field-label" style={{ color: 'var(--c-grad)' }}>Inspector</div>
          {inspector}
        </div>
        <div className="widget-col" style={{ flex: '1 1 330px' }}>
          <div className="field-label">Every tensor, at the probe point ({fmt(probe[0])}, {fmt(probe[1])})</div>
          <div className="readout" style={{ fontSize: 12 }}>
            <span className="k">a⁰ = x </span> [{trace.a[0].map((v) => fmt(v)).join(', ')}]
            {trace.z.map((z, l) => (
              <span key={l}>
                {'\n'}
                <span className="k">z{sup(l + 1)}     </span> [{z.map((v) => fmt(v)).join(', ')}]{'\n'}
                <span className="k">a{sup(l + 1)}     </span> [{trace.a[l + 1].map((v, j) => (m[l + 1][j] ? fmt(v) : '·off·')).join(', ')}]
                {l === trace.z.length - 1 && <span className="k">  ← P(orange)</span>}
              </span>
            ))}
          </div>
          <div className="field-label" style={{ marginTop: 10 }}>Parameters</div>
          <div className="readout" style={{ fontSize: 12 }}>
            {paramBreakdown(sizes).map((p, l) => (
              <span key={l}>
                <span className="k">layer {l + 1}: </span>
                {p.nin} × {p.nout} + {p.nout} = {p.weights + p.biases}
                {'\n'}
              </span>
            ))}
            <span className="k">total:   </span> <strong>{paramTotal(sizes)}</strong>
            {'\n'}
            <span className="k">steps {steps} · loss {fmt(loss, 3)} · accuracy {pct(acc)}</span>
          </div>
        </div>
      </div>
      <PruneBox
        ready={!running && accFull >= 0.95}
        kept={keptHidden}
        total={totalHidden}
        acc={acc}
        best={bestKept}
        optimum={optimum}
        onCompute={() => {
          setOptimum('pending');
          setTimeout(() => setOptimum(minimalNeurons(n, data, 0.95)), 20);
        }}
        onShow={(mk) => {
          revealed.current = true;
          setMask(mk);
        }}
        tooMany={totalHidden > 14}
      />
    </Widget>
  );
}

const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (k: number) => String(k).split('').map((d) => SUP[Number(d)]).join('');

function Bar({ label, v, max, title }: { label: string; v: number; max: number; title?: string }) {
  const w = (Math.abs(v) / max) * 50;
  return (
    <>
      <span title={title}>{label}</span>
      <div style={{ position: 'relative', height: 12, background: 'var(--bg-sunken)', borderRadius: 3 }} title={title}>
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: v >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, background: v >= 0 ? 'var(--c-pos)' : 'var(--c-neg)', borderRadius: 2 }} />
        <div style={{ position: 'absolute', top: -2, bottom: -2, left: '50%', width: 1, background: 'var(--border-strong)' }} />
      </div>
      <span style={{ textAlign: 'right' }}>{fmt(v)}</span>
    </>
  );
}

function PruneBox({
  ready,
  kept,
  total,
  acc,
  best,
  optimum,
  onCompute,
  onShow,
  tooMany,
}: {
  ready: boolean;
  kept: number;
  total: number;
  acc: number;
  best: number | null;
  optimum: ReturnType<typeof minimalNeurons> | 'pending' | undefined;
  onCompute: () => void;
  onShow: (m: Mask) => void;
  tooMany: boolean;
}) {
  return (
    <div style={{ marginTop: 14, border: '1px solid var(--border)', borderRadius: 10, padding: '10px 14px', background: 'var(--bg)' }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>Challenge: prune it</div>
      {!ready ? (
        <div className="muted" style={{ fontSize: 13.5 }}>
          Train a network to at least 95% accuracy, then try to switch off as many hidden neurons as you can while keeping accuracy at 95% or above. No retraining
          allowed — you’re finding out how much of what it learned it actually needs.
        </div>
      ) : (
        <div style={{ fontSize: 13.5 }}>
          Neurons still on: <strong>{kept}</strong> of {total} · accuracy <strong className={acc >= 0.95 ? 'c-good' : 'c-bad'}>{pct(acc)}</strong>
          {acc >= 0.95 ? ' ✓' : ' ✕ (below 95%)'} · your best so far: <strong>{best ?? '—'}</strong>
          <div style={{ marginTop: 6 }}>
            {tooMany ? (
              <span className="muted">This network has more than 14 hidden neurons — too many combinations to check them all here.</span>
            ) : optimum === undefined ? (
              <button className="btn small" onClick={onCompute} disabled={best === null}>
                {best === null ? 'Find a solution first (≥ 95% with something switched off)' : 'Find the true minimum'}
              </button>
            ) : optimum === 'pending' ? (
              <span className="muted">Checking every combination…</span>
            ) : optimum === null ? (
              <span className="muted">No combination keeps 95%.</span>
            ) : (
              <span>
                Fewest neurons that keep 95% for this trained network: <strong>{optimum.count}</strong> (found by trying {optimum.evaluated.toLocaleString()} combinations,
                smallest first, out of {(2 ** total).toLocaleString()}).{' '}
                {best !== null && best <= optimum.count ? 'You found the optimum.' : ''}{' '}
                <button className="btn small" onClick={() => onShow(optimum.mask)}>
                  Show it
                </button>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
