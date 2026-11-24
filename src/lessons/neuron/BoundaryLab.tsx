import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Arrow, Clip, Handle, Label, Plot, Seg, usePlot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import { createRng } from '../../lib/rng';
import { sigmoid } from '../../lib/activations';

type Pt = { x: number; y: number; label: 0 | 1 };

function cluster(rng: ReturnType<typeof createRng>, cx: number, cy: number, n: number, label: 0 | 1, s = 0.45): Pt[] {
  return Array.from({ length: n }, () => ({ x: rng.normal(cx, s), y: rng.normal(cy, s), label }));
}

const DATASETS: Record<string, { label: string; points: Pt[]; story: string }> = (() => {
  const r = createRng(11);
  return {
    emails: {
      label: 'Spam emails',
      story: 'The ten emails from the vectors lesson: x = links, y = exclamation marks. Orange is spam.',
      points: [
        [0, 1, 0], [1, 0, 0], [2, 1, 0], [0, 3, 0], [3, 0, 0], [3, 6, 1], [5, 3, 1], [4, 2, 1], [2, 6, 1], [6, 2, 1],
      ].map(([x, y, l]) => ({ x, y, label: l as 0 | 1 })),
    },
    offset: {
      label: 'Two clusters',
      story: 'Two clouds of points, far from the origin. You’ll need the bias to get the line between them.',
      points: [...cluster(r, 4.5, 4.8, 14, 1, 0.6), ...cluster(r, 2.2, 2.0, 14, 0, 0.6)],
    },
    xor: {
      label: 'XOR',
      story: 'Orange in two opposite corners, blue in the other two. Try to separate them with one straight line.',
      points: [...cluster(r, 1, 1, 8, 0), ...cluster(r, 5, 5, 8, 0), ...cluster(r, 1, 5, 8, 1), ...cluster(r, 5, 1, 8, 1)],
    },
  };
})();
type Key = keyof typeof DATASETS;

function Points({ pts, w, b }: { pts: Pt[]; w: number[]; b: number }) {
  const { x, y } = usePlot();
  return (
    <g>
      {pts.map((p, i) => {
        const pred = w[0] * p.x + w[1] * p.y + b > 0 ? 1 : 0;
        const wrong = pred !== p.label;
        return (
          <g key={i}>
            <circle cx={x(p.x)} cy={y(p.y)} r={6.5} fill={p.label ? 'var(--c-b)' : 'var(--c-a)'} stroke={wrong ? 'var(--text)' : '#fff'} strokeWidth={wrong ? 2.5 : 1.5} />
            {wrong && <circle cx={x(p.x)} cy={y(p.y)} r={11} fill="none" stroke="var(--c-bad)" strokeWidth={2} />}
          </g>
        );
      })}
    </g>
  );
}

export function BoundaryLab() {
  const [key, setKey] = useState<Key>('emails');
  const pts = DATASETS[key].points;
  const [w, setW] = useState([0.3, 1.0]);
  const [b, setB] = useState(-2.5);
  const [training, setTraining] = useState(false);
  const [steps, setSteps] = useState(0);
  const raf = useRef(0);
  const state = useRef({ w, b });
  state.current = { w, b };

  const stats = useMemo(() => {
    let loss = 0;
    let correct = 0;
    for (const p of pts) {
      const z = w[0] * p.x + w[1] * p.y + b;
      const a = sigmoid(z);
      loss += -(p.label * Math.log(Math.max(a, 1e-12)) + (1 - p.label) * Math.log(Math.max(1 - a, 1e-12)));
      if ((z > 0 ? 1 : 0) === p.label) correct++;
    }
    return { loss: loss / pts.length, correct };
  }, [pts, w, b]);

  useEffect(() => {
    if (!training) return;
    const tick = () => {
      let { w: cw, b: cb } = state.current;
      for (let s = 0; s < 4; s++) {
        let gw0 = 0, gw1 = 0, gb = 0;
        for (const p of pts) {
          const d = sigmoid(cw[0] * p.x + cw[1] * p.y + cb) - p.label; // ∂loss/∂z for sigmoid + cross-entropy
          gw0 += d * p.x;
          gw1 += d * p.y;
          gb += d;
        }
        const n = pts.length;
        const lr = 0.08;
        cw = [cw[0] - (lr * gw0) / n, cw[1] - (lr * gw1) / n];
        cb = cb - (lr * gb) / n;
      }
      setW(cw);
      setB(cb);
      setSteps((s) => s + 4);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [training, pts]);

  // Anchor: the point on the boundary closest to the data centre — the weight arrow is drawn from here.
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const wn2 = w[0] * w[0] + w[1] * w[1] || 1e-9;
  const off = (w[0] * cx + w[1] * cy + b) / wn2;
  const anchor: [number, number] = [cx - off * w[0], cy - off * w[1]];
  const dir: [number, number] = [-w[1], w[0]];
  const L = 30;
  const f = useMemo(() => (px: number, py: number) => sigmoid(w[0] * px + w[1] * py + b), [w, b]);
  const cmap = useMemo(() => (t: number) => diverging(1 - t), []);
  const all = stats.correct === pts.length;

  const reset = () => {
    setTraining(false);
    setW([0.3, 1.0]);
    setB(-2.5);
    setSteps(0);
  };

  return (
    <Widget
      title="A neuron draws a line"
      wide
      caption={
        <>
          The background shows the neuron’s output σ(w·x + b) everywhere: <span className="c-b">orange</span> means “class 1”, <span className="c-a">blue</span>{' '}
          “class 0”. The dark line is the <strong>decision boundary</strong>, where z = 0. Drag the <strong>arrow tip</strong> to set the weights (it always points
          towards class 1, perpendicular to the line) and the <strong>square</strong> to slide the line. Misclassified points are ringed in red.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 6 }}>
        <Segmented label="Dataset" value={key} onChange={(k) => { setKey(k); reset(); }} options={(Object.keys(DATASETS) as Key[]).map((k) => ({ value: k, label: DATASETS[k].label }))} />
      </div>
      <p className="muted" style={{ fontSize: 13.5, margin: '0 0 10px' }}>{DATASETS[key].story}</p>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={[-1, 7]} y={[-1, 7]} width={440} height={440} grid={false} ariaLabel="Decision boundary of a neuron over a dataset">
            <Heatmap f={f} range={[0, 1]} colormap={cmap} opacity={0.6} resolution={120} />
            <Clip>
              <Seg from={[anchor[0] - L * dir[0], anchor[1] - L * dir[1]]} to={[anchor[0] + L * dir[0], anchor[1] + L * dir[1]]} color="var(--text)" width={2.5} />
            </Clip>
            <Points pts={pts} w={w} b={b} />
            <Arrow from={anchor} to={[anchor[0] + w[0], anchor[1] + w[1]]} color="var(--accent)" width={3} />
            <Label at={[anchor[0] + w[0], anchor[1] + w[1]]} dx={8} dy={-6} color="var(--accent)">w</Label>
            <Handle
              x={anchor[0] + w[0]}
              y={anchor[1] + w[1]}
              onDrag={(nx, ny) => {
                setTraining(false);
                const nw = [nx - anchor[0], ny - anchor[1]];
                setW(nw);
                setB(-(nw[0] * anchor[0] + nw[1] * anchor[1]));
              }}
              color="var(--accent)"
              label="tip of the weight vector"
            />
            <AnchorHandle
              at={anchor}
              onDrag={(nx, ny) => {
                setTraining(false);
                setB(-(w[0] * nx + w[1] * ny));
              }}
            />
          </Plot>
        </div>
        <div className="widget-col controls">
          <div className="controls inline">
            <Slider label="w₁" value={w[0]} min={-4} max={4} step={0.01} onChange={(v) => { setTraining(false); setW([v, w[1]]); }} />
            <Slider label="w₂" value={w[1]} min={-4} max={4} step={0.01} onChange={(v) => { setTraining(false); setW([w[0], v]); }} />
            <Slider label="bias b" value={b} min={-20} max={20} step={0.05} onChange={(v) => { setTraining(false); setB(v); }} />
          </div>
          <div className="readout">
            <span className="k">accuracy </span> <strong className={all ? 'c-good' : ''}>{stats.correct} / {pts.length}</strong>
            <br />
            <span className="k">avg loss </span> {stats.loss.toFixed(3)} <span className="faint">(cross-entropy)</span>
            <br />
            <span className="k">|w|      </span> {Math.sqrt(wn2).toFixed(2)} <span className="faint">(sharpness of the transition)</span>
            {steps > 0 && (
              <>
                <br />
                <span className="k">training steps</span> {steps}
              </>
            )}
          </div>
          <div className="btn-row">
            <button className={`btn small${training ? '' : ' primary'}`} onClick={() => setTraining(!training)}>
              {training ? '❚❚ Pause training' : '▶ Let gradient descent do it'}
            </button>
            <button className="btn small" onClick={reset}>Reset</button>
          </div>
          <div className="live-note">
            {key === 'xor' && stats.correct >= pts.length * 0.7 ? (
              <>
                <strong>{stats.correct} of {pts.length}.</strong> That’s about as good as any straight line can do here. However you place it, at least one corner
                ends up on the wrong side. Even gradient descent can’t fix this: the problem isn’t the weights, it’s that a single neuron can only draw a line.
              </>
            ) : all ? (
              <>
                <strong>Every point is on the correct side.</strong> {training || steps > 0 ? 'Notice that training keeps going even now: it keeps growing |w| to make the predictions more confident, which keeps lowering the cross-entropy.' : 'Now try “Let gradient descent do it” from a bad starting line, and watch it find a boundary on its own.'}
              </>
            ) : (
              <>
                {pts.length - stats.correct} point{pts.length - stats.correct === 1 ? ' is' : 's are'} on the wrong side. Rotate the arrow to turn the line; drag the
                square to slide it.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

function AnchorHandle({ at, onDrag }: { at: [number, number]; onDrag: (x: number, y: number) => void }) {
  const { x, y, svg } = usePlot();
  const move = (e: React.PointerEvent) => {
    const pt = svg.current!.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.current!.getScreenCTM()!.inverse());
    onDrag(x.inv(p.x), y.inv(p.y));
  };
  return (
    <rect
      x={x(at[0]) - 7}
      y={y(at[1]) - 7}
      width={14}
      height={14}
      rx={2}
      fill="var(--bg-elev)"
      stroke="var(--text)"
      strokeWidth={2.5}
      className="handle"
      role="slider"
      aria-label="slide the decision boundary"
      tabIndex={0}
      onPointerDown={(e) => {
        e.stopPropagation();
        (e.currentTarget as Element).setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        if ((e.currentTarget as Element).hasPointerCapture(e.pointerId)) move(e);
      }}
    />
  );
}
