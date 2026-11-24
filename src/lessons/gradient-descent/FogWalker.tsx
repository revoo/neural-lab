import { useEffect, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Dot, FnPath, Handle, Plot, Seg, Arrow } from '../../components/plot';
import { Tex, fmt, pn } from '../../components/Tex';
import { CURVES, bowlRegime, globalMin } from './descent';

const LOWEST = Object.fromEntries(Object.entries(CURVES).map(([k, c]) => [k, globalMin(c)]));

type Key = keyof typeof CURVES;

export function FogWalker() {
  const [key, setKey] = useState<Key>('bowl');
  const C = CURVES[key];
  const [w, setW] = useState(C.start);
  const [trail, setTrail] = useState<number[]>([C.start]);
  const [lr, setLr] = useState(0.1);
  const [fog, setFog] = useState(true);
  const [running, setRunning] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const state = useRef({ w, lr });
  state.current = { w, lr };

  const reset = (k: Key = key, start = CURVES[k].start) => {
    setRunning(false);
    setW(start);
    setTrail([start]);
  };

  const step = () => {
    const { w: cw, lr: clr } = state.current;
    const nw = cw - clr * C.df(cw);
    if (!Number.isFinite(nw) || Math.abs(nw) > 1e6) {
      setRunning(false);
      return;
    }
    setW(nw);
    setTrail((t) => [...t.slice(-80), nw]);
  };

  useEffect(() => {
    if (!running) return;
    timer.current = window.setInterval(step, 280);
    return () => window.clearInterval(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, key]);

  const slope = C.df(w);
  const loss = C.f(w);
  const stepSize = -lr * slope;
  const [d0, d1] = C.domain;
  const off = w < d0 - 0.5 || w > d1 + 0.5;
  const regime = key === 'bowl' ? bowlRegime(lr) : null;

  let note: React.ReactNode;
  if (off)
    note = (
      <>
        <strong>Off the map.</strong> Each step overshot by more than the last, flinging the walker further and further away. That’s <strong>divergence</strong>: the
        step size is so large that the slope’s advice makes things worse. Reset and try a smaller step size.
      </>
    );
  else if (regime === 'diverging')
    note = (
      <>
        <strong>Step size {lr.toFixed(2)} is too big for this bowl.</strong> Every step overshoots the bottom by <em>more</em> than the distance it started from, so the
        walker swings further out each time. It still knows which way is downhill — it’s just jumping too far.
      </>
    );
  else if (regime === 'bouncing')
    note = (
      <>
        <strong>Exactly 1.0: a perfect bounce.</strong> Each step lands at the mirror-image point on the other side, at exactly the same height. Forever.
      </>
    );
  else if (regime === 'oscillating')
    note = (
      <>
        <strong>Zig-zagging.</strong> The steps overshoot the bottom each time, but by less than before, so it still converges — jumping from side to side as it goes.
      </>
    );
  else if (regime === 'slow')
    note = (
      <>
        <strong>Painfully slow.</strong> Every step goes the right way, but they’re tiny. Fine for a toy; for a model that takes a second per step, unaffordable.
      </>
    );
  else if (Math.abs(slope) < 0.01)
    note = Math.abs(w - LOWEST[key]) > 0.3 ? (
      <>
        <strong>Stuck in a dip.</strong> The ground is flat here, so gradient descent stops — but lift the fog and you’ll see this isn’t the lowest point. It’s a{' '}
        <strong>local minimum</strong>. Gradient descent only ever knows about the ground beneath its feet.
      </>
    ) : (
      <>
        <strong>Arrived.</strong> The slope is (almost) zero, so the steps have shrunk to nothing — the walker has settled at a low point.
      </>
    );
  else
    note = (
      <>
        The slope here is {slope.toFixed(2)}, so the next step moves the walker by −{lr.toFixed(2)} × {pn(slope)} = <strong>{stepSize.toFixed(3)}</strong>. Notice the
        steps shrink as the ground flattens: steep slope, big step; gentle slope, small step.
      </>
    );

  return (
    <Widget
      title="Walking downhill in the fog"
      wide
      caption={
        <>
          You can only see the ground right around your feet. Press <strong>Step</strong> to move by <em>step size × slope</em> in the downhill direction; the trail shows
          where you’ve been. Then press <strong>Run</strong>, and experiment with the step size: try 0.1, 0.45, 0.9, 1.0 and 1.05. Lift the fog when you’re curious
          what the landscape really looked like.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="Landscape" value={key} onChange={(k) => { setKey(k); reset(k); }} options={(Object.keys(CURVES) as Key[]).map((k) => ({ value: k, label: CURVES[k].label }))} />
        <Toggle label="Lift the fog" checked={!fog} onChange={(v) => setFog(!v)} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 440px' }}>
          <Plot x={C.domain} y={C.y} width={480} height={300} xLabel="weight →" yLabel="loss" ariaLabel="Loss curve hidden by fog">
            {!fog && <FnPath f={C.f} color="var(--c-a)" width={2.5} />}
            {fog && !off && <FnPath f={C.f} color="var(--c-a)" width={3.5} domain={[w - 0.45, w + 0.45]} samples={40} />}
            {trail.slice(0, -1).map((t, i) => (
              <Dot key={i} at={[t, C.f(t)]} r={3.5} color="var(--accent)" opacity={0.25 + (0.6 * i) / trail.length} />
            ))}
            {trail.length > 1 &&
              trail.slice(1).map((t, i) => {
                const a = trail[i];
                return Math.abs(t - a) > 0.05 ? <Seg key={'s' + i} from={[a, C.f(a)]} to={[t, C.f(t)]} color="var(--accent)" width={1} opacity={0.35} dash="3 3" /> : null;
              })}
            {!off && <Seg from={[w - 0.9, loss - 0.9 * slope]} to={[w + 0.9, loss + 0.9 * slope]} color="var(--c-c)" width={2} />}
            {!off && Math.abs(stepSize) > 0.05 && <Arrow from={[w, C.y[0] + 0.3]} to={[w + Math.max(-3, Math.min(3, stepSize)), C.y[0] + 0.3]} color="var(--c-grad)" width={2.5} />}
            {!off && <Handle x={w} y={loss} onDrag={(nx) => reset(key, Math.round(nx * 20) / 20)} color="var(--accent)" label="walker position (drag to choose a start)" />}
          </Plot>
        </div>
        <div className="widget-col controls">
          <Slider label="step size (learning rate)" value={lr} min={0.01} max={1.2} step={0.01} onChange={setLr} color="c-grad" />
          <div className="btn-row">
            <button className="btn primary small" onClick={step} disabled={off}>Step</button>
            <button className="btn small" onClick={() => setRunning(!running)} disabled={off}>{running ? '❚❚ Pause' : '▶ Run'}</button>
            <button className="btn small" onClick={() => reset()}>Reset</button>
          </div>
          <div style={{ fontSize: 14.5 }}>
            <Tex block>{`\\underbrace{${fmt(w + stepSize, 3)}}_{\\text{new weight}} = \\underbrace{${fmt(w, 3)}}_{\\text{old weight}} - \\underbrace{${fmt(lr)}}_{\\text{step size}} \\times \\underbrace{${pn(slope, 3)}}_{\\text{slope}}`}</Tex>
          </div>
          <div className="readout">
            <span className="k">loss </span> {off ? '—' : loss.toFixed(4)}
            <br />
            <span className="k">steps</span> {trail.length - 1}
          </div>
          <div className="live-note">{note}</div>
        </div>
      </div>
    </Widget>
  );
}
