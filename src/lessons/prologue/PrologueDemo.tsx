// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Plot, usePlot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import { MAX_STEPS, STEPS_PER_FRAME, accuracy, createDemo, lossOf, makeCircles, train, type Demo } from './prologue';

const DOMAIN = 1.6;
const DATA = makeCircles();
const colour = (t: number) => diverging(1 - t);

function Points({ demo, version }: { demo: Demo; version: number }) {
  const { x, y } = usePlot();
  void version;
  return (
    <g>
      {demo.data.X.map((p, k) => {
        const wrong = (demo.net.predict(p)[0] > 0.5 ? 1 : 0) !== demo.data.y[k];
        return (
          <g key={k}>
            <circle cx={x(p[0])} cy={y(p[1])} r={4} fill={demo.data.y[k] ? 'var(--c-b)' : 'var(--c-a)'} stroke="var(--bg)" strokeWidth={1} />
            {wrong && <circle cx={x(p[0])} cy={y(p[1])} r={7} fill="none" stroke="var(--c-bad)" strokeWidth={1.6} />}
          </g>
        );
      })}
    </g>
  );
}

/** Loss history as a small line chart (log scale on the loss). */
function LossCurve({ history }: { history: number[] }) {
  const W = 260;
  const H = 70;
  if (history.length < 2) {
    return <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Loss curve (empty until you train)" style={{ maxWidth: W }}><rect x={0.5} y={0.5} width={W - 1} height={H - 1} fill="none" stroke="var(--border)" rx={6} /></svg>;
  }
  const lo = Math.log(0.005);
  const hi = Math.log(1);
  const px = (i: number) => 6 + (i / (MAX_STEPS / STEPS_PER_FRAME)) * (W - 12);
  const py = (v: number) => 6 + ((hi - Math.log(Math.max(v, 0.005))) / (hi - lo)) * (H - 12);
  const d = history.map((v, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join('');
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Loss curve: falling as the network learns" style={{ maxWidth: W }}>
      <rect x={0.5} y={0.5} width={W - 1} height={H - 1} fill="none" stroke="var(--border)" rx={6} />
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth={2} />
    </svg>
  );
}

export function PrologueDemo() {
  const [seed, setSeed] = useState(1);
  const demo = useRef<Demo>(createDemo(1, DATA));
  const [version, setVersion] = useState(0);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<number[]>([]);

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const tick = () => {
      const d = demo.current;
      train(d, STEPS_PER_FRAME);
      const l = lossOf(d);
      setHistory((h) => [...h, l]);
      setVersion((v) => v + 1);
      if (d.steps >= MAX_STEPS) {
        setRunning(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  const reset = () => {
    const next = seed + 1;
    setRunning(false);
    setSeed(next);
    demo.current = createDemo(next, DATA);
    setHistory([]);
    setVersion((v) => v + 1);
  };

  const d = demo.current;
  const loss = useMemo(() => lossOf(d), [d, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const acc = useMemo(() => accuracy(d), [d, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const f = useMemo(() => (x: number, y: number) => d.net.predict([x, y])[0], [d, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const wrong = Math.round((1 - acc) * d.data.X.length);

  const note =
    d.steps === 0
      ? `A fresh network (random start #${seed}). Its guesses are the shading behind the points: almost no pattern yet, and ${wrong} of the ${d.data.X.length} points are on the wrong side (ringed).`
      : acc < 1
        ? `Step ${d.steps}: ${wrong} point${wrong === 1 ? '' : 's'} still on the wrong side. Each step nudges all ${d.net.paramCount} numbers inside the network a little, in whichever direction makes the loss smaller.`
        : `Step ${d.steps}: every point is on the right side. Nobody told the network "draw a circle"; it bent its boundary until the mistakes stopped. Training on makes it more confident, so the loss keeps falling.`;

  return (
    <Widget
      title="A learning machine, learning"
      caption={
        <>
          Press <strong>Train</strong> and watch the boundary form. Then press <strong>Reset</strong> and train again: each reset starts the network from different random numbers, so the boundary takes a different path, but it ends up separating the two groups every time.
        </>
      }
    >
      <div className="widget-row" style={{ gap: 16, alignItems: 'flex-start' }}>
        <div className="widget-col" style={{ flex: '0 1 300px' }}>
          <Plot x={[-DOMAIN, DOMAIN]} y={[-DOMAIN, DOMAIN]} width={290} height={290} grid={false} tickLabels={false} ariaLabel="Two groups of points, inner disc and outer ring, with the network's answer shaded behind them">
            <Heatmap f={f} range={[0, 1]} colormap={colour} resolution={60} opacity={0.7} />
            <Points demo={d} version={version} />
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 220px', minWidth: 0 }}>
          <div className="readout" style={{ fontSize: 12.5 }}>
            <span className="k">step      </span>
            {d.steps}
            {'\n'}
            <span className="k">loss      </span>
            {loss.toFixed(3)}
            {'\n'}
            <span className="k">accuracy  </span>
            {(acc * 100).toFixed(1)}%
            {'\n'}
            <span className="k">numbers   </span>
            {d.net.paramCount} (weights and biases)
          </div>
          <div className="field-label" style={{ marginTop: 10 }}>Loss over time</div>
          <LossCurve history={history} />
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button className={`btn${running ? '' : ' primary'}`} onClick={() => setRunning(!running)} disabled={!running && d.steps >= MAX_STEPS}>
              {running ? '❚❚ Pause' : d.steps ? '▶ Keep training' : '▶ Train'}
            </button>
            <button className="btn" onClick={reset}>
              ↺ Reset
            </button>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}
