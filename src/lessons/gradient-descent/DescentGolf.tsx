import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider, Toggle } from '../../components/controls';
import { Clip, Dot, Handle, Plot, usePlot } from '../../components/plot';
import { Heatmap } from '../../components/heatmap';
import { VALLEY_TARGET, par, runValley, valley } from './descent';

const START: [number, number] = [-9, 6];

function Path({ pts, color }: { pts: [number, number][]; color: string }) {
  const { x, y } = usePlot();
  return (
    <polyline
      points={pts.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ')}
      fill="none"
      stroke={color}
      strokeWidth={2}
      strokeLinejoin="round"
    />
  );
}

function Target() {
  const { x, y } = usePlot();
  // valley(x, y) < T is an ellipse with semi-axes sqrt(T/0.02) and sqrt(T/0.5)
  const rx = Math.sqrt(VALLEY_TARGET / 0.02);
  const ry = Math.sqrt(VALLEY_TARGET / 0.5);
  return <ellipse cx={x(0)} cy={y(0)} rx={x(rx) - x(0)} ry={y(0) - y(ry)} fill="#fff" fillOpacity={0.35} stroke="#fff" strokeWidth={2} strokeDasharray="4 3" />;
}

export function DescentGolf() {
  const [lr, setLr] = useState(0.5);
  const [mom, setMom] = useState(0);
  const [useMom, setUseMom] = useState(false);
  const [start, setStart] = useState<[number, number]>(START);
  const [shown, setShown] = useState(0); // animation: how many path points are visible
  const [best, setBest] = useState<number | null>(null);
  const raf = useRef(0);

  const m = useMom ? mom : 0;
  const result = useMemo(() => runValley(start, lr, m), [start, lr, m]);
  const parPlain = useMemo(() => par(start, false), [start]);
  const parMom = useMemo(() => par(start, true), [start]);

  // Animate the path whenever the settings change.
  useEffect(() => {
    cancelAnimationFrame(raf.current);
    setShown(1);
    let i = 1;
    const tick = () => {
      i += Math.max(1, Math.floor(result.path.length / 60));
      setShown(Math.min(i, result.path.length));
      if (i < result.path.length) raf.current = requestAnimationFrame(tick);
      else if (result.reached) setBest((b) => (b === null ? result.steps : Math.min(b, result.steps)));
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [result]);

  const pts = result.path.slice(0, shown).map(([a, b]) => [Math.max(-12, Math.min(12, a)), Math.max(-12, Math.min(12, b))] as [number, number]);
  const done = shown >= result.path.length;
  const theParr = useMom ? parMom : parPlain;

  let verdict: React.ReactNode;
  if (!done) verdict = <span className="status neutral">running…</span>;
  else if (result.diverged)
    verdict = (
      <div className="feedback bad">
        <div className="feedback-title">✕ Flew off the course after {result.steps} steps.</div>
        Look at the first few steps: they bounce across the narrow direction of the valley, each bounce bigger than the last. The steep sides set a hard limit on the step
        size.
      </div>
    );
  else if (!result.reached)
    verdict = (
      <div className="feedback bad">
        <div className="feedback-title">Still not there after {result.steps} steps.</div>
        The path is crawling along the gentle floor of the valley. A bigger step would help — up to the point where the steep sides start to bounce it out.
      </div>
    );
  else
    verdict = (
      <div className="feedback good">
        <div className="feedback-title">✓ Reached the valley floor in {result.steps} steps.</div>
        <div className="muted">
          {result.steps <= theParr.steps
            ? `That matches the best possible${useMom ? ' with momentum' : ' for plain gradient descent'} from this start. You’ve found the sweet spot.`
            : `Optional: the best possible ${useMom ? 'with momentum' : 'for plain gradient descent'} from this start is ${theParr.steps} steps. Can you get there?`}
          {!useMom && result.steps <= parPlain.steps + 3 && ' Now switch on momentum and see how far below that you can go.'}
        </div>
      </div>
    );

  return (
    <Widget
      title="Descent golf"
      wide
      caption={
        <>
          The landscape is a long, narrow valley: steep from side to side, almost flat along its length — very common in real models. Starting from the white handle,
          gradient descent must get into the dashed target at the bottom. <strong>Goal:</strong> get there at all. <strong>Then, optionally:</strong> get there in as
          few steps as possible. Every change re-runs the descent from the start.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={[-11, 11]} y={[-11, 11]} width={440} height={440} grid={false} tickLabels={false} ariaLabel="Narrow valley with a gradient descent path">
            <Heatmap f={valley} range={[0, 70]} contours={16} transform={Math.sqrt} />
            <Target />
            <Clip>
              <Path pts={pts} color="#fff" />
              {pts.map((p, i) => (
                <Dot key={i} at={p} r={2.5} color="#fff" />
              ))}
            </Clip>
            <Handle x={start[0]} y={start[1]} onDrag={(a, b) => setStart([Math.round(a * 2) / 2, Math.round(b * 2) / 2])} color="var(--accent)" label="starting point" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <Slider label="step size (learning rate)" value={lr} min={0.05} max={2.2} step={0.01} onChange={setLr} color="c-grad" />
          <Toggle label="Use momentum (a sneak preview — covered properly later)" checked={useMom} onChange={setUseMom} />
          {useMom && <Slider label="momentum (how much velocity is kept)" value={mom} min={0} max={0.95} step={0.05} onChange={setMom} />}
          <div className="readout">
            <span className="k">steps taken   </span> <strong>{done ? result.steps : shown - 1}</strong>
            <br />
            <span className="k">your best     </span> {best ?? '—'}
            <br />
            <span className="k">par, plain GD </span> {parPlain.steps}
            {useMom && (
              <>
                <br />
                <span className="k">par, momentum </span> {parMom.steps}
              </>
            )}
          </div>
          {verdict}
          {start[0] !== START[0] || start[1] !== START[1] ? (
            <button className="btn small" onClick={() => setStart(START)}>Reset start to the standard tee</button>
          ) : null}
        </div>
      </div>
    </Widget>
  );
}
