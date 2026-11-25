import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Dot, FnPath, Handle, Label, Plot, Seg } from '../../components/plot';
import { Tex, col, fmt } from '../../components/Tex';
import { useFramePlayer } from '../../components/motion';

/** The 'shrink the nudge' animation: h goes from 1 to 0.001 on a log scale, frame by frame. */
export const SHRINK_FRAMES = 60;
export const shrinkLogH = (i: number) => (-3 * i) / SHRINK_FRAMES;

type Fn = { label: string; tex: string; dtex: string; f: (x: number) => number; df: (x: number) => number; y: [number, number] };

export const FUNCTIONS: Record<string, Fn> = {
  square: { label: 'x²', tex: 'f(x) = x^2', dtex: "f'(x) = 2x", f: (x) => x * x, df: (x) => 2 * x, y: [-2, 8] },
  cubic: { label: 'x³ − 3x', tex: 'f(x) = x^3 - 3x', dtex: "f'(x) = 3x^2 - 3", f: (x) => x ** 3 - 3 * x, df: (x) => 3 * x * x - 3, y: [-5, 5] },
  sine: { label: 'sin x', tex: 'f(x) = \\sin x', dtex: "f'(x) = \\cos x", f: Math.sin, df: Math.cos, y: [-2.5, 2.5] },
  bumpy: {
    label: 'a “loss curve”',
    tex: 'f(x) = 0.3x^2 + \\sin(2x) + 1',
    dtex: "f'(x) = 0.6x + 2\\cos(2x)",
    f: (x) => 0.3 * x * x + Math.sin(2 * x) + 1,
    df: (x) => 0.6 * x + 2 * Math.cos(2 * x),
    y: [-1, 6],
  },
  relu: { label: 'ReLU (a kink)', tex: 'f(x) = \\max(0, x)', dtex: "f'(x) = \\begin{cases}1 & x>0\\\\0 & x<0\\end{cases}", f: (x) => Math.max(0, x), df: (x) => (x > 0 ? 1 : 0), y: [-1.5, 3.5] },
};

export function TangentExplorer() {
  const [key, setKey] = useState<keyof typeof FUNCTIONS>('square');
  const fn = FUNCTIONS[key];
  const [x0, setX0] = useState(1);
  const [logH, setLogH] = useState(0); // h = 10^logH
  const [zoom, setZoom] = useState(1);
  const [showDeriv, setShowDeriv] = useState(true);

  const player = useFramePlayer(SHRINK_FRAMES, 50, (i) => setLogH(shrinkLogH(i)));
  const playShrink = () => {
    setZoom(1);
    setLogH(0);
    player.start();
  };

  const h = 10 ** logH;
  const y0 = fn.f(x0);
  const y1 = fn.f(x0 + h);
  const secant = (y1 - y0) / h; // computed in floating point, exactly as code would
  const slope = fn.df(x0);
  const err = Math.abs(secant - slope);

  const span = 3 / zoom;
  const xd: [number, number] = zoom === 1 ? [-3, 3] : [x0 - span, x0 + span];
  const yd: [number, number] = zoom === 1 ? fn.y : [y0 - span * ((fn.y[1] - fn.y[0]) / 6), y0 + span * ((fn.y[1] - fn.y[0]) / 6)];
  const L = 100;

  const kink = key === 'relu' && Math.abs(x0) < 0.01;
  let note: React.ReactNode;
  if (player.running)
    note = (
      <>
        <strong>h = {h >= 0.01 ? h.toFixed(3) : h.toExponential(1)}</strong>: secant slope <span className="c-grad">{secant.toFixed(4)}</span> → true slope{' '}
        <span className="c-c">{slope.toFixed(4)}</span>. The dashed line is turning into the tangent as the second point slides in.
      </>
    );
  else if (kink)
    note = (
      <>
        <strong>Right on the kink.</strong> Nudge right and the slope is 1; nudge left and it’s 0. There’s no single tangent line here, so strictly speaking the
        derivative doesn’t exist at exactly 0. In practice, libraries just pick a value (usually 0) and move on.
      </>
    );
  else if (logH < -11)
    note = (
      <>
        <strong>The nudge is now so small that floating-point rounding dominates.</strong> f(x+h) and f(x) agree in almost every digit, so subtracting them
        throws away most of the precision. Mathematically smaller h is better; numerically, there’s a sweet spot around h ≈ 10⁻⁶ to 10⁻⁸.
      </>
    );
  else if (logH > -1.5)
    note = (
      <>
        The <span className="c-grad">secant line</span> through two points {h.toFixed(2)} apart has slope {secant.toFixed(3)}, while the true slope at the
        point is {slope.toFixed(3)}. <strong>Shrink the nudge</strong> and watch the two lines merge.
      </>
    );
  else
    note = (
      <>
        With a nudge of h = {h.toExponential(0)}, the secant slope ({secant.toFixed(5)}) matches the true derivative ({slope.toFixed(5)}) to within{' '}
        {err.toExponential(1)}. Zoomed in this close, the curve is indistinguishable from a straight line. {slope > 0 ? 'Increasing x increases f.' : slope < 0 ? 'Increasing x decreases f.' : 'The curve is flat here.'}
      </>
    );

  return (
    <Widget
      title="Zooming in on a curve"
      wide
      caption={
        <>
          Drag the point along the curve. The <span className="c-grad">dashed line</span> joins the point to a second point a distance <em>h</em> to the
          right; its slope is “change in output ÷ change in input”. Shrink <em>h</em> and that slope settles down to the slope of the{' '}
          <span className="c-c">tangent line</span> — the derivative. Then try the zoom: any smooth curve looks straight if you zoom in far enough.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="Function" value={key} onChange={(k) => { setKey(k); setZoom(1); }} options={Object.entries(FUNCTIONS).map(([k, v]) => ({ value: k as keyof typeof FUNCTIONS, label: v.label }))} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={xd} y={yd} width={460} height={340} ariaLabel="Function with tangent and secant lines">
            <FnPath f={fn.f} color="var(--c-a)" width={3} />
            <Seg from={[x0 - L, y0 - L * slope]} to={[x0 + L, y0 + L * slope]} color="var(--c-c)" width={2} />
            {Number.isFinite(secant) && <Seg from={[x0 - L, y0 - L * secant]} to={[x0 + L, y0 + L * secant]} color="var(--c-grad)" width={2} dash="6 5" />}
            {h > 0.02 * (xd[1] - xd[0]) && Number.isFinite(y1) && <Dot at={[x0 + h, y1]} r={5} color="var(--c-grad)" />}
            <Label at={[x0, y0]} dx={12} dy={22} color="var(--text-muted)" size={12}>{`(${x0.toFixed(2)}, ${y0.toFixed(2)})`}</Label>
            <Handle x={x0} y={y0} onDrag={(x) => setX0(Math.round(x * 100) / 100)} color="var(--c-a)" label="point on the curve" />
          </Plot>
          {showDeriv && zoom === 1 && (
            <>
              <div className="field-label" style={{ marginTop: 8 }}>The slope at every point — the derivative f′(x)</div>
              <Plot x={xd} y={[-4, 4]} width={460} height={150} ariaLabel="Derivative curve">
                <FnPath f={fn.df} color="var(--c-c)" width={2.5} samples={600} />
                <Seg from={[x0, -4]} to={[x0, 4]} dash="3 3" />
                <Handle x={x0} y={Math.max(-4, Math.min(4, slope))} onDrag={(x) => setX0(Math.round(x * 100) / 100)} color="var(--c-c)" r={6} label="same point on the derivative curve" />
              </Plot>
            </>
          )}
        </div>
        <div className="widget-col controls">
          <div style={{ fontSize: 15 }}>
            <Tex block>{fn.tex}</Tex>
            <Tex block>{col('c-c', fn.dtex)}</Tex>
          </div>
          <Slider label={<>nudge size <em>h</em></>} value={logH} min={-15} max={0.3} step={0.1} onChange={setLogH} format={() => (h >= 0.01 ? h.toFixed(2) : h.toExponential(0))} color="c-grad" />
          <button type="button" className="btn primary" onClick={playShrink} disabled={player.running}>
            {player.running ? 'Shrinking…' : '▶ Play: shrink the nudge'}
          </button>
          <Segmented label="Zoom" value={zoom} onChange={setZoom} options={[{ value: 1, label: '1×' }, { value: 10, label: '10×' }, { value: 100, label: '100×' }, { value: 1000, label: '1000×' }]} />
          <div style={{ fontSize: 14.5 }}>
            <Tex block>{`\\frac{f(x+h) - f(x)}{h} = \\frac{${fmt(y1, 4)} - ${fmt(y0, 4)}}{${h >= 0.01 ? fmt(h, 2) : `10^{${logH.toFixed(1)}}`}} = ${col('c-grad', Number.isFinite(secant) ? fmt(secant, 4) : '?')}`}</Tex>
            <Tex block>{`f'(${fmt(x0)}) = ${col('c-c', kink ? '\\text{undefined}' : fmt(slope, 4))}`}</Tex>
          </div>
          <div className="live-note">{note}</div>
          <Toggle label="Show derivative curve below" checked={showDeriv} onChange={setShowDeriv} />
        </div>
      </div>
    </Widget>
  );
}
